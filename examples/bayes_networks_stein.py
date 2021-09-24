import argparse
from collections import namedtuple
from functools import partial
from pathlib import Path
from random import shuffle
from time import time

import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from numpyro.contrib.callbacks import Progbar
from numpyro.distributions import Normal, Gamma
from numpyro.contrib.einstein import Stein, RBFKernel, IMQKernel

import numpyro
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoDelta, AutoDiagonalNormal
from numpyro.optim import Adagrad

DATADIR = Path(__file__).parent / 'data'
DataState = namedtuple('data', ['xtr', 'xte', 'ytr', 'yte'])


def load_data(name: str) -> DataState:
    data = np.loadtxt(DATADIR / f'{name}.txt')
    x, y = data[:, :-1], data[:, -1]
    xtr, xte, ytr, yte = train_test_split(x, y, train_size=.90)

    return DataState(*map(partial(jnp.array, dtype=float), (xtr, xte, ytr, yte)))


def make_batcher(x, y, rng_key, batch_size=100, ):
    data = np.hstack((x, y.reshape(-1, 1)))
    ds_count = max(data.shape[0] // batch_size, 1)

    def batch_fn(step):
        nonlocal data
        i = step % ds_count
        epoch = step // ds_count
        is_last = i == (ds_count - 1)
        batch = data[i * batch_size: (i + 1) * batch_size]
        if is_last:
            data = shuffle(data)
        return (jnp.array(batch[:, :-1]), jnp.array(batch[:, -1])), {}, epoch, is_last

    return batch_fn


def normalize(val, mean=None, std=None):
    if mean is None and std is None:
        std = jnp.std(val, 0, keepdims=True)
        std = jnp.where(std == 0, 1., std)
        mean = jnp.mean(val, 0, keepdims=True)
    return (val - mean) / std, mean, std


def model(x, y=None, hidden_dim=50, subsample_size=100):
    prec_obs = numpyro.sample("prec_obs", Gamma(1., 0.1))
    prec_nn = numpyro.sample("prec_nn", Gamma(1., 0.1))

    n, m = x.shape
    b1 = numpyro.sample('nn_b1', Normal(jnp.zeros(hidden_dim, ), 1. / prec_nn))
    b2 = numpyro.sample('nn_b2', Normal(0., 1. / prec_nn))
    w1 = numpyro.sample('nn_w1', Normal(jnp.zeros((m, hidden_dim)), 1. / prec_nn))
    w2 = numpyro.sample('nn_w2', Normal(jnp.zeros(hidden_dim), 1. / prec_nn))

    with numpyro.plate('data', x.shape[0], subsample_size=subsample_size, dim=-2):
        batch_x = numpyro.subsample(x, event_dim=1)

        if y is not None:
            batch_y = numpyro.subsample(y, event_dim=0)
        else:
            batch_y = y

        numpyro.sample('y', Normal(jnp.maximum(batch_x @ w1 + b1, 0) @ w2 + b2, 1.0 / prec_obs), obs=batch_y)

def main(args):
    data = load_data(args.dataset)

    inf_key, pred_key, data_key = random.split(random.PRNGKey(args.rng_key), 3)
    x, xtr_mean, xtr_std = normalize(data.xtr)
    y, ytr_mean, ytr_std = normalize(data.ytr)

    if args.method == 0:  # SVI
        svi = SVI(model, AutoDelta(model), Adagrad(.01), Trace_ELBO())

        start = time()
        results = svi.run(inf_key, args.max_iter, x, y, 50)
        print(time() - start)

        plt.plot(results.losses)
        plt.show()
        pred = Predictive(model, guide=svi.guide, params=results.params, num_samples=1)

    if args.method >= 1:
        if args.method == 1:  # SVGD-EinStein
            stein = Stein(model, AutoDelta(model), Adagrad(1.), Trace_ELBO(), RBFKernel(),
                          num_particles=args.num_particles)

        times = []
        scores = []
        for _ in range(21):
            rng_key, inf_key = random.split(inf_key)
            start = time()

            # use keyword params for static (shape etc.)!
            state, losses = stein.run(rng_key, args.max_iter, x, y, hidden_dim=50, subsample_size=args.subsample_size,
                                      callbacks=[Progbar()] if args.progress_bar else None)
            times.append(time() - start)

            # plt.plot(losses)
            # plt.show()
            pred = Predictive(model, guide=stein.guide, params=stein.get_params(state), num_samples=1,
                              num_particles=args.num_particles if args.method != 0 else None)

            preds = pred(pred_key, normalize(data.xte, xtr_mean, xtr_std)[0], subsample_size=data.xte.shape[0])

            y_pred = jnp.mean(
                preds['y'].reshape(-1, data.yte.shape[0]) * ytr_std.reshape(1, 1) + ytr_mean.reshape(1, 1),
                0)

            scores.append(jnp.sqrt(jnp.mean((y_pred - data.yte) ** 2)))

    times = np.array(times)
    scores = np.array(scores)
    print(args.dataset)
    print('all times', times)
    print('all scores', scores)
    print(f'timing {np.mean(times[1:]): .3f}±{np.std(times[1:]):.3f}')
    print(f'rmse {np.mean(scores[1:]):.3f}±{np.std(scores[1:]):.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['boston_housing',
                                              'concrete',
                                              'energy_heating_load',
                                              'kin8nm',
                                              'naval_compressor_decay',
                                              'power',
                                              'protein',
                                              'wine',
                                              'yacht',
                                              'year_prediction_msd'], default='boston_housing')

    parser.add_argument('--subsample_size', type=int, default=1000)
    parser.add_argument('--max_iter', type=int, default=20_000)
    parser.add_argument('--method', type=int, choices=range(2), metavar='[0-1]', default=1)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--num_particles', type=int, default=100)
    parser.add_argument('--progress_bar', type=bool, default=True)
    parser.add_argument('--rng_key', type=int, default=142)

    args = parser.parse_args()

    numpyro.set_platform('gpu')

    main(args)
