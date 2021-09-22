import argparse
from collections import namedtuple
from functools import partial
from pathlib import Path
from time import time

import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from numpyro.contrib.callbacks import Progbar
from numpyro.distributions import Normal, Gamma
from numpyro.contrib.einstein import Stein, RBFKernel

import numpyro
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoDelta, AutoNormal
from numpyro.optim import Adagrad

DATADIR = Path(__file__).parent / 'data'
DataState = namedtuple('data', ['xtr', 'xte', 'ytr', 'yte'])


def load_data(name: str) -> DataState:
    data = np.loadtxt(DATADIR / f'{name}.txt')
    x, y = data[:, :-1], data[:, -1]
    xtr, xte, ytr, yte = train_test_split(x, y, train_size=.90)

    return DataState(*map(partial(jnp.array, dtype=float), (xtr, xte, ytr, yte)))


def normalize(val, mean=None, std=None):
    if mean is None and std is None:
        std = jnp.std(val, 0, keepdims=True)
        std = jnp.where(std == 0, 1., std)
        mean = jnp.mean(val, 0, keepdims=True)
    return (val - mean) / std, mean, std


def model(x, y=None, hidden_dim=50):
    prec_obs = numpyro.sample("prec_obs", Gamma(1., 0.1))
    prec_nn = numpyro.sample("prec_nn", Gamma(1., 0.1))

    n, m = x.shape
    b1 = numpyro.sample('nn_b1', Normal(jnp.zeros(hidden_dim, ), 1. / prec_nn))
    b2 = numpyro.sample('nn_b2', Normal(0., 1. / prec_nn))
    w1 = numpyro.sample('nn_w1', Normal(jnp.zeros((m, hidden_dim)), 1. / prec_nn))
    w2 = numpyro.sample('nn_w2', Normal(jnp.zeros(hidden_dim), 1. / prec_nn))

    with numpyro.plate('data', x.shape[0], dim=-2):
        y_mean = numpyro.deterministic('y_mean', jnp.maximum(x @ w1 + b1, 0) @ w2 + b2)
        numpyro.sample('y', Normal(y_mean, 1.0 / prec_obs), obs=y)


def main(args):
    data = load_data(args.dataset)

    inf_key, pred_key = random.split(random.PRNGKey(args.rng_key), 2)
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

    if args.method == 1:  # SVGD
        pass
    if args.method >= 2:
        if args.method == 2:  # SVGD-EinStein
            stein = Stein(model, AutoDelta(model), Adagrad(1.), Trace_ELBO(), RBFKernel(),
                          num_particles=args.num_particles)

        if args.method == 3:  # SM-EinStein
            stein = Stein(model, AutoNormal(model), Adagrad(1.), Trace_ELBO(), RBFKernel(),
                          num_particles=args.num_particles)

        start = time()
        state, losses = stein.run(inf_key, args.max_iter, x, y, 50)
        print(time() - start)

        plt.plot(losses)
        plt.show()
        pred = Predictive(model, guide=stein.guide, params=stein.get_params(state), num_samples=1,
                          num_particles=args.num_particles if args.method != 0 else None)

    preds = pred(pred_key, normalize(data.xte, xtr_mean, xtr_std)[0])

    y_pred = jnp.mean(preds['y_mean'].reshape(-1, data.yte.shape[0]) * ytr_std.reshape(1, 1) + ytr_mean.reshape(1, 1),
                      0)

    print(jnp.sqrt(jnp.mean((y_pred - data.yte) ** 2)), "±0.")


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
    parser.add_argument('--max_iter', type=int, default=50_000)
    parser.add_argument('--method', type=int, choices=range(5), metavar='[0-4]', default=2)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--num_particles', type=int, default=100)
    parser.add_argument('--rng_key', type=int, default=142)

    args = parser.parse_args()

    numpyro.set_platform('gpu')

    main(args)
