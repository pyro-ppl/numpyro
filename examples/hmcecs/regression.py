from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flax import nn
from flax.nn.activation import relu, tanh
from jax import random, vmap

import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.contrib.module import random_flax_module
from numpyro.infer import MCMC, NUTS, init_to_sample

uci_base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'

numpyro.set_platform("gpu")


def visualize(train_data, train_obs, test_data, predictions):
    fs = 14

    m = predictions.mean(0)
    percentiles = np.percentile(predictions, [2.5, 97.5], axis=0)

    f, ax = plt.subplots(1, 1, figsize=(8, 4))

    # Get upper and lower confidence bounds
    lower, upper = (percentiles[0, :]).flatten(), (percentiles[1, :]).flatten()

    # Plot training data as black stars
    ax.plot(train_data, train_obs, 'x', marker='x', color='forestgreen', rasterized=True, label='Observed Data')
    # Plot predictive means as blue line
    ax.plot(test_data, m, 'b', rasterized=True, label="Mean Prediction")
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_data, lower, upper, alpha=0.5, rasterized=True, label='95% C.I.')
    ax.set_ylim([-2.5, 2.5])
    ax.set_xlim([-2, 2])
    plt.grid()
    ax.legend(fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)

    bbox = {'facecolor': 'white', 'alpha': 0.8, 'pad': 1, 'boxstyle': 'round', 'edgecolor': 'black'}

    plt.tight_layout()
    # plt.savefig('plots/full_hmc.pdf', rasterized=True)

    plt.show()


def load_agw_1d(get_feats=False):
    def features(x):
        return np.hstack([x[:, None] / 2.0, (x[:, None] / 2.0) ** 2])

    data = np.load(str(Path(__file__).parent / 'data' / 'data.npy'))
    x, y = data[:, 0], data[:, 1]
    y = y[:, None]
    f = features(x)

    x_means, x_stds = x.mean(axis=0), x.std(axis=0)
    y_means, y_stds = y.mean(axis=0), y.std(axis=0)
    f_means, f_stds = f.mean(axis=0), f.std(axis=0)

    X = ((x - x_means) / x_stds).astype(np.float32)
    Y = ((y - y_means) / y_stds).astype(np.float32)
    F = ((f - f_means) / f_stds).astype(np.float32)

    if get_feats:
        return F, Y

    return X[:, None], Y


def protein():
    # from hughsalimbeni/bayesian_benchmarks
    # N, D, name = 45730, 9, 'protein'
    url = uci_base_url + '00265/CASP.csv'

    data = pd.read_csv(url).values
    return data[:, 1:], data[:, 0].reshape(-1, 1)


class Network(nn.Module):
    def apply(self, x, out_channels):
        l1 = tanh(nn.Dense(x, features=100))
        l2 = tanh(nn.Dense(l1, features=100))
        means = nn.Dense(l2, features=out_channels)
        return means


def nonlin(x):
    return tanh(x)


def model(data, obs=None):
    module = Network.partial(out_channels=1)

    net = random_flax_module('fnn', module, dist.Normal(0, 1.), input_shape=data.shape[1])

    prec_obs = numpyro.sample("prec_obs", dist.LogNormal(jnp.log(110.4), .0001))
    sigma_obs = 1.0 / jnp.sqrt(prec_obs)  # prior

    numpyro.sample('obs', dist.Normal(net(data), sigma_obs), obs=obs)


def hmc(dataset, data, obs, warmup, num_sample):
    kernel = NUTS(model, max_tree_depth=5, step_size=.0005, init_strategy=init_to_sample)
    mcmc = MCMC(kernel, warmup, num_sample)
    mcmc.run(random.PRNGKey(37), data, obs, extra_fields=('num_steps',))
    mcmc.print_summary()
    return mcmc.get_samples()


# helper function for prediction
def predict(model, rng_key, samples, *args, **kwargs):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = handlers.trace(model).get_trace(*args, **kwargs)
    return model_trace['obs']['value']


def main():
    data, obs = load_agw_1d()
    warmup = 200
    num_samples = 1000
    test_data = np.linspace(-2, 2, 500).reshape(-1, 1)
    samples = hmc('protein', data, obs, warmup, num_samples)
    vmap_args = (samples, random.split(random.PRNGKey(1), num_samples))
    predictions = vmap(lambda samples, rng_key: predict(model, rng_key, samples, test_data))(*vmap_args)
    predictions = predictions[..., 0]
    visualize(data, obs, np.squeeze(test_data), predictions)


if __name__ == '__main__':
    main()
