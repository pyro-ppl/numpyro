# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Bayesian Neural Network
================================

We demonstrate how to use NUTS to do inference on a simple (small)
Bayesian neural network with two hidden layers.
"""

import argparse
import time

import jax.numpy as jnp
import jax.random as random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from flax import nn
from jax import vmap

import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.contrib.module import random_flax_module
from numpyro.examples.datasets import load_dataset, MNIST
from numpyro.infer import MCMC, NUTS

matplotlib.use('Agg')  # noqa: E402


class Network(nn.Module):
    def apply(self, x, hid_channels, out_channels):
        l1 = nn.relu(nn.Dense(x, features=hid_channels))
        l2 = nn.relu(nn.Dense(l1, features=hid_channels))
        logits = nn.Dense(l2, features=out_channels)
        return logits


def mnist_model(features, hid_channels, obs=None):
    module = Network.partial(hid_channels=hid_channels, out_channels=10)
    net = random_flax_module('snn', module, dist.Normal(0, 1.), input_shape=features.shape)
    if obs is not None:
        obs = obs[..., None]
    numpyro.sample('obs', dist.Categorical(logits=net(features)), obs=obs)


def mnist_data(split='train'):
    mnist_init, mnist_batch = load_dataset(MNIST, split=split)
    _, idxs = mnist_init()
    X, Y = mnist_batch(0, idxs)
    _, m, _ = X.shape
    X = X.reshape(-1, m ** 2)
    return X, Y


def mnist_main(args):
    hid_channels = 32
    X, Y = mnist_data()
    rng_key, rng_key_predict = random.split(random.PRNGKey(37))
    samples = run_inference(mnist_model, args, rng_key, X[:args.num_data], hid_channels, Y[:args.num_data])

    # predict Y_test at inputs X_test
    vmap_args = (samples, random.split(rng_key_predict, args.num_samples * args.num_chains))
    X, Y = mnist_data('test')
    predictions = vmap(lambda samples, rng_key: predict(mnist_model, rng_key, samples, X[:100], hid_channels))(
        *vmap_args)
    predictions = predictions[..., 0]


class RegNetwork(nn.Module):
    def apply(self, x, hid_channels, out_channels):
        l1 = nn.tanh(nn.Dense(x, features=hid_channels))
        l2 = nn.tahn(nn.Dense(l1, features=hid_channels))
        mean = nn.Dense(l2, features=out_channels)
        return mean


def reg_model(features, obs, hid_channels):
    in_channels, out_channels = features.shape[1], 1
    module = Network.partial(hid_channels=hid_channels, out_channels=out_channels)

    net = random_flax_module('snn', module, dist.Normal(0, 1.), input_shape=())
    mean = net(features)

    # we put a prior on the observation noise
    prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / jnp.sqrt(prec_obs)  # prior

    numpyro.sample("Y", dist.Normal(mean, sigma_obs), obs=obs[..., None])


# helper function for HMC inference
def run_inference(model, args, rng_key, X, Y, D_H):
    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(kernel, args.num_warmup, args.num_samples, num_chains=args.num_chains)
    mcmc.run(rng_key, X, Y, D_H)
    mcmc.print_summary()
    print('\nMCMC elapsed time:', time.time() - start)
    return mcmc.get_samples()


# helper function for prediction
def predict(model, rng_key, samples, *args, **kwargs):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = handlers.trace(model).get_trace(*args, **kwargs)
    return model_trace['obs']['value']


# create artificial regression dataset
def get_data(N=50, D_X=3, sigma_obs=0.05, N_test=500):
    D_Y = 1  # create 1d outputs
    np.random.seed(0)
    X = jnp.linspace(-1, 1, N)
    X = jnp.power(X[:, np.newaxis], jnp.arange(D_X))
    W = 0.5 * np.random.randn(D_X)
    Y = jnp.dot(X, W) + 0.5 * jnp.power(0.5 + X[:, 1], 2.0) * jnp.sin(4.0 * X[:, 1])
    Y += sigma_obs * np.random.randn(N)
    Y = Y[:, np.newaxis]
    Y -= jnp.mean(Y)
    Y /= jnp.std(Y)

    assert X.shape == (N, D_X)
    assert Y.shape == (N, D_Y)

    X_test = jnp.linspace(-1.3, 1.3, N_test)
    X_test = jnp.power(X_test[:, np.newaxis], jnp.arange(D_X))

    return X, Y, X_test


def main(args):
    N, D_X, D_H = args.num_data, 3, args.num_hidden
    X, Y, X_test = get_data(N=N, D_X=D_X)

    # do inference
    rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    samples = run_inference(reg_model, args, rng_key, X, Y, D_H)

    # predict Y_test at inputs X_test
    vmap_args = (samples, random.split(rng_key_predict, args.num_samples * args.num_chains))
    predictions = vmap(lambda samples, rng_key: predict(reg_model, rng_key, samples, X_test, D_H))(*vmap_args)
    predictions = predictions[..., 0]

    # compute mean prediction and confidence interval around median
    mean_prediction = jnp.mean(predictions, axis=0)
    percentiles = np.percentile(predictions, [5.0, 95.0], axis=0)

    # make plots
    fig, ax = plt.subplots(1, 1)

    # plot training data
    ax.plot(X[:, 1], Y[:, 0], 'kx')
    # plot 90% confidence level of predictions
    ax.fill_between(X_test[:, 1], percentiles[0, :], percentiles[1, :], color='lightblue')
    # plot mean prediction
    ax.plot(X_test[:, 1], mean_prediction, 'blue', ls='solid', lw=2.0)
    ax.set(xlabel="X", ylabel="Y", title="Mean predictions with 90% CI")

    plt.savefig('bnn_plot.pdf')
    plt.tight_layout()


if __name__ == "__main__":
    assert numpyro.__version__.startswith('0.4.1')
    parser = argparse.ArgumentParser(description="Bayesian neural network example")
    parser.add_argument("-n", "--num-samples", nargs="?", default=20, type=int)
    parser.add_argument("--num-warmup", nargs='?', default=10, type=int)
    parser.add_argument("--num-chains", nargs='?', default=1, type=int)
    parser.add_argument("--num-data", nargs='?', default=1000, type=int)
    parser.add_argument("--num-hidden", nargs='?', default=5, type=int)
    parser.add_argument("--device", default='gpu', type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    mnist_main(args)
