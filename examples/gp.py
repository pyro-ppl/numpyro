import matplotlib
matplotlib.use('Agg')  # noqa: E402
import matplotlib.pyplot as plt
import jax

import argparse

import numpy as onp
from jax import vmap
import jax.numpy as np
import jax.random as random

import numpyro.distributions as dist
from numpyro.handlers import sample
from numpyro.hmc_util import initialize_model
from numpyro.mcmc import mcmc

from jax.config import config
config.update('jax_enable_x64', True)

"""
In this example we show how to use NUTS to sample from the posterior
over the hyperparameters of a gaussian process.
"""


# squared exponential kernel with diagonal noise term
def kernel(X, Y, log_var, log_length, log_noise, jitter=1.0e-5, include_noise=True):
    var, length, noise = np.exp(log_var), np.exp(log_length), np.exp(log_noise)
    deltaXsq = np.power((X[:, None] - Y) / length, 2.0)
    k = var * np.exp(-0.5 * deltaXsq)
    if include_noise:
        k += (noise + jitter) * np.eye(X.shape[0])
    return k


def model(X, Y):
    # set uninformative log-normal priors on our three kernel hyperparameters
    log_var = sample("log_var", dist.Normal(0.0, 10.0))
    log_noise = sample("log_noise", dist.Normal(0.0, 10.0))
    log_length = sample("log_length", dist.Normal(0.0, 10.0))

    # compute kernel
    k = kernel(X, X, log_var, log_length, log_noise)

    # there is currently no multivariate normal distribution so we whiten the data with L
    # and do some hacks to make sure the correct log probability is computed
    L = np.linalg.cholesky(k)
    sigma = np.exp(np.trace(np.log(L)) / X.shape[0])
    Y_whitened = sigma * np.matmul(np.linalg.inv(L), Y)
    sample("Y", dist.Normal(np.zeros(X.shape[0]), sigma * np.ones(X.shape[0])), obs=Y_whitened)


# helper function for doing hmc inference
def run_inference(model, args, rng, X, Y):
    init_params, potential_fn, constrain_fn = initialize_model(rng, model, X, Y)
    samples = mcmc(args.num_warmup, args.num_samples, init_params,
                   sampler='hmc', potential_fn=potential_fn, constrain_fn=constrain_fn)
    return samples


# do GP prediction for a given set of hyperparameters. this makes use of the well-known
# formula for gaussian process predictions
def predict(rng, X, Y, X_test, log_var, log_length, log_noise):
    # compute kernels between train and test data, etc.
    k_pp = kernel(X_test, X_test, log_var, log_length, log_noise, include_noise=True)
    k_pX = kernel(X_test, X, log_var, log_length, log_noise, include_noise=False)
    k_XX = kernel(X, X, log_var, log_length, log_noise, include_noise=True)
    K_xx_inv = np.linalg.inv(k_XX)
    K = k_pp - np.matmul(k_pX, np.matmul(K_xx_inv, np.transpose(k_pX)))
    sigma_noise = np.sqrt(np.diag(K)) * jax.random.normal(rng, (X_test.shape[0],))
    mean = np.matmul(k_pX, np.matmul(K_xx_inv, Y))
    # we return both the mean function and a sample from the posterior predictive for the
    # given set of hyperparameters
    return mean, mean + sigma_noise


# create artificial regression dataset
def get_data(N=30, sigma_obs=0.15, N_test=400):
    onp.random.seed(0)
    X = np.linspace(-1, 1, N)
    Y = X + 0.2 * np.power(X, 3.0) + 0.5 * np.power(0.5 + X, 2.0) * np.sin(4.0 * X)
    Y += sigma_obs * onp.random.randn(N)
    Y -= np.mean(Y)
    Y /= np.std(Y)

    assert X.shape == (N,)
    assert Y.shape == (N,)

    X_test = np.linspace(-1.3, 1.3, N_test)

    return X, Y, X_test


def main(args):
    X, Y, X_test = get_data(N=25)

    # do inference
    rng, rng_predict = random.split(random.PRNGKey(0))
    samples = run_inference(model, args, rng, X, Y)

    # do prediction
    vmap_args = (random.split(rng_predict, args.num_samples), samples['log_var'],
                 samples['log_length'], samples['log_noise'])
    means, predictions = vmap(lambda rng, log_var, log_length, log_noise:
                              predict(rng, X, Y, X_test, log_var, log_length, log_noise))(*vmap_args)

    mean_prediction = np.mean(means, axis=0)
    percentiles = onp.percentile(predictions, [5.0, 95.0], axis=0)

    # make plots
    fig, ax = plt.subplots(1, 1)

    # plot training data
    ax.plot(X, Y, 'kx')
    # plot 90% confidence level of predictions
    ax.fill_between(X_test, percentiles[0, :], percentiles[1, :], color='lightblue')
    # plot mean prediction
    ax.plot(X_test, mean_prediction, 'blue', ls='solid', lw=2.0)
    ax.set(xlabel="X", ylabel="Y", title="Mean predictions with 90% CI")

    plt.savefig("gp_plot.pdf")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-samples", nargs="?", default=1000, type=int)
    parser.add_argument("--num-warmup", nargs='?', default=1000, type=int)
    args = parser.parse_args()
    main(args)
