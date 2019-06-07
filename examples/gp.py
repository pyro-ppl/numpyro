import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as onp

import jax
import jax.numpy as np
import jax.random as random
from jax import vmap
from jax.config import config as jax_config

import numpyro.distributions as dist
from numpyro.handlers import sample
from numpyro.hmc_util import initialize_model
from numpyro.mcmc import mcmc

matplotlib.use('Agg')  # noqa: E402

"""
In this example we show how to use NUTS to sample from the posterior
over the hyperparameters of a gaussian process.
"""


# squared exponential kernel with diagonal noise term
def kernel(X, Z, var, length, noise, jitter=1.0e-6, include_noise=True):
    deltaXsq = np.power((X[:, None] - Z) / length, 2.0)
    k = var * np.exp(-0.5 * deltaXsq)
    if include_noise:
        k += (noise + jitter) * np.eye(X.shape[0])
    return k


def model(X, Y):
    # set uninformative log-normal priors on our three kernel hyperparameters
    var = sample("kernel_var", dist.LogNormal(0.0, 10.0))
    noise = sample("kernel_noise", dist.LogNormal(0.0, 10.0))
    length = sample("kernel_length", dist.LogNormal(0.0, 10.0))

    # compute kernel
    k = kernel(X, X, var, length, noise)

    # sample Y according to the standard gaussian process formula
    sample("Y", dist.MultivariateNormal(loc=np.zeros(X.shape[0]), covariance_matrix=k),
           obs=Y)


# helper function for doing hmc inference
def run_inference(model, args, rng, X, Y):
    init_params, potential_fn, constrain_fn = initialize_model(rng, model, X, Y)
    samples = mcmc(args.num_warmup, args.num_samples, init_params,
                   sampler='hmc', potential_fn=potential_fn, constrain_fn=constrain_fn)
    return samples


# do GP prediction for a given set of hyperparameters. this makes use of the well-known
# formula for gaussian process predictions
def predict(rng, X, Y, X_test, var, length, noise):
    # compute kernels between train and test data, etc.
    k_pp = kernel(X_test, X_test, var, length, noise, include_noise=True)
    k_pX = kernel(X_test, X, var, length, noise, include_noise=False)
    k_XX = kernel(X, X, var, length, noise, include_noise=True)
    K_xx_inv = np.linalg.inv(k_XX)
    K = k_pp - np.matmul(k_pX, np.matmul(K_xx_inv, np.transpose(k_pX)))
    sigma_noise = np.sqrt(np.clip(np.diag(K), a_min=0.)) * jax.random.normal(rng, X_test.shape[:1])
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
    jax_config.update('jax_platform_name', args.device)
    X, Y, X_test = get_data(N=args.num_data)

    # do inference
    rng, rng_predict = random.split(random.PRNGKey(0))
    samples = run_inference(model, args, rng, X, Y)

    # do prediction
    vmap_args = (random.split(rng_predict, args.num_samples), samples['kernel_var'],
                 samples['kernel_length'], samples['kernel_noise'])
    means, predictions = vmap(lambda rng, var, length, noise:
                              predict(rng, X, Y, X_test, var, length, noise))(*vmap_args)

    mean_prediction = onp.mean(means, axis=0)
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
    parser = argparse.ArgumentParser(description="Gaussian Process example")
    parser.add_argument("-n", "--num-samples", nargs="?", default=1000, type=int)
    parser.add_argument("--num-warmup", nargs='?', default=1000, type=int)
    parser.add_argument("--num-data", nargs='?', default=25, type=int)
    parser.add_argument("--device", default='cpu', type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()
    main(args)
