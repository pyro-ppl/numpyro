import matplotlib
matplotlib.use('Agg')  # noqa: E402
import matplotlib.pyplot as plt

import argparse

import numpy as onp
from jax import vmap
import jax.numpy as np
import jax.random as random

import numpyro.distributions as dist
from numpyro.handlers import sample, seed, substitute, trace
from numpyro.hmc_util import initialize_model
from numpyro.mcmc import mcmc


# we treat these hyperparameters as known and fixed
K1_var = 0.25
K1_length = 0.25


# squared exponential kernel
def K1(X):
    X1, X2 = X[:, None], X
    deltaXsq = np.power((X1 - X2) / K1_length, 2.0)
    return K1_var * np.exp(-0.5 * deltaXsq)


def model(X, Y):
    # let's be bayesian about one hyperparameter
    prec_obs = sample("prec_obs", dist.Gamma(3.0, 1.0))

    # compute kernel
    k1 = K1(X) + 1.0 / prec_obs
    L1 = np.linalg.cholesky(k1)

    # there is no multivariate normal distribution so we whiten the data with L1
    Y_whitened = np.matmul(np.linalg.inv(L1), Y)
    sample("Y", dist.Normal(np.zeros(Y.shape[0]), np.ones(Y.shape[0])), obs=Y_whitened)


def run_inference(model, args, rng, X, Y):
    init_params, potential_fn, constrain_fn = initialize_model(rng, model, X, Y)
    samples = mcmc(args.num_warmup, args.num_samples, init_params,
                   sampler='hmc', potential_fn=potential_fn, constrain_fn=constrain_fn,
                   algo='nuts')
    return samples


def get_data(N=50, sigma_obs=0.05):
    onp.random.seed(0)
    X = np.linspace(-1, 1, N)
    Y = X + 0.2 * np.power(X, 3.0) + 0.5 * np.power(0.5 + X, 2.0) * np.sin(4.0 * X)
    Y += sigma_obs * onp.random.randn(N)
    Y -= np.mean(Y)
    Y /= np.std(Y)

    assert X.shape == (N,)
    assert Y.shape == (N,)

    return X, Y


def main(args):
    X, Y = get_data(N=20)  # ok for N=10 but freaks out for N=20

    rng, rng_predict = random.split(random.PRNGKey(0))
    samples = run_inference(model, args, rng, X, Y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-samples", nargs="?", default=300, type=int)
    parser.add_argument("--num-warmup", nargs='?', default=300, type=int)
    args = parser.parse_args()
    main(args)
