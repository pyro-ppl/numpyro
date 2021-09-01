# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Horseshoe Regression
================================

We demonstrate how to use NUTS to do sparse regression using
the Horseshoe prior [1].

References:

[1] "Handling Sparsity via the Horseshoe,"
    Carlos M. Carvalho, Nicholas G. Polson, James G. Scott.
"""

import argparse
import os
import time

import numpy as np

import jax.numpy as jnp
import jax.random as random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


# our regression model
def model(X, Y):
    D_X = X.shape[1]

    # sample from horseshoe prior
    lambdas = numpyro.sample("lambdas", dist.HalfCauchy(jnp.ones(D_X)))
    tau = numpyro.sample("tau", dist.HalfCauchy(jnp.ones(1)))

    # note that in practice for a normal likelihood we would probably want to
    # integrate out the coefficients (as is done for example in sparse_regression.py)
    # however, this trick wouldn't be applicable to other likelihoods (e.g. bernoulli),
    # so we do make use of it here.
    unscaled_betas = numpyro.sample("unscaled_betas", dist.Normal(jnp.ones(D_X)))
    scaled_betas = numpyro.deterministic("betas", tau * lambdas * unscaled_betas)

    # compute mean function using linear coefficients
    mean_function = jnp.dot(X, scaled_betas)

    prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / jnp.sqrt(prec_obs)

    # observe data
    numpyro.sample("Y", dist.Normal(mean_function, sigma_obs), obs=Y)


# helper function for HMC inference
def run_inference(model, args, rng_key, X, Y):
    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(rng_key, X, Y)
    mcmc.print_summary(exclude_deterministic=False)
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()


# create artificial regression dataset with 3 non-zero regression coefficients
def get_data(N=50, D_X=3, sigma_obs=0.05):
    assert D_X >= 3
    np.random.seed(0)
    X = np.random.randn(N, D_X)
    W = np.array([1.0, -0.5, 0.25])
    Y = jnp.dot(X[:, :3], W)
    Y += sigma_obs * np.random.randn(N)
    Y -= jnp.mean(Y)
    Y /= jnp.std(Y)

    assert X.shape == (N, D_X)
    assert Y.shape == (N,)

    return X, Y


def main(args):
    N, D_X = args.num_data, 32
    X, Y = get_data(N=N, D_X=D_X)

    # do inference
    rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    run_inference(model, args, rng_key, X, Y)


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.7.2")
    parser = argparse.ArgumentParser(description="Horseshoe regression example")
    parser.add_argument("-n", "--num-samples", nargs="?", default=2000, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=1000, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--num-data", nargs="?", default=100, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
