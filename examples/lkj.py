# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse

from jax import random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC

"""
This simple example is intended to demonstrate how to use an LKJ prior with
a multivariate distribution.

It generates entirely random, uncorrelated data, and then attempts to fit a correlation matrix
and vector of variances.
"""


def model(y):
    d = y.shape[1]
    N = y.shape[0]
    # Vector of variances for each of the d variables
    theta = numpyro.sample("theta", dist.HalfCauchy(jnp.ones(d)))
    # Lower cholesky factor of a correlation matrix
    concentration = jnp.ones(1)  # Implies a uniform distribution over correlation matrices
    L_omega = numpyro.sample("L_omega", dist.LKJCholesky(d, concentration))
    # Lower cholesky factor of the covariance matrix
    L_Omega = jnp.matmul(jnp.diag(jnp.sqrt(theta)), L_omega)

    # Vector of expectations
    mu = jnp.zeros(d)

    with numpyro.plate("observations", N):
        obs = numpyro.sample("obs", dist.MultivariateNormal(mu, scale_tril=L_Omega), obs=y)
    return obs


def main(args):
    rng_key = random.PRNGKey(0)
    y = random.normal(rng_key, (args.n, args.num_variables))
    nuts_kernel = NUTS(model, step_size=1e-5)
    MCMC(nuts_kernel, num_samples=args.num_samples,
         num_warmup=args.num_warmup, num_chains=args.num_chains).run(rng_key, y)


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.6.0")
    parser = argparse.ArgumentParser(description="Demonstrate the use of an LKJ Prior")
    parser.add_argument("--num-samples", nargs="?", default=200, type=int)
    parser.add_argument("--n", nargs="?", default=500, type=int)
    parser.add_argument("--num-chains", nargs='?', default=4, type=int)
    parser.add_argument("--num-variables", nargs='?', default=5, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=100, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
