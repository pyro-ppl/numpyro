# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numpy.testing import assert_allclose
import pytest

from functools import partial

import jax.numpy as jnp
from jax import random
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular

import numpyro
import numpyro.distributions as dist
from numpyro.infer import HMC, MCMC, NUTS, HMCGibbs


def _linear_regression_gibbs_fn(X, XX, XY, Y, rng_key, beta, log_sigma=None, sigma=None):
    N, P = X.shape

    if sigma is None:
        sigma = jnp.exp(log_sigma)

    sigma_sq = jnp.square(sigma)
    covar_inv = XX / sigma_sq + jnp.eye(P)

    L = cho_factor(covar_inv, lower=True)[0]
    L_inv = solve_triangular(L, jnp.eye(P), lower=True)
    loc = cho_solve((L, True), XY) / sigma_sq

    beta_proposal = dist.MultivariateNormal(loc=loc, scale_tril=L_inv).sample(rng_key)

    return {'beta': beta_proposal}


@pytest.mark.parametrize('kernel_cls', [HMC, NUTS])
def test_linear_model_log_sigma(kernel_cls, N=100, P=50, sigma=0.11, warmup_steps=500, num_samples=500):
    np.random.seed(0)
    X = np.random.randn(N * P).reshape((N, P))
    XX = np.matmul(np.transpose(X), X)
    Y = X[:, 0] + sigma * np.random.randn(N)
    XY = np.sum(X * Y[:, None], axis=0)

    def model(X, Y):
        N, P = X.shape

        log_sigma = numpyro.sample("log_sigma", dist.Normal(1.0))
        sigma = jnp.exp(log_sigma)
        beta = numpyro.sample("beta", dist.Normal(jnp.zeros(P), jnp.ones(P)))
        mean = jnp.sum(beta * X, axis=-1)

        numpyro.sample("obs", dist.Normal(mean, sigma), obs=Y)

    gibbs_fn = partial(_linear_regression_gibbs_fn, X, XX, XY, Y)

    hmc_kernel = kernel_cls(model)
    kernel = HMCGibbs(hmc_kernel, gibbs_fn=gibbs_fn, gibbs_sites=['beta'])
    mcmc = MCMC(kernel, warmup_steps, num_samples, progress_bar=False)

    mcmc.run(random.PRNGKey(0), X, Y)

    beta_mean = np.mean(mcmc.get_samples()['beta'], axis=0)
    assert_allclose(beta_mean, np.array([1.0] + [0.0] * (P - 1)), atol=0.05)

    sigma_mean = np.exp(np.mean(mcmc.get_samples()['log_sigma'], axis=0))
    assert_allclose(sigma_mean, sigma, atol=0.25)


@pytest.mark.skip('Problems with constraints? Also seems to hang due to compilation issues?')
@pytest.mark.parametrize('kernel_cls', [HMC, NUTS])
def test_linear_model_sigma(kernel_cls, N=100, P=50, sigma=0.11, warmup_steps=500, num_samples=500):
    np.random.seed(0)
    X = np.random.randn(N * P).reshape((N, P))
    XX = np.matmul(np.transpose(X), X)
    Y = X[:, 0] + sigma * np.random.randn(N)
    XY = np.sum(X * Y[:, None], axis=0)

    def model(X, Y):
        N, P = X.shape

        sigma = numpyro.sample("sigma", dist.HalfCauchy(1.0))
        beta = numpyro.sample("beta", dist.Normal(jnp.zeros(P), jnp.ones(P)))
        mean = jnp.sum(beta * X, axis=-1)

        numpyro.sample("obs", dist.Normal(mean, sigma), obs=Y)

    gibbs_fn = partial(_linear_regression_gibbs_fn, X, XX, XY, Y)

    hmc_kernel = kernel_cls(model)
    kernel = HMCGibbs(hmc_kernel, gibbs_fn=gibbs_fn, gibbs_sites=['beta'])
    mcmc = MCMC(kernel, warmup_steps, num_samples, progress_bar=False)

    mcmc.run(random.PRNGKey(0), X, Y)
    mcmc.print_summary(exclude_deterministic=False)

    beta_mean = np.mean(mcmc.get_samples()['beta'], axis=0)
    assert_allclose(beta_mean, np.array([1.0] + [0.0] * (P - 1)), atol=0.05)

    sigma_mean = np.mean(mcmc.get_samples()['sigma'], axis=0)
    assert_allclose(sigma_mean, sigma, atol=0.25)
