# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numpy.testing import assert_allclose
import pytest

from functools import partial

import jax.numpy as jnp
from jax import random
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular, inv

import numpyro
import numpyro.distributions as dist
from numpyro.infer import HMC, MCMC, NUTS, HMCGibbs, discrete_gibbs_fn, subsample_gibbs_fn


def _linear_regression_gibbs_fn(X, XX, XY, Y, rng_key, gibbs_sites, hmc_sites):
    N, P = X.shape

    sigma = jnp.exp(hmc_sites['log_sigma']) if 'log_sigma' in hmc_sites else hmc_sites['sigma']

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
        numpyro.deterministic("mean", mean)

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


@pytest.mark.parametrize('kernel_cls', [HMC, NUTS])
def test_linear_model_sigma(kernel_cls, N=90, P=40, sigma=0.07, warmup_steps=500, num_samples=500):
    np.random.seed(1)
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

    beta_mean = np.mean(mcmc.get_samples()['beta'], axis=0)
    assert_allclose(beta_mean, np.array([1.0] + [0.0] * (P - 1)), atol=0.05)

    sigma_mean = np.mean(mcmc.get_samples()['sigma'], axis=0)
    assert_allclose(sigma_mean, sigma, atol=0.25)


@pytest.mark.parametrize('kernel_cls', [HMC, NUTS])
def test_gaussian_model(kernel_cls, D=2, warmup_steps=3000, num_samples=5000):
    np.random.seed(0)
    cov = np.random.randn(4 * D * D).reshape((2 * D, 2 * D))
    cov = jnp.matmul(jnp.transpose(cov), cov) + 0.25 * jnp.eye(2 * D)

    cov00 = cov[:D, :D]
    cov01 = cov[:D, D:]
    cov10 = cov[D:, :D]
    cov11 = cov[D:, D:]

    cov_01_cov11_inv = jnp.matmul(cov01, inv(cov11))
    cov_10_cov00_inv = jnp.matmul(cov10, inv(cov00))

    posterior_cov0 = cov00 - jnp.matmul(cov_01_cov11_inv, cov10)
    posterior_cov1 = cov11 - jnp.matmul(cov_10_cov00_inv, cov01)

    # we consider a model in which (x0, x1) ~ MVN(0, cov)

    def gaussian_gibbs_fn(rng_key, hmc_sites, gibbs_sites):
        x1 = hmc_sites['x1']
        posterior_loc0 = jnp.matmul(cov_01_cov11_inv, x1)
        x0_proposal = dist.MultivariateNormal(loc=posterior_loc0, covariance_matrix=posterior_cov0).sample(rng_key)
        return {'x0': x0_proposal}

    def model():
        x0 = numpyro.sample("x0", dist.MultivariateNormal(loc=jnp.zeros(D), covariance_matrix=cov00))
        posterior_loc1 = jnp.matmul(cov_10_cov00_inv, x0)
        numpyro.sample("x1", dist.MultivariateNormal(loc=posterior_loc1, covariance_matrix=posterior_cov1))

    hmc_kernel = kernel_cls(model, dense_mass=True)
    kernel = HMCGibbs(hmc_kernel, gibbs_fn=gaussian_gibbs_fn, gibbs_sites=['x0'])
    mcmc = MCMC(kernel, warmup_steps, num_samples, progress_bar=False)

    mcmc.run(random.PRNGKey(0))

    x0_mean = np.mean(mcmc.get_samples()['x0'], axis=0)
    x1_mean = np.mean(mcmc.get_samples()['x1'], axis=0)

    x0_std = np.std(mcmc.get_samples()['x0'], axis=0)
    x1_std = np.std(mcmc.get_samples()['x1'], axis=0)

    assert_allclose(x0_mean, np.zeros(D), atol=0.15)
    assert_allclose(x1_mean, np.zeros(D), atol=0.2)

    assert_allclose(x0_std, np.sqrt(np.diagonal(cov00)), rtol=0.05)
    assert_allclose(x1_std, np.sqrt(np.diagonal(cov11)), rtol=0.1)


def test_discrete_gibbs():
    def model(probs, locs):
        c = numpyro.sample("c", dist.Categorical(probs))
        numpyro.sample("x", dist.Normal(locs[c], 0.5))

    probs = jnp.array([0.15, 0.3, 0.3, 0.25])
    locs = jnp.array([-2, 0, 2, 4])
    gibbs_fn = discrete_gibbs_fn(model, probs, locs)
    kernel = HMCGibbs(NUTS(model), gibbs_fn, gibbs_sites=["c"])
    mcmc = MCMC(kernel, 1000, 100000, progress_bar=False)
    mcmc.run(random.PRNGKey(0), probs, locs)
    samples = mcmc.get_samples()["x"]
    # actual value 1.3 is obtained by drawing 1e7 samples from prior distribution
    assert_allclose(jnp.mean(samples, 0), 1.3, atol=0.1)


def test_subsample_gibbs():
    def model(data):
        x = numpyro.sample("x", dist.Normal(0, 1))
        with numpyro.plate("N", data.shape[0], subsample_size=100):
            batch = numpyro.subsample(data, event_dim=0)
            numpyro.sample("obs", dist.Normal(x, 1), obs=batch)

    data = random.normal(random.PRNGKey(0), (10000,)) + 1
    gibbs_fn = subsample_gibbs_fn(model, data)
    kernel = HMCGibbs(NUTS(model), gibbs_fn=gibbs_fn, gibbs_sites="N")
    mcmc = MCMC(kernel, 500, 500)
    mcmc.run(random.PRNGKey(1), data)
    samples = mcmc.get_samples()["x"]
    assert_allclose(jnp.mean(samples, 0), 1., atol=0.1)
