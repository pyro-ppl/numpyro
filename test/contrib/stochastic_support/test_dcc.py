# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
from numpy.testing import assert_allclose
import pytest

import jax
from jax import random
import jax.numpy as jnp

import numpyro
from numpyro.contrib.stochastic_support.dcc import DCC
import numpyro.distributions as dist
from numpyro.infer import HMC, NUTS, SA, BarkerMH


@pytest.mark.parametrize(
    "branch_dist",
    [dist.Normal(0, 1), dist.Gamma(1, 1)],
)
@pytest.mark.xfail(raises=RuntimeError)
def test_continuous_branching(branch_dist):
    rng_key = random.PRNGKey(0)

    def model():
        model1 = numpyro.sample("model1", branch_dist, infer={"branching": True})
        mean = 1.0 if model1 == 0 else 2.0
        numpyro.sample("obs", dist.Normal(mean, 1.0), obs=0.2)

    mcmc_kwargs = dict(
        num_warmup=500,
        num_samples=1000,
        num_chains=1,
    )

    dcc = DCC(model, mcmc_kwargs=mcmc_kwargs)
    rng_key, subkey = random.split(rng_key)
    dcc.run(subkey)


def test_different_address_path():
    rng_key = random.PRNGKey(0)

    def model():
        model1 = numpyro.sample(
            "model1", dist.Bernoulli(0.5), infer={"branching": True}
        )
        if model1 == 0:
            numpyro.sample("a1", dist.Normal(9.0, 1.0))
        else:
            numpyro.sample("a2", dist.Normal(9.0, 1.0))
            numpyro.sample("a3", dist.Normal(9.0, 1.0))
        mean = 1.0 if model1 == 0 else 2.0
        numpyro.sample("obs", dist.Normal(mean, 1.0), obs=0.2)

    mcmc_kwargs = dict(
        num_warmup=50,
        num_samples=50,
        num_chains=1,
        progress_bar=False,
    )

    dcc = DCC(model, mcmc_kwargs=mcmc_kwargs)
    rng_key, subkey = random.split(rng_key)
    dcc.run(subkey)


@pytest.mark.parametrize("proposal_scale", [0.1, 1.0, 10.0])
def test_proposal_scale(proposal_scale):
    def model(y):
        z = numpyro.sample("z", dist.Normal(0.0, 1.0))
        model1 = numpyro.sample(
            "model1", dist.Bernoulli(0.5), infer={"branching": True}
        )
        sigma = 1.0 if model1 == 0 else 2.0
        with numpyro.plate("data", y.shape[0]):
            numpyro.sample("obs", dist.Normal(z, sigma), obs=y)

    rng_key = random.PRNGKey(0)

    rng_key, subkey = random.split(rng_key)
    y_train = dist.Normal(0, 1).sample(subkey, (200,))

    mcmc_kwargs = dict(
        num_warmup=50,
        num_samples=50,
        num_chains=2,
        progress_bar=False,
    )

    dcc = DCC(model, mcmc_kwargs=mcmc_kwargs, proposal_scale=proposal_scale)
    rng_key, subkey = random.split(rng_key)
    dcc.run(subkey, y_train)


@pytest.mark.parametrize(
    "chain_method",
    ["sequential", "parallel", "vectorized"],
)
@pytest.mark.parametrize("kernel_cls", [NUTS, HMC, SA, BarkerMH])
def test_kernels(chain_method, kernel_cls):
    if chain_method == "vectorized" and kernel_cls in [SA, BarkerMH]:
        # These methods do not support vectorized execution.
        return

    def model(y):
        z = numpyro.sample("z", dist.Normal(0.0, 1.0))
        model1 = numpyro.sample(
            "model1", dist.Bernoulli(0.5), infer={"branching": True}
        )
        sigma = 1.0 if model1 == 0 else 2.0
        with numpyro.plate("data", y.shape[0]):
            numpyro.sample("obs", dist.Normal(z, sigma), obs=y)

    rng_key = random.PRNGKey(0)

    rng_key, subkey = random.split(rng_key)
    y_train = dist.Normal(0, 1).sample(subkey, (200,))

    mcmc_kwargs = dict(
        num_warmup=50,
        num_samples=50,
        num_chains=2,
        chain_method=chain_method,
        progress_bar=False,
    )

    dcc = DCC(model, mcmc_kwargs=mcmc_kwargs, kernel_cls=kernel_cls)
    rng_key, subkey = random.split(rng_key)
    dcc.run(subkey, y_train)


def test_weight_convergence():
    PRIOR_MEAN, PRIOR_STD = 0.0, 1.0
    LIKELIHOOD1_STD = 2.0
    LIKELIHOOD2_STD = 0.62177

    def log_marginal_likelihood(data, likelihood_std, prior_mean, prior_std):
        """
        Calculate the marginal likelihood of a model with Normal likelihood, unknown mean,
        and Normal prior.

        Taken from Section 2.5 at https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf.
        """
        num_data = data.shape[0]
        likelihood_var = jnp.square(likelihood_std)
        prior_var = jnp.square(prior_std)

        first_term = (
            jnp.log(likelihood_std)
            - num_data * jnp.log(jnp.sqrt(2 * math.pi) * likelihood_std)
            + 0.5 * jnp.log(num_data * prior_var + likelihood_var)
        )
        second_term = -(jnp.sum(jnp.square(data)) / (2 * likelihood_var)) - (
            jnp.square(prior_mean) / (2 * prior_var)
        )
        third_term = (
            (
                prior_var
                * jnp.square(num_data)
                * jnp.square(jnp.mean(data))
                / likelihood_var
            )
            + (likelihood_var * jnp.square(prior_mean) / prior_var)
            + 2 * num_data * jnp.mean(data) * prior_mean
        ) / (2 * (num_data * prior_var + likelihood_var))
        return first_term + second_term + third_term

    def model(y):
        z = numpyro.sample("z", dist.Normal(PRIOR_MEAN, PRIOR_STD))
        model1 = numpyro.sample(
            "model1", dist.Bernoulli(0.5), infer={"branching": True}
        )
        sigma = LIKELIHOOD1_STD if model1 == 0 else LIKELIHOOD2_STD
        with numpyro.plate("data", y.shape[0]):
            numpyro.sample("obs", dist.Normal(z, sigma), obs=y)

    rng_key = random.PRNGKey(1)

    rng_key, subkey = random.split(rng_key)
    y_train = dist.Normal(0, 1).sample(subkey, (200,))

    mcmc_kwargs = dict(
        num_warmup=500,
        num_samples=1000,
        num_chains=1,
    )

    dcc = DCC(model, mcmc_kwargs=mcmc_kwargs)
    rng_key, subkey = random.split(rng_key)
    dcc_result = dcc.run(subkey, y_train)
    slp_weights = jnp.array([dcc_result.slp_weights["0"], dcc_result.slp_weights["1"]])
    assert_allclose(1.0, jnp.sum(slp_weights))

    slp1_lml = log_marginal_likelihood(y_train, LIKELIHOOD1_STD, PRIOR_MEAN, PRIOR_STD)
    slp2_lml = log_marginal_likelihood(y_train, LIKELIHOOD2_STD, PRIOR_MEAN, PRIOR_STD)
    lmls = jnp.array([slp1_lml, slp2_lml])
    analytic_weights = jnp.exp(lmls - jax.scipy.special.logsumexp(lmls))
    close_weights = (  # account for non-identifiability
        np.allclose(analytic_weights, slp_weights, rtol=1e-5, atol=1e-5)
        or np.allclose(analytic_weights, slp_weights[::-1], rtol=1e-5, atol=1e-5)
    )
    assert close_weights
