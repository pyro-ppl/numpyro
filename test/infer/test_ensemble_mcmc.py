# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest

import jax.numpy as jnp
import jax.random as random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import AIES, ESS, MCMC

numpyro.set_host_device_count(2)
# ---
# reused for all smoke-tests
N, dim = 3000, 3

data = random.normal(random.PRNGKey(0), (N, dim))
true_coefs = jnp.arange(1.0, dim + 1.0)
logits = jnp.sum(true_coefs * data, axis=-1)
labels = dist.Bernoulli(logits=logits).sample(random.PRNGKey(1))


def model(labels):
    coefs = numpyro.sample("coefs", dist.Normal(jnp.zeros(dim), jnp.ones(dim)))
    logits = numpyro.deterministic("logits", jnp.sum(coefs * data, axis=-1))
    return numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=labels)


# ---


@pytest.mark.parametrize(
    "kernel_cls, n_chain, method",
    [
        (AIES, 10, "sequential"),
        (AIES, 1, "vectorized"),
        (AIES, 2, "parallel"),
        (ESS, 10, "sequential"),
        (ESS, 1, "vectorized"),
        (ESS, 2, "parallel"),
    ],
)
def test_chain_smoke(kernel_cls, n_chain, method):
    kernel = kernel_cls(model)

    mcmc = MCMC(
        kernel,
        num_warmup=10,
        num_samples=10,
        progress_bar=False,
        num_chains=n_chain,
        chain_method=method,
    )

    with pytest.raises(AssertionError, match="chain_method"):
        mcmc.run(random.PRNGKey(2), labels)


@pytest.mark.parametrize("kernel_cls", [AIES, ESS])
def test_out_shape_smoke(kernel_cls):
    n_chains = 10
    kernel = kernel_cls(model)

    mcmc = MCMC(
        kernel,
        num_warmup=10,
        num_samples=10,
        progress_bar=False,
        num_chains=n_chains,
        chain_method="vectorized",
    )
    mcmc.run(random.PRNGKey(2), labels)

    assert mcmc.get_samples(group_by_chain=True)["coefs"].shape[0] == n_chains


@pytest.mark.parametrize("kernel_cls", [AIES, ESS])
def test_invalid_moves(kernel_cls):
    with pytest.raises(AssertionError, match="Each move"):
        kernel_cls(model, moves={"invalid": 1.0})


@pytest.mark.parametrize("kernel_cls", [AIES, ESS])
def test_multirun(kernel_cls):
    n_chains = 10
    kernel = kernel_cls(model)

    mcmc = MCMC(
        kernel,
        num_warmup=10,
        num_samples=10,
        progress_bar=False,
        num_chains=n_chains,
        chain_method="vectorized",
    )
    mcmc.run(random.PRNGKey(2), labels)
    mcmc.run(random.PRNGKey(3), labels)


@pytest.mark.parametrize("kernel_cls", [AIES, ESS])
def test_warmup(kernel_cls):
    n_chains = 10
    kernel = kernel_cls(model)

    mcmc = MCMC(
        kernel,
        num_warmup=10,
        num_samples=10,
        progress_bar=False,
        num_chains=n_chains,
        chain_method="vectorized",
    )
    mcmc.warmup(random.PRNGKey(2), labels)
    mcmc.run(random.PRNGKey(3), labels)
