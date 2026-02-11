# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import jax.numpy as jnp
import jax.random as random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import AIES, ESS, MCMC
from numpyro.infer.ensemble import EnsembleSampler, EnsembleSamplerState
from numpyro.infer.initialization import init_to_uniform

numpyro.set_host_device_count(2)
# ---
# reused for all smoke-tests
N, dim = 3000, 3

data = np.random.default_rng(0).normal(N, dim)
true_coefs = np.arange(1.0, dim + 1.0)
logits = np.sum(true_coefs * data, axis=-1)


def labels_maker():
    return dist.Bernoulli(logits=logits).sample(random.PRNGKey(1))


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
        mcmc.run(random.PRNGKey(2), labels_maker())


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
    mcmc.run(random.PRNGKey(2), labels_maker())

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
    labels = labels_maker()
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
    labels = labels_maker()
    mcmc.warmup(random.PRNGKey(2), labels)
    mcmc.run(random.PRNGKey(3), labels)


def test_ensemble_sampler_uses_complementary_halves():
    class ToyEnsembleSampler(EnsembleSampler):
        def __init__(self):
            super().__init__(
                potential_fn=lambda z: jnp.array(0.0),
                randomize_split=False,
                init_strategy=init_to_uniform,
            )
            self._num_chains = 4

        def init_inner_state(self, rng_key):
            return jnp.array(0)

        def update_active_chains(self, active, inactive, inner_state):
            # Encode which half was used as inactive in each sub-iteration.
            return inactive + 1.0, inner_state

    sampler = ToyEnsembleSampler()
    state = EnsembleSamplerState(
        # First sub-iteration uses second-half inactive chains [10, 11].
        z=jnp.array([[0.0], [0.0], [10.0], [11.0]]),
        inner_state=jnp.array(0),
        rng_key=random.PRNGKey(0),
    )

    new_state = sampler.sample(state, model_args=(), model_kwargs={})
    expected = jnp.array([[11.0], [12.0], [12.0], [13.0]])
    assert jnp.allclose(new_state.z, expected)
