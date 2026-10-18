# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import jax
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
    return dist.Bernoulli(logits=logits).sample(random.key(1))


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
        mcmc.run(random.key(2), labels_maker())


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
    mcmc.run(random.key(2), labels_maker())

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
    mcmc.run(random.key(2), labels)
    mcmc.run(random.key(3), labels)


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
    mcmc.warmup(random.key(2), labels)
    mcmc.run(random.key(3), labels)


def test_random_move_normalizes_per_walker():
    # Each walker's direction vector must have norm `2 * mu`, independent of
    # the other active walkers (ESS.RandomMove's docstring promises "no
    # chain interaction"). Normalizing on the wrong axis couples walkers
    # through a shared per-dimension scale and leaves per-walker norms
    # scattered around (but not equal to) 2 * mu.
    random_move = ESS.RandomMove()
    inactive = random.normal(random.key(0), (6, 4))
    mu = 1.5

    directions = random_move(random.key(1), inactive, mu)

    row_norms = jnp.linalg.norm(directions, axis=-1)
    assert jnp.allclose(row_norms, 2.0 * mu, atol=1e-5)


def test_random_move_mixed_with_other_move_smoke():
    # Runs RandomMove through ESS.update_active_chains, mixed with another move,
    # so the jax.lax.switch dispatch (every move branch must return the same
    # output shape) is exercised end to end.
    n_chains = 10
    kernel = ESS(model, moves={ESS.DifferentialMove(): 0.5, ESS.RandomMove(): 0.5})

    mcmc = MCMC(
        kernel,
        num_warmup=10,
        num_samples=10,
        progress_bar=False,
        num_chains=n_chains,
        chain_method="vectorized",
    )
    mcmc.run(random.key(2), labels_maker())

    assert mcmc.get_samples(group_by_chain=True)["coefs"].shape[0] == n_chains


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
        z=jnp.array([[0.0], [1.0], [10.0], [11.0]]),
        inner_state=jnp.array(0),
        rng_key=random.PRNGKey(0),
    )

    new_state = sampler.sample(state, model_args=(), model_kwargs={})
    # Expected: first two chains get [11, 12] from second iteration using first half [0, 1] as inactive.
    # Then last two chains get [12, 13] from first iteration using second half [10, 11] as inactive.
    expected = jnp.array([[11.0], [12.0], [12.0], [13.0]])
    assert jnp.allclose(new_state.z, expected)


def _gaussian_potential_fn(z):
    # correlated 2-D Gaussian
    x, y = z["x"], z["y"]
    return 0.5 * (x**2 + ((y - 0.8 * x) / 0.3) ** 2)


@pytest.mark.parametrize("randomize_split", [False, True])
def test_aies_evaluates_only_proposals(randomize_split):
    n_chains, num_samples = 20, 50
    calls = []

    def counted_potential_fn(z):
        jax.debug.callback(lambda n: calls.append(int(n)), 1)
        return _gaussian_potential_fn(z)

    mcmc = MCMC(
        AIES(potential_fn=counted_potential_fn, randomize_split=randomize_split),
        num_warmup=0,
        num_samples=num_samples,
        num_chains=n_chains,
        chain_method="vectorized",
        progress_bar=False,
    )
    init_params = {"x": 0.1 * jnp.arange(n_chains), "y": jnp.zeros(n_chains)}
    mcmc.run(random.key(0), init_params=init_params)
    jax.effects_barrier()
    # 1 batched evaluation of the initial ensemble + 1 per half-step (proposals only)
    assert sum(calls) == 1 + 2 * num_samples
    # the cached log densities describe the final positions
    state = mcmc.last_state
    expected = -jax.vmap(_gaussian_potential_fn)(state.z)
    assert jnp.allclose(state.inner_state.log_density, expected, rtol=1e-5, atol=1e-5)
