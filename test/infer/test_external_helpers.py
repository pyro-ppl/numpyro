# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple

import numpy as np
from numpy.testing import assert_allclose
import pytest

import jax
from jax import random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import (
    MCMC,
    Predictive,
    constrain_samples,
    get_log_density_fn,
)
from numpyro.infer.initialization import init_to_uniform
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import constrain_fn, initialize_model, potential_energy


def _linreg_model(x, y=None):
    """Simple linear regression model used as the integration target."""
    a = numpyro.sample("a", dist.Normal(0.0, 2.0))
    b = numpyro.sample("b", dist.HalfNormal(2.0))
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    mu = numpyro.deterministic("mu", a + b * x)
    with numpyro.plate("data", len(x)):
        numpyro.sample("likelihood", dist.Normal(mu, sigma), obs=y)


def _linreg_data(seed=0, n=80):
    key = random.key(seed)
    k1, k2 = random.split(key)
    x = random.normal(k1, (n,))
    y = 1.0 + 2.0 * x + 0.3 * random.normal(k2, (n,))
    return x, y


def test_initialize_model_bound_populates_logdensity_fn():
    """`bound=True` binds the model args and exposes a negated potential."""
    x, y = _linreg_data()
    info = initialize_model(random.key(0), _linreg_model, bound=True, model_args=(x, y))
    position = info.param_info.z
    # potential_fn is now a single-position callable.
    pe = potential_energy(_linreg_model, (x, y), {}, position)
    assert_allclose(info.potential_fn(position), pe, rtol=1e-5)
    # logdensity_fn is exactly the negation.
    assert info.logdensity_fn is not None
    assert_allclose(
        info.logdensity_fn(position), -info.potential_fn(position), rtol=1e-5
    )


def test_initialize_model_default_has_no_logdensity_fn():
    """Without `bound`, behavior is unchanged and `logdensity_fn` is None."""
    x, y = _linreg_data()
    info = initialize_model(random.key(0), _linreg_model, model_args=(x, y))
    assert info.logdensity_fn is None
    # With dynamic_args=False, postprocess_fn is already a single-position callable.
    out = info.postprocess_fn(info.param_info.z)
    assert "mu" in out


def test_initialize_model_bound_rejects_dynamic_args():
    x, y = _linreg_data()
    with pytest.raises(ValueError, match="incompatible"):
        initialize_model(
            random.key(0),
            _linreg_model,
            bound=True,
            dynamic_args=True,
            model_args=(x, y),
        )


def test_get_log_density_fn_negates_potential():
    """logdensity_fn returns exactly the negation of potential_energy."""
    x, y = _linreg_data()
    info = get_log_density_fn(random.key(0), _linreg_model, model_args=(x, y))
    pe = potential_energy(_linreg_model, (x, y), {}, info.init_position)
    assert_allclose(info.logdensity_fn(info.init_position), -pe, rtol=1e-5)


def test_get_log_density_fn_postprocess_includes_deterministics():
    """postprocess maps unconstrained to constrained and includes deterministics."""
    x, y = _linreg_data()
    info = get_log_density_fn(random.key(0), _linreg_model, model_args=(x, y))
    out = info.postprocess_fn(info.init_position)
    assert set(out.keys()) >= {"a", "b", "sigma", "mu"}
    # sigma has an Exponential prior -> constrained to be positive.
    assert float(out["sigma"]) > 0.0
    # mu (deterministic) shape follows x.
    assert out["mu"].shape == x.shape


@pytest.mark.parametrize("wrapper", [False, True])
def test_constrain_single_position(wrapper):
    """batch_ndims=0 applies a single constrain without vmap."""
    x, y = _linreg_data()
    info = get_log_density_fn(random.key(0), _linreg_model, model_args=(x, y))
    if wrapper:
        out = constrain_samples(
            info.init_position, _linreg_model, model_args=(x, y), batch_ndims=0
        )
    else:
        out = constrain_fn(
            _linreg_model,
            (x, y),
            {},
            info.init_position,
            return_deterministic=True,
            batch_ndims=0,
        )
    assert set(out.keys()) >= {"a", "b", "sigma", "mu"}
    assert out["sigma"].shape == ()  # scalar, no leading batch dim
    assert out["mu"].shape == x.shape
    assert float(out["sigma"]) > 0.0


def test_constrain_fn_batched_chain_matches_loop():
    """batch_ndims=1 vmaps over a chain and matches an explicit Python loop."""
    x, y = _linreg_data()
    info = get_log_density_fn(random.key(0), _linreg_model, model_args=(x, y))
    n = 5
    raw = {
        k: v + 0.1 * jnp.arange(n).reshape((n,) + (1,) * jnp.ndim(v))
        for k, v in info.init_position.items()
    }
    out = constrain_fn(
        _linreg_model, (x, y), {}, raw, return_deterministic=True, batch_ndims=1
    )
    assert out["sigma"].shape == (n,)
    assert out["mu"].shape == (n, x.shape[0])
    # Compare against constraining each draw individually.
    for i in range(n):
        single = constrain_fn(
            _linreg_model,
            (x, y),
            {},
            {k: v[i] for k, v in raw.items()},
            return_deterministic=True,
            batch_ndims=0,
        )
        assert_allclose(out["sigma"][i], single["sigma"], rtol=1e-5)
        assert_allclose(out["mu"][i], single["mu"], rtol=1e-5)


def test_constrain_samples_two_batch_dims():
    """batch_ndims=2 vmaps twice (chains x samples)."""
    x, y = _linreg_data()
    info = get_log_density_fn(random.key(0), _linreg_model, model_args=(x, y))
    chains, draws = 4, 3
    raw = {
        k: jnp.broadcast_to(v, (chains, draws) + jnp.shape(v))
        for k, v in info.init_position.items()
    }
    out = constrain_samples(raw, _linreg_model, model_args=(x, y), batch_ndims=2)
    assert out["sigma"].shape == (chains, draws)
    assert out["mu"].shape == (chains, draws, x.shape[0])


@pytest.mark.parametrize("fn", ["constrain_fn", "constrain_samples"])
def test_constrain_rejects_negative_ndims(fn):
    x, y = _linreg_data()
    info = get_log_density_fn(random.key(0), _linreg_model, model_args=(x, y))
    with pytest.raises(ValueError):
        if fn == "constrain_fn":
            constrain_fn(_linreg_model, (x, y), {}, info.init_position, batch_ndims=-1)
        else:
            constrain_samples(
                info.init_position, _linreg_model, model_args=(x, y), batch_ndims=-1
            )


_RWState = namedtuple("_RWState", ["position", "logdensity", "rng_key"])


class _RandomWalkKernel(MCMCKernel):
    """Minimal :class:`MCMCKernel` driving a random-walk Metropolis sampler.

    Mirrors the notebook pattern (build the bound log-density and postprocess
    via ``initialize_model(bound=True)`` in :meth:`init`) without depending on
    an external sampler library.
    """

    def __init__(self, model, step_size=0.2, init_strategy=init_to_uniform):
        self._model = model
        self._step_size = step_size
        self._init_strategy = init_strategy
        self._logdensity_fn = None
        self._postprocess_fn = None

    @property
    def sample_field(self):
        return "position"

    def postprocess_fn(self, model_args, model_kwargs):
        return self._postprocess_fn

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        info = initialize_model(
            rng_key,
            self._model,
            bound=True,
            init_strategy=self._init_strategy,
            model_args=model_args,
            model_kwargs=model_kwargs,
        )
        self._logdensity_fn = info.logdensity_fn
        self._postprocess_fn = info.postprocess_fn
        position = info.param_info.z if init_params is None else init_params
        return _RWState(position, info.logdensity_fn(position), rng_key)

    def sample(self, state, model_args, model_kwargs):
        rng_key, key_prop, key_accept = random.split(state.rng_key, 3)
        leaves, treedef = jax.tree.flatten(state.position)
        keys = random.split(key_prop, len(leaves))
        proposal_leaves = [
            leaf + self._step_size * random.normal(k, jnp.shape(leaf))
            for leaf, k in zip(leaves, keys)
        ]
        proposal = jax.tree.unflatten(treedef, proposal_leaves)
        proposal_logdensity = self._logdensity_fn(proposal)
        accept_prob = jnp.exp(proposal_logdensity - state.logdensity)
        accept = random.uniform(key_accept) < accept_prob
        position = jax.tree.map(
            lambda p, c: jnp.where(accept, p, c), proposal, state.position
        )
        logdensity = jnp.where(accept, proposal_logdensity, state.logdensity)
        return _RWState(position, logdensity, rng_key)


def test_mcmckernel_subclass_end_to_end():
    """A direct MCMCKernel subclass works with MCMC, postprocess, and Predictive."""
    x, y = _linreg_data()
    kernel = _RandomWalkKernel(_linreg_model, step_size=0.1)
    mcmc = MCMC(kernel, num_warmup=200, num_samples=200, progress_bar=False)
    mcmc.run(random.key(0), x, y)
    samples = mcmc.get_samples()
    # postprocess folded in constrained sites and the deterministic `mu`.
    assert set(samples.keys()) >= {"a", "b", "sigma", "mu"}
    assert np.all(np.asarray(samples["sigma"]) > 0.0)
    assert samples["mu"].shape == (200, x.shape[0])
    # Predictive consumes the constrained posterior samples.
    pred = Predictive(_linreg_model, posterior_samples=samples)(random.key(1), x)
    assert pred["likelihood"].shape[-1] == x.shape[0]
