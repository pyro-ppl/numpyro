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
from numpyro.distributions import constraints
from numpyro.infer import (
    MCMC,
    Predictive,
)
from numpyro.infer.initialization import init_to_uniform
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import (
    ModelInfo,
    constrain_fn,
    initialize_model,
    potential_energy,
)


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


def test_initialize_model_stays_four_fields():
    """Regression guard for the external-sampler integration: ``ModelInfo`` keeps
    exactly four fields so existing four-tuple unpacking of ``initialize_model``
    (used across NumPyro and in user code) does not break."""
    assert ModelInfo._fields == (
        "param_info",
        "potential_fn",
        "postprocess_fn",
        "model_trace",
    )
    x, y = _linreg_data()
    # `a, b, c, d = initialize_model(...)` must keep working.
    param_info, potential_fn, postprocess_fn, model_trace = initialize_model(
        random.key(0), _linreg_model, model_args=(x, y)
    )
    assert set(model_trace.keys()) >= {"a", "b", "sigma"}


def test_initialize_model_inline_logdensity_negates_potential():
    """The inlined `-potential_fn(position)` pattern (used in the notebook in
    place of the removed `get_log_density_fn`) matches `potential_energy`."""
    x, y = _linreg_data()
    model_info = initialize_model(random.key(0), _linreg_model, model_args=(x, y))
    init_position = model_info.param_info.z
    pe = potential_energy(_linreg_model, (x, y), {}, init_position)
    assert_allclose(-model_info.potential_fn(init_position), -pe, rtol=1e-5)
    # postprocess maps unconstrained -> constrained and includes deterministics.
    out = model_info.postprocess_fn(init_position)
    assert set(out.keys()) >= {"a", "b", "sigma", "mu"}
    assert float(out["sigma"]) > 0.0  # Exponential prior -> positive
    assert out["mu"].shape == x.shape


def test_constrain_single_position():
    """batch_ndims=0 applies a single constrain without vmap."""
    x, y = _linreg_data()
    init_position = initialize_model(
        random.key(0), _linreg_model, model_args=(x, y)
    ).param_info.z
    out = constrain_fn(
        _linreg_model,
        (x, y),
        {},
        init_position,
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
    init_position = initialize_model(
        random.key(0), _linreg_model, model_args=(x, y)
    ).param_info.z
    n = 5
    raw = {
        k: v + 0.1 * jnp.arange(n).reshape((n,) + (1,) * jnp.ndim(v))
        for k, v in init_position.items()
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


def _param_site_model(x):
    """Model with a constrained ``numpyro.param`` site to exercise that branch."""
    scale = numpyro.param("scale", 1.0, constraint=constraints.positive)
    loc = numpyro.sample("loc", dist.Normal(0.0, 1.0))
    numpyro.sample("obs", dist.Normal(loc, scale), obs=x)


def test_constrain_fn_batched_param_site():
    """batch_ndims>0 transforms a constrained `param` site and matches a loop."""
    x = jnp.array(0.5)
    n = 4
    raw = {
        "loc": jnp.linspace(-1.0, 1.0, n),
        "scale": jnp.linspace(-0.5, 0.5, n),  # unconstrained value for the param
    }
    out = constrain_fn(_param_site_model, (x,), {}, raw, batch_ndims=1)
    # `scale` carries a positive constraint, so its constrained value is > 0.
    assert out["scale"].shape == (n,)
    assert jnp.all(out["scale"] > 0.0)
    for i in range(n):
        single = constrain_fn(
            _param_site_model,
            (x,),
            {},
            {k: v[i] for k, v in raw.items()},
            batch_ndims=0,
        )
        assert_allclose(out["scale"][i], single["scale"], rtol=1e-5)
        assert_allclose(out["loc"][i], single["loc"], rtol=1e-5)


def test_constrain_fn_two_batch_dims():
    """batch_ndims=2 vmaps twice (chains x samples) with distinct per-cell values."""
    x, y = _linreg_data()
    init_position = initialize_model(
        random.key(0), _linreg_model, model_args=(x, y)
    ).param_info.z
    chains, draws = 4, 3
    # Distinct value per (chain, draw) cell so a wrong vmap axis order would
    # mismatch the manual double loop below (broadcasting alone could not).
    offsets = jnp.arange(chains * draws).reshape(chains, draws)
    raw = {
        k: v + 0.1 * offsets.reshape((chains, draws) + (1,) * jnp.ndim(v))
        for k, v in init_position.items()
    }
    out = constrain_fn(
        _linreg_model, (x, y), {}, raw, return_deterministic=True, batch_ndims=2
    )
    assert out["sigma"].shape == (chains, draws)
    assert out["mu"].shape == (chains, draws, x.shape[0])
    # Match an explicit chain x draw loop of the single-position constrain.
    for c in range(chains):
        for d in range(draws):
            single = constrain_fn(
                _linreg_model,
                (x, y),
                {},
                {k: v[c, d] for k, v in raw.items()},
                return_deterministic=True,
                batch_ndims=0,
            )
            assert_allclose(out["sigma"][c, d], single["sigma"], rtol=1e-5)
            assert_allclose(out["mu"][c, d], single["mu"], rtol=1e-5)


def test_constrain_fn_rejects_negative_ndims():
    x, y = _linreg_data()
    init_position = initialize_model(
        random.key(0), _linreg_model, model_args=(x, y)
    ).param_info.z
    with pytest.raises(ValueError):
        constrain_fn(_linreg_model, (x, y), {}, init_position, batch_ndims=-1)


_RWState = namedtuple("_RWState", ["position", "logdensity", "rng_key"])


class _RandomWalkKernel(MCMCKernel):
    """Minimal :class:`MCMCKernel` driving a random-walk Metropolis sampler.

    Mirrors the notebook pattern (build the log-density and postprocess inline
    from :func:`initialize_model` in :meth:`init`) without depending on an
    external sampler library.
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
        model_info = initialize_model(
            rng_key,
            self._model,
            init_strategy=self._init_strategy,
            model_args=model_args,
            model_kwargs=model_kwargs,
        )

        def logdensity_fn(position):
            return -model_info.potential_fn(position)

        self._logdensity_fn = logdensity_fn
        self._postprocess_fn = model_info.postprocess_fn
        position = model_info.param_info.z if init_params is None else init_params
        return _RWState(position, logdensity_fn(position), rng_key)

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
