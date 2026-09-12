# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the composable-kernel hooks on `MCMCKernel` and the public `HMC` accessors."""

import pytest

from jax import random, value_and_grad
import jax.numpy as jnp

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import HMC, MCMC, NUTS
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import (
    _prepare_model_for_potential,
    _transforms_from_trace,
    _unconstrain_params,
    initialize_model,
    potential_energy,
)
from numpyro.util import _get_nested_attr


def model(scale=1.0):
    x = numpyro.sample("x", dist.Normal(0.0, scale))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    numpyro.deterministic("x2", x**2)
    numpyro.sample("obs", dist.Normal(x, sigma), obs=jnp.array([0.3, -0.2]))


def dynamic_support_model():
    lb = numpyro.sample("lb", dist.Normal(0.0, 1.0))
    numpyro.sample("y", dist.Uniform(lb, lb + 1.0))


def _init(kernel, *args, **kwargs):
    return kernel.init(random.key(0), 10, None, model_args=args, model_kwargs=kwargs)


def test_mcmc_kernel_defaults_raise():
    class Dummy(MCMCKernel):
        sample_field = "z"

        def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
            return None

        def sample(self, state, model_args, model_kwargs):
            return state

    with pytest.raises(NotImplementedError, match="refresh"):
        Dummy().refresh(None, (), {})
    with pytest.raises(NotImplementedError, match="wrap_model"):
        Dummy().wrap_model(lambda m: m)
    # the default constrain function is the (identity) postprocess function
    assert Dummy().get_constrain_fn((), {})({"z": 1.0}) == {"z": 1.0}


@pytest.mark.parametrize("forward_mode", [False, True])
def test_hmc_refresh_matches_value_and_grad(forward_mode):
    kernel = NUTS(model, forward_mode_differentiation=forward_mode)
    with pytest.raises(RuntimeError, match="init"):
        kernel.get_potential_fn((2.0,), {})
    state = _init(kernel, 2.0)
    # pretend the cached values are stale
    stale = state._replace(
        potential_energy=jnp.zeros(()), z_grad={k: 0 * v for k, v in state.z.items()}
    )
    refreshed = kernel.refresh(stale, (2.0,), {})
    pe, z_grad = value_and_grad(kernel.get_potential_fn((2.0,), {}))(state.z)
    assert jnp.allclose(refreshed.potential_energy, pe)
    for k in z_grad:
        assert jnp.allclose(refreshed.z_grad[k], z_grad[k])
    assert refreshed.energy is state.energy
    # a different model argument gives a different potential
    assert not jnp.allclose(kernel.refresh(stale, (0.1,), {}).potential_energy, pe)


def test_hmc_get_constrain_fn():
    kernel = HMC(model)
    with pytest.raises(RuntimeError, match="init"):
        kernel.get_constrain_fn((), {})
    state = _init(kernel)
    transform_only = kernel.get_constrain_fn((), {})(state.z)
    replay = kernel.get_constrain_fn((), {}, return_deterministic=True)(state.z)
    assert set(transform_only) == {"x", "sigma"}
    assert set(replay) == {"x", "sigma", "x2"}
    for k in transform_only:
        assert jnp.allclose(transform_only[k], replay[k])
    assert transform_only["sigma"] > 0
    assert jnp.allclose(replay["x2"], replay["x"] ** 2)


def test_hmc_get_constrain_fn_dynamic_support():
    kernel = HMC(dynamic_support_model)
    state = _init(kernel)
    assert kernel._dynamic_support
    constrained = kernel.get_constrain_fn((), {})(state.z)
    assert constrained["lb"] < constrained["y"] < constrained["lb"] + 1.0


def test_hmc_wrap_model():
    kernel = NUTS(model)
    _init(kernel, 2.0)
    calls = []

    def wrapper(m):
        def wrapped(*args, **kwargs):
            calls.append(1)
            return m(*args, **kwargs)

        return wrapped

    wrapped = kernel.wrap_model(wrapper)
    assert wrapped is not kernel
    assert wrapped.model is not kernel.model
    for attr in ("_init_fn", "_sample_fn", "_potential_fn_gen", "_postprocess_fn"):
        assert getattr(wrapped, attr) is None
        assert getattr(kernel, attr) is not None
    _init(wrapped, 2.0)
    assert calls

    with pytest.raises(ValueError, match="potential function"):
        HMC(potential_fn=lambda z: z["x"] ** 2).wrap_model(wrapper)


def test_get_nested_attr_tuple_index():
    obj = {"a": (0, {"b": 3})}
    assert _get_nested_attr(obj, "a.1.b") == 3
    assert _get_nested_attr(obj, "a.0") == 0


def test_unconstrain_params_on_stack():
    found = {}

    class spy(handlers.Messenger):
        def process_message(self, msg):
            if msg["type"] == "sample" and msg["name"] == "obs":
                for handler in numpyro.primitives._PYRO_STACK[::-1]:
                    if isinstance(handler, _unconstrain_params):
                        found["params"] = handler.params

    params = {"x": jnp.array(0.1), "sigma": jnp.array(-0.3)}
    potential_energy(spy(model), (2.0,), {}, params)
    assert found["params"] is params


def test_prepare_model_for_potential_matches_initialize_model():
    rng_key = random.key(1)
    info = initialize_model(rng_key, model, model_args=(2.0,))
    prepared = _prepare_model_for_potential(model, info.model_trace, enum=False)
    z = info.param_info.z
    assert jnp.allclose(potential_energy(prepared, (2.0,), {}, z), info.potential_fn(z))


def test_transforms_from_trace_flags():
    trace = handlers.trace(handlers.seed(model, random.key(0))).get_trace(2.0)
    info = _transforms_from_trace(trace, raise_warnings=False)
    assert set(info.inv_transforms) == {"x", "sigma"}
    assert info.has_deterministic and not info.dynamic_support
    assert not info.has_enumerate_support
    trace = handlers.trace(
        handlers.seed(dynamic_support_model, random.key(0))
    ).get_trace()
    info = _transforms_from_trace(trace, raise_warnings=False)
    assert info.dynamic_support and not info.has_deterministic


def test_mcmc_still_runs():
    mcmc = MCMC(NUTS(model), num_warmup=5, num_samples=5, progress_bar=False)
    mcmc.run(random.key(0), 2.0)
    assert set(mcmc.get_samples()) == {"x", "sigma", "x2"}
    # re-init of an already-run kernel still works
    mcmc.run(random.key(1), 2.0)
