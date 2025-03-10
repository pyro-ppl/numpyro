# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import pytest

from jax import grad, jit
import jax.numpy as jnp

from numpyro import optim

try:
    import optax
    import optax.contrib

    # the optimizer test is parameterized by different optax optimizers, but we have
    # to define them here to ensure that `optax` is defined. pytest.mark.parameterize
    # decorators are run even if tests are skipped at the top of the file.
    optax_optimizers = [
        (optax.adam, (1e-2,), {}, False),
        # clipped adam
        (optax.chain, (optax.clip(10.0), optax.adam(1e-2)), {}, False),
        (optax.adagrad, (1e-1,), {}, False),
        # SGD with momentum
        (optax.sgd, (1e-2,), {"momentum": 0.9}, False),
        (optax.rmsprop, (1e-2,), {"decay": 0.95}, False),
        # RMSProp with momentum
        (optax.rmsprop, (1e-4,), {"decay": 0.9, "momentum": 0.9}, False),
        (optax.sgd, (1e-2,), {}, False),
        # reduce learning rate on plateau
        (
            optax.chain,
            (
                optax.adam(1e-2),
                optax.contrib.reduce_on_plateau(patience=5, accumulation_size=200),
            ),
            {},
            True,
        ),
    ]
except ImportError:
    pytestmark = pytest.mark.skip(reason="optax is not installed")
    optax_optimizers = []


def loss(params):
    return jnp.sum(params["x"] ** 2 + params["y"] ** 2)


@partial(jit, static_argnums=(1,))
def step(opt_state, optim):
    params = optim.get_params(opt_state)
    g = grad(loss)(params)
    if optim.update_with_value:
        return optim.update(g, opt_state, value=loss(params))
    else:
        return optim.update(g, opt_state)


@pytest.mark.parametrize(
    "optim_class, args, kwargs, uses_value_arg",
    [
        (optim.Adam, (1e-2,), {}, False),
        (optim.ClippedAdam, (1e-2,), {}, False),
        (optim.Adagrad, (1e-1,), {}, False),
        (optim.Momentum, (1e-2, 0.5), {}, False),
        (optim.RMSProp, (1e-2, 0.95), {}, False),
        (optim.RMSPropMomentum, (1e-4,), {}, False),
        (optim.SGD, (1e-2,), {}, False),
    ]
    + optax_optimizers,
)
@pytest.mark.filterwarnings("ignore:.*tree_multimap:FutureWarning")
def test_optim_multi_params(optim_class, args, kwargs, uses_value_arg):
    params = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([-1, -1.0, -1.0])}
    opt = optim_class(*args, **kwargs)
    if not isinstance(opt, optim._NumPyroOptim):
        opt = optim.optax_to_numpyro(opt)
    opt_state = opt.init(params)
    for i in range(2000):
        opt_state = step(opt_state, opt)
    for _, param in opt.get_params(opt_state).items():
        assert jnp.allclose(param, jnp.zeros(3))


# note: this is somewhat of a bruteforce test. testing directly from
# _NumpyroOptim would probably be better
@pytest.mark.parametrize(
    "optim_class, args, kwargs, uses_value_arg",
    [
        (optim.Adam, (1e-2,), {}, False),
        (optim.ClippedAdam, (1e-2,), {}, False),
        (optim.Adagrad, (1e-1,), {}, False),
        (optim.Momentum, (1e-2, 0.5), {}, False),
        (optim.RMSProp, (1e-2, 0.95), {}, False),
        (optim.RMSPropMomentum, (1e-4,), {}, False),
        (optim.SGD, (1e-2,), {}, False),
    ]
    + optax_optimizers,
)
@pytest.mark.filterwarnings("ignore:.*tree_multimap:FutureWarning")
def test_numpyrooptim_no_double_jit(optim_class, args, kwargs, uses_value_arg):
    opt = optim_class(*args, **kwargs)
    if not isinstance(opt, optim._NumPyroOptim):
        opt = optim.optax_to_numpyro(opt)
    state = opt.init(jnp.zeros(10))

    my_fn_calls = 0

    @jit
    def my_fn(state, g):
        nonlocal my_fn_calls
        my_fn_calls += 1

        if opt.update_with_value:
            state = opt.update(g, state, value=0.01)
        else:
            state = opt.update(g, state)
        return state

    state = my_fn(state, jnp.ones(10) * 1.0)
    state = my_fn(state, jnp.ones(10) * 2.0)
    state = my_fn(state, jnp.ones(10) * 3.0)

    if uses_value_arg:
        # Dtype is different on the first call vs the rest of the calls
        assert my_fn_calls == 2
    else:
        assert my_fn_calls == 1
