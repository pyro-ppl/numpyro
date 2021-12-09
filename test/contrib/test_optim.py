# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

from numpy.testing import assert_allclose
import pytest

from jax import grad, jit, random
from jax.lax import fori_loop
import jax.numpy as jnp
from jax.test_util import check_close

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import SVI, RenyiELBO, Trace_ELBO

try:
    import optax

    from numpyro.contrib.optim import optax_to_numpyro

    # the optimizer test is parameterized by different optax optimizers, but we have
    # to define them here to ensure that `optax` is defined. pytest.mark.parameterize
    # decorators are run even if tests are skipped at the top of the file.
    optimizers = [
        (optax.adam, (1e-2,), {}),
        # clipped adam
        (optax.chain, (optax.clip(10.0), optax.adam(1e-2)), {}),
        (optax.adagrad, (1e-1,), {}),
        # SGD with momentum
        (optax.sgd, (1e-2,), {"momentum": 0.9}),
        (optax.rmsprop, (1e-2,), {"decay": 0.95}),
        # RMSProp with momentum
        (optax.rmsprop, (1e-4,), {"decay": 0.9, "momentum": 0.9}),
        (optax.sgd, (1e-2,), {}),
    ]
except ImportError:
    pytestmark = pytest.mark.skip(reason="optax is not installed")
    optimizers = []


def loss(params):
    return jnp.sum(params["x"] ** 2 + params["y"] ** 2)


@partial(jit, static_argnums=(1,))
def step(opt_state, optim):
    params = optim.get_params(opt_state)
    g = grad(loss)(params)
    return optim.update(g, opt_state)


@pytest.mark.parametrize("optim_class, args, kwargs", optimizers)
def test_optim_multi_params(optim_class, args, kwargs):
    params = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([-1, -1.0, -1.0])}
    opt = optax_to_numpyro(optim_class(*args, **kwargs))
    opt_state = opt.init(params)
    for i in range(2000):
        opt_state = step(opt_state, opt)
    for _, param in opt.get_params(opt_state).items():
        assert jnp.allclose(param, jnp.zeros(3))


@pytest.mark.parametrize("elbo", [Trace_ELBO(), RenyiELBO(num_particles=10)])
def test_beta_bernoulli(elbo):
    data = jnp.array([1.0] * 8 + [0.0] * 2)

    def model(data):
        f = numpyro.sample("beta", dist.Beta(1.0, 1.0))
        numpyro.sample("obs", dist.Bernoulli(f), obs=data)

    def guide(data):
        alpha_q = numpyro.param("alpha_q", 1.0, constraint=constraints.positive)
        beta_q = numpyro.param("beta_q", 1.0, constraint=constraints.positive)
        numpyro.sample("beta", dist.Beta(alpha_q, beta_q))

    adam = optax.adam(0.05)
    svi = SVI(model, guide, adam, elbo)
    svi_state = svi.init(random.PRNGKey(1), data)
    assert_allclose(svi.optim.get_params(svi_state.optim_state)["alpha_q"], 0.0)

    def body_fn(i, val):
        svi_state, _ = svi.update(val, data)
        return svi_state

    svi_state = fori_loop(0, 2000, body_fn, svi_state)
    params = svi.get_params(svi_state)
    assert_allclose(
        params["alpha_q"] / (params["alpha_q"] + params["beta_q"]),
        0.8,
        atol=0.05,
        rtol=0.05,
    )


def test_jitted_update_fn():
    data = jnp.array([1.0] * 8 + [0.0] * 2)

    def model(data):
        f = numpyro.sample("beta", dist.Beta(1.0, 1.0))
        numpyro.sample("obs", dist.Bernoulli(f), obs=data)

    def guide(data):
        alpha_q = numpyro.param("alpha_q", 1.0, constraint=constraints.positive)
        beta_q = numpyro.param("beta_q", 1.0, constraint=constraints.positive)
        numpyro.sample("beta", dist.Beta(alpha_q, beta_q))

    adam = optax.adam(0.05)
    svi = SVI(model, guide, adam, Trace_ELBO())
    svi_state = svi.init(random.PRNGKey(1), data)
    expected = svi.get_params(svi.update(svi_state, data)[0])

    actual = svi.get_params(jit(svi.update)(svi_state, data=data)[0])
    check_close(actual, expected, atol=1e-5)
