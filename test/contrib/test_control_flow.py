# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from numpy.testing import assert_allclose
import pytest

from jax import random
import jax.numpy as jnp

import numpyro
from numpyro.contrib.control_flow.scan import scan
import numpyro.distributions as dist
from numpyro.handlers import seed, substitute, trace
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.util import potential_energy


def test_scan():
    def model(T=10, q=1, r=1, phi=0.0, beta=0.0):
        def transition(state, i):
            x0, mu0 = state
            x1 = numpyro.sample("x", dist.Normal(phi * x0, q))
            mu1 = beta * mu0 + x1
            y1 = numpyro.sample("y", dist.Normal(mu1, r))
            numpyro.deterministic("y2", y1 * 2)
            return (x1, mu1), (x1, y1)

        mu0 = x0 = numpyro.sample("x_0", dist.Normal(0, q))
        y0 = numpyro.sample("y_0", dist.Normal(mu0, r))

        _, xy = scan(transition, (x0, mu0), jnp.arange(T))
        x, y = xy

        return jnp.append(x0, x), jnp.append(y0, y)

    T = 10
    num_samples = 100
    kernel = NUTS(model)
    mcmc = MCMC(kernel, 100, num_samples)
    mcmc.run(random.PRNGKey(0), T=T)
    assert set(mcmc.get_samples()) == {"x", "y", "y2", "x_0", "y_0"}
    mcmc.print_summary()

    samples = mcmc.get_samples()
    x = samples.pop("x")[0]  # take 1 sample of x
    # this tests for the composition of condition and substitute
    # this also tests if we can use `vmap` for predictive.
    future = 5
    predictive = Predictive(
        numpyro.handlers.condition(model, {"x": x}),
        samples,
        return_sites=["x", "y", "y2"],
        parallel=True,
    )
    result = predictive(random.PRNGKey(1), T=T + future)
    expected_shape = (num_samples, T + future)
    assert result["x"].shape == expected_shape
    assert result["y"].shape == expected_shape
    assert result["y2"].shape == expected_shape
    assert_allclose(result["x"][:, :T], jnp.broadcast_to(x, (num_samples, T)))
    assert_allclose(result["y"][:, :T], samples["y"])


@pytest.mark.xfail(raises=RuntimeError)
def test_nested_scan_smoke():
    def model():
        def outer_fn(y, val):
            def body_fn(z, val):
                z = numpyro.sample("z", dist.Normal(z, 1))
                return z, z

            y = numpyro.sample("y", dist.Normal(y, 1))
            _, zs = scan(body_fn, y, None, 4)
            return y, zs

        x = numpyro.sample("x", dist.Normal(0, 1))
        _, zs = scan(outer_fn, x, None, 3)
        return zs

    data = jnp.arange(12).reshape((3, 4))
    # we can scan but can't substitute values through multiple levels of scan
    with trace(), seed(rng_seed=0), substitute(data={"z": data}):
        zs = model()
    assert_allclose(zs, data)


def test_scan_constrain_reparam_compatible():
    def model(T, q=1, r=1, phi=0.0, beta=0.0):
        x = 0.0
        mu = 0.0
        for i in range(T):
            x = numpyro.sample(f"x_{i}", dist.LogNormal(phi * x, q))
            mu = beta * mu + x
            numpyro.sample(f"y_{i}", dist.Normal(mu, r))

    def fun_model(T, q=1, r=1, phi=0.0, beta=0.0):
        def transition(state, i):
            x, mu = state
            x = numpyro.sample("x", dist.LogNormal(phi * x, q))
            mu = beta * mu + x
            numpyro.sample("y", dist.Normal(mu, r))
            return (x, mu), None

        scan(transition, (0.0, 0.0), jnp.arange(T))

    T = 10
    params = {}
    for i in range(T):
        params[f"x_{i}"] = (i + 1.0) / 10
        params[f"y_{i}"] = -i / 5
    fun_params = {"x": jnp.arange(1, T + 1) / 10, "y": -jnp.arange(T) / 5}
    actual_log_joint = potential_energy(fun_model, (T,), {}, fun_params)
    expected_log_joint = potential_energy(model, (T,), {}, params)
    assert_allclose(actual_log_joint, expected_log_joint)
