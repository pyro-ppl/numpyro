# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from numpy.testing import assert_allclose
import pytest

from jax import random
import jax.numpy as jnp

import numpyro
from numpyro.contrib.control_flow import cond, scan
import numpyro.distributions as dist
from numpyro.handlers import mask, seed, substitute, trace
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.util import log_density, potential_energy
from numpyro.optim import Adam


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
    mcmc = MCMC(kernel, num_warmup=100, num_samples=num_samples)
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
    assert_allclose(actual_log_joint, expected_log_joint, rtol=1e-6)


def test_scan_without_stack():
    def multiply_and_add_repeatedly(K, c_in):
        def iteration(c_prev, c_in):
            c_next = jnp.dot(c_prev, K) + c_in
            return c_next, (c_next,)

        _, (ys,) = scan(iteration, init=jnp.asarray([1.0, 0.0]), xs=c_in)

        return ys

    result = multiply_and_add_repeatedly(
        K=jnp.asarray([[0.7, 0.3], [0.3, 0.7]]), c_in=jnp.asarray([[1.0, 0.0]])
    )

    assert_allclose(
        result,
        [[1.7, 0.3]],
    )


def test_cond():
    def model():
        def true_fun(_):
            x = numpyro.sample("x", dist.Normal(4.0))
            numpyro.deterministic("z", x - 4.0)

        def false_fun(_):
            x = numpyro.sample("x", dist.Normal(0.0))
            numpyro.deterministic("z", x)

        cluster = numpyro.sample("cluster", dist.Normal())
        cond(cluster > 0, true_fun, false_fun, None)

    def guide():
        m1 = numpyro.param("m1", 2.0)
        s1 = numpyro.param("s1", 0.1, constraint=dist.constraints.positive)
        m2 = numpyro.param("m2", 2.0)
        s2 = numpyro.param("s2", 0.1, constraint=dist.constraints.positive)

        def true_fun(_):
            numpyro.sample("x", dist.Normal(m1, s1))

        def false_fun(_):
            numpyro.sample("x", dist.Normal(m2, s2))

        cluster = numpyro.sample("cluster", dist.Normal())
        cond(cluster > 0, true_fun, false_fun, None)

    svi = SVI(model, guide, numpyro.optim.Adam(1e-2), Trace_ELBO(num_particles=100))
    svi_result = svi.run(random.PRNGKey(0), num_steps=2500)
    params = svi_result.params

    predictive = Predictive(
        model,
        guide=guide,
        params=params,
        num_samples=1000,
        return_sites=["cluster", "x", "z"],
    )
    result = predictive(random.PRNGKey(0))

    assert result["cluster"].shape == (1000,)
    assert result["x"].shape == (1000,)
    assert result["z"].shape == (1000,)

    mcmc = MCMC(
        NUTS(model),
        num_warmup=500,
        num_samples=2500,
        num_chains=4,
        chain_method="sequential",
    )
    mcmc.run(random.PRNGKey(0))

    x = mcmc.get_samples()["x"]
    assert x.shape == (10_000,)
    assert_allclose(
        [x[x > 2.0].mean(), x[x > 2.0].std(), x[x < 2.0].mean(), x[x < 2.0].std()],
        [4.01, 0.965, -0.01, 0.965],
        atol=0.1,
    )
    assert_allclose([x.mean(), x.std()], [2.0, jnp.sqrt(5.0)], atol=0.5)


def test_scan_promote():
    def model():
        def transition_fn(c, val):
            with numpyro.plate("N", 3, dim=-1):
                numpyro.sample("x", dist.Normal(0, 1), obs=1.0)
            return None, None

        scan(transition_fn, None, None, length=10)

    tr = numpyro.handlers.trace(model).get_trace()
    assert tr["x"]["value"].shape == (10, 1)
    assert tr["x"]["fn"].log_prob(tr["x"]["value"]).shape == (10, 3)


def test_scan_plate_mask():
    def model(y=None, T=12):
        def transition(carry, y_curr):
            x_prev, t = carry
            with numpyro.plate("N", 10, dim=-1):
                with mask(mask=(t < T)):
                    x_curr = numpyro.sample(
                        "x",
                        dist.Normal(jnp.zeros((10, 3)), jnp.ones((10, 3))).to_event(1),
                    )
                    y_curr = numpyro.sample(
                        "y",
                        dist.Normal(x_curr, jnp.ones((10, 3))).to_event(1),
                        obs=y_curr,
                    )
                    return (x_curr, t + 1), None

        x0 = numpyro.sample(
            "x_0", dist.Normal(jnp.zeros((10, 3)), jnp.ones((10, 3))).to_event(1)
        )

        x, t = scan(transition, (x0, 0), y, length=T)
        return (x, y)

    with numpyro.handlers.seed(rng_seed=0):
        model_density, model_trace = log_density(model, (None, 12), {}, {})
        assert model_density
        assert model_trace["x"]["fn"].batch_shape == (12, 10)
        assert model_trace["x"]["fn"].event_shape == (3,)


def test_scan_svi():
    T = 3
    N = 5

    def gaussian_hmm(y=None, T=T, N=N):
        def transition(x_prev, y_curr):
            with numpyro.plate("data", N):
                x_curr = numpyro.sample("x", dist.Normal(x_prev, 1.5))
                y_curr = numpyro.sample("y", dist.Normal(x_curr, 0.1), obs=y_curr)
            return x_curr, (x_curr, y_curr)

        with numpyro.plate("data", N):
            x0 = numpyro.sample("x_0", dist.Normal(jnp.zeros(N), 5.0))
        _, (x, y) = scan(transition, x0, y, length=T)
        return (x, y)

    with numpyro.handlers.seed(rng_seed=0):
        x, y = gaussian_hmm()
    with numpyro.handlers.seed(rng_seed=0):
        tr = numpyro.handlers.trace(gaussian_hmm).get_trace(y=y, T=T, N=N)

    guide = AutoNormal(gaussian_hmm)
    svi = SVI(gaussian_hmm, guide, Adam(0.1), Trace_ELBO(), y=y, T=T, N=N)
    results = svi.run(random.PRNGKey(0), 10**3)

    xhat = results.params["x_auto_loc"]
    assert_allclose(xhat, tr["x"]["value"], rtol=0.1, atol=0.2)


def test_scan_mvn():
    def model():
        def transition(c, a):
            with numpyro.plate("foo", 5):
                c2 = numpyro.sample(
                    "val", dist.MultivariateNormal(c + a, scale_tril=jnp.eye(2))
                )
            return c2, c2

        scan(transition, jnp.zeros((5, 2)), jnp.ones((4, 5, 2)))

    with numpyro.handlers.seed(rng_seed=0), numpyro.handlers.trace() as tr:
        model()
    assert tr["val"]["fn"].batch_shape == (4, 5)
