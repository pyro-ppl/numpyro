# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from numpy.testing import assert_allclose
import pytest

from jax import random
import jax.numpy as jnp

import numpyro
from numpyro.contrib.control_flow import cond, scan
import numpyro.distributions as dist
from numpyro.handlers import mask, replay, seed, substitute, trace
from numpyro.infer import (
    MCMC,
    NUTS,
    SVI,
    Predictive,
    RenyiELBO,
    Trace_ELBO,
    TraceGraph_ELBO,
    TraceMeanField_ELBO,
)
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
    mcmc.run(random.key(0), T=T)
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
    result = predictive(random.key(1), T=T + future)
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
    svi_result = svi.run(random.key(0), num_steps=2500)
    params = svi_result.params

    predictive = Predictive(
        model,
        guide=guide,
        params=params,
        num_samples=1000,
        return_sites=["cluster", "x", "z"],
    )
    result = predictive(random.key(0))

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
    mcmc.run(random.key(0))

    x = mcmc.get_samples()["x"]
    assert x.shape == (10_000,)
    assert_allclose(
        [x[x > 2.0].mean(), x[x > 2.0].std(), x[x < 2.0].mean(), x[x < 2.0].std()],
        [4.01, 0.965, -0.01, 0.965],
        atol=0.1,
    )
    assert_allclose([x.mean(), x.std()], [2.0, jnp.sqrt(5.0)], atol=0.5)


def _cond_replay_models():
    def model(y):
        switch = numpyro.sample("switch", dist.Normal(0.0, 1.0))
        level = cond(
            switch > 0,
            lambda _: numpyro.sample("level", dist.Normal(10.0, 2.0)),
            lambda _: numpyro.sample("level", dist.Normal(0.0, 2.0)),
            None,
        )
        with numpyro.plate("data", y.shape[0]):
            numpyro.sample("y", dist.Normal(level, 1.0), obs=y)

    # the same joint density without `cond`
    def model_without_cond(y):
        switch = numpyro.sample("switch", dist.Normal(0.0, 1.0))
        loc = jnp.where(switch > 0, 10.0, 0.0)
        level = numpyro.sample("level", dist.Normal(loc, 2.0))
        with numpyro.plate("data", y.shape[0]):
            numpyro.sample("y", dist.Normal(level, 1.0), obs=y)

    def guide(y):
        numpyro.sample("switch", dist.Normal(1.0, 0.5))
        numpyro.sample("level", dist.Normal(8.0, 0.3))

    return model, model_without_cond, guide


def test_cond_replay():
    model, _, guide = _cond_replay_models()
    y = jnp.array([8.1, 9.2, 7.7])
    guide_trace = trace(seed(guide, 0)).get_trace(y)
    model_trace = trace(seed(replay(model, guide_trace), 1)).get_trace(y)
    # the value replayed inside `cond` is the one used after `cond`
    assert_allclose(model_trace["y"]["fn"].mean, guide_trace["level"]["value"])


@pytest.mark.parametrize(
    "elbo", [Trace_ELBO, TraceMeanField_ELBO, TraceGraph_ELBO, RenyiELBO]
)
def test_cond_replay_elbo(elbo):
    model, model_without_cond, guide = _cond_replay_models()
    y = jnp.array([8.1, 9.2, 7.7])
    loss = elbo(num_particles=100).loss(random.key(0), {}, model, guide, y)
    expected = elbo(num_particles=100).loss(
        random.key(0), {}, model_without_cond, guide, y
    )
    assert_allclose(loss, expected, rtol=1e-5)


@pytest.mark.parametrize("elbo", [Trace_ELBO, TraceGraph_ELBO])
def test_cond_in_scan_replay_elbo(elbo):
    def model(ys):
        def transition(carry, y):
            switch = numpyro.sample("switch", dist.Normal(0.0, 1.0))
            level = cond(
                switch > 0,
                lambda _: numpyro.sample("level", dist.Normal(10.0, 2.0)),
                lambda _: numpyro.sample("level", dist.Normal(0.0, 2.0)),
                None,
            )
            numpyro.sample("y", dist.Normal(level, 1.0), obs=y)
            return carry, None

        scan(transition, 0.0, ys)

    # the same joint density without `cond`
    def model_without_cond(ys):
        def transition(carry, y):
            switch = numpyro.sample("switch", dist.Normal(0.0, 1.0))
            loc = jnp.where(switch > 0, 10.0, 0.0)
            level = numpyro.sample("level", dist.Normal(loc, 2.0))
            numpyro.sample("y", dist.Normal(level, 1.0), obs=y)
            return carry, None

        scan(transition, 0.0, ys)

    def guide(ys):
        def transition(carry, y):
            numpyro.sample("switch", dist.Normal(1.0, 0.5))
            numpyro.sample("level", dist.Normal(8.0, 0.3))
            return carry, None

        scan(transition, 0.0, ys)

    ys = jnp.array([8.1, 9.2, 7.7, 8.8])
    loss = elbo(num_particles=100).loss(random.key(0), {}, model, guide, ys)
    expected = elbo(num_particles=100).loss(
        random.key(0), {}, model_without_cond, guide, ys
    )
    assert_allclose(loss, expected, rtol=1e-5)


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
    results = svi.run(random.key(0), 10**3)

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
