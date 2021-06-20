# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from numpy.testing import assert_allclose
import pytest

import jax
from jax import jit, random, value_and_grad
import jax.numpy as jnp
from jax.test_util import check_close

import numpyro
from numpyro import optim
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.transforms import AffineTransform, SigmoidTransform
from numpyro.handlers import substitute
from numpyro.infer import SVI, RenyiELBO, Trace_ELBO, TraceMeanField_ELBO
from numpyro.util import fori_loop


@pytest.mark.parametrize("alpha", [0.0, 2.0])
def test_renyi_elbo(alpha):
    def model(x):
        numpyro.sample("obs", dist.Normal(0, 1), obs=x)

    def guide(x):
        pass

    def elbo_loss_fn(x):
        return Trace_ELBO().loss(random.PRNGKey(0), {}, model, guide, x)

    def renyi_loss_fn(x):
        return RenyiELBO(alpha=alpha, num_particles=10).loss(
            random.PRNGKey(0), {}, model, guide, x
        )

    elbo_loss, elbo_grad = value_and_grad(elbo_loss_fn)(2.0)
    renyi_loss, renyi_grad = value_and_grad(renyi_loss_fn)(2.0)
    assert_allclose(elbo_loss, renyi_loss, rtol=1e-6)
    assert_allclose(elbo_grad, renyi_grad, rtol=1e-6)


@pytest.mark.parametrize("elbo", [Trace_ELBO(), RenyiELBO(num_particles=10)])
@pytest.mark.parametrize(
    "optimizer", [optim.Adam(0.05), jax.experimental.optimizers.adam(0.05)]
)
def test_beta_bernoulli(elbo, optimizer):
    data = jnp.array([1.0] * 8 + [0.0] * 2)

    def model(data):
        f = numpyro.sample("beta", dist.Beta(1.0, 1.0))
        numpyro.sample("obs", dist.Bernoulli(f), obs=data)

    def guide(data):
        alpha_q = numpyro.param("alpha_q", 1.0, constraint=constraints.positive)
        beta_q = numpyro.param("beta_q", 1.0, constraint=constraints.positive)
        numpyro.sample("beta", dist.Beta(alpha_q, beta_q))

    svi = SVI(model, guide, optimizer, elbo)
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


@pytest.mark.parametrize("progress_bar", [True, False])
def test_run(progress_bar):
    data = jnp.array([1.0] * 8 + [0.0] * 2)

    def model(data):
        f = numpyro.sample("beta", dist.Beta(1.0, 1.0))
        numpyro.sample("obs", dist.Bernoulli(f), obs=data)

    def guide(data):
        alpha_q = numpyro.param(
            "alpha_q", lambda key: random.normal(key), constraint=constraints.positive
        )
        beta_q = numpyro.param(
            "beta_q",
            lambda key: random.exponential(key),
            constraint=constraints.positive,
        )
        numpyro.sample("beta", dist.Beta(alpha_q, beta_q))

    svi = SVI(model, guide, optim.Adam(0.05), Trace_ELBO())
    svi_result = svi.run(random.PRNGKey(1), 1000, data, progress_bar=progress_bar)
    params, losses = svi_result.params, svi_result.losses
    assert losses.shape == (1000,)
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

    adam = optim.Adam(0.05)
    svi = SVI(model, guide, adam, Trace_ELBO())
    svi_state = svi.init(random.PRNGKey(1), data)
    expected = svi.get_params(svi.update(svi_state, data)[0])

    actual = svi.get_params(jit(svi.update)(svi_state, data=data)[0])
    check_close(actual, expected, atol=1e-5)


def test_param():
    # this test the validity of model/guide sites having
    # param constraints contain composed transformed
    rng_keys = random.split(random.PRNGKey(0), 5)
    a_minval = 1
    c_minval = -2
    c_maxval = -1
    a_init = jnp.exp(random.normal(rng_keys[0])) + a_minval
    b_init = jnp.exp(random.normal(rng_keys[1]))
    c_init = random.uniform(rng_keys[2], minval=c_minval, maxval=c_maxval)
    d_init = random.uniform(rng_keys[3])
    obs = random.normal(rng_keys[4])

    def model():
        a = numpyro.param("a", a_init, constraint=constraints.greater_than(a_minval))
        b = numpyro.param("b", b_init, constraint=constraints.positive)
        numpyro.sample("x", dist.Normal(a, b), obs=obs)

    def guide():
        c = numpyro.param(
            "c", c_init, constraint=constraints.interval(c_minval, c_maxval)
        )
        d = numpyro.param("d", d_init, constraint=constraints.unit_interval)
        numpyro.sample("y", dist.Normal(c, d), obs=obs)

    adam = optim.Adam(0.01)
    svi = SVI(model, guide, adam, Trace_ELBO())
    svi_state = svi.init(random.PRNGKey(0))

    params = svi.get_params(svi_state)
    assert_allclose(params["a"], a_init)
    assert_allclose(params["b"], b_init)
    assert_allclose(params["c"], c_init)
    assert_allclose(params["d"], d_init)

    actual_loss = svi.evaluate(svi_state)
    assert jnp.isfinite(actual_loss)
    expected_loss = dist.Normal(c_init, d_init).log_prob(obs) - dist.Normal(
        a_init, b_init
    ).log_prob(obs)
    # not so precisely because we do transform / inverse transform stuffs
    assert_allclose(actual_loss, expected_loss, rtol=1e-6)


def test_elbo_dynamic_support():
    x_prior = dist.TransformedDistribution(
        dist.Normal(),
        [AffineTransform(0, 2), SigmoidTransform(), AffineTransform(0, 3)],
    )
    x_guide = dist.Uniform(0, 3)

    def model():
        numpyro.sample("x", x_prior)

    def guide():
        numpyro.sample("x", x_guide)

    adam = optim.Adam(0.01)
    x = 2.0
    guide = substitute(guide, data={"x": x})
    svi = SVI(model, guide, adam, Trace_ELBO())
    svi_state = svi.init(random.PRNGKey(0))
    actual_loss = svi.evaluate(svi_state)
    assert jnp.isfinite(actual_loss)
    expected_loss = x_guide.log_prob(x) - x_prior.log_prob(x)
    assert_allclose(actual_loss, expected_loss)


@pytest.mark.parametrize("num_steps", [10, 30, 50])
def test_run_with_small_num_steps(num_steps):
    def model():
        pass

    def guide():
        pass

    svi = SVI(model, guide, optim.Adam(1), Trace_ELBO())
    svi.run(random.PRNGKey(0), num_steps)


@pytest.mark.parametrize("stable_run", [True, False])
def test_stable_run(stable_run):
    def model():
        var = numpyro.sample("var", dist.Exponential(1))
        numpyro.sample("obs", dist.Normal(0, jnp.sqrt(var)), obs=0.0)

    def guide():
        loc = numpyro.param("loc", 0.0)
        numpyro.sample("var", dist.Normal(loc, 10))

    svi = SVI(model, guide, optim.Adam(1), Trace_ELBO())
    svi_result = svi.run(random.PRNGKey(0), 1000, stable_update=stable_run)
    assert jnp.isfinite(svi_result.params["loc"]) == stable_run


def test_svi_discrete_latent():
    def model():
        numpyro.sample("x", dist.Bernoulli(0.5))

    def guide():
        probs = numpyro.param("probs", 0.2)
        numpyro.sample("x", dist.Bernoulli(probs))

    svi = SVI(model, guide, optim.Adam(1), Trace_ELBO())
    with pytest.warns(UserWarning, match="SVI does not support models with discrete"):
        svi.run(random.PRNGKey(0), 10)


@pytest.mark.parametrize("stable_update", [True, False])
@pytest.mark.parametrize("num_particles", [1, 10])
@pytest.mark.parametrize("elbo", [Trace_ELBO, TraceMeanField_ELBO])
def test_mutable_state(stable_update, num_particles, elbo):
    def model():
        x = numpyro.sample("x", dist.Normal(-1, 1))
        numpyro.param("x1p", x + 1, infer={"mutable": True})

    def guide():
        loc = numpyro.param("loc", 0.0)
        p = numpyro.param("loc1p", {"value": None}, infer={"mutable": True})
        # we can modify the content of `p` if it is a dict
        p["value"] = loc + 2
        numpyro.sample("x", dist.Normal(loc, 1))

    svi = SVI(model, guide, optim.Adam(0.1), elbo(num_particles=num_particles))
    if num_particles > 1:
        with pytest.raises(ValueError, match="mutable state"):
            svi_result = svi.run(random.PRNGKey(0), 1000, stable_update=stable_update)
        return
    svi_result = svi.run(random.PRNGKey(0), 1000, stable_update=stable_update)
    params = svi_result.params
    assert set(params.keys()) == {"x1p", "loc", "loc1p"}
    assert_allclose(params["loc1p"]["value"], params["loc"] + 2, atol=0.1)
