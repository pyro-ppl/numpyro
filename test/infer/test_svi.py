# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import numpy as np
from numpy.testing import assert_allclose
import pytest

import jax
from jax import jit, lax, random, value_and_grad
from jax.example_libraries import optimizers
import jax.numpy as jnp

import numpyro
from numpyro import optim
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.transforms import AffineTransform, SigmoidTransform
from numpyro.handlers import substitute
from numpyro.infer import (
    SVI,
    RenyiELBO,
    Trace_ELBO,
    TraceGraph_ELBO,
    TraceMeanField_ELBO,
)
from numpyro.infer.elbo import _apply_vmap
from numpyro.primitives import mutable as numpyro_mutable
from numpyro.util import fori_loop


def assert_equal(a, b, prec=0):
    return jax.tree.map(lambda a, b: assert_allclose(a, b, atol=prec), a, b)


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


def test_renyi_local():
    def model(subsample_size=None):
        with numpyro.plate("N", 100, subsample_size=subsample_size):
            numpyro.sample("x", dist.Normal(0, 1))
            numpyro.sample("obs", dist.Bernoulli(0.6), obs=1)

    def guide(subsample_size=None):
        with numpyro.plate("N", 100, subsample_size=subsample_size):
            numpyro.sample("x", dist.Normal(0, 1))

    def renyi_loss_fn(subsample_size=None):
        return RenyiELBO(num_particles=10).loss(
            random.PRNGKey(0), {}, model, guide, subsample_size
        )

    # Test that the scales are applied correctly.
    # Here for each particle, log_p - log_q = log(0.6)
    full_loss = renyi_loss_fn()
    subsample_loss = renyi_loss_fn(subsample_size=2)
    assert_allclose(full_loss, -jnp.log(0.6) * 100, rtol=1e-6)
    assert_allclose(subsample_loss, full_loss, rtol=1e-6)


def test_renyi_nonnested_plates():
    def model():
        with numpyro.plate("N", 10):
            numpyro.sample("x", dist.Normal(0, 1))

        with numpyro.plate("M", 10):
            numpyro.sample("y", dist.Normal(0, 1))

    def get_elbo():
        renyi_elbo = RenyiELBO(num_particles=10)
        return renyi_elbo._single_particle_elbo(
            model,
            model,
            {},
            (),
            {},
            random.PRNGKey(0),
        )

    elbo, _ = get_elbo()
    assert elbo.shape == ()


@pytest.mark.parametrize(
    "n,k",
    [(3, 5), (2, 5), (3, 3), (2, 3)],
    ids=str,
)
def test_renyi_create_plates(n, k):
    P = 10
    N, M, K = 3, 4, 5
    data = jnp.linspace(0.1, 0.9, N * M * K).reshape((N, M, K))

    def model(data, n=N, k=K, fix_indices=True):
        with numpyro.plate("N", N, subsample_size=n, dim=-3):
            with numpyro.plate("M", M, dim=-2):
                with numpyro.plate("K", K, subsample_size=k, dim=-1):
                    if fix_indices:
                        batch = data[:n, :, :k]
                    else:
                        batch = numpyro.subsample(data, event_dim=0)
                    numpyro.sample("data", dist.Bernoulli(batch), obs=1)

    def guide(data, n=N, k=K, fix_indices=True):
        pass

    def get_elbo(n=N, k=K, fix_indices=True):
        renyi_elbo = RenyiELBO(num_particles=P)
        return renyi_elbo._single_particle_elbo(
            model,
            guide,
            {},
            (data,),
            dict(n=n, k=k, fix_indices=fix_indices),
            random.PRNGKey(0),
        )

    def get_renyi(n=N, k=K, fix_indices=True):
        renyi_elbo = RenyiELBO(num_particles=P)
        return -renyi_elbo.loss(
            random.PRNGKey(0), {}, model, guide, data, n=n, k=k, fix_indices=fix_indices
        )

    elbo, scale = get_elbo(n=n, k=k)
    expected_shape = (n, M, k)
    expected_scale = N * K / n / k
    expected_elbo = jnp.log(data)[:n, :, :k]
    assert elbo.shape == expected_shape
    assert_allclose(scale, expected_scale, rtol=1e-6)
    assert_allclose(elbo, expected_elbo, rtol=1e-6)

    renyi = get_renyi(n=n, k=k)
    assert_allclose(renyi, elbo.sum() * scale, rtol=1e-6)

    if (n, k) == (2, 5):
        renyi_random = get_renyi(n=2, fix_indices=False)
        renyi_idx01 = jnp.log(data)[jnp.array([0, 1])].sum() * N / 2
        renyi_idx02 = jnp.log(data)[jnp.array([0, 2])].sum() * N / 2
        renyi_idx12 = jnp.log(data)[jnp.array([1, 2])].sum() * N / 2
        atol = jnp.min(
            jnp.abs(jnp.stack([renyi_idx01, renyi_idx02, renyi_idx12]) - renyi_random)
        )
        assert_allclose(atol, 0.0, atol=1e-5)


def test_assign_vectorize_particles_fn():
    elbo = Trace_ELBO()
    assert elbo._assign_vectorize_particles_fn(True) == _apply_vmap
    assert elbo._assign_vectorize_particles_fn(False) == jax.lax.map
    assert elbo._assign_vectorize_particles_fn(jax.pmap) == jax.pmap
    assert callable(elbo._assign_vectorize_particles_fn(lambda x: x))


@pytest.mark.parametrize(
    argnames="vectorize_particles",
    argvalues=[True, False, jax.pmap, lambda x: x],
    ids=["vmap", "lax", "pmap", "custom"],
)
def test_vectorized_particle(vectorize_particles):
    data = jnp.array([1.0] * 8 + [0.0] * 2)

    def model(data):
        f = numpyro.sample("beta", dist.Beta(1.0, 1.0))
        with numpyro.plate("N", len(data)):
            numpyro.sample("obs", dist.Bernoulli(f), obs=data)

    def guide(data):
        alpha_q = numpyro.param("alpha_q", 1.0, constraint=constraints.positive)
        beta_q = numpyro.param("beta_q", 1.0, constraint=constraints.positive)
        numpyro.sample("beta", dist.Beta(alpha_q, beta_q))

    results = SVI(
        model,
        guide,
        optim.Adam(0.1),
        Trace_ELBO(vectorize_particles=vectorize_particles),
    ).run(random.PRNGKey(0), 100, data)
    map_results = SVI(
        model, guide, optim.Adam(0.1), Trace_ELBO(vectorize_particles=False)
    ).run(random.PRNGKey(0), 100, data)
    assert_allclose(results.losses, map_results.losses, atol=1e-5)


@pytest.mark.parametrize("elbo", [Trace_ELBO(), RenyiELBO(num_particles=10)])
@pytest.mark.parametrize("optimizer", [optim.Adam(0.01), optimizers.adam(0.01)])
def test_beta_bernoulli(elbo, optimizer):
    data = jnp.array([1.0] * 8 + [0.0] * 2)

    def model(data):
        f = numpyro.sample("beta", dist.Beta(1.0, 1.0))
        with numpyro.plate("N", len(data)):
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

    svi_state = fori_loop(0, 10000, body_fn, svi_state)
    params = svi.get_params(svi_state)
    actual_posterior_mean = (data.sum() + 1) / (data.shape[0] + 2)
    assert_allclose(
        params["alpha_q"] / (params["alpha_q"] + params["beta_q"]),
        actual_posterior_mean,
        atol=0.03,
        rtol=0.03,
    )


@pytest.mark.parametrize(
    argnames="vectorize_particles",
    argvalues=[True, False, jax.pmap, lambda x: x],
    ids=["vmap", "lax", "pmap", "custom"],
)
@pytest.mark.parametrize(
    argnames="progress_bar",
    argvalues=[True, False],
    ids=["progress_bar", "no_progress_bar"],
)
def test_run(vectorize_particles, progress_bar):
    data = jnp.array([1.0] * 8 + [0.0] * 2)

    def model(data):
        f = numpyro.sample("beta", dist.Beta(1.0, 1.0))
        with numpyro.plate("N", len(data)):
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

    svi = SVI(
        model,
        guide,
        optim.Adam(0.05),
        Trace_ELBO(vectorize_particles=vectorize_particles),
    )
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
        with numpyro.plate("N", len(data)):
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

    jax.tree.all(jax.tree.map(partial(assert_allclose, atol=1e-5), actual, expected))


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


def test_shared_param_init():
    shared_init = 1.0

    def model():
        # should receive initial value from guide when used in SVI
        shared = numpyro.param("shared")
        assert_allclose(shared, shared_init)

    def guide():
        numpyro.param("shared", lambda _: shared_init)

    svi = SVI(model, guide, optim.Adam(0.01), Trace_ELBO())
    svi_state = svi.init(random.PRNGKey(0))
    params = svi.get_params(svi_state)
    # make sure the correct init ended up in the SVI state
    assert_allclose(params["shared"], shared_init)


def test_shared_param():
    target_value = 5.0

    def model():
        shared = numpyro.param("shared")
        # drive the shared parameter toward a target value
        numpyro.factor("neg_loss", -((shared - target_value) ** 2))

    def guide():
        numpyro.param("shared", 1.0)

    svi = SVI(model, guide, optim.Adam(0.01), Trace_ELBO())
    svi_result = svi.run(random.PRNGKey(0), 1000)
    assert_allclose(svi_result.params["shared"], target_value, atol=0.1)


def test_init_params():
    init_params = {"b": 1.0, "c": 2.0}

    def model():
        numpyro.param("a", 0.0)
        # should receive initial value from init_params
        numpyro.param("b")

    def guide():
        # should receive initial value from init_params
        numpyro.param("c")

    svi = SVI(model, guide, optim.Adam(0.01), Trace_ELBO())
    svi_state = svi.init(random.PRNGKey(0), init_params=init_params)
    params = svi.get_params(svi_state)
    init_params["a"] = 0.0
    # make sure init params ended up in the SVI state
    assert_equal(params, init_params)


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
    cont_inf_only_cls = [RenyiELBO(), Trace_ELBO(), TraceMeanField_ELBO()]
    mixed_inf_cls = [TraceGraph_ELBO()]

    assert not any([c.can_infer_discrete for c in cont_inf_only_cls])
    assert all([c.can_infer_discrete for c in mixed_inf_cls])

    def model():
        numpyro.sample("x", dist.Bernoulli(0.5))

    def guide():
        probs = numpyro.param("probs", 0.2)
        numpyro.sample("x", dist.Bernoulli(probs))

    for elbo in cont_inf_only_cls:
        svi = SVI(model, guide, optim.Adam(1), elbo)
        s_name = type(elbo).__name__
        w_msg = f"Currently, SVI with {s_name} loss does not support models with discrete latent variables"
        with pytest.warns(UserWarning, match=w_msg):
            svi.run(random.PRNGKey(0), 10)


@pytest.mark.parametrize("stable_update", [True, False])
@pytest.mark.parametrize("num_particles", [1, 10])
@pytest.mark.parametrize("elbo", [Trace_ELBO, TraceMeanField_ELBO])
def test_mutable_state(stable_update, num_particles, elbo):
    def model():
        x = numpyro.sample("x", dist.Normal(-1, 1))
        numpyro_mutable("x1p", x + 1)

    def guide():
        loc = numpyro.param("loc", 0.0)
        p = numpyro_mutable("loc1p", {"value": None})
        # we can modify the content of `p` if it is a dict
        p["value"] = loc + 2
        numpyro.sample("x", dist.Normal(loc, 0.1))

    svi = SVI(model, guide, optim.Adam(0.1), elbo(num_particles=num_particles))
    if num_particles > 1:
        with pytest.warns(UserWarning, match="mutable state"):
            svi_result = svi.run(random.PRNGKey(0), 1000, stable_update=stable_update)
        return
    svi_result = svi.run(random.PRNGKey(0), 1000, stable_update=stable_update)
    params = svi_result.params
    mutable_state = svi_result.state.mutable_state
    assert set(mutable_state) == {"x1p", "loc1p"}
    assert_allclose(mutable_state["loc1p"]["value"], params["loc"] + 2, atol=0.1)
    # here, the initial loc has value 0., hence x1p will have init value near 1
    # it won't be updated during SVI run because it is not a mutable state
    assert_allclose(mutable_state["x1p"], 1.0, atol=0.2)


def test_tracegraph_normal_normal():
    # normal-normal; known covariance
    lam0 = jnp.array([0.1, 0.1])  # precision of prior
    loc0 = jnp.array([0.0, 0.5])  # prior mean
    # known precision of observation noise
    lam = jnp.array([6.0, 4.0])
    data = []
    data.append(jnp.array([-0.1, 0.3]))
    data.append(jnp.array([0.0, 0.4]))
    data.append(jnp.array([0.2, 0.5]))
    data.append(jnp.array([0.1, 0.7]))
    n_data = len(data)
    sum_data = data[0] + data[1] + data[2] + data[3]
    analytic_lam_n = lam0 + n_data * lam
    analytic_log_sig_n = -0.5 * jnp.log(analytic_lam_n)
    analytic_loc_n = sum_data * (lam / analytic_lam_n) + loc0 * (lam0 / analytic_lam_n)

    class FakeNormal(dist.Normal):
        reparametrized_params = []

    def model():
        with numpyro.plate("plate", 2):
            loc_latent = numpyro.sample(
                "loc_latent", FakeNormal(loc0, jnp.power(lam0, -0.5))
            )
            for i, x in enumerate(data):
                numpyro.sample(
                    "obs_{}".format(i),
                    dist.Normal(loc_latent, jnp.power(lam, -0.5)),
                    obs=x,
                )
        return loc_latent

    def guide():
        loc_q = numpyro.param("loc_q", analytic_loc_n + jnp.array([0.334, 0.334]))
        log_sig_q = numpyro.param(
            "log_sig_q", analytic_log_sig_n + jnp.array([-0.29, -0.29])
        )
        sig_q = jnp.exp(log_sig_q)
        with numpyro.plate("plate", 2):
            loc_latent = numpyro.sample("loc_latent", FakeNormal(loc_q, sig_q))
        return loc_latent

    adam = optim.Adam(step_size=0.0015, b1=0.97, b2=0.999)
    svi = SVI(model, guide, adam, loss=TraceGraph_ELBO())
    svi_result = svi.run(jax.random.PRNGKey(0), 5000)

    loc_error = jnp.sum(jnp.power(analytic_loc_n - svi_result.params["loc_q"], 2.0))
    log_sig_error = jnp.sum(
        jnp.power(analytic_log_sig_n - svi_result.params["log_sig_q"], 2.0)
    )

    assert_allclose(loc_error, 0, atol=0.05)
    assert_allclose(log_sig_error, 0, atol=0.05)


def test_tracegraph_beta_bernoulli():
    # bernoulli-beta model
    # beta prior hyperparameter
    alpha0 = 1.0
    beta0 = 1.0  # beta prior hyperparameter
    data = jnp.array([0.0, 1.0, 1.0, 1.0])
    n_data = float(len(data))
    data_sum = data.sum()
    alpha_n = alpha0 + data_sum  # posterior alpha
    beta_n = beta0 - data_sum + n_data  # posterior beta
    log_alpha_n = jnp.log(alpha_n)
    log_beta_n = jnp.log(beta_n)

    class FakeBeta(dist.Beta):
        reparametrized_params = []

    def model():
        p_latent = numpyro.sample("p_latent", FakeBeta(alpha0, beta0))
        with numpyro.plate("data", len(data)):
            numpyro.sample("obs", dist.Bernoulli(p_latent), obs=data)
        return p_latent

    def guide():
        alpha_q_log = numpyro.param("alpha_q_log", log_alpha_n + 0.17)
        beta_q_log = numpyro.param("beta_q_log", log_beta_n - 0.143)
        alpha_q, beta_q = jnp.exp(alpha_q_log), jnp.exp(beta_q_log)
        p_latent = numpyro.sample("p_latent", FakeBeta(alpha_q, beta_q))
        with numpyro.plate("data", len(data)):
            pass
        return p_latent

    adam = optim.Adam(step_size=0.0007, b1=0.95, b2=0.999)
    svi = SVI(model, guide, adam, loss=TraceGraph_ELBO())
    svi_result = svi.run(jax.random.PRNGKey(0), 3000)

    alpha_error = jnp.sum(
        jnp.power(log_alpha_n - svi_result.params["alpha_q_log"], 2.0)
    )
    beta_error = jnp.sum(jnp.power(log_beta_n - svi_result.params["beta_q_log"], 2.0))

    assert_allclose(alpha_error, 0, atol=0.03)
    assert_allclose(beta_error, 0, atol=0.04)


def test_tracegraph_gamma_exponential():
    # exponential-gamma model
    # gamma prior hyperparameter
    alpha0 = 1.0
    # gamma prior hyperparameter
    beta0 = 1.0
    n_data = 2
    data = jnp.array([3.0, 2.0])  # two observations
    alpha_n = alpha0 + n_data  # posterior alpha
    beta_n = beta0 + data.sum()  # posterior beta
    log_alpha_n = jnp.log(alpha_n)
    log_beta_n = jnp.log(beta_n)

    class FakeGamma(dist.Gamma):
        reparametrized_params = []

    def model():
        lambda_latent = numpyro.sample("lambda_latent", FakeGamma(alpha0, beta0))
        with numpyro.plate("data", len(data)):
            numpyro.sample("obs", dist.Exponential(lambda_latent), obs=data)
        return lambda_latent

    def guide():
        alpha_q_log = numpyro.param("alpha_q_log", log_alpha_n + 0.17)
        beta_q_log = numpyro.param("beta_q_log", log_beta_n - 0.143)
        alpha_q, beta_q = jnp.exp(alpha_q_log), jnp.exp(beta_q_log)
        numpyro.sample("lambda_latent", FakeGamma(alpha_q, beta_q))
        with numpyro.plate("data", len(data)):
            pass

    adam = optim.Adam(step_size=0.0007, b1=0.95, b2=0.999)
    svi = SVI(model, guide, adam, loss=TraceGraph_ELBO())
    svi_result = svi.run(jax.random.PRNGKey(0), 8000)

    alpha_error = jnp.sum(
        jnp.power(log_alpha_n - svi_result.params["alpha_q_log"], 2.0)
    )
    beta_error = jnp.sum(jnp.power(log_beta_n - svi_result.params["beta_q_log"], 2.0))

    assert_allclose(alpha_error, 0, atol=0.04)
    assert_allclose(beta_error, 0, atol=0.04)


@pytest.mark.parametrize(
    "num_latents,num_steps,step_size,atol,difficulty",
    [
        (3, 5000, 0.003, 0.05, 0.6),
        (5, 6000, 0.003, 0.05, 0.6),
        (7, 8000, 0.003, 0.05, 0.6),
    ],
)
def test_tracegraph_gaussian_chain(num_latents, num_steps, step_size, atol, difficulty):
    loc0 = 0.2
    data = jnp.array([-0.1, 0.03, 0.2, 0.1])
    n_data = data.shape[0]
    sum_data = data.sum()
    N = num_latents
    lambdas = [1.5 * (k + 1) / N for k in range(N + 1)]
    lambdas = list(map(lambda x: jnp.array([x]), lambdas))
    lambda_tilde_posts = [lambdas[0]]
    for k in range(1, N):
        lambda_tilde_k = (lambdas[k] * lambda_tilde_posts[k - 1]) / (
            lambdas[k] + lambda_tilde_posts[k - 1]
        )
        lambda_tilde_posts.append(lambda_tilde_k)
    lambda_posts = [
        None
    ]  # this is never used (just a way of shifting the indexing by 1)
    for k in range(1, N):
        lambda_k = lambdas[k] + lambda_tilde_posts[k - 1]
        lambda_posts.append(lambda_k)
    lambda_N_post = (n_data * lambdas[N]) + lambda_tilde_posts[N - 1]
    lambda_posts.append(lambda_N_post)
    target_kappas = [None]
    target_kappas.extend([lambdas[k] / lambda_posts[k] for k in range(1, N)])
    target_mus = [None]
    target_mus.extend(
        [loc0 * lambda_tilde_posts[k - 1] / lambda_posts[k] for k in range(1, N)]
    )
    target_loc_N = (
        sum_data * lambdas[N] / lambda_N_post
        + loc0 * lambda_tilde_posts[N - 1] / lambda_N_post
    )
    target_mus.append(target_loc_N)
    np.random.seed(0)
    while True:
        mask = np.random.binomial(1, 0.3, (N,))
        if mask.sum() < 0.4 * N and mask.sum() > 0.5:
            which_nodes_reparam = mask
            break

    class FakeNormal(dist.Normal):
        reparametrized_params = []

    def model(difficulty=0.0):
        next_mean = loc0
        for k in range(1, N + 1):
            latent_dist = dist.Normal(next_mean, jnp.power(lambdas[k - 1], -0.5))
            loc_latent = numpyro.sample("loc_latent_{}".format(k), latent_dist)
            next_mean = loc_latent

        loc_N = next_mean
        with numpyro.plate("data", data.shape[0]):
            numpyro.sample(
                "obs", dist.Normal(loc_N, jnp.power(lambdas[N], -0.5)), obs=data
            )
        return loc_N

    def guide(difficulty=0.0):
        previous_sample = None
        for k in reversed(range(1, N + 1)):
            loc_q = numpyro.param(
                f"loc_q_{k}",
                lambda key: target_mus[k]
                + difficulty * (0.1 * random.normal(key) - 0.53),
            )
            log_sig_q = numpyro.param(
                f"log_sig_q_{k}",
                lambda key: -0.5 * jnp.log(lambda_posts[k])
                + difficulty * (0.1 * random.normal(key) - 0.53),
            )
            sig_q = jnp.exp(log_sig_q)
            kappa_q = None
            if k != N:
                kappa_q = numpyro.param(
                    "kappa_q_%d" % k,
                    lambda key: target_kappas[k]
                    + difficulty * (0.1 * random.normal(key) - 0.53),
                )
            mean_function = loc_q if k == N else kappa_q * previous_sample + loc_q
            node_flagged = True if which_nodes_reparam[k - 1] == 1.0 else False
            Normal = dist.Normal if node_flagged else FakeNormal
            loc_latent = numpyro.sample(f"loc_latent_{k}", Normal(mean_function, sig_q))
            previous_sample = loc_latent
        return previous_sample

    adam = optim.Adam(step_size=step_size, b1=0.95, b2=0.999)
    svi = SVI(model, guide, adam, loss=TraceGraph_ELBO())
    svi_result = svi.run(jax.random.PRNGKey(0), num_steps, difficulty=difficulty)

    kappa_errors, log_sig_errors, loc_errors = [], [], []
    for k in range(1, N + 1):
        if k != N:
            kappa_error = jnp.sum(
                jnp.power(svi_result.params[f"kappa_q_{k}"] - target_kappas[k], 2)
            )
            kappa_errors.append(kappa_error)

        loc_errors.append(
            jnp.sum(jnp.power(svi_result.params[f"loc_q_{k}"] - target_mus[k], 2))
        )
        log_sig_error = jnp.sum(
            jnp.power(
                svi_result.params[f"log_sig_q_{k}"] + 0.5 * jnp.log(lambda_posts[k]), 2
            )
        )
        log_sig_errors.append(log_sig_error)

    max_errors = (np.max(loc_errors), np.max(log_sig_errors), np.max(kappa_errors))

    for i in range(3):
        assert_allclose(max_errors[i], 0, atol=atol)


def test_multi_sample_guide():
    actual_loc = 3.0
    actual_scale = 2.0

    def model():
        numpyro.sample("x", dist.Normal(actual_loc, actual_scale))

    def guide():
        loc = numpyro.param("loc", 0.0)
        scale = numpyro.param("scale", 1.0, constraint=constraints.positive)
        numpyro.sample("x", dist.Normal(loc, scale).expand([10]))

    svi = SVI(model, guide, optim.Adam(0.1), Trace_ELBO(multi_sample_guide=True))
    svi_results = svi.run(random.PRNGKey(0), 2000)
    params = svi_results.params
    assert_allclose(params["loc"], actual_loc, rtol=0.1)
    assert_allclose(params["scale"], actual_scale, rtol=0.1)


def test_forward_mode_differentiation():
    def model():
        x = numpyro.sample("x", dist.Normal(0, 1))
        y = lax.while_loop(lambda x: x < 10, lambda x: x + 1, x)
        numpyro.sample("obs", dist.Normal(y, 1), obs=1.0)

    def guide():
        loc = numpyro.param("loc", 0.0)
        scale = numpyro.param("scale", 1.0, constraint=dist.constraints.positive)
        numpyro.sample("x", dist.Normal(loc, scale))

    # this fails in reverse mode
    optimizer = numpyro.optim.Adam(step_size=0.01)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    svi.run(random.PRNGKey(0), 1000, forward_mode_differentiation=True)
