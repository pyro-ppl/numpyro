# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import numpy as np
from numpy.testing import assert_allclose
import pytest

from jax import random
import jax.numpy as jnp

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.transforms import AffineTransform, biject_to
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.initialization import (
    init_to_feasible,
    init_to_mean,
    init_to_median,
    init_to_sample,
    init_to_uniform,
    init_to_value,
)
from numpyro.infer.reparam import TransformReparam
from numpyro.infer.util import (
    Predictive,
    compute_log_probs,
    constrain_fn,
    initialize_model,
    log_density,
    log_likelihood,
    potential_energy,
    transform_fn,
    unconstrain_fn,
)
import numpyro.optim as optim


def beta_bernoulli():
    N = 800
    true_probs = jnp.array([0.2, 0.7])
    data = dist.Bernoulli(true_probs).sample(random.PRNGKey(0), (N,))

    def model(data=None):
        with numpyro.plate("dim", 2):
            beta = numpyro.sample("beta", dist.Beta(1.0, 1.0))
        with numpyro.plate("plate", N, dim=-2):
            numpyro.deterministic("beta_sq", beta**2)
            with numpyro.plate("dim", 2):
                numpyro.sample("obs", dist.Bernoulli(beta), obs=data)

    return model, data, true_probs


def linear_regression():
    N = 800
    X = dist.Normal(0, 1).sample(random.PRNGKey(0), (N,))
    y = 1.5 + X * 0.7

    def model(X, y=None):
        alpha = numpyro.sample("alpha", dist.Normal(0.0, 5))
        beta = numpyro.sample("beta", dist.Normal(0.0, 1.0))
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        with numpyro.plate("plate", len(X)):
            mu = numpyro.deterministic("mu", alpha + X * beta)
            numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)

    return model, X, y


def categorical_probs():
    probs0 = 0.5
    nbatch0, nbatch1 = 2, 1
    probs = jnp.ones((nbatch0, nbatch1, 3)) * probs0

    def model(probs):
        probs = numpyro.deterministic("probs", probs)

        plate = numpyro.plate("plate", size=probs.shape[-1], dim=-1)

        with plate:
            numpyro.sample(
                "counts_categorical",
                dist.Categorical(probs=probs),
                infer={"enumerate": "parallel"},
            )

    return model, probs


@pytest.mark.parametrize("parallel", [True, False])
def test_predictive(parallel):
    model, data, true_probs = beta_bernoulli()
    mcmc = MCMC(NUTS(model), num_warmup=100, num_samples=100)
    mcmc.run(random.PRNGKey(0), data)
    samples = mcmc.get_samples()
    predictive = Predictive(model, samples, parallel=parallel)
    predictive_samples = predictive(random.PRNGKey(1))
    assert predictive_samples.keys() == {"beta_sq", "obs"}

    predictive.return_sites = ["beta", "beta_sq", "obs"]
    predictive_samples = predictive(random.PRNGKey(1))
    # check shapes
    assert predictive_samples["beta"].shape == (100,) + true_probs.shape
    assert predictive_samples["beta_sq"].shape == (100,) + true_probs.shape
    assert predictive_samples["obs"].shape == (100,) + data.shape
    # check sample mean
    obs = predictive_samples["obs"].reshape((-1,) + true_probs.shape).astype(np.float32)
    assert_allclose(obs.mean(0), true_probs, rtol=0.1)


@pytest.mark.parametrize("parallel", [True, False])
def test_predictive_with_deterministic(parallel):
    """Tests that the default behavior when predicting from models with
    deterministic sites doesn't lead to static deterministic sites in the predictive.
    """
    n_preds = 400
    model, X, y = linear_regression()
    mcmc = MCMC(NUTS(model), num_warmup=100, num_samples=100)
    mcmc.run(random.PRNGKey(0), X=X, y=y)
    samples = mcmc.get_samples()
    predictive = Predictive(model, samples, parallel=parallel)
    # change the input (X) shape to make sure the deterministic shape changes
    predictive_samples = predictive(random.PRNGKey(1), X=X[:n_preds])
    assert predictive_samples.keys() == {"mu", "obs"}

    predictive.return_sites = ["beta", "mu", "obs"]
    # change the input (X) shape to make sure the deterministic shape changes
    predictive_samples = predictive(random.PRNGKey(1), X=X[:n_preds])
    # check shapes
    assert predictive_samples["mu"].shape == (100,) + X[:n_preds].shape
    assert predictive_samples["obs"].shape == (100,) + X[:n_preds].shape


@pytest.mark.parametrize(
    argnames="parallel", argvalues=[True, False], ids=["parallel", "sequential"]
)
def test_discrete_predictive_with_deterministic(parallel):
    """Tests that the predictive samples include deterministic sites for discrete models."""
    model, probs = categorical_probs()

    predictive = Predictive(
        model=model,
        posterior_samples=dict(probs=probs),
        infer_discrete=True,
        batch_ndims=2,
        parallel=parallel,
        exclude_deterministic=False,
    )

    predictive_samples = predictive(random.PRNGKey(1), probs=probs)
    assert predictive_samples.keys() == {"counts_categorical"}
    assert predictive_samples["counts_categorical"].shape == probs.shape


def test_predictive_with_guide():
    data = jnp.array([1] * 8 + [0] * 2)

    def model(data):
        f = numpyro.sample("beta", dist.Beta(1.0, 1.0))
        with numpyro.plate("plate", 10):
            numpyro.deterministic("beta_sq", f**2)
            numpyro.sample("obs", dist.Bernoulli(f), obs=data)

    def guide(data):
        alpha_q = numpyro.param("alpha_q", 1.0, constraint=constraints.positive)
        beta_q = numpyro.param("beta_q", 1.0, constraint=constraints.positive)
        numpyro.sample("beta", dist.Beta(alpha_q, beta_q))

    svi = SVI(model, guide, optim.Adam(0.1), Trace_ELBO())
    svi_result = svi.run(random.PRNGKey(1), 5000, data)
    params = svi_result.params
    predictive = Predictive(model, guide=guide, params=params, num_samples=1000)(
        random.PRNGKey(2), data=None
    )
    assert predictive["beta_sq"].shape == (1000,)
    obs_pred = predictive["obs"].astype(np.float32)
    assert_allclose(jnp.mean(obs_pred), 0.8, atol=0.05)


def test_predictive_with_particles():
    num_particles = 5
    num_samples = 2
    fdim = 3
    num_data = 10

    def model(x, y=None):
        latent = numpyro.sample("latent", dist.Normal(0.0, jnp.ones(fdim)).to_event(1))
        with numpyro.plate("data", x.shape[0]):
            numpyro.sample("y", dist.Normal(jnp.matmul(x, latent), 1.0), obs=y)

    def guide(x, y=None):
        latent_loc = numpyro.param(
            "latent_loc", jnp.ones(fdim), constraint=constraints.real
        )
        assert latent_loc.ndim == 1
        numpyro.sample("latent", dist.Normal(latent_loc, 1.0).to_event(1))

    params = {"latent_loc": jnp.zeros((num_particles, fdim))}
    x = dist.Normal(jnp.full(3, 0.2), 1.0).sample(random.PRNGKey(0), (num_data,))
    predictions = Predictive(
        model,
        guide=guide,
        params=params,
        num_samples=num_samples,
        batch_ndims=1,
    )(random.PRNGKey(0), x)
    assert predictions["y"].shape == (num_samples, num_particles, num_data)


def test_predictive_with_improper():
    true_coef = 0.9

    def model(data):
        alpha = numpyro.sample("alpha", dist.Uniform(0, 1))
        with handlers.reparam(config={"loc": TransformReparam()}):
            loc = numpyro.sample(
                "loc",
                dist.TransformedDistribution(
                    dist.Uniform(0, 1).mask(False), AffineTransform(0, alpha)
                ),
            )
        numpyro.sample("obs", dist.Normal(loc, 0.1), obs=data)

    data = true_coef + random.normal(random.PRNGKey(0), (1000,))
    kernel = NUTS(model=model)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000)
    mcmc.run(random.PRNGKey(0), data)
    samples = mcmc.get_samples()
    obs_pred = Predictive(model, samples)(random.PRNGKey(1), data=None)["obs"]
    assert_allclose(jnp.mean(obs_pred), true_coef, atol=0.05)


@pytest.mark.parametrize("batch_ndims", [0, 1, 2])
def test_prior_predictive(batch_ndims):
    model, data, true_probs = beta_bernoulli()
    predictive = Predictive(model, num_samples=100, batch_ndims=batch_ndims)
    predictive_samples = predictive(random.PRNGKey(1))
    assert predictive_samples.keys() == {"beta", "beta_sq", "obs"}

    # check shapes
    batch_shape = (1,) * (batch_ndims - 1) + (100,)
    assert predictive_samples["beta"].shape == batch_shape + true_probs.shape
    assert predictive_samples["obs"].shape == batch_shape + data.shape


@pytest.mark.parametrize("batch_shape", [(), (100,), (2, 50)])
def test_log_likelihood(batch_shape):
    model, data, _ = beta_bernoulli()
    samples = Predictive(model, return_sites=["beta"], num_samples=200)(
        random.PRNGKey(1)
    )
    batch_size = int(np.prod(batch_shape))
    samples = {"beta": samples["beta"][:batch_size].reshape(batch_shape + (1, -1))}

    preds = Predictive(model, samples, batch_ndims=len(batch_shape))(random.PRNGKey(2))
    loglik = log_likelihood(model, samples, data, batch_ndims=len(batch_shape))
    assert preds.keys() == {"beta_sq", "obs"}
    assert loglik.keys() == {"obs"}
    # check shapes
    assert preds["obs"].shape == batch_shape + data.shape
    assert loglik["obs"].shape == batch_shape + data.shape
    assert_allclose(
        loglik["obs"], dist.Bernoulli(samples["beta"]).log_prob(data), rtol=1e-6
    )


def test_compute_log_probs():
    model, data, _ = beta_bernoulli()
    samples = Predictive(model, return_sites=["beta"], num_samples=1)(random.key(7))
    samples = {key: value[0] for key, value in samples.items()}

    logden, _ = log_density(model, (data,), {}, samples)
    assert logden.shape == ()

    logdens, _ = compute_log_probs(model, (data,), {}, samples)
    assert set(logdens) == {"beta", "obs"}
    assert all(x.shape == () for x in logdens.values())

    logdens, _ = compute_log_probs(model, (data,), {}, samples, False)
    assert logdens["beta"].shape == (2,)
    assert logdens["obs"].shape == (800, 2)


def test_model_with_transformed_distribution():
    x_prior = dist.HalfNormal(2)
    y_prior = dist.LogNormal(scale=3.0)  # transformed distribution

    def model():
        numpyro.sample("x", x_prior)
        numpyro.sample("y", y_prior)

    params = {"x": jnp.array(-5.0), "y": jnp.array(7.0)}
    model = handlers.seed(model, random.PRNGKey(0))
    inv_transforms = {"x": biject_to(x_prior.support), "y": biject_to(y_prior.support)}
    expected_samples = partial(transform_fn, inv_transforms)(params)
    expected_potential_energy = (
        -x_prior.log_prob(expected_samples["x"])
        - y_prior.log_prob(expected_samples["y"])
        - inv_transforms["x"].log_abs_det_jacobian(params["x"], expected_samples["x"])
        - inv_transforms["y"].log_abs_det_jacobian(params["y"], expected_samples["y"])
    )

    reparam_model = handlers.reparam(model, {"y": TransformReparam()})
    base_params = {"x": params["x"], "y_base": params["y"]}
    actual_samples = constrain_fn(
        handlers.seed(reparam_model, random.PRNGKey(0)),
        (),
        {},
        base_params,
        return_deterministic=True,
    )
    actual_potential_energy = potential_energy(reparam_model, (), {}, base_params)

    assert_allclose(expected_samples["x"], actual_samples["x"])
    assert_allclose(expected_samples["y"], actual_samples["y"])
    assert_allclose(actual_potential_energy, expected_potential_energy)


def test_constrain_unconstrain():
    x_prior = dist.HalfNormal(2)
    y_prior = dist.LogNormal(scale=3.0)  # transformed distribution
    z_constraint = constraints.positive

    def model():
        numpyro.sample("x", x_prior)
        numpyro.sample("y", y_prior)
        numpyro.param("z", init_value=2.0, constraint=z_constraint)

    params = {"x": jnp.array(-5.0), "y": jnp.array(7.0), "z": jnp.array(3.0)}
    model = handlers.seed(model, random.PRNGKey(0))
    inv_transforms = {
        "x": biject_to(x_prior.support),
        "y": biject_to(y_prior.support),
        "z": biject_to(z_constraint),
    }
    expected_constrained_samples = partial(transform_fn, inv_transforms)(params)
    transforms = {
        "x": biject_to(x_prior.support).inv,
        "y": biject_to(y_prior.support).inv,
        "z": biject_to(z_constraint).inv,
    }
    expected_unconstrained_samples = partial(transform_fn, transforms)(
        expected_constrained_samples
    )

    actual_constrained_samples = constrain_fn(model, (), {}, params)
    actual_unconstrained_samples = unconstrain_fn(
        model, (), {}, actual_constrained_samples
    )

    assert_allclose(expected_constrained_samples["x"], actual_constrained_samples["x"])
    assert_allclose(expected_constrained_samples["y"], actual_constrained_samples["y"])
    assert_allclose(expected_constrained_samples["z"], actual_constrained_samples["z"])
    assert_allclose(
        expected_unconstrained_samples["x"], actual_unconstrained_samples["x"]
    )
    assert_allclose(
        expected_unconstrained_samples["y"], actual_unconstrained_samples["y"]
    )
    assert_allclose(
        expected_unconstrained_samples["z"], actual_unconstrained_samples["z"]
    )


def test_model_with_mask_false():
    def model():
        x = numpyro.sample("x", dist.Normal())
        with numpyro.handlers.mask(mask=False):
            numpyro.sample("y", dist.Normal(x), obs=1)

    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=500, num_chains=1)
    mcmc.run(random.PRNGKey(1))
    assert_allclose(mcmc.get_samples()["x"].mean(), 0.0, atol=0.15)


@pytest.mark.parametrize(
    "init_strategy",
    [
        init_to_feasible(),
        init_to_median(num_samples=2),
        init_to_sample(),
        init_to_uniform(radius=3),
        init_to_value(values={"tau": 0.7}),
        init_to_feasible,
        init_to_mean,
        init_to_median,
        init_to_sample,
        init_to_uniform,
        init_to_value,
    ],
)
def test_initialize_model_change_point(init_strategy):
    def model(data):
        alpha = 1 / jnp.mean(data.astype(np.float32))
        lambda1 = numpyro.sample("lambda1", dist.Exponential(alpha))
        lambda2 = numpyro.sample("lambda2", dist.Exponential(alpha))
        tau = numpyro.sample("tau", dist.Uniform(0, 1))
        lambda12 = jnp.where(jnp.arange(len(data)) < tau * len(data), lambda1, lambda2)
        numpyro.sample("obs", dist.Poisson(lambda12), obs=data)

    # fmt: off
    count_data = jnp.array([
        13, 24, 8, 24, 7, 35, 14, 11, 15, 11, 22, 22, 11, 57, 11, 19, 29, 6, 19, 12, 22,
        12, 18, 72, 32, 9, 7, 13, 19, 23, 27, 20, 6, 17, 13, 10, 14, 6, 16, 15, 7, 2,
        15, 15, 19, 70, 49, 7, 53, 22, 21, 31, 19, 11, 1, 20, 12, 35, 17, 23, 17, 4, 2,
        31, 30, 13, 27, 0, 39, 37, 5, 14, 13, 22])
    # fmt: on

    rng_keys = random.split(random.PRNGKey(1), 2)
    init_params, _, _, _ = initialize_model(
        rng_keys, model, init_strategy=init_strategy, model_args=(count_data,)
    )
    if isinstance(init_strategy, partial) and init_strategy.func is init_to_value:
        expected = biject_to(constraints.unit_interval).inv(
            init_strategy.keywords.get("values")["tau"]
        )
        assert_allclose(init_params[0]["tau"], jnp.repeat(expected, 2))
    for i in range(2):
        init_params_i, _, _, _ = initialize_model(
            rng_keys[i], model, init_strategy=init_strategy, model_args=(count_data,)
        )
        for name, p in init_params[0].items():
            # XXX: the result is equal if we disable fast-math-mode
            assert_allclose(p[i], init_params_i[0][name], atol=1e-6)


@pytest.mark.parametrize(
    "init_strategy",
    [
        init_to_feasible(),
        init_to_median(num_samples=2),
        init_to_sample(),
        init_to_uniform(),
    ],
)
def test_initialize_model_dirichlet_categorical(init_strategy):
    def model(data):
        concentration = jnp.array([1.0, 1.0, 1.0])
        p_latent = numpyro.sample("p_latent", dist.Dirichlet(concentration))
        numpyro.sample("obs", dist.Categorical(p_latent), obs=data)
        return p_latent

    true_probs = jnp.array([0.1, 0.6, 0.3])
    data = dist.Categorical(true_probs).sample(random.PRNGKey(1), (2000,))

    rng_keys = random.split(random.PRNGKey(1), 2)
    init_params, _, _, _ = initialize_model(
        rng_keys, model, init_strategy=init_strategy, model_args=(data,)
    )
    for i in range(2):
        init_params_i, _, _, _ = initialize_model(
            rng_keys[i], model, init_strategy=init_strategy, model_args=(data,)
        )
        for name, p in init_params[0].items():
            # XXX: the result is equal if we disable fast-math-mode
            assert_allclose(p[i], init_params_i[0][name], atol=1e-6)


@pytest.mark.parametrize("event_shape", [(3,), ()])
def test_improper_expand(event_shape):
    def model():
        population = jnp.array([1000.0, 2000.0, 3000.0])
        with numpyro.plate("region", 3):
            d = dist.ImproperUniform(
                support=constraints.interval(0, population),
                batch_shape=(3,),
                event_shape=event_shape,
            )
            incidence = numpyro.sample("incidence", d)
            assert d.log_prob(incidence).shape == (3,)

    model_info = initialize_model(random.PRNGKey(0), model)
    assert model_info.param_info.z["incidence"].shape == (3,) + event_shape


def test_get_mask_optimization():
    def model():
        with numpyro.handlers.seed(rng_seed=0):
            x = numpyro.sample("x", dist.Normal(0, 1))
            numpyro.sample("y", dist.Normal(x, 1), obs=0.0)
            called.add("model-always")
            if numpyro.get_mask() is not False:
                called.add("model-sometimes")
                numpyro.factor("f", x + 1)

    def guide():
        with numpyro.handlers.seed(rng_seed=1):
            x = numpyro.sample("x", dist.Normal(0, 1))
            called.add("guide-always")
            if numpyro.get_mask() is not False:
                called.add("guide-sometimes")
                numpyro.factor("g", 2 - x)

    called = set()
    trace = handlers.trace(guide).get_trace()
    handlers.replay(model, trace)()
    assert "model-always" in called
    assert "guide-always" in called
    assert "model-sometimes" in called
    assert "guide-sometimes" in called

    called = set()
    with handlers.mask(mask=False):
        trace = handlers.trace(guide).get_trace()
        handlers.replay(model, trace)()
    assert "model-always" in called
    assert "guide-always" in called
    assert "model-sometimes" not in called
    assert "guide-sometimes" not in called

    called = set()
    Predictive(model, guide=guide, num_samples=2, parallel=True)(random.PRNGKey(2))
    assert "model-always" in called
    assert "guide-always" in called
    assert "model-sometimes" not in called
    assert "guide-sometimes" not in called
