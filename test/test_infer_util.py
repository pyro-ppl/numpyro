# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import numpy as np
from numpy.testing import assert_allclose
import pytest

from jax import lax, random
import jax.numpy as jnp

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.transforms import AffineTransform, biject_to
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.initialization import (
    init_to_feasible,
    init_to_median,
    init_to_sample,
    init_to_uniform,
    init_to_value
)
from numpyro.infer.reparam import TransformReparam
from numpyro.infer.util import (
    Predictive,
    constrain_fn,
    initialize_model,
    log_likelihood,
    potential_energy,
    transform_fn
)
import numpyro.optim as optim


def beta_bernoulli():
    N = 800
    true_probs = jnp.array([0.2, 0.7])
    data = dist.Bernoulli(true_probs).sample(random.PRNGKey(0), (N,))

    def model(data=None):
        with numpyro.plate("dim", 2):
            beta = numpyro.sample("beta", dist.Beta(1., 1.))
        with numpyro.plate("plate", N, dim=-2):
            numpyro.deterministic("beta_sq", beta ** 2)
            numpyro.sample("obs", dist.Bernoulli(beta), obs=data)

    return model, data, true_probs


@pytest.mark.parametrize('parallel', [True, False])
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
    assert_allclose(predictive_samples["obs"].reshape((-1,) + true_probs.shape).mean(0), true_probs, rtol=0.1)


def test_predictive_with_guide():
    data = jnp.array([1] * 8 + [0] * 2)

    def model(data):
        f = numpyro.sample("beta", dist.Beta(1., 1.))
        with numpyro.plate("plate", 10):
            numpyro.deterministic("beta_sq", f ** 2)
            numpyro.sample("obs", dist.Bernoulli(f), obs=data)

    def guide(data):
        alpha_q = numpyro.param("alpha_q", 1.0,
                                constraint=constraints.positive)
        beta_q = numpyro.param("beta_q", 1.0,
                               constraint=constraints.positive)
        numpyro.sample("beta", dist.Beta(alpha_q, beta_q))

    svi = SVI(model, guide, optim.Adam(0.1), Trace_ELBO())
    svi_state = svi.init(random.PRNGKey(1), data)

    def body_fn(i, val):
        svi_state, _ = svi.update(val, data)
        return svi_state

    svi_state = lax.fori_loop(0, 1000, body_fn, svi_state)
    params = svi.get_params(svi_state)
    predictive = Predictive(model, guide=guide, params=params, num_samples=1000)(random.PRNGKey(2), data=None)
    assert predictive["beta_sq"].shape == (1000,)
    obs_pred = predictive["obs"]
    assert_allclose(jnp.mean(obs_pred), 0.8, atol=0.05)


def test_predictive_with_improper():
    true_coef = 0.9

    def model(data):
        alpha = numpyro.sample('alpha', dist.Uniform(0, 1))
        with handlers.reparam(config={'loc': TransformReparam()}):
            loc = numpyro.sample('loc', dist.TransformedDistribution(
                dist.Uniform(0, 1).mask(False),
                AffineTransform(0, alpha)))
        numpyro.sample('obs', dist.Normal(loc, 0.1), obs=data)

    data = true_coef + random.normal(random.PRNGKey(0), (1000,))
    kernel = NUTS(model=model)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000)
    mcmc.run(random.PRNGKey(0), data)
    samples = mcmc.get_samples()
    obs_pred = Predictive(model, samples)(random.PRNGKey(1), data=None)["obs"]
    assert_allclose(jnp.mean(obs_pred), true_coef, atol=0.05)


@pytest.mark.parametrize('batch_ndims', [0, 1, 2])
def test_prior_predictive(batch_ndims):
    model, data, true_probs = beta_bernoulli()
    predictive = Predictive(model, num_samples=100, batch_ndims=batch_ndims)
    predictive_samples = predictive(random.PRNGKey(1))
    assert predictive_samples.keys() == {"beta", "beta_sq", "obs"}

    # check shapes
    batch_shape = (1,) * (batch_ndims - 1) + (100,)
    assert predictive_samples["beta"].shape == batch_shape + true_probs.shape
    assert predictive_samples["obs"].shape == batch_shape + data.shape


@pytest.mark.parametrize('batch_shape', [(), (100,), (2, 50)])
def test_log_likelihood(batch_shape):
    model, data, _ = beta_bernoulli()
    samples = Predictive(model, return_sites=["beta"], num_samples=200)(random.PRNGKey(1))
    batch_size = int(np.prod(batch_shape))
    samples = {'beta': samples['beta'][:batch_size].reshape(batch_shape + (1, -1))}

    preds = Predictive(model, samples, batch_ndims=len(batch_shape))(random.PRNGKey(2))
    loglik = log_likelihood(model, samples, data, batch_ndims=len(batch_shape))
    assert preds.keys() == {"beta_sq", "obs"}
    assert loglik.keys() == {"obs"}
    # check shapes
    assert preds["obs"].shape == batch_shape + data.shape
    assert loglik["obs"].shape == batch_shape + data.shape
    assert_allclose(loglik["obs"], dist.Bernoulli(samples["beta"]).log_prob(data))


def test_model_with_transformed_distribution():
    x_prior = dist.HalfNormal(2)
    y_prior = dist.LogNormal(scale=3.)  # transformed distribution

    def model():
        numpyro.sample('x', x_prior)
        numpyro.sample('y', y_prior)

    params = {'x': jnp.array(-5.), 'y': jnp.array(7.)}
    model = handlers.seed(model, random.PRNGKey(0))
    inv_transforms = {'x': biject_to(x_prior.support), 'y': biject_to(y_prior.support)}
    expected_samples = partial(transform_fn, inv_transforms)(params)
    expected_potential_energy = (
        - x_prior.log_prob(expected_samples['x']) -
        y_prior.log_prob(expected_samples['y']) -
        inv_transforms['x'].log_abs_det_jacobian(params['x'], expected_samples['x']) -
        inv_transforms['y'].log_abs_det_jacobian(params['y'], expected_samples['y'])
    )

    reparam_model = handlers.reparam(model, {'y': TransformReparam()})
    base_params = {'x': params['x'], 'y_base': params['y']}
    actual_samples = constrain_fn(
        handlers.seed(reparam_model, random.PRNGKey(0)),
        (), {}, base_params, return_deterministic=True)
    actual_potential_energy = potential_energy(reparam_model, (), {}, base_params)

    assert_allclose(expected_samples['x'], actual_samples['x'])
    assert_allclose(expected_samples['y'], actual_samples['y'])
    assert_allclose(actual_potential_energy, expected_potential_energy)


def test_model_with_mask_false():
    def model():
        x = numpyro.sample("x", dist.Normal())
        with numpyro.handlers.mask(mask=False):
            numpyro.sample("y", dist.Normal(x), obs=1)

    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=500, num_chains=1)
    mcmc.run(random.PRNGKey(1))
    assert_allclose(mcmc.get_samples()['x'].mean(), 0., atol=0.1)


@pytest.mark.parametrize('init_strategy', [
    init_to_feasible(),
    init_to_median(num_samples=2),
    init_to_sample(),
    init_to_uniform(radius=3),
    init_to_value(values={'tau': 0.7}),
    init_to_feasible,
    init_to_median,
    init_to_sample,
    init_to_uniform,
    init_to_value,
])
def test_initialize_model_change_point(init_strategy):
    def model(data):
        alpha = 1 / jnp.mean(data)
        lambda1 = numpyro.sample('lambda1', dist.Exponential(alpha))
        lambda2 = numpyro.sample('lambda2', dist.Exponential(alpha))
        tau = numpyro.sample('tau', dist.Uniform(0, 1))
        lambda12 = jnp.where(jnp.arange(len(data)) < tau * len(data), lambda1, lambda2)
        numpyro.sample('obs', dist.Poisson(lambda12), obs=data)

    count_data = jnp.array([
        13,  24,   8,  24,   7,  35,  14,  11,  15,  11,  22,  22,  11,  57,
        11,  19,  29,   6,  19,  12,  22,  12,  18,  72,  32,   9,   7,  13,
        19,  23,  27,  20,   6,  17,  13,  10,  14,   6,  16,  15,   7,   2,
        15,  15,  19,  70,  49,   7,  53,  22,  21,  31,  19,  11,  18,  20,
        12,  35,  17,  23,  17,   4,   2,  31,  30,  13,  27,   0,  39,  37,
        5,  14,  13,  22,
    ])

    rng_keys = random.split(random.PRNGKey(1), 2)
    init_params, _, _, _ = initialize_model(rng_keys, model,
                                            init_strategy=init_strategy,
                                            model_args=(count_data,))
    if isinstance(init_strategy, partial) and init_strategy.func is init_to_value:
        expected = biject_to(constraints.unit_interval).inv(init_strategy.keywords.get('values')['tau'])
        assert_allclose(init_params[0]['tau'], jnp.repeat(expected, 2))
    for i in range(2):
        init_params_i, _, _, _ = initialize_model(rng_keys[i], model,
                                                  init_strategy=init_strategy,
                                                  model_args=(count_data,))
        for name, p in init_params[0].items():
            # XXX: the result is equal if we disable fast-math-mode
            assert_allclose(p[i], init_params_i[0][name], atol=1e-6)


@pytest.mark.parametrize('init_strategy', [
    init_to_feasible(),
    init_to_median(num_samples=2),
    init_to_sample(),
    init_to_uniform(),
])
def test_initialize_model_dirichlet_categorical(init_strategy):
    def model(data):
        concentration = jnp.array([1.0, 1.0, 1.0])
        p_latent = numpyro.sample('p_latent', dist.Dirichlet(concentration))
        numpyro.sample('obs', dist.Categorical(p_latent), obs=data)
        return p_latent

    true_probs = jnp.array([0.1, 0.6, 0.3])
    data = dist.Categorical(true_probs).sample(random.PRNGKey(1), (2000,))

    rng_keys = random.split(random.PRNGKey(1), 2)
    init_params, _, _, _ = initialize_model(rng_keys, model,
                                            init_strategy=init_strategy,
                                            model_args=(data,))
    for i in range(2):
        init_params_i, _, _, _ = initialize_model(rng_keys[i], model,
                                                  init_strategy=init_strategy,
                                                  model_args=(data,))
        for name, p in init_params[0].items():
            # XXX: the result is equal if we disable fast-math-mode
            assert_allclose(p[i], init_params_i[0][name], atol=1e-6)


@pytest.mark.parametrize('event_shape', [(3,), ()])
def test_improper_expand(event_shape):

    def model():
        population = jnp.array([1000., 2000., 3000.])
        with numpyro.plate("region", 3):
            d = dist.ImproperUniform(support=constraints.interval(0, population),
                                     batch_shape=(3,),
                                     event_shape=event_shape)
            incidence = numpyro.sample("incidence", d)
            assert d.log_prob(incidence).shape == (3,)

    model_info = initialize_model(random.PRNGKey(0), model)
    assert model_info.param_info.z['incidence'].shape == (3,) + event_shape
