from functools import partial

from numpy.testing import assert_allclose
import pytest

from jax import lax, random
import jax.numpy as np

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.distributions import constraints, transforms
from numpyro.distributions.transforms import biject_to
from numpyro.infer import ELBO, MCMC, NUTS, SVI
from numpyro.infer.util import (
    Predictive,
    constrain_fn,
    init_to_feasible,
    init_to_median,
    init_to_prior,
    init_to_uniform,
    initialize_model,
    log_likelihood,
    potential_energy,
    transform_fn,
    transformed_potential_energy
)
import numpyro.optim as optim


def beta_bernoulli():
    N = 800
    true_probs = np.array([0.2, 0.7])
    data = dist.Bernoulli(true_probs).sample(random.PRNGKey(0), (N,))

    def model(data=None):
        beta = numpyro.sample("beta", dist.Beta(np.ones(2), np.ones(2)))
        with numpyro.plate("plate", N, dim=-2):
            numpyro.sample("obs", dist.Bernoulli(beta), obs=data)

    return model, data, true_probs


@pytest.mark.parametrize('parallel', [True, False])
def test_predictive(parallel):
    model, data, true_probs = beta_bernoulli()
    mcmc = MCMC(NUTS(model), num_warmup=100, num_samples=100)
    mcmc.run(random.PRNGKey(0), data)
    samples = mcmc.get_samples()
    predictive = Predictive(model, samples, parallel=parallel)
    predictive_samples = predictive.get_samples(random.PRNGKey(1))
    assert predictive_samples.keys() == {"obs"}

    predictive.return_sites = ["beta", "obs"]
    predictive_samples = predictive.get_samples(random.PRNGKey(1))
    # check shapes
    assert predictive_samples["beta"].shape == (100,) + true_probs.shape
    assert predictive_samples["obs"].shape == (100,) + data.shape
    # check sample mean
    assert_allclose(predictive_samples["obs"].reshape((-1,) + true_probs.shape).mean(0), true_probs, rtol=0.1)


def test_predictive_with_guide():
    data = np.array([1] * 8 + [0] * 2)

    def model(data):
        f = numpyro.sample("beta", dist.Beta(1., 1.))
        with numpyro.plate("plate", 10):
            numpyro.sample("obs", dist.Bernoulli(f), obs=data)

    def guide(data):
        alpha_q = numpyro.param("alpha_q", 1.0,
                                constraint=constraints.positive)
        beta_q = numpyro.param("beta_q", 1.0,
                               constraint=constraints.positive)
        numpyro.sample("beta", dist.Beta(alpha_q, beta_q))

    svi = SVI(model, guide, optim.Adam(0.1), ELBO())
    svi_state = svi.init(random.PRNGKey(1), data)

    def body_fn(i, val):
        svi_state, _ = svi.update(val, data)
        return svi_state

    svi_state = lax.fori_loop(0, 1000, body_fn, svi_state)
    params = svi.get_params(svi_state)
    predictive = Predictive(model, guide=guide, params=params, num_samples=1000)
    obs_pred = predictive.get_samples(random.PRNGKey(2), data=None)["obs"]
    assert_allclose(np.mean(obs_pred), 0.8, atol=0.05)


def test_predictive_with_improper():
    true_coef = 0.9

    def model(data):
        alpha = numpyro.sample('alpha', dist.Uniform(0, 1))
        loc = numpyro.param('loc', 0., constraint=constraints.interval(0., alpha))
        numpyro.sample('obs', dist.Normal(loc, 0.1), obs=data)

    data = true_coef + random.normal(random.PRNGKey(0), (1000,))
    kernel = NUTS(model=model)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000)
    mcmc.run(random.PRNGKey(0), data)
    samples = mcmc.get_samples()
    obs_pred = Predictive(model, samples).get_samples(random.PRNGKey(1), data=None)["obs"]
    assert_allclose(np.mean(obs_pred), true_coef, atol=0.05)


def test_prior_predictive():
    model, data, true_probs = beta_bernoulli()
    predictive_samples = Predictive(model, num_samples=100).get_samples(random.PRNGKey(1))
    assert predictive_samples.keys() == {"beta", "obs"}

    # check shapes
    assert predictive_samples["beta"].shape == (100,) + true_probs.shape
    assert predictive_samples["obs"].shape == (100,) + data.shape


def test_log_likelihood():
    model, data, _ = beta_bernoulli()
    samples = Predictive(model, return_sites=["beta"], num_samples=100).get_samples(random.PRNGKey(1))
    loglik = log_likelihood(model, samples, data)
    assert loglik.keys() == {"obs"}
    # check shapes
    assert loglik["obs"].shape == (100,) + data.shape
    assert_allclose(loglik["obs"], dist.Bernoulli(samples["beta"].reshape((100, 1, -1))).log_prob(data))


def test_transformed_potential_energy():
    beta_dist = dist.Beta(np.ones(5), np.ones(5))
    transform = transforms.AffineTransform(3, 4)
    inv_transform = transforms.AffineTransform(-0.75, 0.25)

    z = random.normal(random.PRNGKey(0), (5,))
    pe_expected = -dist.TransformedDistribution(beta_dist, transform).log_prob(z)
    potential_fn = lambda x: -beta_dist.log_prob(x)  # noqa: E731
    pe_actual = transformed_potential_energy(potential_fn, inv_transform, z)
    assert_allclose(pe_actual, pe_expected)


def test_model_with_transformed_distribution():
    x_prior = dist.HalfNormal(2)
    y_prior = dist.LogNormal(scale=3.)  # transformed distribution

    def model():
        numpyro.sample('x', x_prior)
        numpyro.sample('y', y_prior)

    params = {'x': np.array(-5.), 'y': np.array(7.)}
    model = handlers.seed(model, random.PRNGKey(0))
    inv_transforms = {'x': biject_to(x_prior.support), 'y': biject_to(y_prior.support)}
    expected_samples = partial(transform_fn, inv_transforms)(params)
    expected_potential_energy = (
        - x_prior.log_prob(expected_samples['x']) -
        y_prior.log_prob(expected_samples['y']) -
        inv_transforms['x'].log_abs_det_jacobian(params['x'], expected_samples['x']) -
        inv_transforms['y'].log_abs_det_jacobian(params['y'], expected_samples['y'])
    )

    base_inv_transforms = {'x': biject_to(x_prior.support), 'y': biject_to(y_prior.base_dist.support)}
    actual_samples = constrain_fn(
        handlers.seed(model, random.PRNGKey(0)), base_inv_transforms,  (), {}, params)
    actual_potential_energy = potential_energy(model, base_inv_transforms, (), {}, params)

    assert_allclose(expected_samples['x'], actual_samples['x'])
    assert_allclose(expected_samples['y'], actual_samples['y'])
    assert_allclose(actual_potential_energy, expected_potential_energy)


@pytest.mark.parametrize('init_strategy', [
    init_to_feasible(),
    init_to_median(num_samples=2),
    init_to_prior(),
    init_to_uniform(),
])
def test_initialize_model_change_point(init_strategy):
    def model(data):
        alpha = 1 / np.mean(data)
        lambda1 = numpyro.sample('lambda1', dist.Exponential(alpha))
        lambda2 = numpyro.sample('lambda2', dist.Exponential(alpha))
        tau = numpyro.sample('tau', dist.Uniform(0, 1))
        lambda12 = np.where(np.arange(len(data)) < tau * len(data), lambda1, lambda2)
        numpyro.sample('obs', dist.Poisson(lambda12), obs=data)

    count_data = np.array([
        13,  24,   8,  24,   7,  35,  14,  11,  15,  11,  22,  22,  11,  57,
        11,  19,  29,   6,  19,  12,  22,  12,  18,  72,  32,   9,   7,  13,
        19,  23,  27,  20,   6,  17,  13,  10,  14,   6,  16,  15,   7,   2,
        15,  15,  19,  70,  49,   7,  53,  22,  21,  31,  19,  11,  18,  20,
        12,  35,  17,  23,  17,   4,   2,  31,  30,  13,  27,   0,  39,  37,
        5,  14,  13,  22,
    ])

    rng_keys = random.split(random.PRNGKey(1), 2)
    init_params, _, _ = initialize_model(rng_keys, model,
                                         init_strategy=init_strategy,
                                         model_args=(count_data,))
    for i in range(2):
        init_params_i, _, _ = initialize_model(rng_keys[i], model,
                                               init_strategy=init_strategy,
                                               model_args=(count_data,))
        for name, p in init_params.items():
            # XXX: the result is equal if we disable fast-math-mode
            assert_allclose(p[i], init_params_i[name], atol=1e-6)


@pytest.mark.parametrize('init_strategy', [
    init_to_feasible(),
    init_to_median(num_samples=2),
    init_to_prior(),
    init_to_uniform(),
])
def test_initialize_model_dirichlet_categorical(init_strategy):
    def model(data):
        concentration = np.array([1.0, 1.0, 1.0])
        p_latent = numpyro.sample('p_latent', dist.Dirichlet(concentration))
        numpyro.sample('obs', dist.Categorical(p_latent), obs=data)
        return p_latent

    true_probs = np.array([0.1, 0.6, 0.3])
    data = dist.Categorical(true_probs).sample(random.PRNGKey(1), (2000,))

    rng_keys = random.split(random.PRNGKey(1), 2)
    init_params, _, _ = initialize_model(rng_keys, model,
                                         init_strategy=init_strategy,
                                         model_args=(data,))
    for i in range(2):
        init_params_i, _, _ = initialize_model(rng_keys[i], model,
                                               init_strategy=init_strategy,
                                               model_args=(data,))
        for name, p in init_params.items():
            # XXX: the result is equal if we disable fast-math-mode
            assert_allclose(p[i], init_params_i[name], atol=1e-6)
