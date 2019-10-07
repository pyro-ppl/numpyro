from functools import partial

from numpy.testing import assert_allclose
import pytest

from jax import random
import jax.numpy as np

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.distributions import transforms
from numpyro.distributions.transforms import biject_to
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import (
    constrain_fn,
    init_to_feasible,
    init_to_median,
    init_to_prior,
    init_to_uniform,
    initialize_model,
    log_likelihood,
    potential_energy,
    predictive,
    transform_fn,
    transformed_potential_energy
)


def beta_bernoulli():
    N = 800
    true_probs = np.array([0.2, 0.7])
    data = dist.Bernoulli(true_probs).sample(random.PRNGKey(0), (N,))

    def model(data=None):
        beta = numpyro.sample("beta", dist.Beta(np.ones(2), np.ones(2)))
        with numpyro.plate("plate", N, dim=-2):
            numpyro.sample("obs", dist.Bernoulli(beta), obs=data)

    return model, data, true_probs


def test_predictive():
    model, data, true_probs = beta_bernoulli()
    mcmc = MCMC(NUTS(model), num_warmup=100, num_samples=100)
    mcmc.run(random.PRNGKey(0), data)
    samples = mcmc.get_samples()
    predictive_samples = predictive(random.PRNGKey(1), model, samples)
    assert predictive_samples.keys() == {"obs"}

    predictive_samples = predictive(random.PRNGKey(1), model, samples,
                                    return_sites=["beta", "obs"])
    # check shapes
    assert predictive_samples["beta"].shape == (100,) + true_probs.shape
    assert predictive_samples["obs"].shape == (100,) + data.shape
    # check sample mean
    assert_allclose(predictive_samples["obs"].reshape((-1,) + true_probs.shape).mean(0), true_probs, rtol=0.1)


def test_prior_predictive():
    model, data, true_probs = beta_bernoulli()
    predictive_samples = predictive(random.PRNGKey(1), model, {}, num_samples=100)
    assert predictive_samples.keys() == {"beta", "obs"}

    # check shapes
    assert predictive_samples["beta"].shape == (100,) + true_probs.shape
    assert predictive_samples["obs"].shape == (100,) + data.shape


def test_log_likelihood():
    model, data, _ = beta_bernoulli()
    samples = predictive(random.PRNGKey(1), model, {}, return_sites=["beta"], num_samples=100)
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
        handlers.seed(model, random.PRNGKey(0)), (), {}, base_inv_transforms, params)
    actual_potential_energy = potential_energy(model, (), {}, base_inv_transforms, params)

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

    rngs = random.split(random.PRNGKey(1), 2)
    init_params, _, _ = initialize_model(rngs, model, count_data,
                                         init_strategy=init_strategy)
    for i in range(2):
        init_params_i, _, _ = initialize_model(rngs[i], model, count_data,
                                               init_strategy=init_strategy)
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

    rngs = random.split(random.PRNGKey(1), 2)
    init_params, _, _ = initialize_model(rngs, model, data,
                                         init_strategy=init_strategy)
    for i in range(2):
        init_params_i, _, _ = initialize_model(rngs[i], model, data,
                                               init_strategy=init_strategy)
        for name, p in init_params.items():
            # XXX: the result is equal if we disable fast-math-mode
            assert_allclose(p[i], init_params_i[name], atol=1e-6)
