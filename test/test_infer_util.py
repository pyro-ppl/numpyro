from numpy.testing import assert_allclose

from jax import random
import jax.numpy as np

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer_util import log_likelihood, predictive, transformed_potential_energy
from numpyro.mcmc import MCMC, NUTS


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
    transform = constraints.AffineTransform(3, 4)
    inv_transform = constraints.AffineTransform(-0.75, 0.25)

    z = random.normal(random.PRNGKey(0), (5,))
    pe_expected = -dist.TransformedDistribution(beta_dist, transform).log_prob(z)
    potential_fn = lambda x: -beta_dist.log_prob(x)  # noqa: E731
    pe_actual = transformed_potential_energy(potential_fn, inv_transform, z)
    assert_allclose(pe_actual, pe_expected)
