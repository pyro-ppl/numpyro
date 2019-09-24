from numpy.testing import assert_allclose

from jax import random
import jax.numpy as np

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer_util import predictive, transformed_potential_energy
from numpyro.mcmc import MCMC, NUTS


def beta_bernoulli():
    N = 1000
    true_probs = np.array([0.2, 0.3, 0.4, 0.8, 0.5])
    data = dist.Bernoulli(true_probs).sample(random.PRNGKey(0), (N,))

    def model(data=None):
        beta = numpyro.sample("beta", dist.Beta(np.ones(5), np.ones(5)))
        numpyro.sample("obs", dist.Bernoulli(beta), obs=data, sample_shape=(1000,))

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
    assert predictive_samples["beta"].shape == (100, 5)
    assert predictive_samples["obs"].shape == (100, 1000, 5)

    # check sample mean
    assert_allclose(predictive_samples["obs"].reshape([-1, 5]).mean(0), true_probs, rtol=0.1)


def test_transformed_potential_energy():
    beta_dist = dist.Beta(np.ones(5), np.ones(5))
    transform = constraints.AffineTransform(3, 4)
    inv_transform = constraints.AffineTransform(-0.75, 0.25)

    z = random.normal(random.PRNGKey(0), (5,))
    pe_expected = -dist.TransformedDistribution(beta_dist, transform).log_prob(z)
    potential_fn = lambda x: -beta_dist.log_prob(x)
    pe_actual = transformed_potential_energy(potential_fn, inv_transform, z)
    assert_allclose(pe_actual, pe_expected)
