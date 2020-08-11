# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
from numpy.testing import assert_allclose

import jax.numpy as jnp
from jax import random

import numpyro
from numpyro.infer import MCMC, NUTS


@pytest.mark.filterwarnings("ignore")
def test_logistic_regression():
    from numpyro.contrib.tfp import distributions as dist

    N, dim = 3000, 3
    num_warmup, num_samples = (1000, 8000)
    data = random.normal(random.PRNGKey(0), (N, dim))
    true_coefs = jnp.arange(1., dim + 1.)
    logits = jnp.sum(true_coefs * data, axis=-1)
    labels = dist.Bernoulli(logits=logits).sample(seed=random.PRNGKey(1))

    def model(labels):
        coefs = numpyro.sample('coefs', dist.Normal(jnp.zeros(dim), jnp.ones(dim)))
        logits = numpyro.deterministic('logits', jnp.sum(coefs * data, axis=-1))
        return numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=labels)

    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup, num_samples)
    mcmc.run(random.PRNGKey(2), labels)
    mcmc.print_summary()
    samples = mcmc.get_samples()
    assert samples['logits'].shape == (num_samples, N)
    assert_allclose(jnp.mean(samples['coefs'], 0), true_coefs, atol=0.22)


@pytest.mark.filterwarnings("ignore")
def test_beta_bernoulli():
    from numpyro.contrib.tfp import distributions as dist

    warmup_steps, num_samples = (500, 2000)

    def model(data):
        alpha = jnp.array([1.1, 1.1])
        beta = jnp.array([1.1, 1.1])
        p_latent = numpyro.sample('p_latent', dist.Beta(alpha, beta))
        numpyro.sample('obs', dist.Bernoulli(p_latent), obs=data)
        return p_latent

    true_probs = jnp.array([0.9, 0.1])
    data = dist.Bernoulli(true_probs).sample(seed=random.PRNGKey(1), sample_shape=(1000, 2))
    kernel = NUTS(model=model, trajectory_length=0.1)
    mcmc = MCMC(kernel, num_warmup=warmup_steps, num_samples=num_samples)
    mcmc.run(random.PRNGKey(2), data)
    mcmc.print_summary()
    samples = mcmc.get_samples()
    assert_allclose(jnp.mean(samples['p_latent'], 0), true_probs, atol=0.05)
