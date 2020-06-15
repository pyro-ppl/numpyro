# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numpy.testing import assert_allclose

from jax import random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.contrib.funsor import markov


def test_gaussian_mixture_model():
    K, N = 3, 1000

    def gmm(data):
        mix_proportions = numpyro.sample("phi", dist.Dirichlet(jnp.ones(K)))
        with numpyro.plate("num_clusters", K, dim=-1):
            cluster_means = numpyro.sample("cluster_means", dist.Normal(jnp.arange(K), 1.))
        with numpyro.plate("data", data.shape[0], dim=-1):
            assignments = numpyro.sample("assignments", dist.Categorical(mix_proportions))
            numpyro.sample("obs", dist.Normal(cluster_means[assignments], 1.), obs=data)

    true_cluster_means = jnp.array([1., 5., 10.])
    true_mix_proportions = jnp.array([0.1, 0.3, 0.6])
    cluster_assignments = dist.Categorical(true_mix_proportions).sample(random.PRNGKey(0), (N,))
    data = dist.Normal(true_cluster_means[cluster_assignments], 1.0).sample(random.PRNGKey(1))

    nuts_kernel = NUTS(gmm)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=500)
    mcmc.run(random.PRNGKey(2), data)
    samples = mcmc.get_samples()
    assert_allclose(samples["phi"].mean(0).sort(), true_mix_proportions, atol=0.05)
    assert_allclose(samples["cluster_means"].mean(0).sort(), true_cluster_means, atol=0.2)


def test_bernoulli_latent_model():
    def model(data):
        y_prob = numpyro.sample("y_prob", dist.Beta(1., 1.))
        with numpyro.plate("data", data.shape[0]):
            y = numpyro.sample("y", dist.Bernoulli(y_prob))
            z = numpyro.sample("z", dist.Bernoulli(0.65 * y + 0.1))
            numpyro.sample("obs", dist.Normal(2. * z, 1.), obs=data)

    N = 2000
    y_prob = 0.3
    y = dist.Bernoulli(y_prob).sample(random.PRNGKey(0), (N,))
    z = dist.Bernoulli(0.65 * y + 0.1).sample(random.PRNGKey(1))
    data = dist.Normal(2. * z, 1.0).sample(random.PRNGKey(2))

    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=500)
    mcmc.run(random.PRNGKey(3), data)
    samples = mcmc.get_samples()
    assert_allclose(samples["y_prob"].mean(0), y_prob, atol=0.05)


def test_gaussian_hmm():
    dim = 4
    num_steps = 10

    def model(data):
        with numpyro.plate("states", dim):
            transition = numpyro.sample("transition", dist.Dirichlet(jnp.ones(dim)))
            emission_loc = numpyro.sample("emission_loc", dist.Normal(0, 1))
            emission_scale = numpyro.sample("emission_scale", dist.LogNormal(0, 1))

        trans_prob = numpyro.sample("initialize", dist.Dirichlet(jnp.ones(dim)))
        for t, y in markov(enumerate(data)):
            x = numpyro.sample("x_{}".format(t), dist.Categorical(trans_prob))
            numpyro.sample("y_{}".format(t), dist.Normal(emission_loc[x], emission_scale[x]), obs=y)
            trans_prob = transition[x]

    def _generate_data():
        transition_probs = np.random.rand(dim, dim)
        transition_probs = transition_probs / transition_probs.sum(-1, keepdims=True)
        emissions_loc = np.arange(dim)
        emissions_scale = 1.
        state = np.random.choice(3)
        obs = [np.random.normal(emissions_loc[state], emissions_scale)]
        for _ in range(num_steps - 1):
            state = np.random.choice(dim, p=transition_probs[state])
            obs.append(np.random.normal(emissions_loc[state], emissions_scale))
        return np.stack(obs)

    data = _generate_data()
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=500)
    mcmc.run(random.PRNGKey(0), data)
