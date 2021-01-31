import argparse
import os

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import seaborn as sns

import jax
from jax import random
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import numpyro
from numpyro import optim
from numpyro.diagnostics import print_summary
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoBNAFNormal
from numpyro.infer.reparam import NeuTraReparam


class DualMoonDistribution(dist.Distribution):
    support = constraints.real_vector

    def __init__(self):
        super(DualMoonDistribution, self).__init__(event_shape=(2,))

    def sample(self, key, sample_shape=()):
        # it is enough to return an arbitrary sample with correct shape
        return jnp.zeros(sample_shape + self.event_shape)

    def log_prob(self, x):
        term1 = 0.5 * ((jnp.linalg.norm(x, axis=-1) - 2) / 0.4) ** 2
        term2 = -0.5 * ((x[..., :1] + jnp.array([-2., 2.])) / 0.6) ** 2
        pe = term1 - logsumexp(term2, axis=-1)
        return -pe


def dual_moon_model():
    numpyro.sample('x', DualMoonDistribution())


def guide():
    var = numpyro.param('var', jnp.eye(2, dtype=jnp.float32), constraints=constraints.corr_matrix)
    mean = numpyro.param('mean', jnp.zeros(2, dtype=jnp.float32), constraints=constraints.real_vector)
    numpyro.sample('x', dist.MultivariateNormal(mean, var))


def visualize(samples):
    print(samples.shape)
    print(samples)
    sns.kdeplot(x=samples[:, 0], y=samples[:, 1])
    plt.show()


def two_moons(rng_key, noise, shape):
    def make_circle(data, radius, center):
        return jnp.sqrt(radius ** 2 - (data - center) ** 2)

    # TODO: finish compute density

    noise_key, uni_key = random.split(rng_key)
    uni_samples = jax.random.uniform(uni_key, shape)
    noise = noise * jax.random.normal(noise_key, shape)
    upper = uni_samples[:shape[0] // 2] - .25
    upper_noise = noise[:shape[0] // 2]
    lower_noise = noise[shape[0] // 2:]
    lower = uni_samples[shape[0] // 2:] + .25
    upper = jnp.vstack((upper, -make_circle(upper, .5, .25) + .1)).T
    lower = jnp.vstack((lower, make_circle(lower, .5, .75) - .1)).T

    plt.scatter(upper[:, 0], upper[:, 1])
    plt.scatter(lower[:, 0], lower[:, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


if __name__ == '__main__':
    sim_key, guide_key, mcmc_key = random.split(random.PRNGKey(0), 3)
    two_moons(sim_key, noise=.05, shape=(1000,))
    dm = DualMoonDistribution()
    samples = dm.sample(sim_key, (10000,))

    # visualize(samples)
