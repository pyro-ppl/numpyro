# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pickle

import pytest

from jax import random, test_util
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import (
    HMC,
    HMCECS,
    MCMC,
    NUTS,
    SA,
    BarkerMH,
    DiscreteHMCGibbs,
    MixedHMC,
)


def normal_model():
    numpyro.sample("x", dist.Normal(0, 1))


def bernoulli_model():
    numpyro.sample("x", dist.Bernoulli(0.5))


def logistic_regression():
    data = jnp.arange(10)
    x = numpyro.sample("x", dist.Normal(0, 1))
    with numpyro.plate("N", 10, subsample_size=2):
        batch = numpyro.subsample(data, 0)
        numpyro.sample("obs", dist.Bernoulli(logits=x), obs=batch)


@pytest.mark.parametrize("kernel", [BarkerMH, HMC, NUTS, SA])
def test_pickle_hmc(kernel):
    mcmc = MCMC(kernel(normal_model), num_warmup=10, num_samples=10)
    mcmc.run(random.PRNGKey(0))
    pickled_mcmc = pickle.loads(pickle.dumps(mcmc))
    test_util.check_close(mcmc.get_samples(), pickled_mcmc.get_samples())


@pytest.mark.parametrize("kernel", [DiscreteHMCGibbs, MixedHMC])
def test_pickle_discrete_hmc(kernel):
    mcmc = MCMC(kernel(HMC(bernoulli_model)), num_warmup=10, num_samples=10)
    mcmc.run(random.PRNGKey(0))
    pickled_mcmc = pickle.loads(pickle.dumps(mcmc))
    test_util.check_close(mcmc.get_samples(), pickled_mcmc.get_samples())


def test_pickle_hmcecs():
    mcmc = MCMC(HMCECS(NUTS(logistic_regression)), num_warmup=10, num_samples=10)
    mcmc.run(random.PRNGKey(0))
    pickled_mcmc = pickle.loads(pickle.dumps(mcmc))
    test_util.check_close(mcmc.get_samples(), pickled_mcmc.get_samples())
