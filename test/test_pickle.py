# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pickle

import numpy as np
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
    SVI,
    BarkerMH,
    DiscreteHMCGibbs,
    MixedHMC,
    Predictive,
)
from numpyro.infer.autoguide import AutoDelta, AutoDiagonalNormal, AutoNormal


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


def poisson_regression(x, N):
    rate = numpyro.sample("param", dist.Gamma(1.0, 1.0))
    batch_size = len(x) if x is not None else None
    with numpyro.plate("batch", N, batch_size):
        numpyro.sample("x", dist.Poisson(rate), obs=x)


@pytest.mark.parametrize("guide_class", [AutoDelta, AutoDiagonalNormal, AutoNormal])
def test_pickle_autoguide(guide_class):
    x = np.random.poisson(1.0, size=(100,))

    guide = guide_class(poisson_regression)
    optim = numpyro.optim.Adam(1e-2)
    svi = SVI(poisson_regression, guide, optim, numpyro.infer.Trace_ELBO())
    svi_result = svi.run(random.PRNGKey(1), 3, x, len(x))
    pickled_guide = pickle.loads(pickle.dumps(guide))

    predictive = Predictive(
        poisson_regression,
        guide=pickled_guide,
        params=svi_result.params,
        num_samples=1,
        return_sites=["param", "x"],
    )
    samples = predictive(random.PRNGKey(1), None, 1)
    assert set(samples.keys()) == {"param", "x"}
