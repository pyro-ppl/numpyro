# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest

from jax import random

import numpyro
from numpyro import handlers
from numpyro.contrib.stochastic_support.sdvi import SDVI
import numpyro.distributions as dist
from numpyro.infer import (
    RenyiELBO,
    Trace_ELBO,
    TraceEnum_ELBO,
    TraceGraph_ELBO,
    TraceMeanField_ELBO,
)
from numpyro.infer.autoguide import (
    AutoBNAFNormal,
    AutoDAIS,
    AutoDelta,
    AutoDiagonalNormal,
    AutoGuideList,
    AutoIAFNormal,
    AutoLaplaceApproximation,
    AutoLowRankMultivariateNormal,
    AutoMultivariateNormal,
    AutoNormal,
)


@pytest.mark.parametrize(
    "auto_class",
    [
        AutoDiagonalNormal,
        AutoDAIS,
        AutoIAFNormal,
        AutoBNAFNormal,
        AutoMultivariateNormal,
        AutoLaplaceApproximation,
        AutoLowRankMultivariateNormal,
        AutoNormal,
        AutoDelta,
        AutoGuideList,
    ],
)
def test_autoguides(auto_class):
    dim = 2

    def model(y):
        z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([dim]).to_event())
        model1 = numpyro.sample(
            "model1", dist.Bernoulli(0.5), infer={"branching": True}
        )
        sigma = 1.0 if model1 == 0 else 2.0
        with numpyro.plate("data", y.shape[0]):
            numpyro.sample("obs", dist.Normal(z, sigma).to_event(), obs=y)

    rng_key = random.PRNGKey(0)

    rng_key, subkey = random.split(rng_key)
    y = dist.Normal(0, 1).sample(subkey, (200, dim))
    if auto_class == AutoGuideList:

        def guide_init_fn(model):
            guide = AutoGuideList(model)
            guide.append(AutoNormal(handlers.block(model, hide=[])))
            return guide

        auto_class = guide_init_fn

    sdvi = SDVI(
        model,
        optimizer=numpyro.optim.Adam(0.01),
        guide_init=auto_class,
        svi_num_steps=10,
    )

    rng_key, subkey = random.split(rng_key)
    sdvi.run(subkey, y)


@pytest.mark.parametrize(
    "elbo_class",
    [
        Trace_ELBO,
        TraceMeanField_ELBO,
        TraceEnum_ELBO,
        TraceGraph_ELBO,
    ],
)
@pytest.mark.parametrize("num_particles", [1, 4])
def test_elbos(elbo_class, num_particles):
    dim = 2

    def model(y):
        z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([dim]).to_event())
        model1 = numpyro.sample(
            "model1", dist.Bernoulli(0.5), infer={"branching": True}
        )
        sigma = 1.0 if model1 == 0 else 2.0
        with numpyro.plate("data", y.shape[0]):
            numpyro.sample("obs", dist.Normal(z, sigma).to_event(), obs=y)

    rng_key = random.PRNGKey(0)

    rng_key, subkey = random.split(rng_key)
    y = dist.Normal(0, 1).sample(subkey, (200, dim))
    sdvi = SDVI(
        model,
        optimizer=numpyro.optim.Adam(0.01),
        guide_init=AutoNormal,
        svi_num_steps=10,
        loss=elbo_class(num_particles=num_particles),
    )

    rng_key, subkey = random.split(rng_key)
    sdvi.run(subkey, y)


@pytest.mark.parametrize("elbo_class", [RenyiELBO])
@pytest.mark.xfail(raises=ValueError)
def test_fail_elbos(elbo_class):
    dim = 2

    def model(y):
        z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([dim]).to_event())
        model1 = numpyro.sample(
            "model1", dist.Bernoulli(0.5), infer={"branching": True}
        )
        sigma = 1.0 if model1 == 0 else 2.0
        with numpyro.plate("data", y.shape[0]):
            numpyro.sample("obs", dist.Normal(z, sigma).to_event(), obs=y)

    rng_key = random.PRNGKey(0)

    rng_key, subkey = random.split(rng_key)
    y = dist.Normal(0, 1).sample(subkey, (200, dim))
    sdvi = SDVI(
        model,
        optimizer=numpyro.optim.Adam(0.01),
        svi_num_steps=10,
        loss=elbo_class(),
    )

    rng_key, subkey = random.split(rng_key)
    sdvi.run(subkey, y)


def test_progress_bar():
    dim = 2

    def model(y):
        z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([dim]).to_event())
        model1 = numpyro.sample(
            "model1", dist.Bernoulli(0.5), infer={"branching": True}
        )
        sigma = 1.0 if model1 == 0 else 2.0
        with numpyro.plate("data", y.shape[0]):
            numpyro.sample("obs", dist.Normal(z, sigma).to_event(), obs=y)

    rng_key = random.PRNGKey(0)

    rng_key, subkey = random.split(rng_key)
    y = dist.Normal(0, 1).sample(subkey, (200, dim))
    sdvi = SDVI(
        model,
        optimizer=numpyro.optim.Adam(0.01),
        svi_num_steps=10,
        svi_progress_bar=True,
    )
    rng_key, subkey = random.split(rng_key)
    sdvi.run(subkey, y)
