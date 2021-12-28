# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest

from jax import random
import jax.numpy as jnp

import numpyro
from numpyro import handlers
from numpyro.contrib.einstein.reinit_guide import WrappedGuide
from numpyro.distributions import Bernoulli, Normal
from numpyro.distributions.constraints import _Real, softplus_positive
from numpyro.infer import (
    init_to_feasible,
    init_to_median,
    init_to_sample,
    init_to_uniform,
)
from numpyro.infer.autoguide import (
    AutoDelta,
    AutoDiagonalNormal,
    AutoLaplaceApproximation,
    AutoLowRankMultivariateNormal,
    AutoMultivariateNormal,
    AutoNormal,
)


@pytest.mark.parametrize(
    "auto_class",
    [
        AutoMultivariateNormal,
        AutoLaplaceApproximation,
        AutoLowRankMultivariateNormal,
        AutoNormal,
        AutoDelta,
        AutoDiagonalNormal,
        None,
    ],
)
@pytest.mark.parametrize(
    "init_loc_fn",
    [
        init_to_feasible,
        init_to_median,
        init_to_sample,
        init_to_uniform,
    ],
)
@pytest.mark.parametrize("num_particles", [1, 2, 10])
def test_auto_guide(auto_class, init_loc_fn, num_particles):
    latent_dim = 3

    def model(obs):
        a = numpyro.sample("a", Normal(0, 1))
        return numpyro.sample("obs", Bernoulli(logits=a), obs=obs)

    def guide(obs):
        loc_param = numpyro.param("loc_param", jnp.zeros(latent_dim))
        scale_param = numpyro.param(
            "scale_param", jnp.full(latent_dim, 0.1), constraint=softplus_positive
        )
        numpyro.sample("a", Normal(loc_param, scale_param))

    obs = Bernoulli(0.5).sample(random.PRNGKey(0), (10, latent_dim))

    auto_guide = (
        auto_class(model, init_loc_fn=init_loc_fn())
        if auto_class is not None
        else guide
    )

    with handlers.seed(rng_seed=1), handlers.trace() as auto_guide_tr:
        auto_guide(obs)

    # Corresponds to current procedure in `SteinVI.init`
    wrapped_guide = WrappedGuide(auto_guide, init_loc_fn=init_loc_fn())
    rng_keys = random.split(random.PRNGKey(2), num_particles)
    wrapped_guide.find_params(rng_keys, obs)
    init_params = wrapped_guide.init_params()

    for name, (init_value, constraint) in init_params.items():
        assert name in auto_guide_tr
        auto_param = auto_guide_tr[name]
        assert init_value.shape == (num_particles, *auto_param["value"].shape)

        if "constraint" in auto_param["kwargs"]:
            assert constraint == auto_param["kwargs"]["constraint"]
        else:
            constraint == _Real()
