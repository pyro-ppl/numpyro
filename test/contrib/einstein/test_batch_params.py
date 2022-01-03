# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from copy import copy

from numpy.testing import assert_array_equal
import pytest

from jax import random, vmap
import jax.numpy as jnp

import numpyro
from numpyro import handlers
from numpyro.distributions import Bernoulli, Normal, biject_to
from numpyro.distributions.constraints import circular, interval, positive, real
from numpyro.infer import (
    init_to_feasible,
    init_to_median,
    init_to_sample,
    init_to_uniform,
)
from numpyro.infer.autoguide import (
    AutoDelta,
    AutoDiagonalNormal,
    AutoGuide,
    AutoLaplaceApproximation,
    AutoLowRankMultivariateNormal,
    AutoMultivariateNormal,
    AutoNormal,
)


@pytest.mark.parametrize(
    "auto_class",
    [
        AutoMultivariateNormal,
        AutoNormal,
        AutoLowRankMultivariateNormal,
        AutoLaplaceApproximation,
        AutoDelta,
        AutoDiagonalNormal,
    ],
)
@pytest.mark.parametrize(
    "init_loc_fn",
    [
        init_to_uniform,
        init_to_feasible,
        init_to_median,
        init_to_sample,
    ],
)
@pytest.mark.parametrize("num_particles", [1, 2, 10])
def test_auto_guide(auto_class, init_loc_fn, num_particles):
    latent_dim = 3

    def model(obs):
        a = numpyro.sample("a", Normal(0, 1))
        return numpyro.sample("obs", Bernoulli(logits=a), obs=obs)

    obs = Bernoulli(0.5).sample(random.PRNGKey(0), (10, latent_dim))

    inner_guide = auto_class(model, init_loc_fn=init_loc_fn())

    assert isinstance(inner_guide, AutoGuide)  # branch conditional
    rng_key = random.PRNGKey(0)
    guide_key, particle_key = random.split(rng_key)

    with handlers.seed(rng_seed=guide_key), handlers.trace() as inner_guide_tr:
        inner_guide(obs)

    # Corresponds to current procedure in `SteinVI.init`
    init_params = {}
    for name, site in inner_guide_tr.items():
        site = copy(site)
        constraint = site["kwargs"].get("constraint", real)
        if site["type"] == "param":
            site_value = site["value"]
            site_shape = jnp.shape(site_value)
            if (
                isinstance(inner_guide, AutoGuide)
                and "_".join((inner_guide.prefix, "loc")) in name
            ):
                site_key, particle_key = random.split(particle_key)
                unconst_value = site_value[None, ...] + Normal(  # Add gaussian noise
                    scale=0.1
                ).sample(particle_key, (num_particles, *site_shape))
                init_value = biject_to(constraint)(unconst_value)
            else:
                init_value = jnp.full(
                    (num_particles, *jnp.shape(site_value)), site_value
                )

            init_params[name] = (init_value, constraint)

    for name, (init_value, constraint) in init_params.items():
        assert name in inner_guide_tr
        inner_param = inner_guide_tr[name]
        assert init_value.shape == (num_particles, *jnp.shape(inner_param["value"]))

        if "constraint" in inner_param["kwargs"]:
            assert constraint == inner_param["kwargs"]["constraint"]
        else:
            constraint == real


def test_reinit_hide_fn():
    num_particles = 5
    const_params = ["a", "c"]

    def guide():
        numpyro.param("a", 0, constraint=interval(0, 1.0))
        numpyro.param(
            "b", lambda rng_key: Normal(0, 0.1).sample(rng_key), constraint=circular
        )
        numpyro.param("c", 0, constraint=positive)

    # SteinVI logic
    rng_key = random.PRNGKey(0)
    guide_key, particle_key = random.split(rng_key)

    with handlers.seed(rng_seed=guide_key), handlers.trace() as guide_tr:
        guide()

    init_params = {}
    for name, site in guide_tr.items():
        if site["type"] == "param":
            site_value = site["value"]
            if callable(site_value):
                site_key, particle_key = random.split(particle_key)
                keys = random.split(site_key, num_particles).reshape((num_particles, 2))
                init_value = vmap(site["value"])(keys)
            else:
                init_value = jnp.full(
                    (num_particles, *jnp.shape(site_value)), site_value
                )
            constraint = site["kwargs"].get("constraint", real)

            init_params[name] = (init_value, constraint)

    for name, (init_value, constraint) in init_params.items():
        assert name in guide_tr
        inner_param = guide_tr[name]
        assert init_value.shape == (num_particles, *jnp.shape(inner_param["value"]))
        if name in const_params:
            assert_array_equal(init_value, jnp.zeros(num_particles))
        else:
            assert jnp.alltrue(init_value != jnp.zeros(num_particles))

        assert inner_param["kwargs"]["constraint"] == constraint
