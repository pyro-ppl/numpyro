# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from copy import copy

from numpy.testing import assert_array_equal
import pytest

from jax import random, vmap
import jax.numpy as jnp

import numpyro
from numpyro import handlers
from numpyro.contrib.einstein.util import get_parameter_transform
from numpyro.distributions import Bernoulli, Normal
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
        None,
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

    def guide(obs):
        numpyro.param("a", 0, constraint=interval(0, 1.0))
        numpyro.param(
            "b", lambda rng_key: Normal(0, 0.1).sample(rng_key), constraint=circular
        )
        numpyro.param("c", 0, constraint=positive)

    obs = Bernoulli(0.5).sample(random.PRNGKey(0), (10, latent_dim))

    if auto_class is None:
        inner_guide = guide
    else:
        inner_guide = auto_class(model, init_loc_fn=init_loc_fn())

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
                init_value = jnp.expand_dims(
                    site_value, 0
                ) + Normal(  # Add gaussian noise
                    scale=0.1
                ).sample(
                    particle_key, (num_particles, *site_shape)
                )

                init_value = get_parameter_transform(site)(init_value)
            elif callable(site_value):  # param(name, lambda rng_key: ...)
                site_key, particle_key = random.split(particle_key)
                keys = random.split(site_key, num_particles).reshape((num_particles, 2))
                init_value = vmap(site["value"])(keys)
                init_value = get_parameter_transform(site)(init_value)
            else:
                init_value = (
                    jnp.full(  # site_value in constraint space, no need to transform
                        (num_particles, *jnp.shape(site_value)), site_value
                    )
                )

            init_params[name] = (init_value, constraint)

    for name, (init_value, constraint) in init_params.items():
        assert name in inner_guide_tr
        inner_param = inner_guide_tr[name]
        expected_shape = (num_particles, *jnp.shape(inner_param["value"]))
        assert init_value.shape == expected_shape
        if "auto_loc" in name or name == "b":
            assert jnp.alltrue(init_value != jnp.zeros(expected_shape))
        elif "scale" in name:
            assert_array_equal(init_value, jnp.full(expected_shape, 0.1))
        else:
            assert_array_equal(init_value, jnp.zeros(expected_shape))

        if "constraint" in inner_param["kwargs"]:
            assert constraint == inner_param["kwargs"]["constraint"]
        else:
            constraint == real
