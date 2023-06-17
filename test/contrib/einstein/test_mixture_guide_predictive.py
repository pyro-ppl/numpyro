# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from numpy.testing import assert_allclose

from jax import random
import jax.numpy as jnp

import numpyro
from numpyro.contrib.einstein.mixture_guide_predictive import MixtureGuidePredictive
import numpyro.distributions as dist
from numpyro.distributions import constraints


def test_predictive_with_particles():
    num_samples = 20
    fdim = 3
    num_data = 10
    mixture_assignment_sitename = "assigns"
    num_particles = 3

    def model(x, y=None):
        latent = numpyro.sample("latent", dist.Normal(0.0, jnp.ones(fdim)).to_event(1))
        with numpyro.plate("data", x.shape[0]):
            numpyro.sample("y", dist.Normal(x * latent, 1.0).to_event(1), obs=y)

    def guide(x, y=None):
        latent_loc = numpyro.param(
            "latent_loc", jnp.ones(fdim), constraint=constraints.real
        )
        assert latent_loc.ndim == 1
        numpyro.sample("latent", dist.Normal(latent_loc, 0.1).to_event(1))

    params = jnp.array([[-100, -100, -100.0], [0, 0, 0], [100, 100, 100]])
    x = dist.Normal(jnp.full(fdim, 10), 1.0).sample(random.PRNGKey(0), (num_data,))

    predictions = MixtureGuidePredictive(
        model,
        guide=guide,
        params={"latent_loc": params},
        num_samples=num_samples,
        guide_sites=["latent_loc"],
        mixture_assignment_sitename=mixture_assignment_sitename,
    )(random.PRNGKey(0), x)
    assert predictions["y"].shape == (num_samples, num_data, fdim)
    assert mixture_assignment_sitename in predictions
    assert jnp.max(predictions[mixture_assignment_sitename]) <= num_particles - 1
    assert 0 <= jnp.min(predictions[mixture_assignment_sitename])

    # Check we can recover assignments from predictions
    pred_assigns = jnp.argmin(
        jnp.linalg.norm(predictions["y"][:, :, None] - params, axis=-1), axis=-1
    )
    actual_assigns = jnp.repeat(
        predictions[mixture_assignment_sitename][:, None], num_data, 1
    )
    assert_allclose(pred_assigns, actual_assigns)
