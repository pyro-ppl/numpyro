# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from numpy.testing import assert_allclose
from pytest import fail

from jax import numpy as jnp, random, value_and_grad

import numpyro
from numpyro.contrib.einstein.stein_loss import SteinLoss
from numpyro.contrib.einstein.stein_util import batch_ravel_pytree
import numpyro.distributions as dist
from numpyro.infer import Trace_ELBO


def test_single_particle_loss():
    def model(x):
        numpyro.sample("obs", dist.Normal(0, 1), obs=x)

    def guide(x):
        pass

    try:
        SteinLoss(elbo_num_particles=10, stein_num_particles=1).loss(
            random.PRNGKey(0), {}, model, guide, {}, 2.0
        )
        fail()
    except ValueError:
        pass


def test_stein_elbo():
    def model(x):
        numpyro.sample("x", dist.Normal(0, 1))
        numpyro.sample("obs", dist.Normal(0, 1), obs=x)

    def guide(x):
        numpyro.sample("x", dist.Normal(0, 1))

    def elbo_loss_fn(x, param):
        return Trace_ELBO(num_particles=1).loss(
            random.PRNGKey(0), param, model, guide, x
        )

    def stein_loss_fn(x, particles):
        return SteinLoss(elbo_num_particles=1, stein_num_particles=1).loss(
            random.PRNGKey(0), {}, model, guide, particles, x
        )

    elbo_loss, elbo_grad = value_and_grad(elbo_loss_fn)(2.0, {"x": 1.0})
    stein_loss, stein_grad = value_and_grad(stein_loss_fn)(2.0, {"x": jnp.array([1.0])})
    assert_allclose(elbo_loss, stein_loss, rtol=1e-6)
    assert_allclose(elbo_grad, stein_grad, rtol=1e-6)


def test_stein_particle_loss():
    def model(x):
        numpyro.sample("x", dist.Normal(0, 1))
        numpyro.sample("obs", dist.Normal(0, 1), obs=x)

    def guide(x):
        numpyro.sample("x", dist.Normal(0, 1))

    def stein_loss_fn(x, particles, chosen_particle, assign):
        return SteinLoss(
            elbo_num_particles=1, stein_num_particles=3
        ).single_particle_loss(
            random.PRNGKey(0),
            model,
            guide,
            chosen_particle,
            unravel_pytree,
            particles,
            assign,
            (x,),
            {},
            {},
        )

    xs = jnp.array([-1, 0.5, 3.0])
    num_particles = xs.shape[0]
    particles = {"x": xs}

    flat_particles, unravel_pytree, _ = batch_ravel_pytree(particles, nbatch_dims=1)
    losses, grads = [], []
    for i in range(num_particles):
        chosen_particle = unravel_pytree(flat_particles[i])
        loss, grad = value_and_grad(stein_loss_fn)(
            2.0, flat_particles, chosen_particle, i
        )
        losses.append(loss)
        grads.append(grad)

    assert jnp.abs(losses[0] - losses[1]) > 0.1
    assert jnp.abs(losses[1] - losses[2]) > 0.1
    assert_allclose(grads[0], grads[1])
    assert_allclose(grads[1], grads[2])
