# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np
import pytest

import jax
from jax import random
import jax.numpy as jnp

import numpyro as pyro
from numpyro import infer
from numpyro.contrib.funsor import config_enumerate, config_kl
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.ops.indexing import Vindex

logger = logging.getLogger(__name__)


def assert_equal(a, b, prec=0):
    return jax.tree_util.tree_map(
        lambda a, b: np.testing.assert_allclose(a, b, atol=prec), a, b
    )


def model_0(data, params):
    with pyro.plate("data", len(data)):
        z = pyro.sample("z", dist.Categorical(jnp.array([0.3, 0.7])))
        pyro.sample("x", dist.Normal(z, 1), obs=data)


def guide_0(data, params):
    probs = pyro.param("probs", params["probs"], constraint=constraints.simplex)
    with pyro.plate("data", len(data)):
        pyro.sample("z", dist.Categorical(probs))


params_0 = {"probs": jnp.array([[0.4, 0.6], [0.5, 0.5]])}


def model_1(data, params):
    a = pyro.sample("a", dist.Categorical(jnp.array([0.3, 0.7])))
    with pyro.plate("data", len(data)):
        probs_b = jnp.array([[0.1, 0.9], [0.2, 0.8]])
        b = pyro.sample("b", dist.Categorical(probs_b[a]))
        pyro.sample("c", dist.Normal(b, 1), obs=data)


def guide_1(data, params):
    probs_a = pyro.param("probs_a", params["probs_a"], constraint=constraints.simplex)
    probs_b = pyro.param("probs_b", params["probs_b"], constraint=constraints.simplex)
    a = pyro.sample("a", dist.Categorical(probs_a))
    with pyro.plate("data", len(data)) as idx:
        pyro.sample("b", dist.Categorical(Vindex(probs_b)[a, idx]))


params_1 = {
    "probs_a": jnp.array([0.5, 0.5]),
    "probs_b": jnp.array([[[0.5, 0.5], [0.6, 0.4]], [[0.4, 0.6], [0.35, 0.65]]]),
}


def model_2(data, params):
    prob_b = jnp.array([[0.3, 0.7], [0.4, 0.6]])
    prob_c = jnp.array([[0.5, 0.5], [0.6, 0.4]])
    prob_d = jnp.array([[0.2, 0.8], [0.3, 0.7]])
    prob_e = jnp.array([[0.5, 0.5], [0.1, 0.9]])
    a = pyro.sample("a", dist.Categorical(jnp.array([0.3, 0.7])))
    with pyro.plate("data", len(data)):
        b = pyro.sample("b", dist.Categorical(prob_b[a]))
        c = pyro.sample("c", dist.Categorical(prob_c[b]))
        pyro.sample("d", dist.Categorical(prob_d[b]))
        pyro.sample("e", dist.Categorical(prob_e[c]), obs=data)


def guide_2(data, params):
    probs_a = pyro.param("probs_a", params["probs_a"], constraint=constraints.simplex)
    probs_b = pyro.param("probs_b", params["probs_b"], constraint=constraints.simplex)
    probs_c = pyro.param("probs_c", params["probs_c"], constraint=constraints.simplex)
    probs_d = pyro.param("probs_d", params["probs_d"], constraint=constraints.simplex)
    a = pyro.sample("a", dist.Categorical(probs_a))
    with pyro.plate("data", len(data)) as idx:
        b = pyro.sample("b", dist.Categorical(probs_b[a]))
        pyro.sample("c", dist.Categorical(Vindex(probs_c)[b, idx]))
        pyro.sample("d", dist.Categorical(Vindex(probs_d)[b, idx]))


params_2 = {
    "probs_a": jnp.array([0.5, 0.5]),
    "probs_b": jnp.array([[0.4, 0.6], [0.3, 0.7]]),
    "probs_c": jnp.array([[[0.3, 0.7], [0.8, 0.2]], [[0.2, 0.8], [0.5, 0.5]]]),
    "probs_d": jnp.array([[[0.2, 0.8], [0.9, 0.1]], [[0.1, 0.9], [0.4, 0.6]]]),
}


@pytest.mark.parametrize(
    "model,guide,params,data",
    [
        (model_0, guide_0, params_0, jnp.array([-0.5, 2.0])),
        (model_1, guide_1, params_1, jnp.array([-0.5, 2.0])),
        (model_2, guide_2, params_2, jnp.array([0, 1])),
    ],
)
def test_gradient(model, guide, params, data):
    transform = dist.biject_to(dist.constraints.simplex)
    params_raw = jax.tree_util.tree_map(transform.inv, params)

    # Expected grads based on exact integration
    elbo = infer.TraceEnum_ELBO()

    def expected_loss_fn(params_raw):
        params = jax.tree_util.tree_map(transform, params_raw)
        return elbo.loss(
            random.PRNGKey(0), {}, model, config_enumerate(guide), data, params
        )

    expected_loss, expected_grads = jax.value_and_grad(expected_loss_fn)(params_raw)

    # Actual grads averaged over num_particles
    elbo = infer.TraceGraph_ELBO(num_particles=10_000)

    def actual_loss_fn(params_raw):
        params = jax.tree_util.tree_map(transform, params_raw)
        return elbo.loss(random.PRNGKey(0), {}, model, guide, data, params)

    actual_loss, actual_grads = jax.value_and_grad(actual_loss_fn)(params_raw)

    for name in sorted(params):
        logger.info("expected {} = {}".format(name, expected_grads[name]))
        logger.info("actual   {} = {}".format(name, actual_grads[name]))

    assert_equal(actual_grads, expected_grads, prec=0.02)


def kl_model_0_z1z2z3(params):
    # Model
    # z1
    #
    # z2
    #
    # z3
    probs_z1 = jnp.array([0.5, 0.5])
    probs_z2 = jnp.array([0.7, 0.3])
    probs_z3 = jnp.array([0.8, 0.2])
    pyro.sample("z1", dist.Categorical(probs_z1))
    pyro.sample("z2", dist.Categorical(probs_z2))
    pyro.sample("z3", dist.Categorical(probs_z3))


def kl_model_1_z1z2z3(params):
    # Model
    # z1 --+
    # |    |
    # V    |
    # z2   |
    # |    |
    # V    |
    # z3 <-+
    probs_z1 = jnp.array([0.5, 0.5])
    probs_z2 = jnp.array([[0.4, 0.6], [0.6, 0.4]])
    probs_z3 = jnp.array([[[0.4, 0.6], [0.1, 0.9]], [[0.7, 0.3], [0.2, 0.8]]])
    z1 = pyro.sample("z1", dist.Categorical(probs_z1))
    z2 = pyro.sample("z2", dist.Categorical(probs_z2[z1]))
    pyro.sample("z3", dist.Categorical(probs_z3[z1, z2]))


def kl_model_2_z2z3(params):
    # Model (inverted the order of z1 and z2)
    # z2 --+
    # |    |
    # V    |
    # z1   |
    # |    |
    # V    |
    # z3 <-+
    probs_z2 = jnp.array([0.5, 0.5])
    probs_z1 = jnp.array([[0.4, 0.6], [0.6, 0.4]])
    probs_z3 = jnp.array([[[0.4, 0.6], [0.1, 0.9]], [[0.7, 0.3], [0.2, 0.8]]])
    z2 = pyro.sample("z2", dist.Categorical(probs_z2))
    z1 = pyro.sample("z1", dist.Categorical(probs_z1[z2]))
    pyro.sample("z3", dist.Categorical(probs_z3[z1, z2]))


@config_enumerate
def kl_model_3_z2z3(params):
    # Model
    # d
    # |
    # V
    # z1
    #
    # z2
    #
    # z3
    probs_d = jnp.array([0.5, 0.5])
    probs_z1 = jnp.array([[0.4, 0.6], [0.6, 0.4]])
    probs_z2 = jnp.array([0.7, 0.3])
    probs_z3 = jnp.array([0.8, 0.2])
    d = pyro.sample("d", dist.Categorical(probs_d))
    pyro.sample("z1", dist.Categorical(probs_z1[d]))
    pyro.sample("z2", dist.Categorical(probs_z2))
    pyro.sample("z3", dist.Categorical(probs_z3))


@config_enumerate
def kl_model_4_z3(params):
    # Model
    # d ---+
    #      |
    # z1 <-|
    #      |
    # z2 <-+
    #
    # z3
    probs_d = jnp.array([0.5, 0.5])
    probs_z1 = jnp.array([[0.5, 0.5], [0.2, 0.8]])
    probs_z2 = jnp.array([[0.4, 0.6], [0.6, 0.4]])
    probs_z3 = jnp.array([0.8, 0.2])
    d = pyro.sample("d", dist.Categorical(probs_d))
    pyro.sample("z1", dist.Categorical(probs_z1[d]))
    pyro.sample("z2", dist.Categorical(probs_z2[d]))
    pyro.sample("z3", dist.Categorical(probs_z3))


@config_enumerate
def kl_model_5_z2(params):
    # Model
    # d ---+
    #      |
    # z1 <-|
    #      |
    # z2   |
    #      |
    # z3 <-+
    probs_d = jnp.array([0.5, 0.5])
    probs_z1 = jnp.array([[0.5, 0.5], [0.2, 0.8]])
    probs_z2 = jnp.array([0.8, 0.2])
    probs_z3 = jnp.array([[0.4, 0.6], [0.6, 0.4]])
    d = pyro.sample("d", dist.Categorical(probs_d))
    pyro.sample("z1", dist.Categorical(probs_z1[d]))
    pyro.sample("z2", dist.Categorical(probs_z2))
    pyro.sample("z3", dist.Categorical(probs_z3[d]))


@config_enumerate
def kl_model_6_(params):
    # Model
    # d ---+
    #      |
    # z1 <-|
    #      |
    # z2 <-|
    #      |
    # z3 <-+
    probs_d = jnp.array([0.5, 0.5])
    probs_z1 = jnp.array([[0.5, 0.5], [0.2, 0.8]])
    probs_z2 = jnp.array([[0.1, 0.9], [0.3, 0.7]])
    probs_z3 = jnp.array([[0.4, 0.6], [0.6, 0.4]])
    d = pyro.sample("d", dist.Categorical(probs_d))
    pyro.sample("z1", dist.Categorical(probs_z1[d]))
    pyro.sample("z2", dist.Categorical(probs_z2[d]))
    pyro.sample("z3", dist.Categorical(probs_z3[d]))


@config_enumerate
def kl_model_7_z3(params):
    # Model
    # d ---+
    #      |
    # z1 <-|
    #      |
    # z2 <-+
    # |
    # V
    # z3
    probs_d = jnp.array([0.5, 0.5])
    probs_z1 = jnp.array([[0.5, 0.5], [0.2, 0.8]])
    probs_z2 = jnp.array([[0.4, 0.6], [0.6, 0.4]])
    probs_z3 = jnp.array([[0.8, 0.2], [0.6, 0.4]])
    d = pyro.sample("d", dist.Categorical(probs_d))
    pyro.sample("z1", dist.Categorical(probs_z1[d]))
    z2 = pyro.sample("z2", dist.Categorical(probs_z2[d]))
    pyro.sample("z3", dist.Categorical(probs_z3[z2]))


@pytest.mark.parametrize(
    "model,kl_sites,valid_kl",
    [
        (kl_model_0_z1z2z3, set(["z1", "z2", "z3"]), True),
        (kl_model_1_z1z2z3, set(["z1", "z2", "z3"]), True),
        (kl_model_2_z2z3, set(["z1", "z2", "z3"]), False),
        (kl_model_2_z2z3, set(["z2", "z3"]), True),
        (kl_model_3_z2z3, set(["z1", "z2", "z3"]), False),
        (kl_model_3_z2z3, set(["z2", "z3"]), True),
        (kl_model_4_z3, set(["z1", "z2", "z3"]), False),
        (kl_model_4_z3, set(["z3"]), True),
        (kl_model_5_z2, set(["z1", "z2", "z3"]), False),
        (kl_model_5_z2, set(["z2"]), True),
        (kl_model_6_, set(["z1", "z2", "z3"]), False),
        (kl_model_6_, set(), True),
        (kl_model_7_z3, set(["z1", "z2", "z3"]), False),
        (kl_model_7_z3, set(["z3"]), True),
    ][-2:],
)
def test_analytic_kl(model, kl_sites, valid_kl):
    # Guide
    # z1 --+
    # |    |
    # v    |
    # z2   |
    # |    |
    # v    |
    # z3 <-+
    @config_enumerate
    def guide(params):
        probs_z1 = pyro.param(
            "probs_z1", params["probs_z1"], constraint=constraints.simplex
        )
        probs_z2 = pyro.param(
            "probs_z2", params["probs_z2"], constraint=constraints.simplex
        )
        probs_z3 = pyro.param(
            "probs_z3", params["probs_z3"], constraint=constraints.simplex
        )
        z1 = pyro.sample("z1", dist.Categorical(probs_z1))
        z2 = pyro.sample("z2", dist.Categorical(probs_z2[z1]))
        pyro.sample("z3", dist.Categorical(probs_z3[z1, z2]))

    params = {
        "probs_z1": jnp.array([0.3, 0.7]),
        "probs_z2": jnp.array([[0.4, 0.6], [0.5, 0.5]]),
        "probs_z3": jnp.array([[[0.4, 0.6], [0.5, 0.5]], [[0.7, 0.3], [0.9, 0.1]]]),
    }
    transform = dist.biject_to(dist.constraints.simplex)
    params_raw = jax.tree_util.tree_map(transform.inv, params)

    elbo = infer.TraceEnum_ELBO()

    # Expected grads averaged over num_particles
    def expected_loss_fn(params_raw):
        params = jax.tree_util.tree_map(transform, params_raw)
        return elbo.loss(random.PRNGKey(0), {}, model, guide, params)

    expected_loss, expected_grads = jax.value_and_grad(expected_loss_fn)(params_raw)

    # Grads based on exact integration
    def actual_loss_fn(params_raw):
        params = jax.tree_util.tree_map(transform, params_raw)
        return elbo.loss(
            random.PRNGKey(0), {}, model, config_kl(guide, kl_sites), params
        )

    if valid_kl:
        actual_loss, actual_grads = jax.value_and_grad(actual_loss_fn)(params_raw)

        assert_equal(actual_loss, expected_loss, prec=1e-5)
        assert_equal(actual_grads, expected_grads, prec=1e-5)
    else:
        with pytest.raises(
            AssertionError, match="Expected that for use of analytic KL solution"
        ):
            actual_loss, actual_grads = jax.value_and_grad(actual_loss_fn)(params_raw)

            assert_equal(actual_loss, expected_loss, prec=1e-5)
            assert_equal(actual_grads, expected_grads, prec=1e-5)
