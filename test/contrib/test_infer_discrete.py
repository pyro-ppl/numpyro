# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np
from numpy.testing import assert_allclose
import pytest

from jax import random
import jax.numpy as jnp

import numpyro
from numpyro import handlers, infer
from numpyro.contrib.control_flow import scan
import numpyro.distributions as dist
from numpyro.distributions.util import is_identically_one

# put all funsor-related imports here, so test collection works without funsor
try:
    import funsor
    from numpyro.contrib.funsor import config_enumerate, infer_discrete, markov

    funsor.set_backend("jax")
except ImportError:
    pytestmark = pytest.mark.skip(reason="funsor is not installed")

logger = logging.getLogger(__name__)


def log_prob_sum(trace):
    log_joint = jnp.zeros(())
    for site in trace.values():
        if site["type"] == "sample":
            value = site["value"]
            intermediates = site["intermediates"]
            scale = site["scale"]
            if intermediates:
                log_prob = site["fn"].log_prob(value, intermediates)
            else:
                log_prob = site["fn"].log_prob(value)

            if (scale is not None) and (not is_identically_one(scale)):
                log_prob = scale * log_prob

            log_prob = jnp.sum(log_prob)
            log_joint = log_joint + log_prob
    return log_joint


@pytest.mark.parametrize("length", [1, 2, 10])
@pytest.mark.parametrize("temperature", [0, 1])
def test_hmm_smoke(length, temperature):

    # This should match the example in the infer_discrete docstring.
    def hmm(data, hidden_dim=10):
        transition = 0.3 / hidden_dim + 0.7 * jnp.eye(hidden_dim)
        means = jnp.arange(float(hidden_dim))
        states = [0]
        for t in markov(range(len(data))):
            states.append(
                numpyro.sample(
                    "states_{}".format(t), dist.Categorical(transition[states[-1]])
                )
            )
            data[t] = numpyro.sample(
                "obs_{}".format(t), dist.Normal(means[states[-1]], 1.0), obs=data[t]
            )
        return states, data

    true_states, data = handlers.seed(hmm, 0)([None] * length)
    assert len(data) == length
    assert len(true_states) == 1 + len(data)

    decoder = infer_discrete(
        config_enumerate(hmm), temperature=temperature, rng_key=random.PRNGKey(1)
    )
    inferred_states, _ = decoder(data)
    assert len(inferred_states) == len(true_states)

    logger.info("true states: {}".format(list(map(int, true_states))))
    logger.info("inferred states: {}".format(list(map(int, inferred_states))))


@pytest.mark.parametrize("length", [1, 2, 10])
@pytest.mark.parametrize("temperature", [0, 1])
def test_scan_hmm_smoke(length, temperature):

    # This should match the example in the infer_discrete docstring.
    def hmm(data, hidden_dim=10):
        transition = 0.3 / hidden_dim + 0.7 * jnp.eye(hidden_dim)
        means = jnp.arange(float(hidden_dim))

        def transition_fn(state, y):
            state = numpyro.sample("states", dist.Categorical(transition[state]))
            y = numpyro.sample("obs", dist.Normal(means[state], 1.0), obs=y)
            return state, (state, y)

        _, (states, data) = scan(transition_fn, 0, data, length=length)

        return [0] + [s for s in states], data

    true_states, data = handlers.seed(hmm, 0)(None)
    assert len(data) == length
    assert len(true_states) == 1 + len(data)

    decoder = infer_discrete(
        config_enumerate(hmm), temperature=temperature, rng_key=random.PRNGKey(1)
    )
    inferred_states, _ = decoder(data)
    assert len(inferred_states) == len(true_states)

    logger.info("true states: {}".format(list(map(int, true_states))))
    logger.info("inferred states: {}".format(list(map(int, inferred_states))))


def vectorize_model(model, size, dim):
    def fn(*args, **kwargs):
        with numpyro.plate("particles", size=size, dim=dim):
            return model(*args, **kwargs)

    return fn


@pytest.mark.parametrize("temperature", [0, 1])
def test_distribution_1(temperature):
    #      +-------+
    #  z --|--> x  |
    #      +-------+
    num_particles = 10000
    data = np.array([1.0, 2.0, 3.0])

    @config_enumerate
    def model(z=None):
        p = numpyro.param("p", np.array([0.75, 0.25]))
        iz = numpyro.sample("z", dist.Categorical(p), obs=z)
        z = jnp.array([0.0, 1.0])[iz]
        logger.info("z.shape = {}".format(z.shape))
        with numpyro.plate("data", 3):
            numpyro.sample("x", dist.Normal(z, 1.0), obs=data)

    first_available_dim = -3
    vectorized_model = (
        model if temperature == 0 else vectorize_model(model, num_particles, dim=-2)
    )
    sampled_model = infer_discrete(
        vectorized_model, first_available_dim, temperature, rng_key=random.PRNGKey(1)
    )
    sampled_trace = handlers.trace(sampled_model).get_trace()
    conditioned_traces = {
        z: handlers.trace(model).get_trace(z=np.array(z)) for z in [0, 1]
    }

    # Check  posterior over z.
    actual_z_mean = sampled_trace["z"]["value"].astype(float).mean()
    if temperature:
        expected_z_mean = 1 / (
            1
            + np.exp(
                log_prob_sum(conditioned_traces[0])
                - log_prob_sum(conditioned_traces[1])
            )
        )
    else:
        expected_z_mean = (
            log_prob_sum(conditioned_traces[1]) > log_prob_sum(conditioned_traces[0])
        ).astype(float)
        expected_max = max(log_prob_sum(t) for t in conditioned_traces.values())
        actual_max = log_prob_sum(sampled_trace)
        assert_allclose(expected_max, actual_max, atol=1e-5)
    assert_allclose(actual_z_mean, expected_z_mean, atol=1e-2 if temperature else 1e-5)


@pytest.mark.parametrize("temperature", [0, 1])
def test_distribution_2(temperature):
    #       +--------+
    #  z1 --|--> x1  |
    #   |   |        |
    #   V   |        |
    #  z2 --|--> x2  |
    #       +--------+
    num_particles = 10000
    data = np.array([[-1.0, -1.0, 0.0], [-1.0, 1.0, 1.0]])

    @config_enumerate
    def model(z1=None, z2=None):
        p = numpyro.param("p", jnp.array([[0.25, 0.75], [0.1, 0.9]]))
        loc = numpyro.param("loc", jnp.array([-1.0, 1.0]))
        z1 = numpyro.sample("z1", dist.Categorical(p[0]), obs=z1)
        z2 = numpyro.sample("z2", dist.Categorical(p[z1]), obs=z2)
        logger.info("z1.shape = {}".format(z1.shape))
        logger.info("z2.shape = {}".format(z2.shape))
        with numpyro.plate("data", 3):
            numpyro.sample("x1", dist.Normal(loc[z1], 1.0), obs=data[0])
            numpyro.sample("x2", dist.Normal(loc[z2], 1.0), obs=data[1])

    first_available_dim = -3
    vectorized_model = (
        model if temperature == 0 else vectorize_model(model, num_particles, dim=-2)
    )
    sampled_model = infer_discrete(
        vectorized_model, first_available_dim, temperature, rng_key=random.PRNGKey(1)
    )
    sampled_trace = handlers.trace(sampled_model).get_trace()
    conditioned_traces = {
        (z1, z2): handlers.trace(model).get_trace(z1=np.array(z1), z2=np.array(z2))
        for z1 in [0, 1]
        for z2 in [0, 1]
    }

    # Check joint posterior over (z1, z2).
    actual_probs = np.zeros((2, 2))
    expected_probs = np.zeros((2, 2))
    for (z1, z2), tr in conditioned_traces.items():
        expected_probs[z1, z2] = np.exp(log_prob_sum(tr))
        actual_probs[z1, z2] = (
            (
                (sampled_trace["z1"]["value"] == z1)
                & (sampled_trace["z2"]["value"] == z2)
            )
            .astype(float)
            .mean()
        )

    if temperature:
        expected_probs = expected_probs / expected_probs.sum()
    else:
        argmax = np.argmax(expected_probs.reshape(-1))
        expected_max = expected_probs.reshape(-1)[argmax]
        actual_max = log_prob_sum(sampled_trace)
        assert_allclose(np.log(expected_max), actual_max, atol=1e-5)
        expected_probs[:] = 0
        expected_probs.reshape(-1)[argmax] = 1
    assert_allclose(expected_probs, actual_probs, atol=1e-2 if temperature else 1e-5)


@pytest.mark.parametrize("temperature", [0, 1])
def test_distribution_3_simple(temperature):
    #  +---------------+
    #  |  z2 ---> x2   |
    #  |             2 |
    #  +---------------+
    num_particles = 10000
    data = np.array([-1.0, 1.0])

    @config_enumerate
    def model(z2=None):
        p = numpyro.param("p", np.array([0.25, 0.75]))
        loc = numpyro.param("loc", jnp.array([-1.0, 1.0]))
        with numpyro.plate("data", 2):
            z2 = numpyro.sample("z2", dist.Categorical(p), obs=z2)
            numpyro.sample("x2", dist.Normal(loc[z2], 1.0), obs=data)

    first_available_dim = -3
    vectorized_model = (
        model if temperature == 0 else vectorize_model(model, num_particles, dim=-2)
    )
    sampled_model = infer_discrete(
        vectorized_model, first_available_dim, temperature, random.PRNGKey(1)
    )
    sampled_trace = handlers.trace(sampled_model).get_trace()
    conditioned_traces = {
        (z20, z21): handlers.trace(model).get_trace(z2=np.array([z20, z21]))
        for z20 in [0, 1]
        for z21 in [0, 1]
    }

    # Check joint posterior over (z2[0], z2[1]).
    actual_probs = np.zeros((2, 2))
    expected_probs = np.zeros((2, 2))
    for (z20, z21), tr in conditioned_traces.items():
        expected_probs[z20, z21] = np.exp(log_prob_sum(tr))
        actual_probs[z20, z21] = (
            (
                (sampled_trace["z2"]["value"][..., :1] == z20)
                & (sampled_trace["z2"]["value"][..., 1:] == z21)
            )
            .astype(float)
            .mean()
        )
    if temperature:
        expected_probs = expected_probs / expected_probs.sum()
    else:
        argmax = np.argmax(expected_probs.reshape(-1))
        expected_max = expected_probs.reshape(-1)[argmax]
        actual_max = log_prob_sum(sampled_trace)
        assert_allclose(np.log(expected_max), actual_max, atol=1e-5)
        expected_probs[:] = 0
        expected_probs.reshape(-1)[argmax] = 1
    assert_allclose(expected_probs.reshape(-1), actual_probs.reshape(-1), atol=1e-2)


@pytest.mark.parametrize("temperature", [0, 1])
def test_distribution_3(temperature):
    #       +---------+  +---------------+
    #  z1 --|--> x1   |  |  z2 ---> x2   |
    #       |       3 |  |             2 |
    #       +---------+  +---------------+
    num_particles = 10000
    data = [np.array([-1.0, -1.0, 0.0]), np.array([-1.0, 1.0])]

    @config_enumerate
    def model(z1=None, z2=None):
        p = numpyro.param("p", np.array([0.25, 0.75]))
        loc = numpyro.param("loc", jnp.array([-1.0, 1.0]))
        z1 = numpyro.sample("z1", dist.Categorical(p), obs=z1)
        with numpyro.plate("data[0]", 3):
            numpyro.sample("x1", dist.Normal(loc[z1], 1.0), obs=data[0])
        with numpyro.plate("data[1]", 2):
            z2 = numpyro.sample("z2", dist.Categorical(p), obs=z2)
            numpyro.sample("x2", dist.Normal(loc[z2], 1.0), obs=data[1])

    first_available_dim = -3
    vectorized_model = (
        model if temperature == 0 else vectorize_model(model, num_particles, dim=-2)
    )
    sampled_model = infer_discrete(
        vectorized_model, first_available_dim, temperature, rng_key=random.PRNGKey(1)
    )
    sampled_trace = handlers.trace(sampled_model).get_trace()
    conditioned_traces = {
        (z1, z20, z21): handlers.trace(model).get_trace(
            z1=np.array(z1), z2=np.array([z20, z21])
        )
        for z1 in [0, 1]
        for z20 in [0, 1]
        for z21 in [0, 1]
    }

    # Check joint posterior over (z1, z2[0], z2[1]).
    actual_probs = np.zeros((2, 2, 2))
    expected_probs = np.zeros((2, 2, 2))
    for (z1, z20, z21), tr in conditioned_traces.items():
        expected_probs[z1, z20, z21] = jnp.exp(log_prob_sum(tr))
        actual_probs[z1, z20, z21] = (
            (
                (sampled_trace["z1"]["value"] == z1)
                & (sampled_trace["z2"]["value"][..., :1] == z20)
                & (sampled_trace["z2"]["value"][..., 1:] == z21)
            )
            .astype(float)
            .mean()
        )
    if temperature:
        expected_probs = expected_probs / expected_probs.sum()
    else:
        argmax = expected_probs.reshape(-1).argmax()
        expected_max = expected_probs.reshape(-1)[argmax]
        actual_max = np.exp(log_prob_sum(sampled_trace))
        assert_allclose(expected_max, actual_max, atol=1e-5)
        expected_probs[:] = 0
        expected_probs.reshape(-1)[argmax] = 1
    assert_allclose(expected_probs.reshape(-1), actual_probs.reshape(-1), atol=1e-2)


def model_zzxx():
    #                  loc,scale
    #                 /         \
    #       +-------/-+  +--------\------+
    #  z1 --|--> x1   |  |  z2 ---> x2   |
    #       |       3 |  |             2 |
    #       +---------+  +---------------+
    data = [np.array([-1.0, -1.0, 0.0]), np.array([-1.0, 1.0])]
    p = numpyro.param("p", np.array([0.25, 0.75]))
    loc = numpyro.sample("loc", dist.Normal(0, 1).expand([2]).to_event(1))
    # FIXME results in infinite loop in transformeddist_to_funsor.
    # scale = numpyro.sample("scale", dist.LogNormal(0, 1))
    scale = jnp.exp(numpyro.sample("scale", dist.Normal(0, 1)))
    z1 = numpyro.sample("z1", dist.Categorical(p))
    with numpyro.plate("data[0]", 3):
        numpyro.sample("x1", dist.Normal(loc[z1], scale), obs=data[0])
    with numpyro.plate("data[1]", 2):
        z2 = numpyro.sample("z2", dist.Categorical(p))
        numpyro.sample("x2", dist.Normal(loc[z2], scale), obs=data[1])


def model2():

    data = [np.array([-1.0, -1.0, 0.0]), np.array([-1.0, 1.0])]
    p = numpyro.param("p", np.array([0.25, 0.75]))
    loc = numpyro.sample("loc", dist.Normal(0, 1).expand([2]).to_event(1))
    # FIXME results in infinite loop in transformeddist_to_funsor.
    # scale = numpyro.sample("scale", dist.LogNormal(0, 1))
    z1 = numpyro.sample("z1", dist.Categorical(p))
    scale = numpyro.sample("scale", dist.LogNormal(jnp.array([0.0, 1.0])[z1], 1))
    with numpyro.plate("data[0]", 3):
        numpyro.sample("x1", dist.Normal(loc[z1], scale), obs=data[0])
    with numpyro.plate("data[1]", 2):
        z2 = numpyro.sample("z2", dist.Categorical(p))
        numpyro.sample("x2", dist.Normal(loc[z2], scale), obs=data[1])


@pytest.mark.parametrize("model", [model_zzxx, model2])
@pytest.mark.parametrize("temperature", [0, 1])
def test_mcmc_model_side_enumeration(model, temperature):
    mcmc = infer.MCMC(infer.NUTS(model), 0, 1)
    mcmc.run(random.PRNGKey(0))
    mcmc_data = {
        k: v[0] for k, v in mcmc.get_samples().items() if k in ["loc", "scale"]
    }

    # MAP estimate discretes, conditioned on posterior sampled continous latents.
    model = handlers.seed(model, rng_seed=1)
    actual_trace = handlers.trace(
        infer_discrete(
            # TODO support replayed sites in infer_discrete.
            # handlers.replay(config_enumerate(model), mcmc_trace),
            handlers.condition(config_enumerate(model), mcmc_data),
            temperature=temperature,
            rng_key=random.PRNGKey(1),
        )
    ).get_trace()

    # Check site names and shapes.
    expected_trace = handlers.trace(model).get_trace()
    assert set(actual_trace) == set(expected_trace)


@pytest.mark.parametrize("temperature", [0, 1])
def test_distribution_masked(temperature):
    #      +-------+
    #  z --|--> x  |
    #      +-------+
    num_particles = 10000
    data = np.array([1.0, 2.0, 3.0])
    mask = np.array([True, False, False])

    @config_enumerate
    def model(z=None):
        p = numpyro.param("p", np.array([0.75, 0.25]))
        z = numpyro.sample("z", dist.Categorical(p), obs=z)
        logger.info("z.shape = {}".format(z.shape))
        with numpyro.plate("data", 3), handlers.mask(mask=mask):
            numpyro.sample("x", dist.Normal(z, 1.0), obs=data)

    first_available_dim = -3
    vectorized_model = (
        model if temperature == 0 else vectorize_model(model, num_particles, dim=-2)
    )
    sampled_model = infer_discrete(
        vectorized_model, first_available_dim, temperature, rng_key=random.PRNGKey(1)
    )
    sampled_trace = handlers.trace(sampled_model).get_trace()
    conditioned_traces = {
        z: handlers.trace(model).get_trace(z=np.array(z)) for z in [0.0, 1.0]
    }

    # Check  posterior over z.
    actual_z_mean = sampled_trace["z"]["value"].astype(float).mean()
    if temperature:
        expected_z_mean = 1 / (
            1
            + jnp.exp(
                log_prob_sum(conditioned_traces[0])
                - log_prob_sum(conditioned_traces[1])
            )
        )
    else:
        expected_z_mean = (
            log_prob_sum(conditioned_traces[1]) > log_prob_sum(conditioned_traces[0])
        ).astype(float)
    assert_allclose(actual_z_mean, expected_z_mean, atol=1e-2)
