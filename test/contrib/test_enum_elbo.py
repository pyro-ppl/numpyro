# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np
import pytest

import jax
from jax import random
import jax.numpy as jnp

import numpyro as pyro
from numpyro import handlers, infer
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.ops.indexing import Vindex

# put all funsor-related imports here, so test collection works without funsor
try:
    import funsor
    import numpyro.contrib.funsor
    from numpyro.contrib.funsor import config_enumerate

    funsor.set_backend("jax")
except ImportError:
    pytestmark = pytest.mark.skip(reason="funsor is not installed")

logger = logging.getLogger(__name__)

transform = dist.biject_to(dist.constraints.simplex)


def assert_equal(a, b, prec=0):
    return jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, atol=prec), a, b)


def xfail_param(*args, **kwargs):
    kwargs.setdefault("reason", "unknown")
    return pytest.param(*args, marks=[pytest.mark.xfail(**kwargs)])


@pytest.mark.parametrize("inner_dim", [2])
@pytest.mark.parametrize("outer_dim", [2])
def test_elbo_plate_plate(outer_dim, inner_dim):
    q = jnp.array([0.75, 0.25])
    p = 0.2693204236205713  # for which kl(Categorical(q), Categorical(p)) = 0.5
    p = jnp.array([p, 1 - p])

    def model(q):
        d = dist.Categorical(p)
        context1 = pyro.plate("outer", outer_dim, dim=-1)
        context2 = pyro.plate("inner", inner_dim, dim=-2)
        pyro.sample("w", d)
        with context1:
            pyro.sample("x", d)
        with context2:
            pyro.sample("y", d)
        with context1, context2:
            pyro.sample("z", d)

    def guide(q):
        d = dist.Categorical(q)
        context1 = pyro.plate("outer", outer_dim, dim=-1)
        context2 = pyro.plate("inner", inner_dim, dim=-2)
        pyro.sample("w", d, infer={"enumerate": "parallel"})
        with context1:
            pyro.sample("x", d, infer={"enumerate": "parallel"})
        with context2:
            pyro.sample("y", d, infer={"enumerate": "parallel"})
        with context1, context2:
            pyro.sample("z", d, infer={"enumerate": "parallel"})

    def expected_loss_fn(q):
        kl_node = pyro.distributions.kl_divergence(
            dist.Categorical(q), dist.Categorical(p)
        )
        kl = (1 + outer_dim + inner_dim + outer_dim * inner_dim) * kl_node
        return kl

    expected_loss, expected_grad = jax.value_and_grad(expected_loss_fn)(q)

    def actual_loss_fn(q):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, model, guide, q)

    actual_loss, actual_grad = jax.value_and_grad(actual_loss_fn)(q)

    assert_equal(actual_loss, expected_loss, prec=1e-5)
    assert_equal(actual_grad, expected_grad, prec=1e-5)


@pytest.mark.parametrize("scale", [1, 10])
def test_elbo_enumerate_1(scale):
    params = {}
    params["guide_probs_x"] = jnp.array([0.1, 0.9])
    params["model_probs_x"] = jnp.array([0.4, 0.6])
    params["model_probs_y"] = jnp.array([[0.75, 0.25], [0.55, 0.45]])
    params["model_probs_z"] = jnp.array([0.3, 0.7])

    @handlers.scale(scale=scale)
    def auto_model(params):
        probs_x = params["model_probs_x"]
        probs_y = params["model_probs_y"]
        probs_z = params["model_probs_z"]
        x = pyro.sample("x", dist.Categorical(probs_x))
        pyro.sample("y", dist.Categorical(probs_y[x]), infer={"enumerate": "parallel"})
        pyro.sample("z", dist.Categorical(probs_z), obs=jnp.array(0))

    @handlers.scale(scale=scale)
    def hand_model(params):
        probs_x = params["model_probs_x"]
        probs_z = params["model_probs_z"]
        pyro.sample("x", dist.Categorical(probs_x))
        pyro.sample("z", dist.Categorical(probs_z), obs=jnp.array(0))

    @handlers.scale(scale=scale)
    def guide(params):
        probs_x = params["guide_probs_x"]
        pyro.sample("x", dist.Categorical(probs_x), infer={"enumerate": "parallel"})

    def auto_loss_fn(params_raw):
        params = jax.tree.map(transform, params_raw)
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, auto_model, guide, params)

    def hand_loss_fn(params_raw):
        params = jax.tree.map(transform, params_raw)
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, hand_model, guide, params)

    params_raw = jax.tree.map(transform.inv, params)
    auto_loss, auto_grad = jax.value_and_grad(auto_loss_fn)(params_raw)
    hand_loss, hand_grad = jax.value_and_grad(hand_loss_fn)(params_raw)

    assert_equal(auto_loss, hand_loss, prec=1e-5)
    assert_equal(auto_grad, hand_grad, prec=1e-5)


@pytest.mark.parametrize("scale", [1, 10])
def test_elbo_enumerate_2(scale):
    params = {}
    params["guide_probs_x"] = jnp.array([0.1, 0.9])
    params["model_probs_x"] = jnp.array([0.4, 0.6])
    params["model_probs_y"] = jnp.array([[0.75, 0.25], [0.55, 0.45]])
    params["model_probs_z"] = jnp.array([[0.3, 0.7], [0.2, 0.8]])

    @handlers.scale(scale=scale)
    def auto_model(params):
        probs_x = params["model_probs_x"]
        probs_y = params["model_probs_y"]
        probs_z = params["model_probs_z"]
        x = pyro.sample("x", dist.Categorical(probs_x))
        y = pyro.sample(
            "y", dist.Categorical(probs_y[x]), infer={"enumerate": "parallel"}
        )
        pyro.sample("z", dist.Categorical(probs_z[y]), obs=0)

    @handlers.scale(scale=scale)
    def hand_model(params):
        probs_x = params["model_probs_x"]
        probs_y = params["model_probs_y"]
        probs_z = params["model_probs_z"]
        probs_yz = probs_y @ probs_z
        x = pyro.sample("x", dist.Categorical(probs_x))
        pyro.sample("z", dist.Categorical(probs_yz[x]), obs=0)

    @config_enumerate
    @handlers.scale(scale=scale)
    def guide(params):
        probs_x = params["guide_probs_x"]
        pyro.sample("x", dist.Categorical(probs_x))

    def auto_loss_fn(params_raw):
        params = jax.tree.map(transform, params_raw)
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, auto_model, guide, params)

    def hand_loss_fn(params_raw):
        params = jax.tree.map(transform, params_raw)
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, hand_model, guide, params)

    params_raw = jax.tree.map(transform.inv, params)
    auto_loss, auto_grad = jax.value_and_grad(auto_loss_fn)(params_raw)
    hand_loss, hand_grad = jax.value_and_grad(hand_loss_fn)(params_raw)

    assert_equal(auto_loss, hand_loss, prec=1e-5)
    assert_equal(auto_grad, hand_grad, prec=1e-5)


@pytest.mark.parametrize("scale", [1, 10])
def test_elbo_enumerate_3(scale):
    params = {}
    params["guide_probs_x"] = jnp.array([0.1, 0.9])
    params["model_probs_x"] = jnp.array([0.4, 0.6])
    params["model_probs_y"] = jnp.array([[0.75, 0.25], [0.55, 0.45]])
    params["model_probs_z"] = jnp.array([[0.3, 0.7], [0.2, 0.8]])

    def auto_model(params):
        probs_x = params["model_probs_x"]
        probs_y = params["model_probs_y"]
        probs_z = params["model_probs_z"]
        x = pyro.sample("x", dist.Categorical(probs_x))
        with handlers.scale(scale=scale):
            y = pyro.sample(
                "y", dist.Categorical(probs_y[x]), infer={"enumerate": "parallel"}
            )
            pyro.sample("z", dist.Categorical(probs_z[y]), obs=0)

    def hand_model(params):
        probs_x = params["model_probs_x"]
        probs_y = params["model_probs_y"]
        probs_z = params["model_probs_z"]
        probs_yz = probs_y @ probs_z
        x = pyro.sample("x", dist.Categorical(probs_x))
        with handlers.scale(scale=scale):
            pyro.sample("z", dist.Categorical(probs_yz[x]), obs=0)

    @config_enumerate
    def guide(params):
        probs_x = params["guide_probs_x"]
        pyro.sample("x", dist.Categorical(probs_x))

    def auto_loss_fn(params_raw):
        params = jax.tree.map(transform, params_raw)
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, auto_model, guide, params)

    def hand_loss_fn(params_raw):
        params = jax.tree.map(transform, params_raw)
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, hand_model, guide, params)

    params_raw = jax.tree.map(transform.inv, params)
    auto_loss, auto_grad = jax.value_and_grad(auto_loss_fn)(params_raw)
    hand_loss, hand_grad = jax.value_and_grad(hand_loss_fn)(params_raw)

    assert_equal(auto_loss, hand_loss, prec=1e-5)
    assert_equal(auto_grad, hand_grad, prec=1e-5)


@pytest.mark.parametrize("scale", [1, 10])
@pytest.mark.parametrize(
    "num_samples,num_masked", [(2, 2), (3, 2)], ids=["batch", "masked"]
)
def test_elbo_enumerate_plate_1(num_samples, num_masked, scale):
    #              +---------+
    #  x ----> y ----> z     |
    #              |       N |
    #              +---------+
    params = {}
    params["guide_probs_x"] = jnp.array([0.1, 0.9])
    params["model_probs_x"] = jnp.array([0.4, 0.6])
    params["model_probs_y"] = jnp.array([[0.75, 0.25], [0.55, 0.45]])
    params["model_probs_z"] = jnp.array([[0.3, 0.7], [0.2, 0.8]])

    def auto_model(data, params):
        probs_x = params["model_probs_x"]
        probs_y = params["model_probs_y"]
        probs_z = params["model_probs_z"]
        x = pyro.sample("x", dist.Categorical(probs_x))
        with handlers.scale(scale=scale):
            y = pyro.sample(
                "y", dist.Categorical(probs_y[x]), infer={"enumerate": "parallel"}
            )
            if num_masked == num_samples:
                with pyro.plate("data", len(data)):
                    pyro.sample("z", dist.Categorical(probs_z[y]), obs=data)
            else:
                with pyro.plate("data", len(data)):
                    with handlers.mask(mask=jnp.arange(num_samples) < num_masked):
                        pyro.sample("z", dist.Categorical(probs_z[y]), obs=data)

    def hand_model(data, params):
        probs_x = params["model_probs_x"]
        probs_y = params["model_probs_y"]
        probs_z = params["model_probs_z"]
        x = pyro.sample("x", dist.Categorical(probs_x))
        with handlers.scale(scale=scale):
            y = pyro.sample(
                "y", dist.Categorical(probs_y[x]), infer={"enumerate": "parallel"}
            )
            for i in range(num_masked):
                pyro.sample(f"z_{i}", dist.Categorical(probs_z[y]), obs=data[i])

    @config_enumerate
    def guide(data, params):
        probs_x = params["guide_probs_x"]
        pyro.sample("x", dist.Categorical(probs_x))

    data = dist.Categorical(jnp.array([0.3, 0.7])).sample(
        random.PRNGKey(0), (num_samples,)
    )

    def auto_loss_fn(params_raw):
        params = jax.tree.map(transform, params_raw)
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, auto_model, guide, data, params)

    def hand_loss_fn(params_raw):
        params = jax.tree.map(transform, params_raw)
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, hand_model, guide, data, params)

    params_raw = jax.tree.map(transform.inv, params)
    auto_loss, auto_grad = jax.value_and_grad(auto_loss_fn)(params_raw)
    hand_loss, hand_grad = jax.value_and_grad(hand_loss_fn)(params_raw)

    assert_equal(auto_loss, hand_loss, prec=1e-5)
    assert_equal(auto_grad, hand_grad, prec=1e-5)


@pytest.mark.parametrize("scale", [1, 10])
@pytest.mark.parametrize(
    "num_samples,num_masked", [(2, 2), (3, 2)], ids=["batch", "masked"]
)
def test_elbo_enumerate_plate_2(num_samples, num_masked, scale):
    #      +-----------------+
    #  x ----> y ----> z     |
    #      |               N |
    #      +-----------------+
    params = {}
    params["guide_probs_x"] = jnp.array([0.1, 0.9])
    params["model_probs_x"] = jnp.array([0.4, 0.6])
    params["model_probs_y"] = jnp.array([[0.75, 0.25], [0.55, 0.45]])
    params["model_probs_z"] = jnp.array([[0.3, 0.7], [0.2, 0.8]])

    def auto_model(data, params):
        probs_x = params["model_probs_x"]
        probs_y = params["model_probs_y"]
        probs_z = params["model_probs_z"]
        x = pyro.sample("x", dist.Categorical(probs_x))
        with handlers.scale(scale=scale):
            with pyro.plate("data", len(data)):
                if num_masked == num_samples:
                    y = pyro.sample(
                        "y",
                        dist.Categorical(probs_y[x]),
                        infer={"enumerate": "parallel"},
                    )
                    pyro.sample("z", dist.Categorical(probs_z[y]), obs=data)
                else:
                    with handlers.mask(mask=jnp.arange(num_samples) < num_masked):
                        y = pyro.sample(
                            "y",
                            dist.Categorical(probs_y[x]),
                            infer={"enumerate": "parallel"},
                        )
                        pyro.sample("z", dist.Categorical(probs_z[y]), obs=data)

    def hand_model(data, params):
        probs_x = params["model_probs_x"]
        probs_y = params["model_probs_y"]
        probs_z = params["model_probs_z"]
        x = pyro.sample("x", dist.Categorical(probs_x))
        with handlers.scale(scale=scale):
            for i in range(num_masked):
                y = pyro.sample(
                    f"y_{i}",
                    dist.Categorical(probs_y[x]),
                    infer={"enumerate": "parallel"},
                )
                pyro.sample(f"z_{i}", dist.Categorical(probs_z[y]), obs=data[i])

    @config_enumerate
    def guide(data, params):
        probs_x = params["guide_probs_x"]
        pyro.sample("x", dist.Categorical(probs_x))

    data = dist.Categorical(jnp.array([0.3, 0.7])).sample(
        random.PRNGKey(0), (num_samples,)
    )

    def auto_loss_fn(params_raw):
        params = jax.tree.map(transform, params_raw)
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, auto_model, guide, data, params)

    def hand_loss_fn(params_raw):
        params = jax.tree.map(transform, params_raw)
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, hand_model, guide, data, params)

    params_raw = jax.tree.map(transform.inv, params)
    auto_loss, auto_grad = jax.value_and_grad(auto_loss_fn)(params_raw)
    hand_loss, hand_grad = jax.value_and_grad(hand_loss_fn)(params_raw)

    assert_equal(auto_loss, hand_loss, prec=1e-5)
    assert_equal(auto_grad, hand_grad, prec=1e-5)


@pytest.mark.parametrize("scale", [1, 10])
@pytest.mark.parametrize(
    "num_samples,num_masked", [(2, 2), (3, 2)], ids=["batch", "masked"]
)
def test_elbo_enumerate_plate_3(num_samples, num_masked, scale):
    #  +-----------------------+
    #  | x ----> y ----> z     |
    #  |                     N |
    #  +-----------------------+
    # This plate should remain unreduced since all enumeration is in a single plate.
    params = {}
    params["guide_probs_x"] = jnp.array([0.1, 0.9])
    params["model_probs_x"] = jnp.array([0.4, 0.6])
    params["model_probs_y"] = jnp.array([[0.75, 0.25], [0.55, 0.45]])
    params["model_probs_z"] = jnp.array([[0.3, 0.7], [0.2, 0.8]])

    @handlers.scale(scale=scale)
    def auto_model(data, params):
        probs_x = params["model_probs_x"]
        probs_y = params["model_probs_y"]
        probs_z = params["model_probs_z"]
        with pyro.plate("data", len(data)):
            if num_masked == num_samples:
                x = pyro.sample("x", dist.Categorical(probs_x))
                y = pyro.sample(
                    "y", dist.Categorical(probs_y[x]), infer={"enumerate": "parallel"}
                )
                pyro.sample("z", dist.Categorical(probs_z[y]), obs=data)
            else:
                with handlers.mask(mask=jnp.arange(num_samples) < num_masked):
                    x = pyro.sample("x", dist.Categorical(probs_x))
                    y = pyro.sample(
                        "y",
                        dist.Categorical(probs_y[x]),
                        infer={"enumerate": "parallel"},
                    )
                    pyro.sample("z", dist.Categorical(probs_z[y]), obs=data)

    @handlers.scale(scale=scale)
    @config_enumerate
    def auto_guide(data, params):
        probs_x = params["guide_probs_x"]
        with pyro.plate("data", len(data)):
            if num_masked == num_samples:
                pyro.sample("x", dist.Categorical(probs_x))
            else:
                with handlers.mask(mask=jnp.arange(num_samples) < num_masked):
                    pyro.sample("x", dist.Categorical(probs_x))

    @handlers.scale(scale=scale)
    def hand_model(data, params):
        probs_x = params["model_probs_x"]
        probs_y = params["model_probs_y"]
        probs_z = params["model_probs_z"]
        for i in range(num_masked):
            x = pyro.sample(f"x_{i}", dist.Categorical(probs_x))
            y = pyro.sample(
                f"y_{i}",
                dist.Categorical(probs_y[x]),
                infer={"enumerate": "parallel"},
            )
            pyro.sample("z_{}".format(i), dist.Categorical(probs_z[y]), obs=data[i])

    @handlers.scale(scale=scale)
    @config_enumerate
    def hand_guide(data, params):
        probs_x = params["guide_probs_x"]
        for i in range(num_masked):
            pyro.sample(f"x_{i}", dist.Categorical(probs_x))

    data = dist.Categorical(jnp.array([0.3, 0.7])).sample(
        random.PRNGKey(0), (num_samples,)
    )

    def auto_loss_fn(params_raw):
        params = jax.tree.map(transform, params_raw)
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, auto_model, auto_guide, data, params)

    def hand_loss_fn(params_raw):
        params = jax.tree.map(transform, params_raw)
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, hand_model, hand_guide, data, params)

    params_raw = jax.tree.map(transform.inv, params)
    auto_loss, auto_grad = jax.value_and_grad(auto_loss_fn)(params_raw)
    hand_loss, hand_grad = jax.value_and_grad(hand_loss_fn)(params_raw)

    assert_equal(auto_loss, hand_loss, prec=1e-5)
    assert_equal(auto_grad, hand_grad, prec=1e-5)


@pytest.mark.parametrize("scale", [1, 10])
@pytest.mark.parametrize(
    "outer_obs,inner_obs", [(False, True), (True, False), (True, True)]
)
def test_elbo_enumerate_plate_4(outer_obs, inner_obs, scale):
    #    a ---> outer_obs
    #      \
    #  +-----\------------------+
    #  |       \                |
    #  | b ---> inner_obs   N=2 |
    #  +------------------------+
    # This tests two different observations, one outside and one inside an plate.
    params = {}
    params["probs_a"] = jnp.array([0.4, 0.6])
    params["probs_b"] = jnp.array([0.6, 0.4])
    params["locs"] = jnp.array([-1.0, 1.0])
    params["scales"] = jnp.array([1.0, 2.0])

    outer_data = 2.0
    inner_data = jnp.array([0.5, 1.5])

    @handlers.scale(scale=scale)
    def auto_model(params):
        probs_a = pyro.param(
            "probs_a", params["probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["probs_b"], constraint=constraints.simplex
        )
        locs = pyro.param("locs", params["locs"])
        scales = pyro.param("scales", params["scales"], constraint=constraints.positive)
        a = pyro.sample("a", dist.Categorical(probs_a), infer={"enumerate": "parallel"})
        if outer_obs:
            pyro.sample("outer_obs", dist.Normal(0.0, scales[a]), obs=outer_data)
        with pyro.plate("inner", 2):
            b = pyro.sample(
                "b", dist.Categorical(probs_b), infer={"enumerate": "parallel"}
            )
            if inner_obs:
                pyro.sample(
                    "inner_obs", dist.Normal(locs[b], scales[a]), obs=inner_data
                )

    @handlers.scale(scale=scale)
    def hand_model(params):
        probs_a = pyro.param(
            "probs_a", params["probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["probs_b"], constraint=constraints.simplex
        )
        locs = pyro.param("locs", params["locs"])
        scales = pyro.param("scales", params["scales"], constraint=constraints.positive)
        a = pyro.sample("a", dist.Categorical(probs_a), infer={"enumerate": "parallel"})
        if outer_obs:
            pyro.sample("outer_obs", dist.Normal(0.0, scales[a]), obs=outer_data)
        for i in range(2):
            b = pyro.sample(
                f"b_{i}",
                dist.Categorical(probs_b),
                infer={"enumerate": "parallel"},
            )
            if inner_obs:
                pyro.sample(
                    f"inner_obs_{i}",
                    dist.Normal(locs[b], scales[a]),
                    obs=inner_data[i],
                )

    def guide(params):
        pass

    def auto_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, auto_model, guide, params)

    def hand_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, hand_model, guide, params)

    auto_loss, auto_grad = jax.value_and_grad(auto_loss_fn)(params)
    hand_loss, hand_grad = jax.value_and_grad(hand_loss_fn)(params)

    assert_equal(auto_loss, hand_loss, prec=1e-5)
    assert_equal(auto_grad, hand_grad, prec=1e-5)


def test_elbo_enumerate_plate_5():
    #        Guide   Model
    #                  a
    #  +---------------|--+
    #  | M=2           V  |
    #  |       b ----> c  |
    #  +------------------+
    params = {}
    params["model_probs_a"] = jnp.array([0.45, 0.55])
    params["model_probs_b"] = jnp.array([0.6, 0.4])
    params["model_probs_c"] = jnp.array(
        [[[0.4, 0.5, 0.1], [0.3, 0.5, 0.2]], [[0.3, 0.4, 0.3], [0.4, 0.4, 0.2]]]
    )
    params["guide_probs_b"] = jnp.array([0.8, 0.2])
    data = jnp.array([1, 2])

    @config_enumerate
    def model_plate(params):
        probs_a = pyro.param(
            "model_probs_a", params["model_probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "model_probs_b", params["model_probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "model_probs_c",
            params["model_probs_c"],
            constraint=constraints.simplex,
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("b_axis", 2):
            b = pyro.sample("b", dist.Categorical(probs_b))
            pyro.sample("c", dist.Categorical(Vindex(probs_c)[a, b]), obs=data)

    @config_enumerate
    def guide_plate(params):
        probs_b = pyro.param(
            "guide_probs_b", params["guide_probs_b"], constraint=constraints.simplex
        )
        with pyro.plate("b_axis", 2):
            pyro.sample("b", dist.Categorical(probs_b))

    @config_enumerate
    def model_iplate(params):
        probs_a = pyro.param(
            "model_probs_a", params["model_probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "model_probs_b", params["model_probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "model_probs_c",
            params["model_probs_c"],
            constraint=constraints.simplex,
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in range(2):
            b = pyro.sample(f"b_{i}", dist.Categorical(probs_b))
            pyro.sample(f"c_{i}", dist.Categorical(Vindex(probs_c)[a, b]), obs=data[i])

    @config_enumerate
    def guide_iplate(params):
        probs_b = pyro.param(
            "guide_probs_b", params["guide_probs_b"], constraint=constraints.simplex
        )
        for i in range(2):
            pyro.sample(f"b_{i}", dist.Categorical(probs_b))

    def auto_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, model_plate, guide_plate, params)

    def hand_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, model_iplate, guide_iplate, params)

    with pytest.raises(
        ValueError, match="Expected model enumeration to be no more global than guide"
    ):
        hand_loss, hand_grad = jax.value_and_grad(hand_loss_fn)(params)
        # This never gets run because we don't support this yet.
        auto_loss, auto_grad = jax.value_and_grad(auto_loss_fn)(params)

        assert_equal(auto_loss, hand_loss, prec=1e-5)
        assert_equal(auto_grad, hand_grad, prec=1e-5)


@pytest.mark.parametrize("enumerate1", ["parallel", "sequential"])
def test_elbo_enumerate_plate_6(enumerate1):
    #     Guide           Model
    #           +-------+
    #       b ----> c <---- a
    #           |  M=2  |
    #           +-------+
    # This tests that sequential enumeration over b works, even though
    # model-side enumeration moves c into b's plate via contraction.
    params = {}
    params["model_probs_a"] = jnp.array([0.45, 0.55])
    params["model_probs_b"] = jnp.array([0.6, 0.4])
    params["model_probs_c"] = jnp.array(
        [[[0.4, 0.5, 0.1], [0.3, 0.5, 0.2]], [[0.3, 0.4, 0.3], [0.4, 0.4, 0.2]]]
    )
    params["guide_probs_b"] = jnp.array([0.8, 0.2])
    data = jnp.array([1, 2])

    @config_enumerate
    def model_plate(params):
        probs_a = pyro.param(
            "model_probs_a", params["model_probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "model_probs_b", params["model_probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "model_probs_c",
            params["model_probs_c"],
            constraint=constraints.simplex,
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        b = pyro.sample("b", dist.Categorical(probs_b))
        with pyro.plate("b_axis", 2):
            pyro.sample("c", dist.Categorical(Vindex(probs_c)[a, b]), obs=data)

    @config_enumerate
    def model_iplate(params):
        probs_a = pyro.param(
            "model_probs_a", params["model_probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "model_probs_b", params["model_probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "model_probs_c",
            params["model_probs_c"],
            constraint=constraints.simplex,
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        b = pyro.sample("b", dist.Categorical(probs_b))
        for i in range(2):
            pyro.sample(
                "c_{}".format(i), dist.Categorical(Vindex(probs_c)[a, b]), obs=data[i]
            )

    @config_enumerate(default=enumerate1)
    def guide(params):
        probs_b = pyro.param(
            "guide_probs_b", params["guide_probs_b"], constraint=constraints.simplex
        )
        pyro.sample("b", dist.Categorical(probs_b))

    def auto_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, model_plate, guide, params)

    def hand_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, model_iplate, guide, params)

    auto_loss, auto_grad = jax.value_and_grad(auto_loss_fn)(params)
    hand_loss, hand_grad = jax.value_and_grad(hand_loss_fn)(params)

    assert_equal(auto_loss, hand_loss, prec=1e-5)
    assert_equal(auto_grad, hand_grad, prec=1e-5)


@pytest.mark.parametrize("scale", [1, 10])
def test_elbo_enumerate_plate_7(scale):
    #  Guide    Model
    #    a -----> b
    #    |        |
    #  +-|--------|----------------+
    #  | V        V                |
    #  | c -----> d -----> e   N=2 |
    #  +---------------------------+
    # This tests a mixture of model and guide enumeration.
    params = {}
    params["model_probs_a"] = jnp.array([0.45, 0.55])
    params["model_probs_b"] = jnp.array([[0.6, 0.4], [0.4, 0.6]])
    params["model_probs_c"] = jnp.array([[0.75, 0.25], [0.55, 0.45]])
    params["model_probs_d"] = jnp.array(
        [[[0.4, 0.6], [0.3, 0.7]], [[0.3, 0.7], [0.2, 0.8]]]
    )
    params["model_probs_e"] = jnp.array([[0.75, 0.25], [0.55, 0.45]])
    params["guide_probs_a"] = jnp.array([0.35, 0.64])
    params["guide_probs_c"] = jnp.array([[0.0, 1.0], [1.0, 0.0]])  # deterministic

    @handlers.scale(scale=scale)
    def auto_model(data, params):
        probs_a = pyro.param(
            "model_probs_a", params["model_probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "model_probs_b", params["model_probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "model_probs_c",
            params["model_probs_c"],
            constraint=constraints.simplex,
        )
        probs_d = pyro.param(
            "model_probs_d",
            params["model_probs_d"],
            constraint=constraints.simplex,
        )
        probs_e = pyro.param(
            "model_probs_e",
            params["model_probs_e"],
            constraint=constraints.simplex,
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        b = pyro.sample(
            "b", dist.Categorical(probs_b[a]), infer={"enumerate": "parallel"}
        )
        with pyro.plate("data", 2):
            c = pyro.sample("c", dist.Categorical(probs_c[a]))
            d = pyro.sample(
                "d",
                dist.Categorical(Vindex(probs_d)[b, c]),
                infer={"enumerate": "parallel"},
            )
            pyro.sample("obs", dist.Categorical(probs_e[d]), obs=data)

    @handlers.scale(scale=scale)
    def auto_guide(data, params):
        probs_a = pyro.param(
            "guide_probs_a", params["guide_probs_a"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "guide_probs_c", params["guide_probs_c"], constraint=constraints.simplex
        )
        a = pyro.sample("a", dist.Categorical(probs_a), infer={"enumerate": "parallel"})
        with pyro.plate("data", 2):
            pyro.sample("c", dist.Categorical(probs_c[a]))

    @handlers.scale(scale=scale)
    def hand_model(data, params):
        probs_a = pyro.param(
            "model_probs_a", params["model_probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "model_probs_b", params["model_probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "model_probs_c",
            params["model_probs_c"],
            constraint=constraints.simplex,
        )
        probs_d = pyro.param(
            "model_probs_d",
            params["model_probs_d"],
            constraint=constraints.simplex,
        )
        probs_e = pyro.param(
            "model_probs_e",
            params["model_probs_e"],
            constraint=constraints.simplex,
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        b = pyro.sample(
            "b", dist.Categorical(probs_b[a]), infer={"enumerate": "parallel"}
        )
        for i in range(2):
            c = pyro.sample(f"c_{i}", dist.Categorical(probs_c[a]))
            d = pyro.sample(
                f"d_{i}",
                dist.Categorical(Vindex(probs_d)[b, c]),
                infer={"enumerate": "parallel"},
            )
            pyro.sample(f"obs_{i}", dist.Categorical(probs_e[d]), obs=data[i])

    @handlers.scale(scale=scale)
    def hand_guide(data, params):
        probs_a = pyro.param(
            "guide_probs_a", params["guide_probs_a"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "guide_probs_c", params["guide_probs_c"], constraint=constraints.simplex
        )
        a = pyro.sample("a", dist.Categorical(probs_a), infer={"enumerate": "parallel"})
        for i in range(2):
            pyro.sample(f"c_{i}", dist.Categorical(probs_c[a]))

    data = jnp.array([0, 0])

    def auto_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, auto_model, auto_guide, data, params)

    def hand_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, hand_model, hand_guide, data, params)

    auto_loss, auto_grad = jax.value_and_grad(auto_loss_fn)(params)
    hand_loss, hand_grad = jax.value_and_grad(hand_loss_fn)(params)

    assert_equal(auto_loss, hand_loss, prec=1e-5)
    assert_equal(auto_grad, hand_grad, prec=1e-5)


@pytest.mark.parametrize("scale", [1, 10])
def test_elbo_enumerate_plates_1(scale):
    #  +-----------------+
    #  | a ----> b   M=2 |
    #  +-----------------+
    #  +-----------------+
    #  | c ----> d   N=3 |
    #  +-----------------+
    # This tests two unrelated plates.
    # Each should remain uncontracted.
    params = {}
    params["probs_a"] = jnp.array([0.45, 0.55])
    params["probs_b"] = jnp.array([[0.6, 0.4], [0.4, 0.6]])
    params["probs_c"] = jnp.array([0.75, 0.25])
    params["probs_d"] = jnp.array([[0.4, 0.6], [0.3, 0.7]])

    b_data = jnp.array([0, 1])
    d_data = jnp.array([0, 0, 1])

    @config_enumerate
    @handlers.scale(scale=scale)
    def auto_model(params):
        probs_a = pyro.param(
            "probs_a", params["probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["probs_c"], constraint=constraints.simplex
        )
        probs_d = pyro.param(
            "probs_d", params["probs_d"], constraint=constraints.simplex
        )
        with pyro.plate("a_axis", 2):
            a = pyro.sample("a", dist.Categorical(probs_a))
            pyro.sample("b", dist.Categorical(probs_b[a]), obs=b_data)
        with pyro.plate("c_axis", 3):
            c = pyro.sample("c", dist.Categorical(probs_c))
            pyro.sample("d", dist.Categorical(probs_d[c]), obs=d_data)

    @config_enumerate
    @handlers.scale(scale=scale)
    def hand_model(params):
        probs_a = pyro.param(
            "probs_a", params["probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["probs_c"], constraint=constraints.simplex
        )
        probs_d = pyro.param(
            "probs_d", params["probs_d"], constraint=constraints.simplex
        )
        for i in range(2):
            a = pyro.sample(f"a_{i}", dist.Categorical(probs_a))
            pyro.sample(f"b_{i}", dist.Categorical(probs_b[a]), obs=b_data[i])
        for j in range(3):
            c = pyro.sample(f"c_{j}", dist.Categorical(probs_c))
            pyro.sample(f"d_{j}", dist.Categorical(probs_d[c]), obs=d_data[j])

    def guide(params):
        pass

    def auto_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, auto_model, guide, params)

    def hand_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, hand_model, guide, params)

    auto_loss, auto_grad = jax.value_and_grad(auto_loss_fn)(params)
    hand_loss, hand_grad = jax.value_and_grad(hand_loss_fn)(params)

    assert_equal(auto_loss, hand_loss, prec=1e-5)
    assert_equal(auto_grad, hand_grad, prec=1e-5)


@pytest.mark.parametrize("scale", [1, 10])
def test_elbo_enumerate_plates_2(scale):
    #  +---------+       +---------+
    #  |     b <---- a ----> c     |
    #  | M=2     |       |     N=3 |
    #  +---------+       +---------+
    # This tests two different plates with recycled dimension.
    params = {}
    params["probs_a"] = jnp.array([0.45, 0.55])
    params["probs_b"] = jnp.array([[0.6, 0.4], [0.4, 0.6]])
    params["probs_c"] = jnp.array([[0.75, 0.25], [0.55, 0.45]])

    b_data = jnp.array([0, 1])
    c_data = jnp.array([0, 0, 1])

    @config_enumerate
    @handlers.scale(scale=scale)
    def auto_model(params):
        probs_a = pyro.param(
            "probs_a", params["probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["probs_c"], constraint=constraints.simplex
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("b_axis", 2):
            pyro.sample("b", dist.Categorical(probs_b[a]), obs=b_data)
        with pyro.plate("c_axis", 3):
            pyro.sample("c", dist.Categorical(probs_c[a]), obs=c_data)

    @config_enumerate
    @handlers.scale(scale=scale)
    def hand_model(params):
        probs_a = pyro.param(
            "probs_a", params["probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["probs_c"], constraint=constraints.simplex
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in range(2):
            pyro.sample(f"b_{i}", dist.Categorical(probs_b[a]), obs=b_data[i])
        for j in range(3):
            pyro.sample(f"c_{j}", dist.Categorical(probs_c[a]), obs=c_data[j])

    def guide(params):
        pass

    def auto_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, auto_model, guide, params)

    def hand_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, hand_model, guide, params)

    auto_loss, auto_grad = jax.value_and_grad(auto_loss_fn)(params)
    hand_loss, hand_grad = jax.value_and_grad(hand_loss_fn)(params)

    assert_equal(auto_loss, hand_loss, prec=1e-5)
    assert_equal(auto_grad, hand_grad, prec=1e-5)


@pytest.mark.parametrize("scale", [1, 10])
def test_elbo_enumerate_plates_3(scale):
    #      +--------------------+
    #      |  +----------+      |
    #  a -------> b      |      |
    #      |  |      N=2 |      |
    #      |  +----------+  M=2 |
    #      +--------------------+
    # This is tests the case of multiple plate contractions in
    # a single step.
    params = {}
    params["probs_a"] = jnp.array([0.45, 0.55])
    params["probs_b"] = jnp.array([[0.6, 0.4], [0.4, 0.6]])
    data = jnp.array([[0, 1], [0, 0]])

    @config_enumerate
    @handlers.scale(scale=scale)
    def auto_model(params):
        probs_a = pyro.param(
            "probs_a", params["probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["probs_b"], constraint=constraints.simplex
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("outer", 2):
            with pyro.plate("inner", 2):
                pyro.sample("b", dist.Categorical(probs_b[a]), obs=data)

    @config_enumerate
    @handlers.scale(scale=scale)
    def hand_model(params):
        probs_a = pyro.param(
            "probs_a", params["probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["probs_b"], constraint=constraints.simplex
        )
        inner = range(2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in range(2):
            for j in inner:
                pyro.sample(f"b_{i}_{j}", dist.Categorical(probs_b[a]), obs=data[i, j])

    def guide(params):
        pass

    def auto_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, auto_model, guide, params)

    def hand_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, hand_model, guide, params)

    auto_loss, auto_grad = jax.value_and_grad(auto_loss_fn)(params)
    hand_loss, hand_grad = jax.value_and_grad(hand_loss_fn)(params)

    assert_equal(auto_loss, hand_loss, prec=1e-5)
    assert_equal(auto_grad, hand_grad, prec=1e-5)


@pytest.mark.parametrize("scale", [1, 10])
def test_elbo_enumerate_plates_4(scale):
    #      +--------------------+
    #      |       +----------+ |
    #  a ----> b ----> c      | |
    #      |       |      N=2 | |
    #      | M=2   +----------+ |
    #      +--------------------+
    params = {}
    params["probs_a"] = jnp.array([0.45, 0.55])
    params["probs_b"] = jnp.array([[0.6, 0.4], [0.4, 0.6]])
    params["probs_c"] = jnp.array([[0.4, 0.6], [0.3, 0.7]])

    @config_enumerate
    @handlers.scale(scale=scale)
    def auto_model(data, params):
        probs_a = pyro.param(
            "probs_a", params["probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["probs_c"], constraint=constraints.simplex
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("outer", 2):
            b = pyro.sample("b", dist.Categorical(probs_b[a]))
            with pyro.plate("inner", 2):
                pyro.sample("c", dist.Categorical(probs_c[b]), obs=data)

    @config_enumerate
    @handlers.scale(scale=scale)
    def hand_model(data, params):
        probs_a = pyro.param(
            "probs_a", params["probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["probs_c"], constraint=constraints.simplex
        )
        inner = range(2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in range(2):
            b = pyro.sample(f"b_{i}", dist.Categorical(probs_b[a]))
            for j in inner:
                pyro.sample(f"c_{i}_{j}", dist.Categorical(probs_c[b]), obs=data[i, j])

    def guide(data, params):
        pass

    data = jnp.array([[0, 1], [0, 0]])

    def auto_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, auto_model, guide, data, params)

    def hand_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, hand_model, guide, data, params)

    auto_loss, auto_grad = jax.value_and_grad(auto_loss_fn)(params)
    hand_loss, hand_grad = jax.value_and_grad(hand_loss_fn)(params)

    assert_equal(auto_loss, hand_loss, prec=1e-5)
    assert_equal(auto_grad, hand_grad, prec=1e-5)


@pytest.mark.parametrize("scale", [1, 10])
def test_elbo_enumerate_plates_5(scale):
    #     a
    #     | \
    #  +--|---\------------+
    #  |  V   +-\--------+ |
    #  |  b ----> c      | |
    #  |      |      N=2 | |
    #  | M=2  +----------+ |
    #  +-------------------+
    params = {}
    params["probs_a"] = jnp.array([0.45, 0.55])
    params["probs_b"] = jnp.array([[0.6, 0.4], [0.4, 0.6]])
    params["probs_c"] = jnp.array([[[0.4, 0.6], [0.3, 0.7]], [[0.2, 0.8], [0.1, 0.9]]])
    data = jnp.array([[0, 1], [0, 0]])

    @config_enumerate
    @handlers.scale(scale=scale)
    def auto_model(params):
        probs_a = pyro.param(
            "probs_a", params["probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["probs_c"], constraint=constraints.simplex
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("outer", 2):
            b = pyro.sample("b", dist.Categorical(probs_b[a]))
            with pyro.plate("inner", 2):
                pyro.sample("c", dist.Categorical(Vindex(probs_c)[a, b]), obs=data)

    @config_enumerate
    @handlers.scale(scale=scale)
    def hand_model(params):
        probs_a = pyro.param(
            "probs_a", params["probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["probs_c"], constraint=constraints.simplex
        )
        inner = range(2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in range(2):
            b = pyro.sample(f"b_{i}", dist.Categorical(probs_b[a]))
            for j in inner:
                pyro.sample(
                    f"c_{i}_{j}",
                    dist.Categorical(Vindex(probs_c)[a, b]),
                    obs=data[i, j],
                )

    def guide(params):
        pass

    def auto_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, auto_model, guide, params)

    def hand_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, hand_model, guide, params)

    auto_loss, auto_grad = jax.value_and_grad(auto_loss_fn)(params)
    hand_loss, hand_grad = jax.value_and_grad(hand_loss_fn)(params)

    assert_equal(auto_loss, hand_loss, prec=1e-5)
    assert_equal(auto_grad, hand_grad, prec=1e-5)


@pytest.mark.parametrize("scale", [1, 10])
def test_elbo_enumerate_plates_6(scale):
    #         +----------+
    #         |      M=2 |
    #     a ----> b      |
    #     |   |   |      |
    #  +--|-------|--+   |
    #  |  V   |   V  |   |
    #  |  c ----> d  |   |
    #  |      |      |   |
    #  | N=2  +------|---+
    #  +-------------+
    # This tests different ways of mixing two independence contexts,
    # where each can be either sequential or vectorized plate.
    params = {}
    params["probs_a"] = jnp.array([0.45, 0.55])
    params["probs_b"] = jnp.array([[0.6, 0.4], [0.4, 0.6]])
    params["probs_c"] = jnp.array([[0.75, 0.25], [0.55, 0.45]])
    params["probs_d"] = jnp.array([[[0.4, 0.6], [0.3, 0.7]], [[0.3, 0.7], [0.2, 0.8]]])

    @config_enumerate
    @handlers.scale(scale=scale)
    @handlers.trace
    def model_iplate_iplate(data, params):
        probs_a = pyro.param(
            "probs_a", params["probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["probs_c"], constraint=constraints.simplex
        )
        probs_d = pyro.param(
            "probs_d", params["probs_d"], constraint=constraints.simplex
        )
        b_axis = range(2)
        c_axis = range(2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        b = [pyro.sample(f"b_{i}", dist.Categorical(probs_b[a])) for i in b_axis]
        c = [pyro.sample(f"c_{j}", dist.Categorical(probs_c[a])) for j in c_axis]
        for i in b_axis:
            for j in c_axis:
                pyro.sample(
                    f"d_{i}_{j}",
                    dist.Categorical(Vindex(probs_d)[b[i], c[j]]),
                    obs=data[i, j],
                )

    @config_enumerate
    @handlers.scale(scale=scale)
    def model_iplate_plate(data, params):
        probs_a = pyro.param(
            "probs_a", params["probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["probs_c"], constraint=constraints.simplex
        )
        probs_d = pyro.param(
            "probs_d", params["probs_d"], constraint=constraints.simplex
        )
        b_axis = range(2)
        c_axis = pyro.plate("c_axis", 2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        with c_axis:
            c = pyro.sample("c", dist.Categorical(probs_c[a]))
        for i in b_axis:
            b_i = pyro.sample(f"b_{i}", dist.Categorical(probs_b[a]))
            with c_axis:
                pyro.sample(
                    f"d_{i}",
                    dist.Categorical(Vindex(probs_d)[b_i, c]),
                    obs=data[i],
                )

    @config_enumerate
    @handlers.scale(scale=scale)
    @handlers.trace
    def model_plate_iplate(data, params):
        probs_a = pyro.param(
            "probs_a", params["probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["probs_c"], constraint=constraints.simplex
        )
        probs_d = pyro.param(
            "probs_d", params["probs_d"], constraint=constraints.simplex
        )
        b_axis = pyro.plate("b_axis", 2)
        c_axis = range(2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        with b_axis:
            b = pyro.sample("b", dist.Categorical(probs_b[a]))
        c = [pyro.sample(f"c_{j}", dist.Categorical(probs_c[a])) for j in c_axis]
        with b_axis:
            for j in c_axis:
                pyro.sample(
                    f"d_{j}",
                    dist.Categorical(Vindex(probs_d)[b, c[j]]),
                    obs=data[:, j],
                )

    @config_enumerate
    @handlers.scale(scale=scale)
    def model_plate_plate(data, params):
        probs_a = pyro.param(
            "probs_a", params["probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["probs_c"], constraint=constraints.simplex
        )
        probs_d = pyro.param(
            "probs_d", params["probs_d"], constraint=constraints.simplex
        )
        b_axis = pyro.plate("b_axis", 2, dim=-1)
        c_axis = pyro.plate("c_axis", 2, dim=-2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        with b_axis:
            b = pyro.sample("b", dist.Categorical(probs_b[a]))
        with c_axis:
            c = pyro.sample("c", dist.Categorical(probs_c[a]))
        with b_axis, c_axis:
            pyro.sample("d", dist.Categorical(Vindex(probs_d)[b, c]), obs=data)

    def guide(data, params):
        pass

    # Check that either one of the sequential plates can be promoted to be vectorized.
    data = jnp.array([[0, 1], [0, 0]])

    def iplate_iplate_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(
            random.PRNGKey(0), {}, model_iplate_iplate, guide, data, params
        )

    def plate_iplate_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, model_plate_iplate, guide, data, params)

    def iplate_plate_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, model_iplate_plate, guide, data, params)

    iplate_iplate_loss, iplate_iplate_grad = jax.value_and_grad(iplate_iplate_loss_fn)(
        params
    )
    plate_iplate_loss, plate_iplate_grad = jax.value_and_grad(plate_iplate_loss_fn)(
        params
    )
    iplate_plate_loss, iplate_plate_grad = jax.value_and_grad(iplate_plate_loss_fn)(
        params
    )

    assert_equal(iplate_iplate_loss, plate_iplate_loss, prec=1e-5)
    assert_equal(iplate_iplate_grad, plate_iplate_grad, prec=1e-5)
    assert_equal(iplate_iplate_loss, iplate_plate_loss, prec=1e-5)
    assert_equal(iplate_iplate_grad, iplate_plate_grad, prec=1e-5)

    # But promoting both to plates should result in an error.
    with pytest.raises(ValueError, match="intractable!"):
        elbo = infer.TraceEnum_ELBO()
        elbo.loss(random.PRNGKey(0), {}, model_plate_plate, guide, data, params)


@pytest.mark.parametrize("scale", [1, 10])
def test_elbo_enumerate_plates_7(scale):
    #         +-------------+
    #         |         N=2 |
    #     a -------> c      |
    #     |   |      |      |
    #  +--|----------|--+   |
    #  |  |   |      V  |   |
    #  |  V   |      e  |   |
    #  |  b ----> d     |   |
    #  |      |         |   |
    #  | M=2  +---------|---+
    #  +----------------+
    # This tests tree-structured dependencies among variables but
    # non-tree dependencies among plate nestings.
    params = {}
    params["probs_a"] = jnp.array([0.45, 0.55])
    params["probs_b"] = jnp.array([[0.6, 0.4], [0.4, 0.6]])
    params["probs_c"] = jnp.array([[0.75, 0.25], [0.55, 0.45]])
    params["probs_d"] = jnp.array([[0.3, 0.7], [0.2, 0.8]])
    params["probs_e"] = jnp.array([[0.4, 0.6], [0.3, 0.7]])

    @config_enumerate
    @handlers.scale(scale=scale)
    @handlers.trace
    def model_iplate_iplate(data, params):
        probs_a = pyro.param(
            "probs_a", params["probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["probs_c"], constraint=constraints.simplex
        )
        probs_d = pyro.param(
            "probs_d", params["probs_d"], constraint=constraints.simplex
        )
        probs_e = pyro.param(
            "probs_e", params["probs_e"], constraint=constraints.simplex
        )
        b_axis = range(2)
        c_axis = range(2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        b = [pyro.sample(f"b_{i}", dist.Categorical(probs_b[a])) for i in b_axis]
        c = [pyro.sample(f"c_{j}", dist.Categorical(probs_c[a])) for j in c_axis]
        for i in b_axis:
            for j in c_axis:
                pyro.sample(
                    f"d_{i}_{j}",
                    dist.Categorical(probs_d[b[i]]),
                    obs=data[i, j],
                )
                pyro.sample(
                    f"e_{i}_{j}",
                    dist.Categorical(probs_e[c[j]]),
                    obs=data[i, j],
                )

    @config_enumerate
    @handlers.scale(scale=scale)
    def model_iplate_plate(data, params):
        probs_a = pyro.param(
            "probs_a", params["probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["probs_c"], constraint=constraints.simplex
        )
        probs_d = pyro.param(
            "probs_d", params["probs_d"], constraint=constraints.simplex
        )
        probs_e = pyro.param(
            "probs_e", params["probs_e"], constraint=constraints.simplex
        )
        b_axis = range(2)
        c_axis = pyro.plate("c_axis", 2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        with c_axis:
            c = pyro.sample("c", dist.Categorical(probs_c[a]))
        for i in b_axis:
            b_i = pyro.sample(f"b_{i}", dist.Categorical(probs_b[a]))
            with c_axis:
                pyro.sample(f"d_{i}", dist.Categorical(probs_d[b_i]), obs=data[i])
                pyro.sample(f"e_{i}", dist.Categorical(probs_e[c]), obs=data[i])

    @config_enumerate
    @handlers.scale(scale=scale)
    @handlers.trace
    def model_plate_iplate(data, params):
        probs_a = pyro.param(
            "probs_a", params["probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["probs_c"], constraint=constraints.simplex
        )
        probs_d = pyro.param(
            "probs_d", params["probs_d"], constraint=constraints.simplex
        )
        probs_e = pyro.param(
            "probs_e", params["probs_e"], constraint=constraints.simplex
        )
        b_axis = pyro.plate("b_axis", 2)
        c_axis = range(2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        with b_axis:
            b = pyro.sample("b", dist.Categorical(probs_b[a]))
        c = [pyro.sample(f"c_{j}", dist.Categorical(probs_c[a])) for j in c_axis]
        with b_axis:
            for j in c_axis:
                pyro.sample(f"d_{j}", dist.Categorical(probs_d[b]), obs=data[:, j])
                pyro.sample(f"e_{j}", dist.Categorical(probs_e[c[j]]), obs=data[:, j])

    @config_enumerate
    @handlers.scale(scale=scale)
    def model_plate_plate(data, params):
        probs_a = pyro.param(
            "probs_a", params["probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["probs_c"], constraint=constraints.simplex
        )
        probs_d = pyro.param(
            "probs_d", params["probs_d"], constraint=constraints.simplex
        )
        probs_e = pyro.param(
            "probs_e", params["probs_e"], constraint=constraints.simplex
        )
        b_axis = pyro.plate("b_axis", 2, dim=-1)
        c_axis = pyro.plate("c_axis", 2, dim=-2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        with b_axis:
            b = pyro.sample("b", dist.Categorical(probs_b[a]))
        with c_axis:
            c = pyro.sample("c", dist.Categorical(probs_c[a]))
        with b_axis, c_axis:
            pyro.sample("d", dist.Categorical(probs_d[b]), obs=data)
            pyro.sample("e", dist.Categorical(probs_e[c]), obs=data)

    def guide(data, params):
        pass

    # Check that any combination of sequential plates can be promoted to be vectorized.
    data = jnp.array([[0, 1], [0, 0]])

    def iplate_iplate_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(
            random.PRNGKey(0), {}, model_iplate_iplate, guide, data, params
        )

    def plate_iplate_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, model_plate_iplate, guide, data, params)

    def iplate_plate_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, model_iplate_plate, guide, data, params)

    def plate_plate_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, model_plate_plate, guide, data, params)

    iplate_iplate_loss, iplate_iplate_grad = jax.value_and_grad(iplate_iplate_loss_fn)(
        params
    )
    plate_iplate_loss, plate_iplate_grad = jax.value_and_grad(plate_iplate_loss_fn)(
        params
    )
    iplate_plate_loss, iplate_plate_grad = jax.value_and_grad(iplate_plate_loss_fn)(
        params
    )
    plate_plate_loss, plate_plate_grad = jax.value_and_grad(plate_plate_loss_fn)(params)

    assert_equal(iplate_iplate_loss, plate_iplate_loss, prec=1e-4)
    assert_equal(iplate_iplate_grad, plate_iplate_grad, prec=1e-4)
    assert_equal(iplate_iplate_loss, iplate_plate_loss, prec=1e-4)
    assert_equal(iplate_iplate_grad, iplate_plate_grad, prec=1e-4)
    assert_equal(iplate_iplate_loss, plate_plate_loss, prec=1e-4)
    assert_equal(iplate_iplate_grad, plate_plate_grad, prec=1e-4)


@pytest.mark.parametrize("guide_scale", [1])
@pytest.mark.parametrize("model_scale", [1])
@pytest.mark.parametrize("outer_vectorized", [False, True])
@pytest.mark.parametrize("inner_vectorized", [False, True])
def test_elbo_enumerate_plates_8(
    model_scale, guide_scale, inner_vectorized, outer_vectorized
):
    #        Guide   Model
    #                  a
    #      +-----------|--------+
    #      | M=2   +---|------+ |
    #      |       |   V  N=2 | |
    #      |   b ----> c      | |
    #      |       +----------+ |
    #      +--------------------+
    params = {}
    params["model_probs_a"] = jnp.array([0.45, 0.55])
    params["model_probs_b"] = jnp.array([0.6, 0.4])
    params["model_probs_c"] = jnp.array(
        [[[0.4, 0.5, 0.1], [0.3, 0.5, 0.2]], [[0.3, 0.4, 0.3], [0.4, 0.4, 0.2]]]
    )
    params["guide_probs_b"] = jnp.array([0.8, 0.2])
    data = jnp.array([[0, 1], [0, 2]])

    @config_enumerate
    @handlers.scale(scale=model_scale)
    def model_plate_plate(params):
        probs_a = pyro.param(
            "probs_a", params["model_probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["model_probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["model_probs_c"], constraint=constraints.simplex
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("outer", 2):
            b = pyro.sample("b", dist.Categorical(probs_b))
            with pyro.plate("inner", 2):
                pyro.sample("c", dist.Categorical(Vindex(probs_c)[a, b]), obs=data)

    @config_enumerate
    @handlers.scale(scale=model_scale)
    def model_iplate_plate(params):
        probs_a = pyro.param(
            "probs_a", params["model_probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["model_probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["model_probs_c"], constraint=constraints.simplex
        )
        inner = pyro.plate("inner", 2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in range(2):
            b = pyro.sample(f"b_{i}", dist.Categorical(probs_b))
            with inner:
                pyro.sample(
                    f"c_{i}",
                    dist.Categorical(Vindex(probs_c)[a, b]),
                    obs=data[:, i],
                )

    @config_enumerate
    @handlers.scale(scale=model_scale)
    def model_plate_iplate(params):
        probs_a = pyro.param(
            "probs_a", params["model_probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["model_probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["model_probs_c"], constraint=constraints.simplex
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("outer", 2):
            b = pyro.sample("b", dist.Categorical(probs_b))
            for j in range(2):
                pyro.sample(
                    f"c_{j}",
                    dist.Categorical(Vindex(probs_c)[a, b]),
                    obs=data[j],
                )

    @config_enumerate
    @handlers.scale(scale=model_scale)
    def model_iplate_iplate(params):
        probs_a = pyro.param(
            "probs_a", params["model_probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["model_probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["model_probs_c"], constraint=constraints.simplex
        )
        inner = range(2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in range(2):
            b = pyro.sample(f"b_{i}", dist.Categorical(probs_b))
            for j in inner:
                pyro.sample(
                    f"c_{i}_{j}",
                    dist.Categorical(Vindex(probs_c)[a, b]),
                    obs=data[j, i],
                )

    @config_enumerate
    @handlers.scale(scale=guide_scale)
    def guide_plate(params):
        probs_b = pyro.param(
            "probs_b", params["guide_probs_b"], constraint=constraints.simplex
        )
        with pyro.plate("outer", 2):
            pyro.sample("b", dist.Categorical(probs_b))

    @config_enumerate
    @handlers.scale(scale=guide_scale)
    def guide_iplate(params):
        probs_b = pyro.param(
            "probs_b", params["guide_probs_b"], constraint=constraints.simplex
        )
        for i in range(2):
            pyro.sample(f"b_{i}", dist.Categorical(probs_b))

    def iplate_iplate_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(
            random.PRNGKey(0), {}, model_iplate_iplate, guide_iplate, params
        )

    def plate_iplate_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, model_plate_iplate, guide_plate, params)

    def iplate_plate_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(
            random.PRNGKey(0), {}, model_iplate_plate, guide_iplate, params
        )

    def plate_plate_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, model_plate_plate, guide_plate, params)

    expected_loss, expected_grad = jax.value_and_grad(iplate_iplate_loss_fn)(params)
    if inner_vectorized:
        if outer_vectorized:
            with pytest.raises(
                ValueError,
                match="Expected model enumeration to be no more global than guide",
            ):
                actual_loss, actual_grad = jax.value_and_grad(plate_plate_loss_fn)(
                    params
                )
                assert_equal(actual_loss, expected_loss, prec=1e-4)
                assert_equal(actual_grad, expected_grad, prec=1e-4)
        else:
            actual_loss, actual_grad = jax.value_and_grad(iplate_plate_loss_fn)(params)
            assert_equal(actual_loss, expected_loss, prec=1e-4)
            assert_equal(actual_grad, expected_grad, prec=1e-4)
    else:
        if outer_vectorized:
            with pytest.raises(
                ValueError,
                match="Expected model enumeration to be no more global than guide",
            ):
                actual_loss, actual_grad = jax.value_and_grad(plate_iplate_loss_fn)(
                    params
                )
                assert_equal(actual_loss, expected_loss, prec=1e-4)
                assert_equal(actual_grad, expected_grad, prec=1e-4)
        else:
            actual_loss, actual_grad = jax.value_and_grad(iplate_iplate_loss_fn)(params)
            assert_equal(actual_loss, expected_loss, prec=1e-4)
            assert_equal(actual_grad, expected_grad, prec=1e-4)


def test_elbo_enumerate_plate_9():
    #        Model   Guide
    #          a
    #  +-------|-------+
    #  | M=2   V       |
    #  |       b -> c  |
    #  +---------------+
    params = {}
    params["model_probs_a"] = jnp.array([0.45, 0.55])
    params["model_probs_b"] = jnp.array([[0.3, 0.7], [0.6, 0.4]])
    params["model_probs_c"] = jnp.array([[0.3, 0.4, 0.3], [0.4, 0.4, 0.2]])
    params["guide_probs_a"] = jnp.array([0.45, 0.55])
    params["guide_probs_b"] = jnp.array([[0.3, 0.7], [0.8, 0.2]])

    data = jnp.array([1, 2])

    @config_enumerate
    def model_plate(params):
        probs_a = pyro.param(
            "probs_a", params["model_probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["model_probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["model_probs_c"], constraint=constraints.simplex
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("b_axis", 2):
            b = pyro.sample("b", dist.Categorical(probs_b[a]))
            pyro.sample("c", dist.Categorical(probs_c[b]), obs=data)

    @config_enumerate
    def guide_plate(params):
        probs_a = pyro.param(
            "probs_a", params["guide_probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["guide_probs_b"], constraint=constraints.simplex
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("b_axis", 2):
            pyro.sample("b", dist.Categorical(probs_b[a]))

    @config_enumerate
    def model_iplate(params):
        probs_a = pyro.param(
            "probs_a", params["model_probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["model_probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["model_probs_c"], constraint=constraints.simplex
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in range(2):
            b = pyro.sample(f"b_{i}", dist.Categorical(probs_b[a]))
            pyro.sample(f"c_{i}", dist.Categorical(probs_c[b]), obs=data[i])

    @config_enumerate
    def guide_iplate(params):
        probs_a = pyro.param(
            "probs_a", params["guide_probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["guide_probs_b"], constraint=constraints.simplex
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in range(2):
            pyro.sample(f"b_{i}", dist.Categorical(probs_b[a]))

    def expected_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, model_iplate, guide_iplate, params)

    def actual_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, model_plate, guide_plate, params)

    expected_loss, expected_grad = jax.value_and_grad(expected_loss_fn)(params)
    actual_loss, actual_grad = jax.value_and_grad(actual_loss_fn)(params)

    assert_equal(expected_loss, actual_loss, prec=1e-5)
    assert_equal(expected_grad, actual_grad, prec=1e-5)


def test_elbo_enumerate_plate_10():
    # Model
    # a -> [ [ bij -> cij ] ]
    # Guide
    # a -> [ [ bij ] ]
    params = {}
    params["model_probs_a"] = jnp.array([0.45, 0.55])
    params["model_probs_b"] = jnp.array([[0.3, 0.7], [0.6, 0.4]])
    params["model_probs_c"] = jnp.array([[0.3, 0.4, 0.3], [0.4, 0.4, 0.2]])
    params["guide_probs_a"] = jnp.array([0.45, 0.55])
    params["guide_probs_b"] = jnp.array([[0.3, 0.7], [0.8, 0.2]])
    data = jnp.array([[0, 1, 2], [1, 2, 2]])

    @config_enumerate
    def model_plate(params):
        probs_a = pyro.param(
            "probs_a", params["model_probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["model_probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["model_probs_c"], constraint=constraints.simplex
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("i", 2, dim=-2):
            with pyro.plate("j", 3, dim=-1):
                b = pyro.sample("b", dist.Categorical(probs_b[a]))
                pyro.sample("c", dist.Categorical(probs_c[b]), obs=data)

    @config_enumerate
    def guide_plate(params):
        probs_a = pyro.param(
            "probs_a", params["guide_probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["guide_probs_b"], constraint=constraints.simplex
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("i", 2, dim=-2):
            with pyro.plate("j", 3, dim=-1):
                pyro.sample("b", dist.Categorical(probs_b[a]))

    @config_enumerate
    def model_iplate(params):
        probs_a = pyro.param(
            "probs_a", params["model_probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["model_probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["model_probs_c"], constraint=constraints.simplex
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in range(2):
            for j in range(3):
                b = pyro.sample(f"b_{i}_{j}", dist.Categorical(probs_b[a]))
                pyro.sample(f"c_{i}_{j}", dist.Categorical(probs_c[b]), obs=data[i, j])

    @config_enumerate
    def guide_iplate(params):
        probs_a = pyro.param(
            "probs_a", params["guide_probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["guide_probs_b"], constraint=constraints.simplex
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in range(2):
            for j in range(3):
                pyro.sample(f"b_{i}_{j}", dist.Categorical(probs_b[a]))

    def expected_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, model_iplate, guide_iplate, params)

    def actual_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, model_plate, guide_plate, params)

    expected_loss, expected_grad = jax.value_and_grad(expected_loss_fn)(params)
    actual_loss, actual_grad = jax.value_and_grad(actual_loss_fn)(params)

    assert_equal(expected_loss, actual_loss, prec=1e-5)
    assert_equal(expected_grad, actual_grad, prec=1e-5)


def test_elbo_enumerate_plate_11():
    # Model
    # [ ai -> [ bij -> cij ] ]
    # Guide
    # [ ai -> [ bij ] ]
    params = {}
    params["model_probs_a"] = jnp.array([0.45, 0.55])
    params["model_probs_b"] = jnp.array([[0.3, 0.7], [0.6, 0.4]])
    params["model_probs_c"] = jnp.array([[0.3, 0.4, 0.3], [0.4, 0.4, 0.2]])
    params["guide_probs_a"] = jnp.array([0.45, 0.55])
    params["guide_probs_b"] = jnp.array([[0.3, 0.7], [0.8, 0.2]])
    data = jnp.array([[0, 1, 2], [1, 2, 2]])

    @config_enumerate
    def model_plate(params):
        probs_a = pyro.param(
            "probs_a", params["model_probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["model_probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["model_probs_c"], constraint=constraints.simplex
        )
        with pyro.plate("i", 2, dim=-2):
            a = pyro.sample("a", dist.Categorical(probs_a))
            with pyro.plate("j", 3, dim=-1):
                b = pyro.sample("b", dist.Categorical(probs_b[a]))
                pyro.sample("c", dist.Categorical(probs_c[b]), obs=data)

    @config_enumerate
    def guide_plate(params):
        probs_a = pyro.param(
            "probs_a", params["guide_probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["guide_probs_b"], constraint=constraints.simplex
        )
        with pyro.plate("i", 2, dim=-2):
            a = pyro.sample("a", dist.Categorical(probs_a))
            with pyro.plate("j", 3, dim=-1):
                pyro.sample("b", dist.Categorical(probs_b[a]))

    @config_enumerate
    def model_iplate(params):
        probs_a = pyro.param(
            "probs_a", params["model_probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["model_probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["model_probs_c"], constraint=constraints.simplex
        )
        for i in range(2):
            a = pyro.sample(f"a_{i}", dist.Categorical(probs_a))
            for j in range(3):
                b = pyro.sample(f"b_{i}_{j}", dist.Categorical(probs_b[a]))
                pyro.sample(f"c_{i}_{j}", dist.Categorical(probs_c[b]), obs=data[i, j])

    @config_enumerate
    def guide_iplate(params):
        probs_a = pyro.param(
            "probs_a", params["guide_probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["guide_probs_b"], constraint=constraints.simplex
        )
        for i in range(2):
            a = pyro.sample(f"a_{i}", dist.Categorical(probs_a))
            for j in range(3):
                pyro.sample(f"b_{i}_{j}", dist.Categorical(probs_b[a]))

    def expected_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, model_iplate, guide_iplate, params)

    def actual_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, model_plate, guide_plate, params)

    expected_loss, expected_grad = jax.value_and_grad(expected_loss_fn)(params)
    actual_loss, actual_grad = jax.value_and_grad(actual_loss_fn)(params)

    assert_equal(expected_loss, actual_loss, prec=1e-5)
    assert_equal(expected_grad, actual_grad, prec=1e-5)


def test_elbo_enumerate_plate_12():
    # Model
    # a -> [ bi -> [ cij -> dij ] ]
    # Guide
    # a -> [ bi -> [ cij ] ]
    params = {}
    params["model_probs_a"] = jnp.array([0.45, 0.55])
    params["model_probs_b"] = jnp.array([[0.3, 0.7], [0.6, 0.4]])
    params["model_probs_c"] = jnp.array([[0.3, 0.4, 0.3], [0.4, 0.4, 0.2]])
    params["model_probs_d"] = jnp.array(
        [[0.1, 0.6, 0.3], [0.3, 0.4, 0.3], [0.4, 0.4, 0.2]]
    )
    params["guide_probs_a"] = jnp.array([0.45, 0.55])
    params["guide_probs_b"] = jnp.array([[0.3, 0.7], [0.8, 0.2]])
    params["guide_probs_c"] = jnp.array([[0.3, 0.3, 0.4], [0.2, 0.4, 0.4]])
    data = jnp.array([[0, 1, 2], [1, 2, 2]])

    @config_enumerate
    def model_plate(params):
        probs_a = pyro.param(
            "probs_a", params["model_probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["model_probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["model_probs_c"], constraint=constraints.simplex
        )
        probs_d = pyro.param(
            "probs_d", params["model_probs_d"], constraint=constraints.simplex
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("i", 2, dim=-2):
            b = pyro.sample("b", dist.Categorical(probs_b[a]))
            with pyro.plate("j", 3, dim=-1):
                c = pyro.sample("c", dist.Categorical(probs_c[b]))
                pyro.sample("d", dist.Categorical(probs_d[c]), obs=data)

    @config_enumerate
    def guide_plate(params):
        probs_a = pyro.param(
            "probs_a", params["guide_probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["guide_probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["guide_probs_c"], constraint=constraints.simplex
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("i", 2, dim=-2):
            b = pyro.sample("b", dist.Categorical(probs_b[a]))
            with pyro.plate("j", 3, dim=-1):
                pyro.sample("c", dist.Categorical(probs_c[b]))

    @config_enumerate
    def model_iplate(params):
        probs_a = pyro.param(
            "probs_a", params["model_probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["model_probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["model_probs_c"], constraint=constraints.simplex
        )
        probs_d = pyro.param(
            "probs_d", params["model_probs_d"], constraint=constraints.simplex
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in range(2):
            b = pyro.sample(f"b_{i}", dist.Categorical(probs_b[a]))
            for j in range(3):
                c = pyro.sample(f"c_{i}_{j}", dist.Categorical(probs_c[b]))
                pyro.sample(f"d_{i}_{j}", dist.Categorical(probs_d[c]), obs=data[i, j])

    @config_enumerate
    def guide_iplate(params):
        probs_a = pyro.param(
            "probs_a", params["guide_probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["guide_probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["guide_probs_c"], constraint=constraints.simplex
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in range(2):
            b = pyro.sample(f"b_{i}", dist.Categorical(probs_b[a]))
            for j in range(3):
                pyro.sample(f"c_{i}_{j}", dist.Categorical(probs_c[b]))

    def expected_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, model_iplate, guide_iplate, params)

    def actual_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, model_plate, guide_plate, params)

    expected_loss, expected_grad = jax.value_and_grad(expected_loss_fn)(params)
    actual_loss, actual_grad = jax.value_and_grad(actual_loss_fn)(params)

    assert_equal(expected_loss, actual_loss, prec=1e-5)
    assert_equal(expected_grad, actual_grad, prec=1e-5)


def test_elbo_enumerate_plate_13():
    # Model
    # a -> [ cj -> [ dij ] ]
    # |
    # v
    # [ bi ]
    # Guide
    # a -> [ cj ]
    # |
    # v
    # [ bi ]
    params = {}
    params["model_probs_a"] = jnp.array([0.45, 0.55])
    params["model_probs_b"] = jnp.array([[0.3, 0.7], [0.6, 0.4]])
    params["model_probs_c"] = jnp.array([[0.3, 0.7], [0.4, 0.6]])
    params["model_probs_d"] = jnp.array(
        [[0.1, 0.6, 0.3], [0.3, 0.4, 0.3], [0.4, 0.4, 0.2]]
    )
    params["guide_probs_a"] = jnp.array([0.45, 0.55])
    params["guide_probs_b"] = jnp.array([[0.3, 0.7], [0.8, 0.2]])
    params["guide_probs_c"] = jnp.array([[0.2, 0.8], [0.9, 0.1]])
    data = jnp.array([[0, 1, 2], [1, 2, 2]])

    @config_enumerate
    def model_plate(params):
        probs_a = pyro.param(
            "probs_a", params["model_probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["model_probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["model_probs_c"], constraint=constraints.simplex
        )
        probs_d = pyro.param(
            "probs_d", params["model_probs_d"], constraint=constraints.simplex
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("i", 2, dim=-2):
            pyro.sample("b", dist.Categorical(probs_b[a]))
            with pyro.plate("j", 3, dim=-1):
                c = pyro.sample("c", dist.Categorical(probs_c[a]))
                pyro.sample("d", dist.Categorical(probs_d[c]), obs=data)

    @config_enumerate
    def guide_plate(params):
        probs_a = pyro.param(
            "probs_a", params["guide_probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["guide_probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["guide_probs_c"], constraint=constraints.simplex
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("i", 2, dim=-2):
            pyro.sample("b", dist.Categorical(probs_b[a]))
            with pyro.plate("j", 3, dim=-1):
                pyro.sample("c", dist.Categorical(probs_c[a]))

    @config_enumerate
    def model_iplate(params):
        probs_a = pyro.param(
            "probs_a", params["model_probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["model_probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["model_probs_c"], constraint=constraints.simplex
        )
        probs_d = pyro.param(
            "probs_d", params["model_probs_d"], constraint=constraints.simplex
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in range(2):
            pyro.sample(f"b_{i}", dist.Categorical(probs_b[a]))
            for j in range(3):
                c = pyro.sample(f"c_{i}_{j}", dist.Categorical(probs_c[a]))
                pyro.sample(f"d_{i}_{j}", dist.Categorical(probs_d[c]), obs=data[i, j])

    @config_enumerate
    def guide_iplate(params):
        probs_a = pyro.param(
            "probs_a", params["guide_probs_a"], constraint=constraints.simplex
        )
        probs_b = pyro.param(
            "probs_b", params["guide_probs_b"], constraint=constraints.simplex
        )
        probs_c = pyro.param(
            "probs_c", params["guide_probs_c"], constraint=constraints.simplex
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in range(2):
            pyro.sample(f"b_{i}", dist.Categorical(probs_b[a]))
            for j in range(3):
                pyro.sample(f"c_{i}_{j}", dist.Categorical(probs_c[a]))

    def expected_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, model_iplate, guide_iplate, params)

    def actual_loss_fn(params):
        elbo = infer.TraceEnum_ELBO()
        return elbo.loss(random.PRNGKey(0), {}, model_plate, guide_plate, params)

    expected_loss, expected_grad = jax.value_and_grad(expected_loss_fn)(params)
    actual_loss, actual_grad = jax.value_and_grad(actual_loss_fn)(params)

    assert_equal(expected_loss, actual_loss, prec=1e-5)
    assert_equal(expected_grad, actual_grad, prec=1e-5)


@pytest.mark.parametrize("scale", [1, 10])
def test_model_enum_subsample_1(scale):
    # Enumerate: a
    # Subsample: b
    #  a - [-> b  ]
    @config_enumerate
    @handlers.scale(scale=scale)
    def model(params):
        locs = pyro.param("locs", params["locs"])
        probs_a = pyro.param(
            "probs_a", params["probs_a"], constraint=constraints.simplex
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("b_axis", size=3):
            pyro.sample("b", dist.Normal(locs[a], 1.0), obs=0)

    @config_enumerate
    @handlers.scale(scale=scale)
    def model_subsample(params):
        locs = pyro.param("locs", params["locs"])
        probs_a = pyro.param(
            "probs_a", params["probs_a"], constraint=constraints.simplex
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("b_axis", size=3, subsample_size=2):
            pyro.sample("b", dist.Normal(locs[a], 1.0), obs=0)

    def guide(params):
        pass

    params = {
        "locs": jnp.array([0.0, 1.0]),
        "probs_a": jnp.array([0.4, 0.6]),
    }
    transform = dist.biject_to(dist.constraints.simplex)
    params_raw = {"locs": params["locs"], "probs_a": transform.inv(params["probs_a"])}

    elbo = infer.TraceEnum_ELBO()

    # Expected grads w/o subsampling
    def expected_loss_fn(params_raw):
        params = {
            "locs": params_raw["locs"],
            "probs_a": transform(params_raw["probs_a"]),
        }
        return elbo.loss(random.PRNGKey(0), {}, model, guide, params)

    expected_loss, expected_grads = jax.value_and_grad(expected_loss_fn)(params_raw)

    # Actual grads w/ subsampling
    def actual_loss_fn(params_raw):
        params = {
            "locs": params_raw["locs"],
            "probs_a": transform(params_raw["probs_a"]),
        }
        return elbo.loss(random.PRNGKey(0), {}, model_subsample, guide, params)

    with pytest.raises(
        ValueError, match="Expected all enumerated sample sites to share a common scale"
    ):
        # This never gets run because we don't support this yet.
        actual_loss, actual_grads = jax.value_and_grad(actual_loss_fn)(params_raw)

        assert_equal(actual_loss, expected_loss, prec=1e-5)
        assert_equal(actual_grads, expected_grads, prec=1e-5)


@pytest.mark.parametrize("scale", [1, 10])
def test_model_enum_subsample_2(scale):
    # Enumerate: a
    # Subsample: b, c
    #  a - [-> b  ]
    #   \
    #    - [-> c  ]
    @config_enumerate
    @handlers.scale(scale=scale)
    def model(params):
        locs = pyro.param("locs", params["locs"])
        probs_a = pyro.param(
            "probs_a", params["probs_a"], constraint=constraints.simplex
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("b_axis", size=3):
            pyro.sample("b", dist.Normal(locs[a], 1.0), obs=0)

        with pyro.plate("c_axis", size=6):
            pyro.sample("c", dist.Normal(locs[a], 1.0), obs=1)

    @config_enumerate
    @handlers.scale(scale=scale)
    def model_subsample(params):
        locs = pyro.param("locs", params["locs"])
        probs_a = pyro.param(
            "probs_a", params["probs_a"], constraint=constraints.simplex
        )
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("b_axis", size=3, subsample_size=2):
            pyro.sample("b", dist.Normal(locs[a], 1.0), obs=0)

        with pyro.plate("c_axis", size=6, subsample_size=3):
            pyro.sample("c", dist.Normal(locs[a], 1.0), obs=1)

    def guide(params):
        pass

    params = {
        "locs": jnp.array([0.0, 1.0]),
        "probs_a": jnp.array([0.4, 0.6]),
    }
    transform = dist.biject_to(dist.constraints.simplex)
    params_raw = {"locs": params["locs"], "probs_a": transform.inv(params["probs_a"])}

    elbo = infer.TraceEnum_ELBO()

    # Expected grads w/o subsampling
    def expected_loss_fn(params_raw):
        params = {
            "locs": params_raw["locs"],
            "probs_a": transform(params_raw["probs_a"]),
        }
        return elbo.loss(random.PRNGKey(0), {}, model, guide, params)

    expected_loss, expected_grads = jax.value_and_grad(expected_loss_fn)(params_raw)

    # Actual grads w/ subsampling
    def actual_loss_fn(params_raw):
        params = {
            "locs": params_raw["locs"],
            "probs_a": transform(params_raw["probs_a"]),
        }
        return elbo.loss(random.PRNGKey(0), {}, model_subsample, guide, params)

    with pytest.raises(
        ValueError, match="Expected all enumerated sample sites to share a common scale"
    ):
        # This never gets run because we don't support this yet.
        actual_loss, actual_grads = jax.value_and_grad(actual_loss_fn)(params_raw)

        assert_equal(actual_loss, expected_loss, prec=1e-5)
        assert_equal(actual_grads, expected_grads, prec=1e-5)


@pytest.mark.parametrize("scale", [1, 10])
def test_model_enum_subsample_3(scale):
    # Enumerate: a
    # Subsample: a, b, c
    # [ a - [----> b    ]
    # [  \  [           ]
    # [   - [- [-> c  ] ]
    @config_enumerate
    @handlers.scale(scale=scale)
    def model(params):
        locs = pyro.param("locs", params["locs"])
        probs_a = pyro.param(
            "probs_a", params["probs_a"], constraint=constraints.simplex
        )
        with pyro.plate("a_axis", size=3):
            a = pyro.sample("a", dist.Categorical(probs_a))
            with pyro.plate("b_axis", size=6):
                pyro.sample("b", dist.Normal(locs[a], 1.0), obs=0)
                with pyro.plate("c_axis", size=9):
                    pyro.sample("c", dist.Normal(locs[a], 1.0), obs=1)

    @config_enumerate
    @handlers.scale(scale=scale)
    def model_subsample(params):
        locs = pyro.param("locs", params["locs"])
        probs_a = pyro.param(
            "probs_a", params["probs_a"], constraint=constraints.simplex
        )
        with pyro.plate("a_axis", size=3, subsample_size=2):
            a = pyro.sample("a", dist.Categorical(probs_a))
            with pyro.plate("b_axis", size=6, subsample_size=3):
                pyro.sample("b", dist.Normal(locs[a], 1.0), obs=0)
                with pyro.plate("c_axis", size=9, subsample_size=4):
                    pyro.sample("c", dist.Normal(locs[a], 1.0), obs=1)

    def guide(params):
        pass

    params = {
        "locs": jnp.array([0.0, 1.0]),
        "probs_a": jnp.array([0.4, 0.6]),
    }
    transform = dist.biject_to(dist.constraints.simplex)
    params_raw = {"locs": params["locs"], "probs_a": transform.inv(params["probs_a"])}

    elbo = infer.TraceEnum_ELBO()

    # Expected grads w/o subsampling
    def expected_loss_fn(params_raw):
        params = {
            "locs": params_raw["locs"],
            "probs_a": transform(params_raw["probs_a"]),
        }
        return elbo.loss(random.PRNGKey(0), {}, model, guide, params)

    expected_loss, expected_grads = jax.value_and_grad(expected_loss_fn)(params_raw)

    # Actual grads w/ subsampling
    def actual_loss_fn(params_raw):
        params = {
            "locs": params_raw["locs"],
            "probs_a": transform(params_raw["probs_a"]),
        }
        return elbo.loss(random.PRNGKey(0), {}, model_subsample, guide, params)

    with pytest.raises(
        ValueError, match="Expected all enumerated sample sites to share a common scale"
    ):
        # This never gets run because we don't support this yet.
        actual_loss, actual_grads = jax.value_and_grad(actual_loss_fn)(params_raw)

        assert_equal(actual_loss, expected_loss, prec=1e-3)
        assert_equal(actual_grads, expected_grads, prec=1e-5)


def test_guide_plate_contraction():
    def model(params):
        with pyro.plate("a_axis", size=2):
            a = pyro.sample("a", dist.Poisson(jnp.array(3.0)))
        pyro.sample("b", dist.Normal(jnp.sum(a), 1.0), obs=1)

    def guide(params):
        probs_a = pyro.param(
            "probs_a", params["probs_a"], constraint=constraints.positive
        )
        with pyro.plate("a_axis", size=2):
            pyro.sample("a", dist.Poisson(probs_a))

    params = {
        "probs_a": jnp.array([3.0, 2.5]),
    }
    transform = dist.biject_to(dist.constraints.positive)
    params_raw = jax.tree.map(transform.inv, params)

    # TraceGraph_ELBO grads averaged over num_particles
    elbo = infer.TraceGraph_ELBO(num_particles=50_000)

    def graph_loss_fn(params_raw):
        params = jax.tree.map(transform, params_raw)
        return elbo.loss(random.PRNGKey(0), {}, model, guide, params)

    graph_loss, graph_grads = jax.value_and_grad(graph_loss_fn)(params_raw)

    # TraceEnum_ELBO grads averaged over num_particles (no enumeration)
    elbo = infer.TraceEnum_ELBO(num_particles=50_000)

    def enum_loss_fn(params_raw):
        params = jax.tree.map(transform, params_raw)
        return elbo.loss(random.PRNGKey(0), {}, model, guide, params)

    enum_loss, enum_grads = jax.value_and_grad(enum_loss_fn)(params_raw)

    assert_equal(enum_loss, graph_loss, prec=1e-3)
    assert_equal(enum_grads, graph_grads, prec=2e-2)
