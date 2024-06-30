# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numpy.testing import assert_allclose
import pytest

import jax
from jax import random
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.util import check_model_guide_match, fori_collect, format_shapes, soft_vmap


def test_fori_collect_thinning():
    def f(x):
        return x + 1

    actual2 = fori_collect(0, 9, f, np.array([-1]), thinning=2)
    expected2 = np.array([[2], [4], [6], [8]])
    assert_allclose(actual2, expected2)

    actual3 = fori_collect(0, 9, f, np.array([-1]), thinning=3)
    expected3 = np.array([[2], [5], [8]])
    assert_allclose(actual3, expected3)

    actual4 = fori_collect(0, 9, f, np.array([-1]), thinning=4)
    expected4 = np.array([[4], [8]])
    assert_allclose(actual4, expected4)

    actual5 = fori_collect(12, 37, f, np.array([-1]), thinning=5)
    expected5 = np.array([[16], [21], [26], [31], [36]])
    assert_allclose(actual5, expected5)


def test_fori_collect():
    def f(x):
        return {"i": x["i"] + x["j"], "j": x["i"] - x["j"]}

    a = {"i": np.array([0.0]), "j": np.array([1.0])}
    expected_tree = {"i": np.array([[0.0], [2.0]])}
    actual_tree = fori_collect(1, 3, f, a, transform=lambda a: {"i": a["i"]})

    jax.tree.all(jax.tree.map(assert_allclose, actual_tree, expected_tree))


@pytest.mark.parametrize("progbar", [False, True])
def test_fori_collect_return_last(progbar):
    def f(x):
        x["i"] = x["i"] + 1
        return x

    tree, init_state = fori_collect(
        2,
        4,
        f,
        {"i": 0},
        transform=lambda a: {"i": a["i"]},
        return_last_val=True,
        progbar=progbar,
    )
    expected_tree = {"i": np.array([3, 4])}
    expected_last_state = {"i": np.array(4)}
    jax.tree.all(jax.tree.map(assert_allclose, init_state, expected_last_state))
    jax.tree.all(jax.tree.map(assert_allclose, tree, expected_tree))


@pytest.mark.parametrize(
    "pytree",
    [
        {"a": np.array(0.0), "b": np.array([[1.0, 2.0], [3.0, 4.0]])},
        {"a": np.array(0), "b": np.array([[1, 2], [3, 4]])},
        {"a": np.array(0), "b": np.array([[1.0, 2.0], [3.0, 4.0]])},
        {"a": 0.0, "b": np.array([[1.0, 2.0], [3.0, 4.0]])},
        {"a": False, "b": np.array([[1.0, 2.0], [3.0, 4.0]])},
        [False, True, 0.0, np.array([[1.0, 2.0], [3.0, 4.0]])],
    ],
)
def test_ravel_pytree(pytree):
    flat, unravel_fn = ravel_pytree(pytree)
    unravel = unravel_fn(flat)
    jax.tree.flatten(jax.tree.map(lambda x, y: assert_allclose(x, y), unravel, pytree))
    assert all(
        jax.tree.flatten(
            jax.tree.map(
                lambda x, y: jnp.result_type(x) == jnp.result_type(y), unravel, pytree
            )
        )[0]
    )


@pytest.mark.parametrize("batch_shape", [(), (1,), (10,), (3, 4)])
@pytest.mark.parametrize("chunk_size", [None, 1, 5, 16])
def test_soft_vmap(batch_shape, chunk_size):
    def f(x):
        return {
            k: ((v[..., None] * jnp.ones(4)) if k == "a" else ~v) for k, v in x.items()
        }

    xs = {"a": jnp.ones(batch_shape + (4,)), "b": jnp.zeros(batch_shape).astype(bool)}
    ys = soft_vmap(f, xs, len(batch_shape), chunk_size)
    assert set(ys.keys()) == {"a", "b"}
    assert_allclose(ys["a"], xs["a"][..., None] * jnp.ones(4))
    assert_allclose(ys["b"], ~xs["b"])


def test_format_shapes():
    data = jnp.arange(100)

    def model_test():
        mean = numpyro.param("mean", jnp.zeros(len(data)))
        scale = numpyro.sample("scale", dist.Normal(0, 1).expand([3]).to_event(1))
        scale = scale.sum()
        with numpyro.plate("data", len(data), subsample_size=10) as ind:
            batch = data[ind]
            mean_batch = mean[ind]
            numpyro.sample("x", dist.Normal(mean_batch, scale), obs=batch)

    with numpyro.handlers.seed(rng_seed=0), numpyro.handlers.trace() as t:
        model_test()

    assert (
        format_shapes(t) == "Trace Shapes:         \n"
        " Param Sites:         \n"
        "         mean    100  \n"
        "Sample Sites:         \n"
        "   scale dist      | 3\n"
        "        value      | 3\n"
        "   data plate 10   |  \n"
        "       x dist 10   |  \n"
        "        value 10   |  "
    )
    assert (
        format_shapes(t, compute_log_prob=True) == "Trace Shapes:         \n"
        " Param Sites:         \n"
        "         mean    100  \n"
        "Sample Sites:         \n"
        "   scale dist      | 3\n"
        "        value      | 3\n"
        "     log_prob      |  \n"
        "   data plate 10   |  \n"
        "       x dist 10   |  \n"
        "        value 10   |  \n"
        "     log_prob 10   |  "
    )
    assert (
        format_shapes(t, compute_log_prob=lambda site: site["name"] == "scale")
        == "Trace Shapes:         \n"
        " Param Sites:         \n"
        "         mean    100  \n"
        "Sample Sites:         \n"
        "   scale dist      | 3\n"
        "        value      | 3\n"
        "     log_prob      |  \n"
        "   data plate 10   |  \n"
        "       x dist 10   |  \n"
        "        value 10   |  "
    )
    assert (
        format_shapes(t, last_site="data") == "Trace Shapes:         \n"
        " Param Sites:         \n"
        "         mean    100  \n"
        "Sample Sites:         \n"
        "   scale dist      | 3\n"
        "        value      | 3\n"
        "   data plate 10   |  "
    )


def _run_svi_check_warnings(model, guide, expected_string):
    with pytest.warns(UserWarning, match=expected_string) as ws:
        adam = numpyro.optim.Adam(1e-3)
        svi = numpyro.infer.SVI(model, guide, adam, numpyro.infer.Trace_ELBO())
        svi.run(random.PRNGKey(42), num_steps=5)
        assert len(ws) == 1
        assert expected_string in str(ws[0].message)


def _create_traces_check_error_string(model, guide, expected_string):
    model_trace = numpyro.handlers.trace(
        numpyro.handlers.seed(model, rng_seed=42)
    ).get_trace()
    guide_trace = numpyro.handlers.trace(
        numpyro.handlers.seed(guide, rng_seed=42)
    ).get_trace()
    with pytest.raises(ValueError, match=expected_string):
        check_model_guide_match(model_trace, guide_trace)


def test_check_model_guide_match():
    # 1. Auxiliary vars in the model
    def model():
        numpyro.sample("x", dist.Normal())

    def guide():
        numpyro.sample("x", dist.Normal(), infer={"is_auxiliary": True})

    _run_svi_check_warnings(model, guide, "Found auxiliary vars in the model")

    # 2. Non-auxiliary vars in guide but not model
    def model():
        numpyro.sample("x1", dist.Normal())

    def guide():
        numpyro.sample("x1", dist.Normal())
        numpyro.sample("x2", dist.Normal())

    _run_svi_check_warnings(
        model, guide, "Found non-auxiliary vars in guide but not model"
    )

    # 3. Vars in model but not guide
    def model():
        numpyro.sample("x1", dist.Normal())
        numpyro.sample("x2", dist.Normal())

    def guide():
        numpyro.sample("x1", dist.Normal())

    _run_svi_check_warnings(model, guide, "Found vars in model but not guide")

    # 4. Check event_dims agree
    def model():
        numpyro.sample("x", dist.MultivariateNormal(jnp.zeros(4), jnp.identity(4)))

    def guide():
        numpyro.sample("x", dist.Normal().expand((3, 5)))

    _create_traces_check_error_string(
        model, guide, "Model and guide event_dims disagree"
    )

    # 5. Check shapes agree
    def model():
        with numpyro.plate("a", 3, dim=-2):
            with numpyro.plate("b", 2, dim=-1):
                numpyro.sample("x", dist.Normal().expand((3, 2)))

    def guide():
        numpyro.sample("x", dist.Normal().expand((3, 5)))

    _create_traces_check_error_string(model, guide, "Model and guide shapes disagree")

    # 6. Check subsample sites introduced by plate
    def model():
        with numpyro.plate("a", 10):
            numpyro.sample("x", dist.Normal().expand((10,)))

    def guide():
        with numpyro.plate("data", 100, subsample_size=10):
            numpyro.sample("x", dist.Normal())

    _run_svi_check_warnings(
        model, guide, "Found plate statements in guide but not model"
    )


def test_missing_plate_in_model():
    def model():
        x = numpyro.sample("x", dist.Normal(0, 1))
        with numpyro.plate("N", 10, dim=-2):
            numpyro.sample("obs", dist.Normal(x, 1), obs=jnp.ones((10, 2)))

    def guide():
        numpyro.sample("x", dist.Normal(0, 1))

    _run_svi_check_warnings(model, guide, "Missing a plate statement")
