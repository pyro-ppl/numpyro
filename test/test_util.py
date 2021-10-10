# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import warnings

from numpy.testing import assert_allclose
import pytest

from jax import random
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
from jax.test_util import check_eq
from jax.tree_util import tree_flatten, tree_multimap

import numpyro
import numpyro.distributions as dist
from numpyro.util import check_model_guide_match, fori_collect, format_shapes, soft_vmap


def test_fori_collect_thinning():
    def f(x):
        return x + 1.0

    actual2 = fori_collect(0, 9, f, jnp.array([-1]), thinning=2)
    expected2 = jnp.array([[2], [4], [6], [8]])
    check_eq(actual2, expected2)

    actual3 = fori_collect(0, 9, f, jnp.array([-1]), thinning=3)
    expected3 = jnp.array([[2], [5], [8]])
    check_eq(actual3, expected3)

    actual4 = fori_collect(0, 9, f, jnp.array([-1]), thinning=4)
    expected4 = jnp.array([[4], [8]])
    check_eq(actual4, expected4)

    actual5 = fori_collect(12, 37, f, jnp.array([-1]), thinning=5)
    expected5 = jnp.array([[16], [21], [26], [31], [36]])
    check_eq(actual5, expected5)


def test_fori_collect():
    def f(x):
        return {"i": x["i"] + x["j"], "j": x["i"] - x["j"]}

    a = {"i": jnp.array([0.0]), "j": jnp.array([1.0])}
    expected_tree = {"i": jnp.array([[0.0], [2.0]])}
    actual_tree = fori_collect(1, 3, f, a, transform=lambda a: {"i": a["i"]})
    check_eq(actual_tree, expected_tree)


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
    expected_tree = {"i": jnp.array([3, 4])}
    expected_last_state = {"i": jnp.array(4)}
    check_eq(init_state, expected_last_state)
    check_eq(tree, expected_tree)


@pytest.mark.parametrize(
    "pytree",
    [
        {"a": jnp.array(0.0), "b": jnp.array([[1.0, 2.0], [3.0, 4.0]])},
        {"a": jnp.array(0), "b": jnp.array([[1, 2], [3, 4]])},
        {"a": jnp.array(0), "b": jnp.array([[1.0, 2.0], [3.0, 4.0]])},
        {"a": 0.0, "b": jnp.array([[1.0, 2.0], [3.0, 4.0]])},
        {"a": False, "b": jnp.array([[1.0, 2.0], [3.0, 4.0]])},
        [False, True, 0.0, jnp.array([[1.0, 2.0], [3.0, 4.0]])],
    ],
)
def test_ravel_pytree(pytree):
    flat, unravel_fn = ravel_pytree(pytree)
    unravel = unravel_fn(flat)
    tree_flatten(tree_multimap(lambda x, y: assert_allclose(x, y), unravel, pytree))
    assert all(
        tree_flatten(
            tree_multimap(
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


def test_check_model_guide_match():
    def _run_svi(model, guide):
        adam = numpyro.optim.Adam(1e-3)
        svi = numpyro.infer.SVI(model, guide, adam, numpyro.infer.Trace_ELBO())
        svi.run(random.PRNGKey(42), num_steps=50)

    def _assert_single_warning_and_contains_string(ws, s):
        assert len(ws) == 1
        assert issubclass(ws[0].category, UserWarning)
        assert s in str(ws[0].message)

    def _run_svi_check_warnings(model, guide, expected_string):
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            _run_svi(model, guide)
            _assert_single_warning_and_contains_string(ws, expected_string)

    def _create_traces_check_error_string(model, guide, expected_string):
        model_trace = numpyro.handlers.trace(
            numpyro.handlers.seed(model, rng_seed=42)
        ).get_trace()
        guide_trace = numpyro.handlers.trace(
            numpyro.handlers.seed(guide, rng_seed=42)
        ).get_trace()
        with pytest.raises(ValueError, match=expected_string):
            check_model_guide_match(model_trace, guide_trace)

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
        numpyro.sample("x", dist.Normal().expand((3, 2)))

    def guide():
        numpyro.sample("x", dist.Normal().expand((3, 5)))

    _create_traces_check_error_string(model, guide, "Model and guide shapes disagree")

    # 6. Check subsample sites introduced by plate
    def model():
        numpyro.sample("x", dist.Normal().expand((10,)))

    def guide():
        with numpyro.handlers.plate("data", 100, subsample_size=10):
            numpyro.sample("x", dist.Normal())

    _run_svi_check_warnings(
        model, guide, "Found plate statements in guide but not model"
    )
