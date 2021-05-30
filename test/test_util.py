# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from numpy.testing import assert_allclose
import pytest

from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
from jax.test_util import check_eq
from jax.tree_util import tree_flatten, tree_multimap

from numpyro.util import fori_collect, soft_vmap


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
