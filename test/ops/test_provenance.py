# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import inspect

import pytest

import jax
from jax.api_util import flatten_fun_nokwargs
import jax.core as core

try:
    import jax.extend.linear_util as lu
except ImportError:
    import jax.linear_util as lu
import jax.numpy as jnp

from numpyro.ops.provenance import eval_provenance


@pytest.mark.parametrize(
    "f, inputs, expected_output",
    [
        (lambda a, b: a + 1, ("a", "b"), {"a"}),
        (lambda a, b: jax.scipy.special.xlogy(a, b), ("a", "b"), {"a", "b"}),
        (lambda a, b, c: a + b, ("a", "b", "c"), {"a", "b"}),
        (
            lambda a, b: {"sum": a + b, "zero": 0},
            ("a", "b"),
            {"sum": {"a", "b"}, "zero": set()},
        ),
    ],
)
def test_provenance(f, inputs, expected_output):
    inputs = {p: 0 for p in inspect.getfullargspec(f).args}
    assert eval_provenance(f, **inputs) == expected_output


def test_provenance_const():
    def f(x):
        with jax.ensure_compile_time_eval():
            y = jnp.ones(4)
        return x + y

    jaxpr = jax.make_jaxpr(f)(jnp.zeros((3, 4), jnp.float32))
    assert len(jaxpr.consts) == 1
    assert eval_provenance(f, x=3) == {"x"}


def test_provenance_fori():
    def f(x, y, z):
        del z
        return jax.lax.fori_loop(0, 5, lambda _, x: x + y, x)

    assert eval_provenance(f, x=3, y=2, z=1) == {"x", "y"}


def test_provenance_vmap():
    def f(x, y):
        del x
        return jax.vmap(jnp.sin)(y)

    assert eval_provenance(f, x=3, y=jnp.ones(3)) == {"y"}


def test_provenance_pytree_in():
    def f(x, y):
        return x["v"] * y, x["u"]

    assert eval_provenance(f, x={"v": 2, "u": 1}, y=1) == ({"x", "y"}, {"x"})


def test_provenance_call():
    def identity(x):
        args, in_tree = jax.tree.flatten((x,))
        fn, out_tree = flatten_fun_nokwargs(lu.wrap_init(lambda x: x), in_tree)
        out = core.closed_call_p.bind(fn, *args)
        return jax.tree.unflatten(out_tree(), out)

    assert eval_provenance(identity, x={"v": 2}) == {"v": frozenset({"x"})}


def test_provenance_closed_call():
    def identity(x):
        args, in_tree = jax.tree.flatten((x,))
        fn, out_tree = flatten_fun_nokwargs(lu.wrap_init(lambda x: x), in_tree)
        out = core.closed_call_p.bind(fn, *args)
        return jax.tree.unflatten(out_tree(), out)

    assert eval_provenance(identity, x={"v": 2}) == {"v": frozenset({"x"})}
