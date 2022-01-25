# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer.inspect import get_dependencies


class NonreparameterizedNormal(dist.Normal):
    has_rsample = False


def test_get_dependencies():
    def model(data):
        a = numpyro.sample("a", dist.Normal(0, 1))
        b = numpyro.sample("b", NonreparameterizedNormal(a, 0))
        c = numpyro.sample("c", dist.Normal(b, 1))
        d = numpyro.sample("d", dist.Normal(a, jnp.exp(c)))

        e = numpyro.sample("e", dist.Normal(0, 1))
        f = numpyro.sample("f", dist.Normal(0, 1))
        g = numpyro.sample("g", dist.Bernoulli(logits=e + f), obs=0.0)

        with numpyro.plate("p", len(data)):
            d_ = jax.lax.stop_gradient(d)  # this results in a known failure
            h = numpyro.sample("h", dist.Normal(c, jnp.exp(d_)))
            i = numpyro.deterministic("i", h + 1)
            j = numpyro.sample("j", dist.Delta(h + 1), obs=h + 1)
            k = numpyro.sample("k", dist.Normal(a, jnp.exp(j)), obs=data)

        return [a, b, c, d, e, f, g, h, i, j, k]

    data = np.random.randn(3)
    actual = get_dependencies(model, (data,))
    _ = set()
    expected = {
        "prior_dependencies": {
            "a": {"a": _},
            "b": {"b": _, "a": _},
            "c": {"c": _, "b": _},
            "d": {"d": _, "c": _, "a": _},
            "e": {"e": _},
            "f": {"f": _},
            "g": {"g": _, "e": _, "f": _},
            "h": {"h": _, "c": _, "d": _},
            "k": {"k": _, "a": _, "h": _},
        },
        "posterior_dependencies": {
            "a": {"a": _, "b": _, "c": _, "d": _, "h": _, "k": _},
            "b": {"b": _, "c": _},
            "c": {"c": _, "d": _, "h": _},
            "d": {"d": _, "h": _},
            "e": {"e": _, "g": _, "f": _},
            "f": {"f": _, "g": _},
            "h": {"h": _, "k": _},
        },
    }
    assert actual == expected


def test_docstring_example_1():
    def model_1():
        a = numpyro.sample("a", dist.Normal(0, 1))
        numpyro.sample("b", dist.Normal(a, 1), obs=0.0)

    actual = get_dependencies(model_1)
    expected = {
        "prior_dependencies": {
            "a": {"a": set()},
            "b": {"a": set(), "b": set()},
        },
        "posterior_dependencies": {
            "a": {"a": set(), "b": set()},
        },
    }
    assert actual == expected


def test_docstring_example_2():
    def model_2():
        a = numpyro.sample("a", dist.Normal(0, 1))
        b = numpyro.sample("b", dist.LogNormal(0, 1))
        c = numpyro.sample("c", dist.Normal(a, b))
        numpyro.sample("d", dist.Normal(c, 1), obs=0.0)

    actual = get_dependencies(model_2)
    expected = {
        "prior_dependencies": {
            "a": {"a": set()},
            "b": {"b": set()},
            "c": {"a": set(), "b": set(), "c": set()},
            "d": {"c": set(), "d": set()},
        },
        "posterior_dependencies": {
            "a": {"a": set(), "b": set(), "c": set()},
            "b": {"b": set(), "c": set()},
            "c": {"c": set(), "d": set()},
        },
    }
    assert actual == expected


def test_docstring_example_3():
    def model_3():
        with numpyro.plate("p", 5):
            a = numpyro.sample("a", dist.Normal(0, 1))
        numpyro.sample("b", dist.Normal(a.sum(), 1), obs=0.0)

    actual = get_dependencies(model_3)
    expected = {
        "prior_dependencies": {
            "a": {"a": set()},
            "b": {"a": set(), "b": set()},
        },
        "posterior_dependencies": {
            "a": {"a": {"p"}, "b": set()},
        },
    }
    assert actual == expected


def test_factor():
    def model():
        a = numpyro.sample("a", dist.Normal(0, 1))
        numpyro.factor("b", 0.0)
        numpyro.factor("c", a)

    actual = get_dependencies(model)
    expected = {
        "prior_dependencies": {
            "a": {"a": set()},
            "b": {"b": set()},
            "c": {"c": set(), "a": set()},
        },
        "posterior_dependencies": {
            "a": {"a": set(), "c": set()},
        },
    }
    assert actual == expected


def test_discrete_obs():
    def model():
        a = numpyro.sample("a", dist.Normal(0, 1))
        b = numpyro.sample("b", dist.Normal(a[..., None], jnp.ones(3)).to_event(1))
        c = numpyro.sample(
            "c", dist.MultivariateNormal(jnp.zeros(3) + a[..., None], jnp.eye(3))
        )
        with numpyro.plate("i", 2):
            d = numpyro.sample("d", dist.Dirichlet(jnp.exp(b + c)))
            numpyro.sample("e", dist.Categorical(logits=d), obs=jnp.array([0, 0]))
        return a, b, c, d

    actual = get_dependencies(model)
    expected = {
        "prior_dependencies": {
            "a": {"a": set()},
            "b": {"a": set(), "b": set()},
            "c": {"a": set(), "c": set()},
            "d": {"b": set(), "c": set(), "d": set()},
            "e": {"d": set(), "e": set()},
        },
        "posterior_dependencies": {
            "a": {"a": set(), "b": set(), "c": set()},
            "b": {"b": set(), "c": set(), "d": set()},
            "c": {"c": set(), "d": set()},
            "d": {"d": set(), "e": set()},
        },
    }
    assert actual == expected


def test_discrete():
    def model():
        a = numpyro.sample("a", dist.Dirichlet(jnp.ones(3)))
        b = numpyro.sample("b", dist.Categorical(a))
        c = numpyro.sample("c", dist.Normal(jnp.zeros(3), 1).to_event(1))
        d = numpyro.sample("d", dist.Poisson(jnp.exp(c[b])))
        numpyro.sample("e", dist.Normal(d, 1), obs=jnp.ones(()))

    actual = get_dependencies(model)
    expected = {
        "prior_dependencies": {
            "a": {"a": set()},
            "b": {"a": set(), "b": set()},
            "c": {"c": set()},
            "d": {"b": set(), "c": set(), "d": set()},
            "e": {"d": set(), "e": set()},
        },
        "posterior_dependencies": {
            "a": {"a": set(), "b": set()},
            "b": {"b": set(), "c": set(), "d": set()},
            "c": {"c": set(), "d": set()},
            "d": {"d": set(), "e": set()},
        },
    }
    assert actual == expected


def test_plate_coupling():
    #   x  x
    #    ||
    #    y
    #
    # This results in posterior dependency structure:
    #
    #     x x y
    #   x ? ? ?
    #   x ? ? ?

    def model(data):
        with numpyro.plate("p", len(data)):
            x = numpyro.sample("x", dist.Normal(0, 1))
        numpyro.sample("y", dist.Normal(x.sum(), 1), obs=data.sum())

    data = np.random.randn(2)
    actual = get_dependencies(model, (data,))
    expected = {
        "prior_dependencies": {
            "x": {"x": set()},
            "y": {"y": set(), "x": set()},
        },
        "posterior_dependencies": {
            "x": {"x": {"p"}, "y": set()},
        },
    }
    assert actual == expected


def test_plate_coupling_2():
    #   x x
    #     \\   y y
    #      \\ //
    #        z
    #
    # This results in posterior dependency structure:
    #
    #     x x y y z
    #   x ? ? ? ? ?
    #   x ? ? ? ? ?
    #   y     ? ? ?
    #   y     ? ? ?

    def model(data):
        with numpyro.plate("p", len(data)):
            x = numpyro.sample("x", dist.Normal(0, 1))
            y = numpyro.sample("y", dist.Normal(0, 1))
        numpyro.sample("z", dist.Normal(x.sum(), jnp.exp(y.sum())), obs=data.sum())

    data = np.random.randn(2)
    actual = get_dependencies(model, (data,))
    expected = {
        "prior_dependencies": {
            "x": {"x": set()},
            "y": {"y": set()},
            "z": {"z": set(), "x": set(), "y": set()},
        },
        "posterior_dependencies": {
            "x": {"x": {"p"}, "y": {"p"}, "z": set()},
            "y": {"y": {"p"}, "z": set()},
        },
    }
    assert actual == expected


def test_plate_coupling_3():
    #    x x x x
    #     // \\
    #   y y   z z
    #
    # This results in posterior dependency structure:
    #
    #     x x y y z
    #   x ? ? ? ? ?
    #   x ? ? ? ? ?
    #   y     ? ? ?
    #   y     ? ? ?

    def model(data):
        i_plate = numpyro.plate("i", data.shape[0], dim=-2)
        j_plate = numpyro.plate("j", data.shape[1], dim=-1)
        with i_plate, j_plate:
            x = numpyro.sample("x", dist.Normal(0, 1))
        with i_plate:
            numpyro.sample(
                "y",
                dist.Normal(x.sum(-1, keepdims=True), 1),
                obs=data.sum(-1, keepdims=True),
            )
        with j_plate:
            numpyro.sample(
                "z",
                dist.Normal(x.sum(-2, keepdims=True), 1),
                obs=data.sum(-2, keepdims=True),
            )

    data = np.random.randn(3, 2)
    actual = get_dependencies(model, (data,))
    expected = {
        "prior_dependencies": {
            "x": {"x": set()},
            "y": {"y": set(), "x": set()},
            "z": {"z": set(), "x": set()},
        },
        "posterior_dependencies": {
            "x": {"x": {"i", "j"}, "y": set(), "z": set()},
        },
    }
    assert actual == expected


def test_plate_collider():
    #   x x    y y
    #     \\  //
    #      zzzz
    #
    # This results in posterior dependency structure:
    #
    #     x x y y z z z z
    #   x ?   ? ? ? ?
    #   x   ? ? ?     ? ?
    #   y     ?   ?   ?
    #   y       ?   ?   ?

    def model(data):
        i_plate = numpyro.plate("i", data.shape[0], dim=-2)
        j_plate = numpyro.plate("j", data.shape[1], dim=-1)

        with i_plate:
            x = numpyro.sample("x", dist.Normal(0, 1))
        with j_plate:
            y = numpyro.sample("y", dist.Normal(0, 1))
        with i_plate, j_plate:
            numpyro.sample("z", dist.Normal(x, jnp.exp(y)), obs=data)

    data = np.random.randn(3, 2)
    actual = get_dependencies(model, (data,))
    _ = set()
    expected = {
        "prior_dependencies": {
            "x": {"x": _},
            "y": {"y": _},
            "z": {"x": _, "y": _, "z": _},
        },
        "posterior_dependencies": {
            "x": {"x": _, "y": _, "z": _},
            "y": {"y": _, "z": _},
        },
    }
    assert actual == expected


def test_plate_dependency():
    #   w                              w
    #     \  x1 x2      unroll    x1  / \  x2
    #      \  || y1 y2  =====>  y1 | /   \ | y2
    #       \ || //               \|/     \|/
    #        z1 z2                z1       z2
    #
    # This allows posterior dependency structure:
    #
    #     w x x y y z z
    #   w ? ? ? ? ? ? ?
    #   x   ?   ?   ?
    #   x     ?   ?   ?
    #   y       ?   ?
    #   y         ?   ?

    def model(data):
        w = numpyro.sample("w", dist.Normal(0, 1))
        with numpyro.plate("p", len(data)):
            x = numpyro.sample("x", dist.Normal(0, 1))
            y = numpyro.sample("y", dist.Normal(0, 1))
            numpyro.sample("z", dist.Normal(w + x + y, 1), obs=data)

    data = np.random.rand(2)
    actual = get_dependencies(model, (data,))
    _ = set()
    expected = {
        "prior_dependencies": {
            "w": {"w": _},
            "x": {"x": _},
            "y": {"y": _},
            "z": {"w": _, "x": _, "y": _, "z": _},
        },
        "posterior_dependencies": {
            "w": {"w": _, "x": _, "y": _, "z": _},
            "x": {"x": _, "y": _, "z": _},
            "y": {"y": _, "z": _},
        },
    }
    assert actual == expected


def test_nested_plate_collider():
    # a a       b b
    #  a a     b b
    #    \\   //
    #      c c
    #       |
    #       d

    def model():
        plate_i = numpyro.plate("i", 2, dim=-1)
        plate_j = numpyro.plate("j", 3, dim=-2)
        plate_k = numpyro.plate("k", 3, dim=-2)

        with plate_i:
            with plate_j:
                a = numpyro.sample("a", dist.Normal(0, 1))
            with plate_k:
                b = numpyro.sample("b", dist.Normal(0, 1))
            c = numpyro.sample("c", dist.Normal(a.sum(0) + b.sum([0, 1]), 1))
        numpyro.sample("d", dist.Normal(c.sum(), 1), obs=jnp.zeros(()))

    actual = get_dependencies(model)
    _ = set()
    expected = {
        "prior_dependencies": {
            "a": {"a": _},
            "b": {"b": _},
            "c": {"c": _, "a": _, "b": _},
            "d": {"d": _, "c": _},
        },
        "posterior_dependencies": {
            "a": {"a": {"j"}, "b": _, "c": _},
            "b": {"b": {"k"}, "c": _},
            "c": {"c": {"i"}, "d": _},
        },
    }
    assert actual == expected
