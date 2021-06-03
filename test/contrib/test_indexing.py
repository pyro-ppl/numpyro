# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import itertools

import numpy as np
import pytest

import jax.lax as lax
import jax.numpy as jnp
import jax.random as random

from numpyro.contrib.indexing import Vindex
import numpyro.distributions as dist


def z(*shape):
    return jnp.zeros(shape, dtype=jnp.int32)


SHAPE_EXAMPLES = [
    ("Vindex(z())[...]", ()),
    ("Vindex(z(2))[...]", (2,)),
    ("Vindex(z(2))[...,0]", ()),
    ("Vindex(z(2))[...,:]", (2,)),
    ("Vindex(z(2))[...,z(3)]", (3,)),
    ("Vindex(z(2))[0]", ()),
    ("Vindex(z(2))[:]", (2,)),
    ("Vindex(z(2))[z(3)]", (3,)),
    ("Vindex(z(2,3))[...]", (2, 3)),
    ("Vindex(z(2,3))[...,0]", (2,)),
    ("Vindex(z(2,3))[...,:]", (2, 3)),
    ("Vindex(z(2,3))[...,z(2)]", (2,)),
    ("Vindex(z(2,3))[...,z(4,1)]", (4, 2)),
    ("Vindex(z(2,3))[...,0,0]", ()),
    ("Vindex(z(2,3))[...,0,:]", (3,)),
    ("Vindex(z(2,3))[...,0,z(4)]", (4,)),
    ("Vindex(z(2,3))[...,:,0]", (2,)),
    ("Vindex(z(2,3))[...,:,:]", (2, 3)),
    ("Vindex(z(2,3))[...,:,z(4)]", (4, 2)),
    ("Vindex(z(2,3))[...,z(4),0]", (4,)),
    ("Vindex(z(2,3))[...,z(4),:]", (4, 3)),
    ("Vindex(z(2,3))[...,z(4),z(4)]", (4,)),
    ("Vindex(z(2,3))[...,z(5,1),z(4)]", (5, 4)),
    ("Vindex(z(2,3))[...,z(4),z(5,1)]", (5, 4)),
    ("Vindex(z(2,3))[0,0]", ()),
    ("Vindex(z(2,3))[0,:]", (3,)),
    ("Vindex(z(2,3))[0,z(4)]", (4,)),
    ("Vindex(z(2,3))[:,0]", (2,)),
    ("Vindex(z(2,3))[:,:]", (2, 3)),
    ("Vindex(z(2,3))[:,z(4)]", (4, 2)),
    ("Vindex(z(2,3))[z(4),0]", (4,)),
    ("Vindex(z(2,3))[z(4),:]", (4, 3)),
    ("Vindex(z(2,3))[z(4)]", (4, 3)),
    ("Vindex(z(2,3))[z(4),z(4)]", (4,)),
    ("Vindex(z(2,3))[z(5,1),z(4)]", (5, 4)),
    ("Vindex(z(2,3))[z(4),z(5,1)]", (5, 4)),
    ("Vindex(z(2,3,4))[...]", (2, 3, 4)),
    ("Vindex(z(2,3,4))[...,z(3)]", (2, 3)),
    ("Vindex(z(2,3,4))[...,z(2,1)]", (2, 3)),
    ("Vindex(z(2,3,4))[...,z(2,3)]", (2, 3)),
    ("Vindex(z(2,3,4))[...,z(5,1,1)]", (5, 2, 3)),
    ("Vindex(z(2,3,4))[...,z(2),0]", (2,)),
    ("Vindex(z(2,3,4))[...,z(5,1),0]", (5, 2)),
    ("Vindex(z(2,3,4))[...,z(2),:]", (2, 4)),
    ("Vindex(z(2,3,4))[...,z(5,1),:]", (5, 2, 4)),
    ("Vindex(z(2,3,4))[...,z(5),0,0]", (5,)),
    ("Vindex(z(2,3,4))[...,z(5),0,:]", (5, 4)),
    ("Vindex(z(2,3,4))[...,z(5),:,0]", (5, 3)),
    ("Vindex(z(2,3,4))[...,z(5),:,:]", (5, 3, 4)),
    ("Vindex(z(2,3,4))[0,0,z(5)]", (5,)),
    ("Vindex(z(2,3,4))[0,:,z(5)]", (5, 3)),
    ("Vindex(z(2,3,4))[0,z(5),0]", (5,)),
    ("Vindex(z(2,3,4))[0,z(5),:]", (5, 4)),
    ("Vindex(z(2,3,4))[0,z(5),z(5)]", (5,)),
    ("Vindex(z(2,3,4))[0,z(5,1),z(6)]", (5, 6)),
    ("Vindex(z(2,3,4))[0,z(6),z(5,1)]", (5, 6)),
    ("Vindex(z(2,3,4))[:,0,z(5)]", (5, 2)),
    ("Vindex(z(2,3,4))[:,:,z(5)]", (5, 2, 3)),
    ("Vindex(z(2,3,4))[:,z(5),0]", (5, 2)),
    ("Vindex(z(2,3,4))[:,z(5),:]", (5, 2, 4)),
    ("Vindex(z(2,3,4))[:,z(5),z(5)]", (5, 2)),
    ("Vindex(z(2,3,4))[:,z(5,1),z(6)]", (5, 6, 2)),
    ("Vindex(z(2,3,4))[:,z(6),z(5,1)]", (5, 6, 2)),
    ("Vindex(z(2,3,4))[z(5),0,0]", (5,)),
    ("Vindex(z(2,3,4))[z(5),0,:]", (5, 4)),
    ("Vindex(z(2,3,4))[z(5),:,0]", (5, 3)),
    ("Vindex(z(2,3,4))[z(5),:,:]", (5, 3, 4)),
    ("Vindex(z(2,3,4))[z(5),0,z(5)]", (5,)),
    ("Vindex(z(2,3,4))[z(5,1),0,z(6)]", (5, 6)),
    ("Vindex(z(2,3,4))[z(6),0,z(5,1)]", (5, 6)),
    ("Vindex(z(2,3,4))[z(5),:,z(5)]", (5, 3)),
    ("Vindex(z(2,3,4))[z(5,1),:,z(6)]", (5, 6, 3)),
    ("Vindex(z(2,3,4))[z(6),:,z(5,1)]", (5, 6, 3)),
]


@pytest.mark.parametrize("expression,expected_shape", SHAPE_EXAMPLES, ids=str)
def test_shape(expression, expected_shape):
    result = eval(expression)
    assert result.shape == expected_shape


@pytest.mark.parametrize("event_shape", [(), (7,)], ids=str)
@pytest.mark.parametrize("j_shape", [(), (2,), (3, 1), (4, 1, 1), (4, 3, 2)], ids=str)
@pytest.mark.parametrize("i_shape", [(), (2,), (3, 1), (4, 1, 1), (4, 3, 2)], ids=str)
@pytest.mark.parametrize("x_shape", [(), (2,), (3, 1), (4, 1, 1), (4, 3, 2)], ids=str)
def test_value(x_shape, i_shape, j_shape, event_shape):
    x = jnp.array(np.random.rand(*(x_shape + (5, 6) + event_shape)))
    i = dist.Categorical(jnp.ones((5,))).sample(random.PRNGKey(1), i_shape)
    j = dist.Categorical(jnp.ones((6,))).sample(random.PRNGKey(2), j_shape)
    if event_shape:
        actual = Vindex(x)[..., i, j, :]
    else:
        actual = Vindex(x)[..., i, j]

    shape = lax.broadcast_shapes(x_shape, i_shape, j_shape)
    x = jnp.broadcast_to(x, shape + (5, 6) + event_shape)
    i = jnp.broadcast_to(i, shape)
    j = jnp.broadcast_to(j, shape)
    expected = np.empty(shape + event_shape, dtype=x.dtype)
    for ind in itertools.product(*map(range, shape)) if shape else [()]:
        expected[ind] = x[ind + (i[ind].item(), j[ind].item())]
    assert jnp.all(actual == jnp.array(expected, dtype=x.dtype))


@pytest.mark.parametrize("prev_enum_dim,curr_enum_dim", [(-3, -4), (-4, -5), (-5, -3)])
def test_hmm_example(prev_enum_dim, curr_enum_dim):
    hidden_dim = 8
    probs_x = jnp.array(np.random.rand(hidden_dim, hidden_dim, hidden_dim))
    x_prev = jnp.arange(hidden_dim).reshape((-1,) + (1,) * (-1 - prev_enum_dim))
    x_curr = jnp.arange(hidden_dim).reshape((-1,) + (1,) * (-1 - curr_enum_dim))

    expected = probs_x[
        x_prev.reshape(x_prev.shape + (1,)),
        x_curr.reshape(x_curr.shape + (1,)),
        jnp.arange(hidden_dim),
    ]

    actual = Vindex(probs_x)[x_prev, x_curr, :]
    assert jnp.all(actual == expected)
