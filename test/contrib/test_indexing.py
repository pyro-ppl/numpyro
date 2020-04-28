# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import itertools

import jax.lax as lax
import jax.numpy as np
import jax.random as random
import numpy as onp
import pytest

import numpyro.distributions as dist
from numpyro.contrib.indexing import Vindex


def z(*shape):
    return np.zeros(shape, dtype=np.int32)


SHAPE_EXAMPLES = [
    ('Vindex(z())[...]', ()),
    ('Vindex(z(2))[...]', (2,)),
    ('Vindex(z(2))[...,0]', ()),
    ('Vindex(z(2))[...,:]', (2,)),
    ('Vindex(z(2))[...,z(3)]', (3,)),
    ('Vindex(z(2))[0]', ()),
    ('Vindex(z(2))[:]', (2,)),
    ('Vindex(z(2))[z(3)]', (3,)),
    ('Vindex(z(2,3))[...]', (2, 3)),
    ('Vindex(z(2,3))[...,0]', (2,)),
    ('Vindex(z(2,3))[...,:]', (2, 3)),
    ('Vindex(z(2,3))[...,z(2)]', (2,)),
    ('Vindex(z(2,3))[...,z(4,1)]', (4, 2)),
    ('Vindex(z(2,3))[...,0,0]', ()),
    ('Vindex(z(2,3))[...,0,:]', (3,)),
    ('Vindex(z(2,3))[...,0,z(4)]', (4,)),
    ('Vindex(z(2,3))[...,:,0]', (2,)),
    ('Vindex(z(2,3))[...,:,:]', (2, 3)),
    ('Vindex(z(2,3))[...,:,z(4)]', (4, 2)),
    ('Vindex(z(2,3))[...,z(4),0]', (4,)),
    ('Vindex(z(2,3))[...,z(4),:]', (4, 3)),
    ('Vindex(z(2,3))[...,z(4),z(4)]', (4,)),
    ('Vindex(z(2,3))[...,z(5,1),z(4)]', (5, 4)),
    ('Vindex(z(2,3))[...,z(4),z(5,1)]', (5, 4)),
    ('Vindex(z(2,3))[0,0]', ()),
    ('Vindex(z(2,3))[0,:]', (3,)),
    ('Vindex(z(2,3))[0,z(4)]', (4,)),
    ('Vindex(z(2,3))[:,0]', (2,)),
    ('Vindex(z(2,3))[:,:]', (2, 3)),
    ('Vindex(z(2,3))[:,z(4)]', (4, 2)),
    ('Vindex(z(2,3))[z(4),0]', (4,)),
    ('Vindex(z(2,3))[z(4),:]', (4, 3)),
    ('Vindex(z(2,3))[z(4)]', (4, 3)),
    ('Vindex(z(2,3))[z(4),z(4)]', (4,)),
    ('Vindex(z(2,3))[z(5,1),z(4)]', (5, 4)),
    ('Vindex(z(2,3))[z(4),z(5,1)]', (5, 4)),
    ('Vindex(z(2,3,4))[...]', (2, 3, 4)),
    ('Vindex(z(2,3,4))[...,z(3)]', (2, 3)),
    ('Vindex(z(2,3,4))[...,z(2,1)]', (2, 3)),
    ('Vindex(z(2,3,4))[...,z(2,3)]', (2, 3)),
    ('Vindex(z(2,3,4))[...,z(5,1,1)]', (5, 2, 3)),
    ('Vindex(z(2,3,4))[...,z(2),0]', (2,)),
    ('Vindex(z(2,3,4))[...,z(5,1),0]', (5, 2)),
    ('Vindex(z(2,3,4))[...,z(2),:]', (2, 4)),
    ('Vindex(z(2,3,4))[...,z(5,1),:]', (5, 2, 4)),
    ('Vindex(z(2,3,4))[...,z(5),0,0]', (5,)),
    ('Vindex(z(2,3,4))[...,z(5),0,:]', (5, 4)),
    ('Vindex(z(2,3,4))[...,z(5),:,0]', (5, 3)),
    ('Vindex(z(2,3,4))[...,z(5),:,:]', (5, 3, 4)),
    ('Vindex(z(2,3,4))[0,0,z(5)]', (5,)),
    ('Vindex(z(2,3,4))[0,:,z(5)]', (5, 3)),
    ('Vindex(z(2,3,4))[0,z(5),0]', (5,)),
    ('Vindex(z(2,3,4))[0,z(5),:]', (5, 4)),
    ('Vindex(z(2,3,4))[0,z(5),z(5)]', (5,)),
    ('Vindex(z(2,3,4))[0,z(5,1),z(6)]', (5, 6)),
    ('Vindex(z(2,3,4))[0,z(6),z(5,1)]', (5, 6)),
    ('Vindex(z(2,3,4))[:,0,z(5)]', (5, 2)),
    ('Vindex(z(2,3,4))[:,:,z(5)]', (5, 2, 3)),
    ('Vindex(z(2,3,4))[:,z(5),0]', (5, 2)),
    ('Vindex(z(2,3,4))[:,z(5),:]', (5, 2, 4)),
    ('Vindex(z(2,3,4))[:,z(5),z(5)]', (5, 2)),
    ('Vindex(z(2,3,4))[:,z(5,1),z(6)]', (5, 6, 2)),
    ('Vindex(z(2,3,4))[:,z(6),z(5,1)]', (5, 6, 2)),
    ('Vindex(z(2,3,4))[z(5),0,0]', (5,)),
    ('Vindex(z(2,3,4))[z(5),0,:]', (5, 4)),
    ('Vindex(z(2,3,4))[z(5),:,0]', (5, 3)),
    ('Vindex(z(2,3,4))[z(5),:,:]', (5, 3, 4)),
    ('Vindex(z(2,3,4))[z(5),0,z(5)]', (5,)),
    ('Vindex(z(2,3,4))[z(5,1),0,z(6)]', (5, 6)),
    ('Vindex(z(2,3,4))[z(6),0,z(5,1)]', (5, 6)),
    ('Vindex(z(2,3,4))[z(5),:,z(5)]', (5, 3)),
    ('Vindex(z(2,3,4))[z(5,1),:,z(6)]', (5, 6, 3)),
    ('Vindex(z(2,3,4))[z(6),:,z(5,1)]', (5, 6, 3)),
]


@pytest.mark.parametrize('expression,expected_shape', SHAPE_EXAMPLES, ids=str)
def test_shape(expression, expected_shape):
    result = eval(expression)
    assert result.shape == expected_shape


@pytest.mark.parametrize('event_shape', [(), (7,)], ids=str)
@pytest.mark.parametrize('j_shape', [(), (2,), (3, 1), (4, 1, 1), (4, 3, 2)], ids=str)
@pytest.mark.parametrize('i_shape', [(), (2,), (3, 1), (4, 1, 1), (4, 3, 2)], ids=str)
@pytest.mark.parametrize('x_shape', [(), (2,), (3, 1), (4, 1, 1), (4, 3, 2)], ids=str)
def test_value(x_shape, i_shape, j_shape, event_shape):
    x = np.array(onp.random.rand(*(x_shape + (5, 6) + event_shape)))
    i = dist.Categorical(np.ones((5,))).sample(random.PRNGKey(1), i_shape)
    j = dist.Categorical(np.ones((6,))).sample(random.PRNGKey(2), j_shape)
    if event_shape:
        actual = Vindex(x)[..., i, j, :]
    else:
        actual = Vindex(x)[..., i, j]

    shape = lax.broadcast_shapes(x_shape, i_shape, j_shape)
    x = np.broadcast_to(x, shape + (5, 6) + event_shape)
    i = np.broadcast_to(i, shape)
    j = np.broadcast_to(j, shape)
    expected = onp.empty(shape + event_shape, dtype=x.dtype)
    for ind in (itertools.product(*map(range, shape)) if shape else [()]):
        expected[ind] = x[ind + (i[ind].item(), j[ind].item())]
    assert np.all(actual == np.array(expected, dtype=x.dtype))


@pytest.mark.parametrize('prev_enum_dim,curr_enum_dim', [(-3, -4), (-4, -5), (-5, -3)])
def test_hmm_example(prev_enum_dim, curr_enum_dim):
    hidden_dim = 8
    probs_x = np.array(onp.random.rand(hidden_dim, hidden_dim, hidden_dim))
    x_prev = np.arange(hidden_dim).reshape((-1,) + (1,) * (-1 - prev_enum_dim))
    x_curr = np.arange(hidden_dim).reshape((-1,) + (1,) * (-1 - curr_enum_dim))

    expected = probs_x[x_prev.reshape(x_prev.shape + (1,)),
                       x_curr.reshape(x_curr.shape + (1,)),
                       np.arange(hidden_dim)]

    actual = Vindex(probs_x)[x_prev, x_curr, :]
    assert np.all(actual == expected)
