# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from itertools import chain

import numpy as np
from numpy.testing import assert_allclose
import pytest
import scipy

from jax import numpy as jnp
from jax.tree_util import tree_flatten, tree_map

from numpyro.contrib.einstein.util import (
    batch_ravel_pytree,
    posdef,
    sqrth,
    sqrth_and_inv_sqrth,
)

pd_matrices = [
    np.array(
        [
            [3.37299503, -1.71077041, 1.82588055],  # positive definite
            [-1.71077041, 1.96674198, -0.73149742],
            [1.82588055, -0.73149742, 1.32398149],
        ]
    ),
    np.eye(2),
]
nd_matices = [
    np.array(
        [
            [-3.37299503, 1.71077041, -1.82588055],  # negative definite
            [1.71077041, -1.96674198, 0.73149742],
            [-1.82588055, 0.73149742, -1.32398149],
        ]
    ),
]
matrices = chain(pd_matrices, nd_matices)


@pytest.mark.parametrize("m", matrices)
def test_posdef(m):
    pd_m = posdef(m)
    assert jnp.alltrue(jnp.linalg.eigvals(pd_m) > 0)


@pytest.mark.parametrize("batch_shape", [(), (2,), (3, 1)])
def test_posdef_shape(batch_shape):
    dim = 4
    x = np.random.normal(size=batch_shape + (dim, dim + 1))
    m = x @ np.swapaxes(x, -2, -1)
    assert_allclose(posdef(m), m, rtol=1e-5)


@pytest.mark.parametrize("m", matrices)
def test_sqrth(m):
    assert_allclose(sqrth(m), scipy.linalg.sqrtm(posdef(m)), atol=1e-5)


@pytest.mark.parametrize("batch_shape", [(), (2,), (3, 1)])
def test_sqrth_shape(batch_shape):
    dim = 4
    x = np.random.normal(size=batch_shape + (dim, dim + 1))
    m = x @ np.swapaxes(x, -2, -1)
    s = sqrth(m)
    assert_allclose(s @ np.swapaxes(s, -2, -1), m, rtol=1e-5)


@pytest.mark.parametrize("m", pd_matrices)
def test_sqrt_inv_sqrth(m):
    msqrt, minv, minv_sqrt = sqrth_and_inv_sqrth(m)
    assert_allclose(msqrt, scipy.linalg.sqrtm(m), atol=1e-5)
    assert_allclose(minv, np.linalg.inv(m), atol=1e-4)
    assert_allclose(minv_sqrt, np.linalg.inv(scipy.linalg.sqrtm(m)), atol=1e-5)


@pytest.mark.parametrize("batch_shape", [(), (2,), (3, 1)])
def test_sqrth_and_inv_sqrth_shape(batch_shape):
    dim = 4
    x = np.random.normal(size=batch_shape + (dim, dim + 1))
    m = x @ np.swapaxes(x, -2, -1)
    s, i, si = sqrth_and_inv_sqrth(m)
    assert_allclose(s @ np.swapaxes(s, -2, -1), m, rtol=1e-5)
    assert_allclose(i, np.linalg.inv(m), rtol=1e-5)
    assert_allclose(si @ np.swapaxes(si, -2, -1), i, rtol=1e-5)


@pytest.mark.parametrize(
    "pytree",
    [
        {
            "a": np.array([[[1.0], [0.0], [3]], [[1], [0.0], [-1]]]),
        },
        {
            "b": np.array([[1.0, 0.0, 3], [1, 0.0, -1]]),
        },
        [
            np.array([[1.0, 0.0, 3], [1, 0.0, -1]]),
            np.array([[1.0, 0.0, 3], [1, 0.0, -1]]),
        ],
    ],
)
@pytest.mark.parametrize("nbatch_dims", [0, 1, 2])
def test_ravel_pytree_batched(pytree, nbatch_dims):
    flat, _, unravel_fn = batch_ravel_pytree(pytree, nbatch_dims)
    unravel = unravel_fn(flat)
    tree_flatten(tree_map(lambda x, y: assert_allclose(x, y), unravel, pytree))
    assert all(
        tree_flatten(
            tree_map(
                lambda x, y: jnp.result_type(x) == jnp.result_type(y), unravel, pytree
            )
        )[0]
    )
