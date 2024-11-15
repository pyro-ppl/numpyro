# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import reduce
from operator import mul

import numpy as np
import pytest

import jax.numpy as jnp
from jax.typing import ArrayLike

from numpyro.contrib.hsgp.laplacian import (
    _convert_ell,
    eigenfunctions,
    eigenindices,
    sqrt_eigenvalues,
)


@pytest.mark.parametrize(
    argnames="m, dim, xfail",
    argvalues=[
        (1, 1, False),
        (2, 1, False),
        (10, 1, False),
        (100, 1, False),
        (10, 2, False),
        ([2, 2, 3], 3, False),
        (2, 3, False),
        ([2, 2, 3], 2, True),
    ],
    ids=[
        "m=1",
        "m=2",
        "m=10",
        "m=100",
        "m=10,d=2",
        "m=[2,2,3],d=3",
        "m=2,d=3",
        "m=[2,2,3],d=2",
    ],
)
def test_eigenindices(m, dim, xfail):
    if xfail:
        with pytest.raises(ValueError):
            S = eigenindices(m, dim)
    else:
        S = eigenindices(m, dim)
        if isinstance(m, int):
            m_ = [m] * dim
        else:
            m_ = m
        m_star = reduce(mul, m_)
        assert str(S.dtype)[0:3] == "int"  # matrix is integer-valued
        assert S.shape == (dim, m_star)  # matrix has the right shape
        assert jnp.all(S >= 1)  # indices are greater than or equal to one
        assert jnp.all(S <= m_star)  # maximum possible index value is m_star
        if m == [2, 2, 3]:  # eq 10 in Riutort-Mayol et al
            assert (
                S
                == jnp.array(
                    [
                        [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
                        [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
                        [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
                    ]
                )
            ).all()


@pytest.mark.parametrize(
    argnames="ell, m, dim",
    argvalues=[
        (0.1, 1, 1),
        (0.2, 2, 1),
        (0.3, 10, 1),
        (0.1, 100, 1),
        (0.1, 10, 2),
        (0.1, [2, 2, 3], 3),
    ],
    ids=["m=1", "m=2", "m=10", "m=100", "m=10,d=2", "m=[2,2,3],d=3"],
)
def test_sqrt_eigenvalues(ell: float | int, m: int | list[int], dim: int):
    sqrt_eigenvalues_ = sqrt_eigenvalues(ell=ell, m=m, dim=dim)
    diff_sqrt_eigenvalues = jnp.diff(sqrt_eigenvalues_)
    if isinstance(m, int):
        m = [m] * dim
    assert sqrt_eigenvalues_.shape == (dim, reduce(mul, m))
    assert jnp.all(sqrt_eigenvalues_ > 0.0)
    if dim == 1:
        assert jnp.all(diff_sqrt_eigenvalues > 0.0)


@pytest.mark.parametrize(
    argnames="x, ell, m",
    argvalues=[
        (np.linspace(0, 1, 10), 1, 1),
        (np.linspace(-1, 1, 10), 1, 21),
        (np.linspace(-2, -1, 10), 2, 10),
        (np.linspace(0, 100, 500), 120, 100),
        (np.linspace(np.zeros(3), np.ones(3), 10), 2, [2, 2, 3]),
        (
            np.linspace(np.zeros(3), np.ones(3), 100).reshape((10, 10, 3)),
            2,
            [2, 2, 3],
        ),
    ],
    ids=["x_pos", "x_contains_zero", "x_neg2", "x_pos2-large", "x_2d", "x_batch"],
)
def test_eigenfunctions(x: ArrayLike, ell: float | int, m: int | list[int]):
    phi = eigenfunctions(x=x, ell=ell, m=m)
    if isinstance(m, int):
        m = [m]
    if x.ndim == 1:
        x_ = x[..., None]
    else:
        x_ = x
    assert phi.shape == x_.shape[:-1] + (reduce(mul, m),)
    assert phi.max() <= 1.0
    assert phi.min() >= -1.0


@pytest.mark.parametrize(
    argnames="ell, dim, xfail",
    argvalues=[
        (1.0, 1, False),
        (1, 1, False),
        (1, 2, False),
        ([1, 1], 2, False),
        (np.array([1, 1])[..., None], 2, False),
        (jnp.array([1, 1])[..., None], 2, False),
        (np.array([1, 1]), 2, True),
        (jnp.array([1, 1]), 2, True),
        ([1, 1], 1, True),
        (np.array([1, 1]), 1, True),
        (jnp.array([1, 1]), 1, True),
    ],
    ids=[
        "ell-float",
        "ell-int",
        "ell-int-multdim",
        "ell-list",
        "ell-array",
        "ell-jax-array",
        "ell-array-fail",
        "ell-jax-array-fail",
        "dim-fail",
        "dim-fail-array",
        "dim-fail-jax",
    ],
)
def test_convert_ell(ell, dim, xfail):
    if xfail:
        with pytest.raises(ValueError):
            _convert_ell(ell, dim)
    else:
        assert (_convert_ell(ell, dim) == jnp.array([1.0] * dim)[..., None]).all()
