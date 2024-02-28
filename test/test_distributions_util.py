# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from numbers import Number

import numpy as np
from numpy.testing import assert_allclose
import pytest
import scipy

from jax import lax, random, vmap
import jax.numpy as jnp
from jax.scipy.special import expit, xlog1py, xlogy

from numpyro.distributions.util import (
    add_diag,
    binary_cross_entropy_with_logits,
    binomial,
    categorical,
    cholesky_update,
    multinomial,
    safe_normalize,
    vec_to_tril_matrix,
    von_mises_centered,
)


@pytest.mark.parametrize("x, y", [(0.2, 10.0), (0.6, -10.0)])
def test_binary_cross_entropy_with_logits(x, y):
    actual = -y * jnp.log(expit(x)) - (1 - y) * jnp.log(expit(-x))
    expect = binary_cross_entropy_with_logits(x, y)
    assert_allclose(actual, expect, rtol=1e-6)


@pytest.mark.parametrize("prim", [xlogy, xlog1py])
def test_binop_batch_rule(prim):
    bx = np.array([1.0, 2.0, 3.0])
    by = np.array([2.0, 3.0, 4.0])
    x = np.array(1.0)
    y = np.array(2.0)

    actual_bx_by = vmap(lambda x, y: prim(x, y))(bx, by)
    for i in range(3):
        assert_allclose(actual_bx_by[i], prim(bx[i], by[i]))

    actual_x_by = vmap(lambda y: prim(x, y))(by)
    for i in range(3):
        assert_allclose(actual_x_by[i], prim(x, by[i]))

    actual_bx_y = vmap(lambda x: prim(x, y))(bx)
    for i in range(3):
        assert_allclose(actual_bx_y[i], prim(bx[i], y))


@pytest.mark.parametrize(
    "p, shape",
    [
        (np.array([0.1, 0.9]), ()),
        (np.array([0.2, 0.8]), (2,)),
        (np.array([[0.1, 0.9], [0.2, 0.8]]), ()),
        (np.array([[0.1, 0.9], [0.2, 0.8]]), (3, 2)),
    ],
)
def test_categorical_shape(p, shape):
    rng_key = random.PRNGKey(0)
    expected_shape = lax.broadcast_shapes(p.shape[:-1], shape)
    assert jnp.shape(categorical(rng_key, p, shape)) == expected_shape


@pytest.mark.parametrize("p", [np.array([0.2, 0.3, 0.5]), np.array([0.8, 0.1, 0.1])])
def test_categorical_stats(p):
    rng_key = random.PRNGKey(0)
    n = 10000
    z = categorical(rng_key, p, (n,))
    _, counts = np.unique(z, return_counts=True)
    assert_allclose(counts / float(n), p, atol=0.01)


@pytest.mark.parametrize(
    "p, shape",
    [
        (np.array([0.1, 0.9]), ()),
        (np.array([0.2, 0.8]), (2,)),
        (np.array([[0.1, 0.9], [0.2, 0.8]]), ()),
        (np.array([[0.1, 0.9], [0.2, 0.8]]), (3, 2)),
    ],
)
def test_multinomial_shape(p, shape):
    rng_key = random.PRNGKey(0)
    n = 10000
    expected_shape = lax.broadcast_shapes(p.shape[:-1], shape) + p.shape[-1:]
    assert jnp.shape(multinomial(rng_key, p, n, shape)) == expected_shape


@pytest.mark.parametrize("n", [0, 1, np.array([0, 0]), np.array([2, 1, 0])])
@pytest.mark.parametrize("device_array", [True, False])
def test_multinomial_inhomogeneous(n, device_array):
    if device_array:
        n = jnp.asarray(n)

    p = np.array([0.5, 0.5])
    x = multinomial(random.PRNGKey(0), p, n)
    assert x.shape == jnp.shape(n) + jnp.shape(p)
    assert_allclose(x.sum(-1), n)


@pytest.mark.parametrize("p", [np.array([0.2, 0.3, 0.5]), np.array([0.8, 0.1, 0.1])])
@pytest.mark.parametrize("n", [10000, np.array([10000, 20000])])
def test_multinomial_stats(p, n):
    rng_key = random.PRNGKey(0)
    z = multinomial(rng_key, p, n)
    n = float(n) if isinstance(n, Number) else jnp.expand_dims(n.astype(p.dtype), -1)
    p = jnp.broadcast_to(p, z.shape)
    assert_allclose(z / n, p, atol=0.01)


@pytest.mark.parametrize("shape", [(6,), (5, 10), (3, 4, 3)])
@pytest.mark.parametrize("diagonal", [0, -1, -2])
def test_vec_to_tril_matrix(shape, diagonal):
    rng_key = random.PRNGKey(0)
    x = random.normal(rng_key, shape)
    actual = vec_to_tril_matrix(x, diagonal)
    expected = np.zeros(shape[:-1] + actual.shape[-2:])
    tril_idxs = np.tril_indices(expected.shape[-1], diagonal)
    expected[..., tril_idxs[0], tril_idxs[1]] = x
    assert_allclose(actual, expected)


@pytest.mark.parametrize("chol_batch_shape", [(), (3,)])
@pytest.mark.parametrize("vec_batch_shape", [(), (3,)])
@pytest.mark.parametrize("dim", [1, 4])
@pytest.mark.parametrize("coef", [1, -1])
def test_cholesky_update(chol_batch_shape, vec_batch_shape, dim, coef):
    A = random.normal(random.PRNGKey(0), chol_batch_shape + (dim, dim))
    A = A @ jnp.swapaxes(A, -2, -1) + jnp.eye(dim)
    x = random.normal(random.PRNGKey(0), vec_batch_shape + (dim,)) * 0.1
    xxt = x[..., None] @ x[..., None, :]
    expected = jnp.linalg.cholesky(A + coef * xxt)
    actual = cholesky_update(jnp.linalg.cholesky(A), x, coef)
    assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("n", [10, 100, 1000])
@pytest.mark.parametrize("p", [0.0, 0.01, 0.05, 0.3, 0.5, 0.7, 0.95, 1.0])
def test_binomial_mean(n, p):
    samples = binomial(random.PRNGKey(1), p, n, shape=(100, 100)).astype(np.float32)
    expected_mean = n * p
    assert_allclose(jnp.mean(samples), expected_mean, rtol=0.05)


@pytest.mark.parametrize("concentration", [1, 10, 100])
def test_von_mises_centered(concentration):
    samples = von_mises_centered(random.PRNGKey(0), concentration, shape=(10000,))
    cdf = scipy.stats.vonmises(kappa=concentration).cdf
    assert scipy.stats.kstest(samples, cdf).pvalue > 0.01


@pytest.mark.parametrize("dim", [2, 3, 4, 5])
def test_safe_normalize(dim):
    data = random.normal(random.PRNGKey(0), (100, dim))
    x = safe_normalize(data)
    assert_allclose((x * x).sum(-1), jnp.ones(x.shape[:-1]), rtol=1e-6)
    assert_allclose((x * data).sum(-1) ** 2, (data * data).sum(-1), rtol=1e-6)

    data = jnp.zeros((10, dim))
    x = safe_normalize(data)
    assert_allclose((x * x).sum(-1), jnp.ones(x.shape[:-1]), rtol=1e-6)


@pytest.mark.parametrize(
    "matrix_shape, diag_shape",
    [
        ((5, 5), ()),
        ((7, 7), (7,)),
        ((10, 3, 3), (10, 3)),
        ((7, 5, 9, 9), (5, 1)),
    ],
)
def test_add_diag(matrix_shape: tuple, diag_shape: tuple) -> None:
    matrix = random.normal(random.key(0), matrix_shape)
    diag = random.normal(random.key(1), diag_shape)
    expected = matrix + diag[..., None] * jnp.eye(matrix.shape[-1])
    actual = add_diag(matrix, diag)
    np.testing.assert_allclose(actual, expected)
