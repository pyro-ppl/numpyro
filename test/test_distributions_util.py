from numbers import Number

import numpy as onp
from numpy.testing import assert_allclose
import pytest

from jax import jacobian, lax, random, vmap
import jax.numpy as np
from jax.scipy.special import expit, xlog1py, xlogy

from numpyro.distributions.util import (
    binary_cross_entropy_with_logits,
    categorical,
    cholesky_update,
    cumprod,
    cumsum,
    multinomial,
    poisson,
    vec_to_tril_matrix,
)


@pytest.mark.parametrize('x, y', [
    (0.2, 10.),
    (0.6, -10.),
])
def test_binary_cross_entropy_with_logits(x, y):
    actual = -y * np.log(expit(x)) - (1 - y) * np.log(expit(-x))
    expect = binary_cross_entropy_with_logits(x, y)
    assert_allclose(actual, expect, rtol=1e-6)


@pytest.mark.parametrize('shape', [
    (3,),
    (5, 3),
])
def test_cumsum_jac(shape):
    rng_key = random.PRNGKey(0)
    x = random.normal(rng_key, shape=shape)

    def test_fn(x):
        return np.stack([x[..., 0], x[..., 0] + x[..., 1], x[..., 0] + x[..., 1] + x[..., 2]], -1)

    assert_allclose(cumsum(x), test_fn(x))
    assert_allclose(jacobian(cumsum)(x), jacobian(test_fn)(x))


@pytest.mark.parametrize('shape', [
    (3,),
    (5, 3),
])
def test_cumprod_jac(shape):
    rng_key = random.PRNGKey(0)
    x = random.uniform(rng_key, shape=shape)

    def test_fn(x):
        return np.stack([x[..., 0], x[..., 0] * x[..., 1], x[..., 0] * x[..., 1] * x[..., 2]], -1)

    assert_allclose(cumprod(x), test_fn(x))
    assert_allclose(jacobian(cumprod)(x), jacobian(test_fn)(x), atol=1e-7)


@pytest.mark.parametrize('prim', [
    xlogy,
    xlog1py,
])
def test_binop_batch_rule(prim):
    bx = np.array([1., 2., 3.])
    by = np.array([2., 3., 4.])
    x = np.array(1.)
    y = np.array(2.)

    actual_bx_by = vmap(lambda x, y: prim(x, y))(bx, by)
    for i in range(3):
        assert_allclose(actual_bx_by[i], prim(bx[i], by[i]))

    actual_x_by = vmap(lambda y: prim(x, y))(by)
    for i in range(3):
        assert_allclose(actual_x_by[i], prim(x, by[i]))

    actual_bx_y = vmap(lambda x: prim(x, y))(bx)
    for i in range(3):
        assert_allclose(actual_bx_y[i], prim(bx[i], y))


@pytest.mark.parametrize('prim', [
    cumsum,
    cumprod,
])
def test_unop_batch_rule(prim):
    rng_key = random.PRNGKey(0)
    bx = random.normal(rng_key, (3, 5))

    actual = vmap(prim)(bx)
    for i in range(3):
        assert_allclose(actual[i], prim(bx[i]))


@pytest.mark.parametrize('p, shape', [
    (np.array([0.1, 0.9]), ()),
    (np.array([0.2, 0.8]), (2,)),
    (np.array([[0.1, 0.9], [0.2, 0.8]]), ()),
    (np.array([[0.1, 0.9], [0.2, 0.8]]), (3, 2)),
])
def test_categorical_shape(p, shape):
    rng_key = random.PRNGKey(0)
    expected_shape = lax.broadcast_shapes(p.shape[:-1], shape)
    assert np.shape(categorical(rng_key, p, shape)) == expected_shape


@pytest.mark.parametrize("p", [
    np.array([0.2, 0.3, 0.5]),
    np.array([0.8, 0.1, 0.1]),
])
def test_categorical_stats(p):
    rng_key = random.PRNGKey(0)
    n = 10000
    z = categorical(rng_key, p, (n,))
    _, counts = onp.unique(z, return_counts=True)
    assert_allclose(counts / float(n), p, atol=0.01)


@pytest.mark.parametrize('p, shape', [
    (np.array([0.1, 0.9]), ()),
    (np.array([0.2, 0.8]), (2,)),
    (np.array([[0.1, 0.9], [0.2, 0.8]]), ()),
    (np.array([[0.1, 0.9], [0.2, 0.8]]), (3, 2)),
])
def test_multinomial_shape(p, shape):
    rng_key = random.PRNGKey(0)
    n = 10000
    expected_shape = lax.broadcast_shapes(p.shape[:-1], shape) + p.shape[-1:]
    assert np.shape(multinomial(rng_key, p, n, shape)) == expected_shape


@pytest.mark.parametrize("p", [
    np.array([0.2, 0.3, 0.5]),
    np.array([0.8, 0.1, 0.1]),
])
@pytest.mark.parametrize("n", [
    10000,
    np.array([10000, 20000]),
])
def test_multinomial_stats(p, n):
    rng_key = random.PRNGKey(0)
    z = multinomial(rng_key, p, n)
    n = float(n) if isinstance(n, Number) else np.expand_dims(n.astype(p.dtype), -1)
    p = np.broadcast_to(p, z.shape)
    assert_allclose(z / n, p, atol=0.01)


def test_poisson():
    mu = rate = 1000
    N = 2 ** 18

    key = random.PRNGKey(64)
    B = poisson(key, rate=rate, shape=(N,))
    assert_allclose(B.mean(), mu, rtol=0.001)


@pytest.mark.parametrize("shape", [
    (6,),
    (5, 10),
    (3, 4, 3),
])
@pytest.mark.parametrize("diagonal", [
    0,
    -1,
    -2,
])
def test_vec_to_tril_matrix(shape, diagonal):
    rng_key = random.PRNGKey(0)
    x = random.normal(rng_key, shape)
    actual = vec_to_tril_matrix(x, diagonal)
    expected = onp.zeros(shape[:-1] + actual.shape[-2:])
    tril_idxs = onp.tril_indices(expected.shape[-1], diagonal)
    expected[..., tril_idxs[0], tril_idxs[1]] = x
    assert_allclose(actual, expected)


@pytest.mark.parametrize("chol_batch_shape", [(), (3,)])
@pytest.mark.parametrize("vec_batch_shape", [(), (3,)])
@pytest.mark.parametrize("dim", [1, 4])
@pytest.mark.parametrize("coef", [1, -1])
def test_cholesky_update(chol_batch_shape, vec_batch_shape, dim, coef):
    A = random.normal(random.PRNGKey(0), chol_batch_shape + (dim, dim))
    A = A @ np.swapaxes(A, -2, -1) + np.eye(dim)
    x = random.normal(random.PRNGKey(0), vec_batch_shape + (dim,)) * 0.1
    xxt = x[..., None] @ x[..., None, :]
    expected = np.linalg.cholesky(A + coef * xxt)
    actual = cholesky_update(np.linalg.cholesky(A), x, coef)
    assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
