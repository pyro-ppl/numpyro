# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from numbers import Number

import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
import pytest
import scipy

import jax
from jax import grad, lax, random, vmap
import jax.numpy as jnp
from jax.scipy.special import expit, xlog1py, xlogy
from jax.test_util import check_grads

import numpyro.distributions as dist
from numpyro.distributions.util import (
    add_diag,
    binary_cross_entropy_with_logits,
    binomial,
    categorical,
    cholesky_update,
    log1mexp,
    logdiffexp,
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


@pytest.mark.parametrize("x", [-80.5632, -0.32523, -0.5, -20.53, -8.032])
def test_log1mexp_grads(x):
    check_grads(log1mexp, (x,), order=3)


@pytest.mark.parametrize(
    "x, expected",
    [
        (np.array([0.01, 0, -np.inf]), np.array([np.nan, -np.inf, 0])),
        (0.001, np.nan),
        (0, -np.inf),
        (-np.inf, 0),
    ],
)
def test_log1mexp_bounds_handling(x, expected):
    """
    log1mexp(x) should be nan for x > 0.

    log1mexp(x) should be -inf for x == 0.

    log1mexp(-inf) should be 0.

    This should work vectorized and not interfere
    with other calculations.
    """
    assert_array_equal(log1mexp(x), expected)


@pytest.mark.parametrize("x", [np.array([-0.6, -8.32, -3]), -2.5, -0.01])
def test_log1mexp_agrees_with_basic(x):
    """
    log1mexp should agree with a basic implementation
    for values where the basic implementation is stable.
    """
    assert_array_almost_equal(log1mexp(x), jnp.log(1 - jnp.exp(x)))


def test_log1mexp_stable():
    """
    log1mexp should be stable at (negative) values of
    x that very small and very large in absolute
    value, where the basic implementation is not.
    """

    def basic(x):
        return jnp.log(1 - jnp.exp(x))

    # this should perhaps be made finfo-aware
    assert jnp.isinf(basic(-1e-20))
    assert not jnp.isinf(log1mexp(-1e-20))
    assert_array_almost_equal(log1mexp(-1e-20), jnp.log(-jnp.expm1(-1e-20)))
    assert abs(basic(-50)) < abs(log1mexp(-50))
    assert_array_almost_equal(log1mexp(-50), jnp.log1p(-jnp.exp(-50)))


@pytest.mark.parametrize("x", [-30.0, -2.53, -1e-4, -1e-9, -1e-15, -1e-40])
def test_log1mexp_grad_stable(x):
    """
    Custom JVP for log1mexp should make gradient computation
    numerically stable, even near zero, where the basic approach
    can encounter divide-by-zero problems and yield nan.
    The two approaches should produce almost equal answers elsewhere.
    """

    def log1mexp_no_custom(x):
        return jnp.where(
            x > -0.6931472,  # approx log(2)
            jnp.log(-jnp.expm1(x)),
            jnp.log1p(-jnp.exp(x)),
        )

    grad_custom = grad(log1mexp)(x)
    grad_no_custom = grad(log1mexp_no_custom)(x)

    assert_array_almost_equal(grad_custom, -1 / jnp.expm1(-x))

    if not jnp.isnan(grad_no_custom):
        assert_array_almost_equal(grad_custom, grad_no_custom)


@pytest.mark.parametrize(
    "a, b", [(-20.0, -35.0), (-0.32523, -0.34), (20.53, 19.035), (8.032, 7.032)]
)
def test_logdiffexp_grads(a, b):
    check_grads(logdiffexp, (a, b), order=3, rtol=0.01)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (
            np.array([np.inf, 0, 6.5, 4.99999, -np.inf]),
            np.array([5, 0, 6.5, 5, -np.inf]),
            np.array([np.nan, -np.inf, -np.inf, np.nan, -np.inf]),
        ),
        (np.inf, 0.3532, np.nan),
        (0, 0, -np.inf),
        (-np.inf, -np.inf, -np.inf),
        (5.6, 5.6, -np.inf),
        (1e34, 1e34, -np.inf),
        (1e34, 1e34 / 0.9999, np.nan),
        (np.inf, np.inf, np.nan),
    ],
)
def test_logdiffexp_bounds_handling(a, b, expected):
    """
    Test bounds handling for logdiffexp.

    logdiffexp(jnp.inf, anything) should be nan,
    including logdiffexp(jnp.inf, jnp.inf).

    logdiffexp(a, b) for a < b should be nan, even if numbers
    are very close.

    logdiffexp(a, b) for a == b should be -jnp.inf
    even if a == b == -jnp.inf (log(0 - 0)).
    """
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    assert_array_equal(logdiffexp(a, b), expected)


@pytest.mark.parametrize(
    "a, b", [(np.array([53, 23.532, 8, -1.35]), np.array([56, -63.2, 2, -5.32]))]
)
def test_logdiffexp_agrees_with_basic(a, b):
    """
    logdiffexp should agree with a basic implementation
    for values at which the basic implementation is stable.
    """
    assert_array_almost_equal(logdiffexp(a, b), jnp.log(jnp.exp(a) - jnp.exp(b)))


@pytest.mark.parametrize("a, b", [(500, 499), (-499, -500), (500, 500)])
def test_logdiffexp_stable(a, b):
    """
    logdiffexp should be numerically stable at values
    where the basic implementation is not.
    """

    def basic(a, b):
        return jnp.log(jnp.exp(a) - jnp.exp(b))

    if a > 0 or a == b:
        assert jnp.isnan(basic(a, b))
    else:
        assert basic(a, b) == -jnp.inf
    result = logdiffexp(a, b)
    assert not jnp.isnan(result)
    if not a == b:
        assert result < a
    else:
        assert result == -jnp.inf


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
    key1, key2 = random.split(random.PRNGKey(0))
    A = random.normal(key1, chol_batch_shape + (dim, dim))
    A = A @ jnp.swapaxes(A, -2, -1) + jnp.eye(dim)
    x = random.normal(key2, vec_batch_shape + (dim,)) * 0.1
    xxt = x[..., None] @ x[..., None, :]
    expected = jnp.linalg.cholesky(A + coef * xxt)
    actual = cholesky_update(jnp.linalg.cholesky(A), x, coef)
    assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)


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


@pytest.mark.parametrize(
    "my_dist",
    [
        lambda: dist.TruncatedNormal(low=-1.0, high=2.0),
        lambda: dist.TruncatedCauchy(low=-5, high=10),
        lambda: dist.TruncatedDistribution(dist.StudentT(3), low=1.5),
    ],
)
def test_no_tracer_leak_at_lazy_property_log_prob(my_dist):
    """
    Tests that truncated distributions, which use @lazy_property
    values in their log_prob() methods, do not
    have tracer leakage when log_prob() is called.
    Reference: https://github.com/pyro-ppl/numpyro/issues/1836, and
    https://github.com/CDCgov/multisignal-epi-inference/issues/282
    """
    my_dist = my_dist()
    jit_lp = jax.jit(my_dist.log_prob)
    with jax.check_tracer_leaks():
        jit_lp(1.0)


@pytest.mark.parametrize(
    "my_dist",
    [
        lambda: dist.TruncatedNormal(low=-1.0, high=2.0),
        lambda: dist.TruncatedCauchy(low=-5, high=10),
        lambda: dist.TruncatedDistribution(dist.StudentT(3), low=1.5),
    ],
)
def test_no_tracer_leak_at_lazy_property_sample(my_dist):
    """
    Tests that truncated distributions, which use @lazy_property
    values in their sample() methods, do not
    have tracer leakage when sample() is called.
    Reference: https://github.com/pyro-ppl/numpyro/issues/1836, and
    https://github.com/CDCgov/multisignal-epi-inference/issues/282
    """
    my_dist = my_dist()
    jit_sample = jax.jit(my_dist.sample)
    with jax.check_tracer_leaks():
        jit_sample(jax.random.key(5))
