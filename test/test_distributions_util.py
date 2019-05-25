from numbers import Number

import numpy as onp
import pytest
import scipy.special as osp_special
import scipy.stats as osp_stats
from numpy.testing import assert_allclose

import jax.numpy as np
from jax import grad, jacobian, jit, lax, random, vmap
from jax.scipy.special import expit
from jax.util import partial

from numpyro.distributions.util import (
    binary_cross_entropy_with_logits,
    categorical,
    cumprod,
    cumsum,
    multinomial,
    standard_gamma,
    vec_to_tril_matrix,
    xlog1py,
    xlogy
)

_zeros = partial(lax.full_like, fill_value=0)


@pytest.mark.parametrize('x, y', [
    (np.array([1]), np.array([1, 2, 3])),
    (np.array([0]), np.array([0, 0])),
    (np.array([[0.], [0.]]), np.array([1., 2.])),
])
@pytest.mark.parametrize('jit_fn', [False, True])
def test_xlogy(x, y, jit_fn):
    fn = xlogy if not jit_fn else jit(xlogy)
    assert_allclose(fn(x, y), osp_special.xlogy(x, y))


@pytest.mark.parametrize('x, y, grad1, grad2', [
    (np.array([1., 1., 1.]), np.array([1., 2., 3.]),
     np.log(np.array([1, 2, 3])), np.array([1., 0.5, 1./3])),
    (np.array([1.]), np.array([1., 2., 3.]),
     np.sum(np.log(np.array([1, 2, 3]))), np.array([1., 0.5, 1./3])),
    (np.array([1., 2., 3.]), np.array([2.]),
     np.log(np.array([2., 2., 2.])), np.array([3.])),
    (np.array([0.]), np.array([0, 0]),
     np.array([-float('inf')]), np.array([0, 0])),
    (np.array([[0.], [0.]]), np.array([1., 2.]),
     np.array([[np.log(2.)], [np.log(2.)]]), np.array([0, 0])),
])
def test_xlogy_jac(x, y, grad1, grad2):
    assert_allclose(grad(lambda x, y: np.sum(xlogy(x, y)))(x, y), grad1)
    assert_allclose(grad(lambda x, y: np.sum(xlogy(x, y)), 1)(x, y), grad2)


@pytest.mark.parametrize('x, y', [
    (np.array([1]), np.array([0, 1, 2])),
    (np.array([0]), np.array([-1, -1])),
    (np.array([[0.], [0.]]), np.array([1., 2.])),
])
@pytest.mark.parametrize('jit_fn', [False, True])
def test_xlog1py(x, y, jit_fn):
    fn = xlog1py if not jit_fn else jit(xlog1py)
    assert_allclose(fn(x, y), osp_special.xlog1py(x, y))


@pytest.mark.parametrize('x, y, grad1, grad2', [
    (np.array([1., 1., 1.]), np.array([0., 1., 2.]),
     np.log(np.array([1, 2, 3])), np.array([1., 0.5, 1./3])),
    (np.array([1., 1., 1.]), np.array([-1., 0., 1.]),
     np.log(np.array([0, 1, 2])), np.array([float('inf'), 1., 0.5])),
    (np.array([1.]), np.array([0., 1., 2.]),
     np.sum(np.log(np.array([1, 2, 3]))), np.array([1., 0.5, 1./3])),
    (np.array([1., 2., 3.]), np.array([1.]),
     np.log(np.array([2., 2., 2.])), np.array([3.])),
    (np.array([0.]), np.array([-1, -1]),
     np.array([-float('inf')]), np.array([0, 0])),
    (np.array([[0.], [0.]]), np.array([1., 2.]),
     np.array([[np.log(6.)], [np.log(6.)]]), np.array([0, 0])),
])
def test_xlog1py_jac(x, y, grad1, grad2):
    assert_allclose(grad(lambda x, y: np.sum(xlog1py(x, y)))(x, y), grad1)
    assert_allclose(grad(lambda x, y: np.sum(xlog1py(x, y)), 1)(x, y), grad2)


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
    rng = random.PRNGKey(0)
    x = random.normal(rng, shape=shape)

    def test_fn(x):
        return np.stack([x[..., 0], x[..., 0] + x[..., 1], x[..., 0] + x[..., 1] + x[..., 2]], -1)

    assert_allclose(cumsum(x), test_fn(x))
    assert_allclose(jacobian(cumsum)(x), jacobian(test_fn)(x))


@pytest.mark.parametrize('shape', [
    (3,),
    (5, 3),
])
def test_cumprod_jac(shape):
    rng = random.PRNGKey(0)
    x = random.uniform(rng, shape=shape)

    def test_fn(x):
        return np.stack([x[..., 0], x[..., 0] * x[..., 1], x[..., 0] * x[..., 1] * x[..., 2]], -1)

    assert_allclose(cumprod(x), test_fn(x))
    assert_allclose(jacobian(cumprod)(x), jacobian(test_fn)(x), atol=1e-7)


@pytest.mark.parametrize('alpha, shape', [
    (1., ()),
    (1., (2,)),
    (np.array([1., 2.]), ()),
    (np.array([1., 2.]), (3, 2)),
])
def test_standard_gamma_shape(alpha, shape):
    rng = random.PRNGKey(0)
    expected_shape = lax.broadcast_shapes(np.shape(alpha), shape)
    assert np.shape(standard_gamma(rng, alpha, shape=shape)) == expected_shape


@pytest.mark.parametrize("alpha", [0.6, 2., 10.])
def test_standard_gamma_stats(alpha):
    rng = random.PRNGKey(0)
    z = standard_gamma(rng, np.full((1000,), alpha))
    assert_allclose(np.mean(z), alpha, rtol=0.06)
    assert_allclose(np.var(z), alpha, rtol=0.2)


@pytest.mark.parametrize("alpha", [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4])
def test_standard_gamma_grad(alpha):
    rng = random.PRNGKey(0)
    alphas = np.full((100,), alpha)
    z = standard_gamma(rng, alphas)
    actual_grad = grad(lambda x: np.sum(standard_gamma(rng, x)))(alphas)

    eps = 0.01 * alpha / (1.0 + np.sqrt(alpha))
    cdf_dot = (osp_stats.gamma.cdf(z, alpha + eps)
               - osp_stats.gamma.cdf(z, alpha - eps)) / (2 * eps)
    pdf = osp_stats.gamma.pdf(z, alpha)
    expected_grad = -cdf_dot / pdf

    assert_allclose(actual_grad, expected_grad, atol=1e-8, rtol=0.0005)


def test_standard_gamma_batch():
    rng = random.PRNGKey(0)
    alphas = np.array([1., 2., 3.])
    rngs = random.split(rng, 3)

    samples = vmap(lambda rng, alpha: standard_gamma(rng, alpha))(rngs, alphas)
    for i in range(3):
        assert_allclose(samples[i], standard_gamma(rngs[i], alphas[i]))


@pytest.mark.parametrize('p, shape', [
    (np.array([0.1, 0.9]), ()),
    (np.array([0.2, 0.8]), (2,)),
    (np.array([[0.1, 0.9], [0.2, 0.8]]), ()),
    (np.array([[0.1, 0.9], [0.2, 0.8]]), (3, 2)),
])
def test_categorical_shape(p, shape):
    rng = random.PRNGKey(0)
    expected_shape = lax.broadcast_shapes(p.shape[:-1], shape)
    assert np.shape(categorical(rng, p, shape)) == expected_shape


@pytest.mark.parametrize("p", [
    np.array([0.2, 0.3, 0.5]),
    np.array([0.8, 0.1, 0.1]),
])
def test_categorical_stats(p):
    rng = random.PRNGKey(0)
    n = 10000
    z = categorical(rng, p, (n,))
    _, counts = onp.unique(z, return_counts=True)
    assert_allclose(counts / float(n), p, atol=0.01)


@pytest.mark.parametrize('p, shape', [
    (np.array([0.1, 0.9]), ()),
    (np.array([0.2, 0.8]), (2,)),
    (np.array([[0.1, 0.9], [0.2, 0.8]]), ()),
    (np.array([[0.1, 0.9], [0.2, 0.8]]), (3, 2)),
])
def test_multinomial_shape(p, shape):
    rng = random.PRNGKey(0)
    n = 10000
    expected_shape = lax.broadcast_shapes(p.shape[:-1], shape) + p.shape[-1:]
    assert np.shape(multinomial(rng, p, n, shape)) == expected_shape


@pytest.mark.parametrize("p", [
    np.array([0.2, 0.3, 0.5]),
    np.array([0.8, 0.1, 0.1]),
])
@pytest.mark.parametrize("n", [
    10000,
    np.array([10000, 20000]),
])
def test_multinomial_stats(p, n):
    rng = random.PRNGKey(0)
    z = multinomial(rng, p, n)
    n = float(n) if isinstance(n, Number) else np.expand_dims(n.astype(p.dtype), -1)
    p = np.broadcast_to(p, z.shape)
    assert_allclose(z / n, p, atol=0.01)


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
    rng = random.PRNGKey(0)
    x = random.normal(rng, shape)
    actual = vec_to_tril_matrix(x, diagonal)
    expected = onp.zeros(shape[:-1] + actual.shape[-2:])
    tril_idxs = onp.tril_indices(expected.shape[-1], diagonal)
    expected[..., tril_idxs[0], tril_idxs[1]] = x
    assert_allclose(actual, expected)
