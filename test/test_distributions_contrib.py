from collections import namedtuple

import pytest
import scipy.stats as osp
from numpy.testing import assert_allclose

import jax
import jax.numpy as np
import jax.random as random

import numpyro.contrib.distributions as dist
from numpyro.contrib.distributions.discrete import _to_probs_bernoulli


def _identity(x): return x


class T(namedtuple('TestCase', ['jax_dist', 'sp_dist', 'params'])):
    def __new__(cls, jax_dist, *params):
        sp_dist = _DIST_MAP[jax_dist]
        return super(cls, T).__new__(cls, jax_dist, sp_dist, params)


_DIST_MAP = {
    dist.Bernoulli: lambda probs: osp.bernoulli(p=probs),
    dist.BernoulliWithLogits: lambda logits: osp.bernoulli(p=_to_probs_bernoulli(logits)),
    dist.Binomial: lambda probs, total_count: osp.binom(n=total_count, p=probs),
    dist.BinomialWithLogits: lambda logits, total_count: osp.binom(n=total_count, p=_to_probs_bernoulli(logits)),
    dist.Cauchy: lambda loc, scale: osp.cauchy(loc=loc, scale=scale),
    dist.Exponential: lambda rate: osp.expon(scale=np.reciprocal(rate)),
    dist.HalfCauchy: lambda scale: osp.halfcauchy(scale=scale),
    dist.Normal: lambda loc, scale: osp.norm(loc=loc, scale=scale),
    dist.Uniform: lambda a, b: osp.uniform(a, b - a),
}


CONTINUOUS = [
    T(dist.Cauchy, 0., 1.),
    T(dist.Cauchy, 0., np.array([1., 2.])),
    T(dist.Cauchy, np.array([0., 1.]), np.array([[1.], [2.]])),
    T(dist.Exponential, 2.,),
    T(dist.Exponential, np.array([4., 2.])),
    T(dist.HalfCauchy, 1.),
    T(dist.HalfCauchy, np.array([1., 2.])),
    T(dist.Normal, 0., 1.),
    T(dist.Normal, 1., np.array([1., 2.])),
    T(dist.Normal, np.array([0., 1.]), np.array([[1.], [2.]])),
    T(dist.Uniform, 0., 2.),
    T(dist.Uniform, 1., np.array([2., 3.])),
    T(dist.Uniform, np.array([0., 0.]), np.array([[2.], [3.]])),
]


DISCRETE = [
    T(dist.Bernoulli, 0.2),
    T(dist.Bernoulli, np.array([0.2, 0.7])),
    T(dist.BernoulliWithLogits, np.array([-1., 3.])),
    T(dist.Binomial, np.array([0.2, 0.7]), np.array([10, 2])),
    T(dist.Binomial, np.array([0.2, 0.7]), np.array([5, 8])),
    T(dist.BinomialWithLogits, np.array([-1., 3.]), np.array([5, 8])),
]


@pytest.mark.parametrize('jax_dist, sp_dist, params', CONTINUOUS + DISCRETE)
@pytest.mark.parametrize('prepend_shape', [
    (),
    (2,),
    (2, 3),
])
def test_dist_shape(jax_dist, sp_dist, params, prepend_shape):
    jax_dist = jax_dist(*params)
    sp_dist = sp_dist(*params)
    rng = random.PRNGKey(0)
    expected_shape = prepend_shape + jax_dist.batch_shape
    samples = jax_dist.sample(key=rng, size=prepend_shape)
    sp_samples = sp_dist.rvs(size=expected_shape)
    assert isinstance(samples, jax.interpreters.xla.DeviceArray)
    assert np.shape(samples) == expected_shape
    assert np.shape(sp_samples) == expected_shape


@pytest.mark.parametrize('jax_dist, sp_dist, params', CONTINUOUS)
def test_sample_gradient(jax_dist, sp_dist, params):
    if not jax_dist.is_reparametrized:
        pass

    rng = random.PRNGKey(0)

    def fn(args):
        return np.sum(jax_dist(*args).sample(key=rng))

    actual_grad = jax.grad(fn)(params)
    assert len(actual_grad) == len(params)

    eps = 1e-5
    for i in range(len(params)):
        if np.result_type(params[i]) in (np.int32, np.int64):
            continue
        args_lhs = [p if j != i else p - eps for j, p in enumerate(params)]
        args_rhs = [p if j != i else p + eps for j, p in enumerate(params)]
        fn_lhs = fn(args_lhs)
        fn_rhs = fn(args_rhs)
        # finite diff approximation
        expected_grad = (fn_rhs - fn_lhs) / (2. * eps)
        assert np.shape(actual_grad[i]) == np.shape(params[i])
        assert_allclose(np.sum(actual_grad[i]), expected_grad, rtol=0.10)


@pytest.mark.parametrize('jax_dist, sp_dist, params', CONTINUOUS + DISCRETE)
@pytest.mark.parametrize('prepend_shape', [
    (),
    (2,),
    (2, 3),
])
@pytest.mark.parametrize('jit', [False, True])
def test_log_prob(jax_dist, sp_dist, params, prepend_shape, jit):
    jit_fn = _identity if not jit else jax.jit
    jax_dist = jax_dist(*params)
    sp_dist = sp_dist(*params)
    rng = random.PRNGKey(0)
    samples = jax_dist.sample(key=rng, size=prepend_shape)
    try:
        expected = sp_dist.logpdf(samples)
    except AttributeError:
        expected = sp_dist.logpmf(samples)
    assert_allclose(jit_fn(jax_dist.log_prob)(samples), expected, atol=1e-5)


@pytest.mark.parametrize('jax_dist, sp_dist, params', CONTINUOUS + DISCRETE)
def test_log_prob_gradient(jax_dist, sp_dist, params):
    if not jax_dist.is_reparametrized:
        pass

    rng = random.PRNGKey(0)

    def fn(args, value):
        return np.sum(jax_dist(*args).log_prob(value))

    value = jax_dist(*params).sample(rng)
    actual_grad = jax.grad(fn)(params, value)
    assert len(actual_grad) == len(params)

    eps = 1e-5
    for i in range(len(params)):
        if np.result_type(params[i]) in (np.int32, np.int64):
            continue
        args_lhs = [p if j != i else p - eps for j, p in enumerate(params)]
        args_rhs = [p if j != i else p + eps for j, p in enumerate(params)]
        fn_lhs = fn(args_lhs, value)
        fn_rhs = fn(args_rhs, value)
        # finite diff approximation
        expected_grad = (fn_rhs - fn_lhs) / (2. * eps)
        assert np.shape(actual_grad[i]) == np.shape(params[i])
        assert_allclose(np.sum(actual_grad[i]), expected_grad, rtol=0.10)


@pytest.mark.parametrize('jax_dist, sp_dist, params', CONTINUOUS + DISCRETE)
def test_mean_var(jax_dist, sp_dist, params):
    n = 100000
    d_jax = jax_dist(*params)
    d_sp = sp_dist(*params)
    k = random.PRNGKey(0)
    samples = d_jax.sample(k, size=(n,))
    sp_mean, sp_var = d_sp.stats(moments='mv')
    assert_allclose(d_jax.mean, sp_mean)
    assert_allclose(d_jax.variance, sp_var)
    if np.all(np.isfinite(sp_mean)):
        assert_allclose(np.mean(samples, 0), d_jax.mean, rtol=0.05, atol=1e-2)
    if np.all(np.isfinite(sp_var)):
        assert_allclose(np.std(samples, 0), np.sqrt(d_jax.variance), rtol=0.05, atol=1e-2)
