from collections import namedtuple

import pytest
import scipy.stats as osp
from numpy.testing import assert_allclose

import jax
import jax.numpy as np
import jax.random as random

import numpyro.contrib.distributions as dist


class T(namedtuple('TestCase', ['jax_dist', 'sp_dist', 'params', 'sp_params'])):
    def __new__(cls, jax_dist, sp_dist, params, sp_params=None):
        if sp_params is None:
            sp_params = params
        return super(cls, T).__new__(cls, jax_dist, sp_dist, params, sp_params)


CONTINUOUS = [
    T(dist.Normal, osp.norm, (0., 1.)),
    T(dist.Normal, osp.norm, (1., np.array([1., 2.]))),
    T(dist.Normal, osp.norm, (np.array([0., 1.]), np.array([[1.], [2.]]))),
    T(dist.Uniform, osp.uniform, (0., 2.)),
    T(dist.Uniform, osp.uniform, (1., np.array([2., 3.])), (1., np.array([1., 2.]))),
    T(dist.Uniform, osp.uniform, (np.array([0., 0.]), np.array([[2.], [3.]]))),
]


DISCRETE = [
    T(dist.Bernoulli, osp.bernoulli, (0.2,)),
    T(dist.Bernoulli, osp.bernoulli, (np.array([0.2, 0.7]),)),
]


@pytest.mark.parametrize('jax_dist, sp_dist, params, sp_params', CONTINUOUS + DISCRETE)
@pytest.mark.parametrize('prepend_shape', [
    (),
    (2,),
    (2, 3),
])
def test_continuous_shape(jax_dist, sp_dist, params, sp_params, prepend_shape):
    jax_dist = jax_dist(*params)
    sp_dist = sp_dist(*sp_params)
    rng = random.PRNGKey(0)
    expected_shape = prepend_shape + jax_dist.batch_shape
    samples = jax_dist.sample(key=rng, size=expected_shape)
    sp_samples = sp_dist.rvs(size=expected_shape)
    assert isinstance(samples, jax.interpreters.xla.DeviceArray)
    assert np.shape(samples) == expected_shape
    assert np.shape(sp_samples) == expected_shape


@pytest.mark.parametrize('jax_dist, sp_dist, params, sp_params', CONTINUOUS)
def test_sample_gradient(jax_dist, sp_dist, params, sp_params):
    rng = random.PRNGKey(0)

    def fn(args):
        return jax_dist(*args).sample(key=rng).sum()

    actual_grad = jax.grad(fn)(params)
    assert len(actual_grad) == len(params)

    eps = 1e-6
    for i in range(len(params)):
        args_lhs = [p if j != i else p - eps for j, p in enumerate(params)]
        args_rhs = [p if j != i else p + eps for j, p in enumerate(params)]
        fn_lhs = fn(args_lhs)
        fn_rhs = fn(args_rhs)
        # finite diff approximation
        expected_grad = (fn_rhs - fn_lhs) / (2. * eps)
        assert np.shape(actual_grad[i]) == np.shape(params[i])
        assert_allclose(np.sum(actual_grad[i]), expected_grad, rtol=0.12)


@pytest.mark.parametrize('jax_dist, sp_dist, params, sp_params', CONTINUOUS + DISCRETE)
@pytest.mark.parametrize('prepend_shape', [
    (),
    (2,),
    (2, 3),
])
def test_log_prob(jax_dist, sp_dist, params, sp_params, prepend_shape):
    jax_dist = jax_dist(*params)
    sp_dist = sp_dist(*sp_params)
    rng = random.PRNGKey(0)
    expected_shape = prepend_shape + jax_dist.batch_shape
    samples = jax_dist.sample(key=rng, size=expected_shape)
    try:
        expected = sp_dist.logpdf(samples)
    except AttributeError:
        expected = sp_dist.logpmf(samples)
    assert_allclose(jax_dist.log_prob(samples), expected, atol=1e-5)
