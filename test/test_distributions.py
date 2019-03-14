from functools import reduce
from operator import mul

import jax
import jax.numpy as np
import jax.random as random
import numpy as onp
import pytest
import scipy.stats as sp
from jax import lax, grad
from numpy.testing import assert_allclose

import numpyro.distributions as dist
from numpyro.distributions.util import standard_gamma


@pytest.mark.parametrize('jax_dist', [
    dist.beta,
    dist.cauchy,
    dist.expon,
    dist.gamma,
    dist.lognorm,
    dist.norm,
    dist.uniform,
], ids=lambda jax_dist: jax_dist.name)
@pytest.mark.parametrize('loc, scale', [
    (1, 1),
    (1., np.array([1., 2.])),
])
@pytest.mark.parametrize('prepend_shape', [
    None,
    (),
    (2,),
    (2, 3),
])
def test_shape(jax_dist, loc, scale, prepend_shape):
    rng = random.PRNGKey(0)
    args = (1,) * jax_dist.numargs
    expected_shape = lax.broadcast_shapes(*[np.shape(loc), np.shape(scale)])
    samples = jax_dist.rvs(*args, loc=loc, scale=scale, random_state=rng)
    assert isinstance(samples, jax.interpreters.xla.DeviceArray)
    assert np.shape(samples) == expected_shape
    assert np.shape(jax_dist(*args, loc=loc, scale=scale).rvs(random_state=rng)) == expected_shape
    if prepend_shape is not None:
        expected_shape = prepend_shape + lax.broadcast_shapes(*[np.shape(loc), np.shape(scale)])
        assert np.shape(jax_dist.rvs(*args, loc=loc, scale=scale,
                                     size=expected_shape, random_state=rng)) == expected_shape
        assert np.shape(jax_dist(*args, loc=loc, scale=scale)
                        .rvs(random_state=rng, size=expected_shape)) == expected_shape


def idfn(param):
    if isinstance(param, (sp._distn_infrastructure.rv_generic,
                          sp._multivariate.multi_rv_generic)):
        return param.name
    return repr(param)


@pytest.mark.parametrize('jax_dist, dist_args', [
    (dist.bernoulli, (0.1,)),
    (dist.bernoulli, (np.array([0.3, 0.5]))),
    (dist.binom, (10, 0.4)),
    (dist.binom, (np.array([10]), np.array([0.4, 0.3]))),
    (dist.multinomial, (10, np.array([0.1, 0.4, 0.5]))),
    (dist.multinomial, (10, np.array([1]))),
], ids=idfn)
@pytest.mark.parametrize('prepend_shape', [
    None,
    (),
    (2,),
    (2, 3),
])
def test_discrete_shape(jax_dist, dist_args, prepend_shape):
    rng = random.PRNGKey(0)
    sp_dist = getattr(sp, jax_dist.name)
    expected_shape = np.shape(sp_dist.rvs(*dist_args))
    samples = jax_dist.rvs(*dist_args, random_state=rng)
    assert isinstance(samples, jax.interpreters.xla.DeviceArray)
    assert np.shape(samples) == expected_shape
    if prepend_shape is not None:
        shape = prepend_shape + lax.broadcast_shapes(*[np.shape(arg) for arg in dist_args])
        expected_shape = np.shape(sp_dist.rvs(*dist_args, size=shape))
        assert np.shape(jax_dist.rvs(*dist_args, size=shape, random_state=rng)) == expected_shape


@pytest.mark.parametrize('jax_dist', [
    dist.beta,
    dist.cauchy,
    dist.expon,
    dist.gamma,
    dist.lognorm,
    dist.norm,
    dist.uniform,
], ids=lambda jax_dist: jax_dist.name)
@pytest.mark.parametrize('loc, scale', [
    (1., 1.),
    (1., np.array([1., 2.])),
])
def test_sample_gradient(jax_dist, loc, scale):
    rng = random.PRNGKey(0)
    args = (1,) * jax_dist.numargs
    expected_shape = lax.broadcast_shapes(*[np.shape(loc), np.shape(scale)])

    def fn(args, loc, scale):
        return jax_dist.rvs(*args, loc=loc, scale=scale, random_state=rng).sum()

    assert_allclose(grad(fn, 1)(args, loc, scale),
                    loc * reduce(mul, expected_shape[:len(expected_shape) - np.ndim(loc)], 1.))
    assert_allclose(grad(fn, 2)(args, loc, scale),
                    jax_dist.rvs(*args, size=expected_shape, random_state=rng))


@pytest.mark.parametrize('jax_dist', [
    dist.beta,
    dist.gamma,
    dist.lognorm,
    dist.t,
], ids=lambda jax_dist: jax_dist.name)
@pytest.mark.parametrize('arg', [1e-2, 1e-1, 1e0, 1e1, 1e2])
def test_pathwise_gradient(jax_dist, arg):
    rng = random.PRNGKey(0)
    num_args = jax_dist.numargs
    num_samples = 100
    sp_dist = getattr(sp, jax_dist.name)
    arg = np.full((num_samples,), arg)

    def _make_args(i, val):
        # create a list with i-th value is val
        return [1 if j != i else val for j in range(num_args)]

    for i in range(num_args):
        z = jax_dist.rvs(*_make_args(i, arg), random_state=rng)
        actual_grad = grad(lambda x: np.sum(jax_dist.rvs(*_make_args(i, arg), random_state=rng)))(arg)

        eps = 0.01 * arg / (1.0 + np.sqrt(arg))
        cdf_dot = ((sp_dist.cdf(z, *_make_args(i, arg + eps)) - sp_dist.cdf(z, *_make_args(i, arg - eps)))
                   / (2 * eps))
        pdf = sp_dist.pdf(z, *args)
        expected_grad = -cdf_dot / pdf

        assert_allclose(actual_grad, expected_grad, rtol=0.005)


@pytest.mark.parametrize('jax_dist', [
    dist.beta,
    dist.cauchy,
    dist.expon,
    dist.gamma,
    dist.lognorm,
    dist.norm,
    pytest.param(dist.uniform,
                 marks=pytest.mark.xfail(
                     reason="jax.scipy.uniform.logpdf is not correctly implemented, "
                     "see https://github.com/google/jax/pull/510")),
], ids=lambda jax_dist: jax_dist.name)
@pytest.mark.parametrize('loc_scale', [
    (),
    (1,),
    (1, 1),
    (1., np.array([1., 2.])),
])
def test_logprob(jax_dist, loc_scale):
    rng = random.PRNGKey(0)
    args = (1,) * jax_dist.numargs + loc_scale
    samples = jax_dist.rvs(*args, random_state=rng)
    sp_dist = getattr(sp, jax_dist.name)
    assert_allclose(jax_dist.logpdf(samples, *args), sp_dist.logpdf(samples, *args))


@pytest.mark.parametrize('jax_dist, dist_args', [
    (dist.bernoulli, (0.1,)),
    (dist.bernoulli, (np.array([0.3, 0.5]))),
    (dist.binom, (10, 0.4)),
    (dist.binom, (np.array([10]), np.array([0.4, 0.3]))),
    (dist.binom, [np.array([2, 5]), np.array([[0.4], [0.5]])]),
    (dist.multinomial, (10, np.array([0.1, 0.4, 0.5]))),
    (dist.multinomial, (10, np.array([1.]))),
], ids=idfn)
@pytest.mark.parametrize('shape', [
    None,
    (),
    (2,),
    (2, 3),
])
def test_discrete_logpmf(jax_dist, dist_args, shape):
    rng = random.PRNGKey(0)
    sp_dist = getattr(sp, jax_dist.name)
    samples = jax_dist.rvs(*dist_args, random_state=rng)
    assert_allclose(jax_dist.logpmf(samples, *dist_args),
                    sp_dist.logpmf(onp.asarray(samples), *dist_args),
                    rtol=1e-5)
    if shape is not None:
        shape = shape + lax.broadcast_shapes(*[np.shape(arg) for arg in dist_args])
        samples = jax_dist.rvs(*dist_args, size=shape, random_state=rng)
        assert_allclose(jax_dist.logpmf(samples, *dist_args),
                        sp_dist.logpmf(onp.asarray(samples), *dist_args),
                        rtol=1e-5)


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
    cdf_dot = (sp.gamma.cdf(z, alpha + eps) - sp.gamma.cdf(z, alpha - eps)) / (2 * eps)
    pdf = sp.gamma.pdf(z, alpha)
    expected_grad = -cdf_dot / pdf

    assert_allclose(actual_grad, expected_grad, rtol=0.0005)
