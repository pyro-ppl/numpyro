# Copyright (c) 2017-2020 Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import reduce
from operator import mul

import numpy as onp
from numpy.testing import assert_allclose
import pytest
import scipy.stats as osp_stats

import jax
from jax import grad, lax, random
import jax.numpy as np
from jax.scipy.special import logit

import numpyro.contrib.distributions as dist
from numpyro.contrib.distributions import jax_multivariate, validation_enabled
from numpyro.distributions import constraints


def idfn(param):
    if isinstance(param, (osp_stats._distn_infrastructure.rv_generic,
                          osp_stats._multivariate.multi_rv_generic)):
        return param.name
    elif isinstance(param, constraints.Constraint):
        return param.__class__.__name__
    return repr(param)


@pytest.mark.parametrize('jax_dist', [
    dist.beta,
    dist.cauchy,
    dist.expon,
    dist.gamma,
    dist.halfcauchy,
    dist.halfnorm,
    dist.lognorm,
    dist.pareto,
    dist.trunccauchy,
    dist.truncnorm,
    dist.norm,
    dist.t,
    dist.uniform,
], ids=idfn)
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
def test_continuous_shape(jax_dist, loc, scale, prepend_shape):
    rng_key = random.PRNGKey(0)
    args = [i + 1 for i in range(jax_dist.numargs)]
    expected_shape = lax.broadcast_shapes(*[np.shape(loc), np.shape(scale)])
    samples = jax_dist.rvs(*args, loc=loc, scale=scale, random_state=rng_key)
    assert isinstance(samples, jax.interpreters.xla.DeviceArray)
    assert np.shape(samples) == expected_shape
    assert np.shape(jax_dist(*args, loc=loc, scale=scale).rvs(random_state=rng_key)) == expected_shape
    if prepend_shape is not None:
        expected_shape = prepend_shape + lax.broadcast_shapes(*[np.shape(loc), np.shape(scale)])
        assert np.shape(jax_dist.rvs(*args, loc=loc, scale=scale,
                                     size=expected_shape, random_state=rng_key)) == expected_shape
        assert np.shape(jax_dist(*args, loc=loc, scale=scale)
                        .rvs(random_state=rng_key, size=expected_shape)) == expected_shape


@pytest.mark.parametrize('jax_dist, dist_args, sample', [
    (dist.beta, (-1, 1), -1),
    (dist.beta, (2, np.array([1., -3])), np.array([1., -2])),
    (dist.cauchy, (), np.inf),
    (dist.cauchy, (), np.array([1., np.nan])),
    (dist.expon, (), -1),
    (dist.expon, (), np.array([1., -2])),
    (dist.gamma, (-1,), -1),
    (dist.gamma, (np.array([-2., 3]),), np.array([1., -2])),
    (dist.halfcauchy, (), -1),
    (dist.halfcauchy, (), np.array([1., -2])),
    (dist.halfnorm, (), -1),
    (dist.halfnorm, (), np.array([1., -2])),
    (dist.lognorm, (-1,), -1),
    (dist.lognorm, (np.array([-2., 3]),), np.array([1., -2])),
    (dist.norm, (), np.inf),
    (dist.norm, (), np.array([1., np.nan])),
    (dist.pareto, (-1,), -1),
    (dist.pareto, (np.array([-2., 3]),), np.array([1., -2])),
    (dist.t, (-1,), np.inf),
    (dist.t, (np.array([-2., 3]),), np.array([1., np.nan])),
    (dist.trunccauchy, (), -1),
    (dist.trunccauchy, (), np.array([1., -2])),
    (dist.truncnorm, (), -1),
    (dist.truncnorm, (), np.array([1., -2])),
    (dist.uniform, (), -1),
    (dist.uniform, (), np.array([0.5, -2])),
], ids=idfn)
def test_continuous_validate_args(jax_dist, dist_args, sample):
    valid_args = [i + 1 for i in range(jax_dist.numargs)]
    with validation_enabled():
        if dist_args:
            with pytest.raises(ValueError, match='Invalid parameters'):
                jax_dist(*dist_args)

        with pytest.raises(ValueError, match='Invalid scale parameter'):
            jax_dist(*valid_args, scale=-1)

        frozen_dist = jax_dist(*valid_args)
        with pytest.raises(ValueError, match='Invalid values'):
            frozen_dist.logpdf(sample)


@pytest.mark.parametrize('jax_dist, dist_args', [
    (dist.categorical, (np.array([0.1, 0.9]),)),
    (dist.categorical, (np.array([[0.1, 0.9], [0.2, 0.8]]),)),
    (dist.dirichlet, (np.ones(3),)),
    (dist.dirichlet, (np.ones((2, 3)),)),
    (dist.multinomial, (10, np.array([0.1, 0.9]),)),
    (dist.multinomial, (10, np.array([[0.1, 0.9], [0.2, 0.8]]),)),
], ids=idfn)
@pytest.mark.parametrize('prepend_shape', [
    None,
    (),
    (2,),
    (2, 3),
])
def test_multivariate_shape(jax_dist, dist_args, prepend_shape):
    rng_key = random.PRNGKey(0)
    expected_shape = jax_dist._batch_shape(*dist_args) + jax_dist._event_shape(*dist_args)
    samples = jax_dist.rvs(*dist_args, random_state=rng_key)
    assert isinstance(samples, jax.interpreters.xla.DeviceArray)
    assert np.shape(samples) == expected_shape
    assert np.shape(jax_dist(*dist_args).rvs(random_state=rng_key)) == expected_shape
    if prepend_shape is not None:
        size = prepend_shape + jax_dist._batch_shape(*dist_args)
        expected_shape = size + jax_dist._event_shape(*dist_args)
        samples = jax_dist.rvs(*dist_args, size=size, random_state=rng_key)
        assert np.shape(samples) == expected_shape
        samples = jax_dist(*dist_args).rvs(random_state=rng_key, size=size)
        assert np.shape(samples) == expected_shape


@pytest.mark.parametrize('jax_dist, valid_args, invalid_args, invalid_sample', [
    (dist.categorical, (np.array([0.1, 0.9]),), (np.array([0.1, 0.8]),), np.array([1, 4])),
    (dist.dirichlet, (np.ones(3),), (np.array([-1., 2., 3.]),), np.array([0.1, 0.7, 0.1])),
    (dist.multinomial, (10, np.array([0.1, 0.9]),), (10, np.array([0.2, 0.9]),), np.array([-1, 9])),
], ids=idfn)
def test_multivariate_validate_args(jax_dist, valid_args, invalid_args, invalid_sample):
    with validation_enabled():
        with pytest.raises(ValueError, match='Invalid parameters'):
            jax_dist(*invalid_args)

        frozen_dist = jax_dist(*valid_args)
        with pytest.raises(ValueError, match='Invalid values'):
            frozen_dist.logpmf(invalid_sample)


@pytest.mark.parametrize('jax_dist, dist_args', [
    (dist.bernoulli, (0.1,)),
    (dist.bernoulli, (np.array([0.3, 0.5]),)),
    (dist.binom, (10, 0.4)),
    (dist.binom, (np.array([10]), np.array([0.4, 0.3]))),
    (dist.poisson, (1.,)),
    (dist.poisson, (np.array([1., 4., 10.]),)),
], ids=idfn)
@pytest.mark.parametrize('prepend_shape', [
    None,
    (),
    (2,),
    (2, 3),
])
def test_discrete_shape(jax_dist, dist_args, prepend_shape):
    rng_key = random.PRNGKey(0)
    sp_dist = getattr(osp_stats, jax_dist.name)
    expected_shape = np.shape(sp_dist.rvs(*dist_args))
    samples = jax_dist.rvs(*dist_args, random_state=rng_key)
    assert isinstance(samples, jax.interpreters.xla.DeviceArray)
    assert np.shape(samples) == expected_shape
    if prepend_shape is not None:
        shape = prepend_shape + lax.broadcast_shapes(*[np.shape(arg) for arg in dist_args])
        expected_shape = np.shape(sp_dist.rvs(*dist_args, size=shape))
        assert np.shape(jax_dist.rvs(*dist_args, size=shape, random_state=rng_key)) == expected_shape


@pytest.mark.parametrize('jax_dist, valid_args, invalid_args, invalid_sample', [
    (dist.bernoulli, (0.8,), (np.nan,), 2),
    (dist.binom, (10, 0.8), (-10, 0.8), -10),
    (dist.binom, (10, 0.8), (10, 1.1), -1),
    (dist.poisson, (4.,), (-1.,), -1),
], ids=idfn)
def test_discrete_validate_args(jax_dist, valid_args, invalid_args, invalid_sample):
    with validation_enabled():
        with pytest.raises(ValueError, match='Invalid parameters'):
            jax_dist(*invalid_args)

        frozen_dist = jax_dist(*valid_args)
        with pytest.raises(ValueError, match='Invalid values'):
            frozen_dist.logpmf(invalid_sample)


@pytest.mark.parametrize('jax_dist', [
    dist.beta,
    dist.cauchy,
    dist.expon,
    dist.gamma,
    dist.halfcauchy,
    dist.halfnorm,
    dist.lognorm,
    dist.norm,
    dist.pareto,
    dist.t,
    dist.trunccauchy,
    dist.truncnorm,
    dist.uniform,
], ids=idfn)
@pytest.mark.parametrize('loc, scale', [
    (1., 1.),
    (1., np.array([1., 2.])),
])
def test_sample_gradient(jax_dist, loc, scale):
    rng_key = random.PRNGKey(0)
    args = [i + 1. for i in range(jax_dist.numargs)]
    expected_shape = lax.broadcast_shapes(*[np.shape(loc), np.shape(scale)])

    def fn(args, loc, scale):
        return jax_dist.rvs(*args, loc=loc, scale=scale, random_state=rng_key).sum()

    # FIXME: find a proper test for gradients of arg parameters
    assert len(grad(fn)(args, loc, scale)) == jax_dist.numargs
    assert_allclose(grad(fn, 1)(args, loc, scale),
                    loc * reduce(mul, expected_shape[:len(expected_shape) - np.ndim(loc)], 1.))
    assert_allclose(grad(fn, 2)(args, loc, scale),
                    jax_dist.rvs(*args, size=expected_shape, random_state=rng_key))


@pytest.mark.parametrize('jax_dist, dist_args', [
    (dist.dirichlet, (np.ones(3),)),
    (dist.dirichlet, (np.ones((2, 3)),)),
], ids=idfn)
def test_mvsample_gradient(jax_dist, dist_args):
    rng_key = random.PRNGKey(0)

    def fn(args):
        return jax_dist.rvs(*args, random_state=rng_key).sum()

    # FIXME: find a proper test for gradients of arg parameters
    assert len(grad(fn)(dist_args)) == jax_dist.numargs


@pytest.mark.parametrize('jax_dist', [
    dist.beta,
    dist.cauchy,
    dist.expon,
    dist.gamma,
    dist.halfcauchy,
    dist.halfnorm,
    dist.lognorm,
    dist.norm,
    dist.pareto,
    dist.t,
    dist.trunccauchy,
    dist.truncnorm,
    dist.uniform,
], ids=idfn)
@pytest.mark.parametrize('loc_scale', [
    (),
    (1,),
    (1, 1),
    (1., np.array([1., 2.])),
])
def test_continuous_logpdf(jax_dist, loc_scale):
    rng_key = random.PRNGKey(0)
    args = [i + 1 for i in range(jax_dist.numargs)] + list(loc_scale)
    samples = jax_dist.rvs(*args, random_state=rng_key)
    if jax_dist is dist.trunccauchy:
        sp_dist = osp_stats.cauchy
        assert_allclose(jax_dist.logpdf(samples, args[0], args[1]),
                        sp_dist.logpdf(samples) - np.log(sp_dist.cdf(args[1]) - sp_dist.cdf(args[0])),
                        atol=1e-6)
    else:
        sp_dist = getattr(osp_stats, jax_dist.name)
        assert_allclose(jax_dist.logpdf(samples, *args), sp_dist.logpdf(samples, *args), atol=1.3e-6)


@pytest.mark.parametrize('jax_dist, dist_args', [
    (dist.dirichlet, (np.array([1., 2., 3.]),)),
], ids=idfn)
@pytest.mark.parametrize('shape', [
    None,
    (),
    (2,),
    (2, 3),
])
def test_multivariate_continuous_logpdf(jax_dist, dist_args, shape):
    rng_key = random.PRNGKey(0)
    samples = jax_dist.rvs(*dist_args, size=shape, random_state=rng_key)
    # XXX scipy.stats.dirichlet does not work with batch
    if samples.ndim == 1:
        sp_dist = getattr(osp_stats, jax_dist.name)
        assert_allclose(jax_dist.logpdf(samples, *dist_args),
                        sp_dist.logpdf(samples, *dist_args), atol=1e-6)

    event_dim = len(jax_dist._event_shape(*dist_args))
    batch_shape = samples.shape if event_dim == 0 else samples.shape[:-1]
    assert jax_dist.logpdf(samples, *dist_args).shape == batch_shape


@pytest.mark.parametrize('jax_dist, dist_args', [
    (dist.categorical, (np.array([0.7, 0.3]),)),
    (dist.multinomial, (10, np.array([0.3, 0.7]),)),
], ids=idfn)
@pytest.mark.parametrize('shape', [
    None,
    (),
    (2,),
    (2, 3),
])
def test_multivariate_discrete_logpmf(jax_dist, dist_args, shape):
    rng_key = random.PRNGKey(0)
    samples = jax_dist.rvs(*dist_args, size=shape, random_state=rng_key)
    # XXX scipy.stats.multinomial does not work with batch
    if samples.ndim == 1:
        if jax_dist is dist.categorical:
            # test against PyTorch
            assert_allclose(jax_dist.logpmf(np.array([1, 0]), *dist_args),
                            np.array([-1.2040, -0.3567]), atol=1e-4)
        else:
            sp_dist = getattr(osp_stats, jax_dist.name)
            assert_allclose(jax_dist.logpmf(samples, *dist_args),
                            sp_dist.logpmf(samples, *dist_args), atol=1e-5)

    event_dim = len(jax_dist._event_shape(*dist_args))
    batch_shape = samples.shape if event_dim == 0 else samples.shape[:-1]
    assert jax_dist.logpmf(samples, *dist_args).shape == batch_shape


@pytest.mark.parametrize('jax_dist, dist_args', [
    (dist.bernoulli, (0.1,)),
    (dist.bernoulli, (np.array([0.3, 0.5]),)),
    (dist.binom, (10, 0.4)),
    (dist.binom, (np.array([10]), np.array([0.4, 0.3]))),
    (dist.binom, (np.array([2, 5]), np.array([[0.4], [0.5]]))),
    (dist.poisson, (4.,)),
    (dist.poisson, (np.array([1., 4., 10.]),)),
], ids=idfn)
@pytest.mark.parametrize('shape', [
    None,
    (),
    (2,),
    (2, 3),
])
def test_discrete_logpmf(jax_dist, dist_args, shape):
    rng_key = random.PRNGKey(0)
    sp_dist = getattr(osp_stats, jax_dist.name)
    samples = jax_dist.rvs(*dist_args, random_state=rng_key)
    assert_allclose(jax_dist.logpmf(samples, *dist_args),
                    sp_dist.logpmf(onp.asarray(samples), *dist_args),
                    rtol=1e-5)
    if shape is not None:
        shape = shape + lax.broadcast_shapes(*[np.shape(arg) for arg in dist_args])
        samples = jax_dist.rvs(*dist_args, size=shape, random_state=rng_key)
        assert_allclose(jax_dist.logpmf(samples, *dist_args),
                        sp_dist.logpmf(onp.asarray(samples), *dist_args),
                        rtol=1e-5)

        def fn(sample, *args):
            return np.sum(jax_dist.logpmf(sample, *args))

        for i in range(len(dist_args)):
            if np.result_type(dist_args[i]) in (np.int32, np.int64):
                continue
            logpmf_grad = grad(fn, i + 1)(samples, *dist_args)
            assert np.all(np.isfinite(logpmf_grad))


@pytest.mark.parametrize('jax_dist, dist_args', [
    (dist.bernoulli, (0.1,)),
    (dist.bernoulli, (np.array([0.3, 0.5]),)),
    (dist.binom, (10, 0.4)),
    (dist.binom, (np.array([10]), np.array([0.4, 0.3]))),
    (dist.binom, (np.array([2, 5]), np.array([[0.4], [0.5]]))),
    (dist.categorical, (np.array([0.1, 0.9]),)),
    (dist.categorical, (np.array([[0.1, 0.9], [0.2, 0.8]]),)),
    (dist.multinomial, (10, np.array([0.1, 0.9]),)),
    (dist.multinomial, (10, np.array([[0.1, 0.9], [0.2, 0.8]]),)),
], ids=idfn)
def test_discrete_with_logits(jax_dist, dist_args):
    rng_key = random.PRNGKey(0)
    logit_to_prob = np.log if isinstance(jax_dist, jax_multivariate) else logit
    logit_args = dist_args[:-1] + (logit_to_prob(dist_args[-1]),)

    actual_sample = jax_dist.rvs(*dist_args, random_state=rng_key)
    expected_sample = jax_dist(*logit_args, is_logits=True).rvs(random_state=rng_key)
    assert_allclose(actual_sample, expected_sample)

    actual_pmf = jax_dist.logpmf(actual_sample, *dist_args)
    expected_pmf = jax_dist(*logit_args, is_logits=True).logpmf(actual_sample)
    assert_allclose(actual_pmf, expected_pmf, rtol=1e-6)
