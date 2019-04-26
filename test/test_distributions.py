from functools import reduce
from operator import mul

import numpy as onp
import pytest
import scipy.stats as osp_stats
from numpy.testing import assert_allclose, assert_array_equal

import jax
import jax.numpy as np
from jax import grad, lax, random
from jax.scipy.special import logit

import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.constraint_registry import biject_to
from numpyro.distributions.distribution import jax_multivariate, validation_enabled


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
    dist.lognorm,
    dist.pareto,
    dist.trunccauchy,
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
    rng = random.PRNGKey(0)
    args = [i + 1 for i in range(jax_dist.numargs)]
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
    rng = random.PRNGKey(0)
    expected_shape = jax_dist._batch_shape(*dist_args) + jax_dist._event_shape(*dist_args)
    samples = jax_dist.rvs(*dist_args, random_state=rng)
    assert isinstance(samples, jax.interpreters.xla.DeviceArray)
    assert np.shape(samples) == expected_shape
    assert np.shape(jax_dist(*dist_args).rvs(random_state=rng)) == expected_shape
    if prepend_shape is not None:
        size = prepend_shape + jax_dist._batch_shape(*dist_args)
        expected_shape = size + jax_dist._event_shape(*dist_args)
        samples = jax_dist.rvs(*dist_args, size=size, random_state=rng)
        assert np.shape(samples) == expected_shape
        samples = jax_dist(*dist_args).rvs(random_state=rng, size=size)
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
], ids=idfn)
@pytest.mark.parametrize('prepend_shape', [
    None,
    (),
    (2,),
    (2, 3),
])
def test_discrete_shape(jax_dist, dist_args, prepend_shape):
    rng = random.PRNGKey(0)
    sp_dist = getattr(osp_stats, jax_dist.name)
    expected_shape = np.shape(sp_dist.rvs(*dist_args))
    samples = jax_dist.rvs(*dist_args, random_state=rng)
    assert isinstance(samples, jax.interpreters.xla.DeviceArray)
    assert np.shape(samples) == expected_shape
    if prepend_shape is not None:
        shape = prepend_shape + lax.broadcast_shapes(*[np.shape(arg) for arg in dist_args])
        expected_shape = np.shape(sp_dist.rvs(*dist_args, size=shape))
        assert np.shape(jax_dist.rvs(*dist_args, size=shape, random_state=rng)) == expected_shape


@pytest.mark.parametrize('jax_dist, valid_args, invalid_args, invalid_sample', [
    (dist.bernoulli, (0.8,), (np.nan,), 2),
    (dist.binom, (10, 0.8), (-10, 0.8), -10),
    (dist.binom, (10, 0.8), (10, 1.1), -1),
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
    dist.lognorm,
    dist.norm,
    dist.pareto,
    dist.t,
    pytest.param(dist.trunccauchy, marks=pytest.mark.xfail(
        reason='jvp rule for np.arctan is not yet available')),
    dist.uniform,
], ids=idfn)
@pytest.mark.parametrize('loc, scale', [
    (1., 1.),
    (1., np.array([1., 2.])),
])
def test_sample_gradient(jax_dist, loc, scale):
    rng = random.PRNGKey(0)
    args = [i + 1 for i in range(jax_dist.numargs)]
    expected_shape = lax.broadcast_shapes(*[np.shape(loc), np.shape(scale)])

    def fn(args, loc, scale):
        return jax_dist.rvs(*args, loc=loc, scale=scale, random_state=rng).sum()

    # FIXME: find a proper test for gradients of arg parameters
    assert len(grad(fn)(args, loc, scale)) == jax_dist.numargs
    assert_allclose(grad(fn, 1)(args, loc, scale),
                    loc * reduce(mul, expected_shape[:len(expected_shape) - np.ndim(loc)], 1.))
    assert_allclose(grad(fn, 2)(args, loc, scale),
                    jax_dist.rvs(*args, size=expected_shape, random_state=rng))


@pytest.mark.parametrize('jax_dist, dist_args', [
    (dist.dirichlet, (np.ones(3),)),
    (dist.dirichlet, (np.ones((2, 3)),)),
], ids=idfn)
def test_mvsample_gradient(jax_dist, dist_args):
    rng = random.PRNGKey(0)

    def fn(args):
        return jax_dist.rvs(*args, random_state=rng).sum()

    # FIXME: find a proper test for gradients of arg parameters
    assert len(grad(fn)(dist_args)) == jax_dist.numargs


@pytest.mark.parametrize('jax_dist', [
    dist.beta,
    dist.cauchy,
    dist.expon,
    dist.gamma,
    dist.halfcauchy,
    dist.lognorm,
    dist.norm,
    dist.pareto,
    dist.t,
    dist.trunccauchy,
    dist.uniform,
], ids=idfn)
@pytest.mark.parametrize('loc_scale', [
    (),
    (1,),
    (1, 1),
    (1., np.array([1., 2.])),
])
def test_continuous_logpdf(jax_dist, loc_scale):
    rng = random.PRNGKey(0)
    args = [i + 1 for i in range(jax_dist.numargs)] + list(loc_scale)
    samples = jax_dist.rvs(*args, random_state=rng)
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
    rng = random.PRNGKey(0)
    samples = jax_dist.rvs(*dist_args, size=shape, random_state=rng)
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
    rng = random.PRNGKey(0)
    samples = jax_dist.rvs(*dist_args, size=shape, random_state=rng)
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
], ids=idfn)
@pytest.mark.parametrize('shape', [
    None,
    (),
    (2,),
    (2, 3),
])
def test_discrete_logpmf(jax_dist, dist_args, shape):
    rng = random.PRNGKey(0)
    sp_dist = getattr(osp_stats, jax_dist.name)
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

        def fn(sample, *args):
            return np.sum(jax_dist.logpmf(sample, *args))

        for i in range(len(dist_args)):
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
    rng = random.PRNGKey(0)
    logit_to_prob = np.log if isinstance(jax_dist, jax_multivariate) else logit
    logit_args = dist_args[:-1] + (logit_to_prob(dist_args[-1]),)

    actual_sample = jax_dist.rvs(*dist_args, random_state=rng)
    expected_sample = jax_dist(*logit_args, is_logits=True).rvs(random_state=rng)
    assert_allclose(actual_sample, expected_sample)

    actual_pmf = jax_dist.logpmf(actual_sample, *dist_args)
    expected_pmf = jax_dist(*logit_args, is_logits=True).logpmf(actual_sample)
    assert_allclose(actual_pmf, expected_pmf, rtol=1e-6)


########################################
# Tests for constraints and transforms #
########################################


@pytest.mark.parametrize('constraint, x, expected', [
    (constraints.boolean, np.array([True, False]), np.array([True, True])),
    (constraints.boolean, np.array([1, 1]), np.array([True, True])),
    (constraints.boolean, np.array([-1, 1]), np.array([False, True])),
    (constraints.greater_than(1), 3, True),
    (constraints.greater_than(1), np.array([-1, 1, 5]), np.array([False, False, True])),
    (constraints.integer_interval(-3, 5), 0, True),
    (constraints.integer_interval(-3, 5), np.array([-5, -3, 0, 1.1, 5, 7]),
     np.array([False, True, True, False, True, False])),
    (constraints.interval(-3, 5), 0, True),
    (constraints.interval(-3, 5), np.array([-5, -3, 0, 5, 7]),
     np.array([False, False, True, False, False])),
    (constraints.nonnegative_integer, 3, True),
    (constraints.nonnegative_integer, np.array([-1., 0., 5.]), np.array([False, True, True])),
    (constraints.positive, 3, True),
    (constraints.positive, np.array([-1, 0, 5]), np.array([False, False, True])),
    (constraints.positive_integer, 3, True),
    (constraints.positive_integer, np.array([-1., 0., 5.]), np.array([False, False, True])),
    (constraints.real, -1, True),
    (constraints.real, np.array([np.inf, np.NINF, np.nan, np.pi]),
     np.array([False, False, False, True])),
    (constraints.simplex, np.array([0.1, 0.3, 0.6]), True),
    (constraints.simplex, np.array([[0.1, 0.3, 0.6], [-0.1, 0.6, 0.5], [0.1, 0.6, 0.5]]),
     np.array([True, False, False])),
    (constraints.unit_interval, 0.1, True),
    (constraints.unit_interval, np.array([-5, 0, 0.5, 1, 7]),
     np.array([False, False, True, False, False])),
], ids=idfn)
def test_constraints(constraint, x, expected):
    assert_array_equal(constraint(x), expected)


@pytest.mark.parametrize('shape', [(), (1,), (3,), (5,), (3, 1), (1, 3), (5, 3)], ids=idfn)
@pytest.mark.parametrize('constraint', [
    constraints.greater_than(2),
    constraints.interval(-3, 5),
    constraints.positive,
    constraints.real,
    constraints.simplex,
    constraints.unit_interval,
], ids=idfn)
def test_biject_to(constraint, shape):
    if constraint is constraints.simplex and not shape:
        return
    transform = biject_to(constraint)
    rng = random.PRNGKey(0)
    x = random.normal(rng, shape)
    y = transform(x)

    # test codomain
    batch_shape = shape if transform.event_dim == 0 else shape[:-1]
    assert_array_equal(transform.codomain(y), np.ones(batch_shape))

    # test inv
    z = transform.inv(y)
    assert_allclose(x, z, atol=1e-6)

    # test domain, currently all is constraints.real
    assert_array_equal(transform.domain(z), np.ones(shape))

    # test log_abs_det_jacobian
    actual = transform.log_abs_det_jacobian(x, y)
    assert np.shape(actual) == batch_shape
    if len(shape) == transform.event_dim:
        if constraint is constraints.simplex:
            expected = np.linalg.slogdet(jax.jacobian(transform)(x)[:-1, :])[1]
            inv_expected = np.linalg.slogdet(jax.jacobian(transform.inv)(y)[:, :-1])[1]
        else:
            expected = np.log(np.abs(grad(transform)(x)))
            inv_expected = np.log(np.abs(grad(transform.inv)(y)))

        assert_allclose(actual, expected, atol=1e-6)
        assert_allclose(actual, -inv_expected, atol=1e-6)
