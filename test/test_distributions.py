# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from functools import partial
import inspect
import os

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
import scipy.stats as osp

import jax
from jax import grad, jacfwd, lax, vmap
import jax.numpy as jnp
import jax.random as random
from jax.scipy.special import logsumexp

import numpyro.distributions as dist
from numpyro.distributions import constraints, kl_divergence, transforms
from numpyro.distributions.discrete import _to_probs_bernoulli, _to_probs_multinom
from numpyro.distributions.flows import InverseAutoregressiveTransform
from numpyro.distributions.transforms import LowerCholeskyAffine, PermuteTransform, PowerTransform, biject_to
from numpyro.distributions.util import (matrix_to_tril_vec, multinomial, signed_stick_breaking_tril,
                                        sum_rightmost, vec_to_tril_matrix)
from numpyro.nn import AutoregressiveNN


def _identity(x): return x


class T(namedtuple('TestCase', ['jax_dist', 'sp_dist', 'params'])):
    def __new__(cls, jax_dist, *params):
        sp_dist = None
        if jax_dist in _DIST_MAP:
            sp_dist = _DIST_MAP[jax_dist]
        return super(cls, T).__new__(cls, jax_dist, sp_dist, params)


def _mvn_to_scipy(loc, cov, prec, tril):
    jax_dist = dist.MultivariateNormal(loc, cov, prec, tril)
    mean = jax_dist.mean
    cov = jax_dist.covariance_matrix
    return osp.multivariate_normal(mean=mean, cov=cov)


def _lowrank_mvn_to_scipy(loc, cov_fac, cov_diag):
    jax_dist = dist.LowRankMultivariateNormal(loc, cov_fac, cov_diag)
    mean = jax_dist.mean
    cov = jax_dist.covariance_matrix
    return osp.multivariate_normal(mean=mean, cov=cov)


class _ImproperWrapper(dist.ImproperUniform):
    def sample(self, key, sample_shape=()):
        transform = biject_to(self.support)
        prototype_value = jnp.zeros(self.event_shape)
        unconstrained_event_shape = jnp.shape(transform.inv(prototype_value))
        shape = sample_shape + self.batch_shape + unconstrained_event_shape
        unconstrained_samples = random.uniform(key, shape,
                                               minval=-2,
                                               maxval=2)
        return transform(unconstrained_samples)


_DIST_MAP = {
    dist.BernoulliProbs: lambda probs: osp.bernoulli(p=probs),
    dist.BernoulliLogits: lambda logits: osp.bernoulli(p=_to_probs_bernoulli(logits)),
    dist.Beta: lambda con1, con0: osp.beta(con1, con0),
    dist.BinomialProbs: lambda probs, total_count: osp.binom(n=total_count, p=probs),
    dist.BinomialLogits: lambda logits, total_count: osp.binom(n=total_count, p=_to_probs_bernoulli(logits)),
    dist.Cauchy: lambda loc, scale: osp.cauchy(loc=loc, scale=scale),
    dist.Chi2: lambda df: osp.chi2(df),
    dist.Dirichlet: lambda conc: osp.dirichlet(conc),
    dist.Exponential: lambda rate: osp.expon(scale=jnp.reciprocal(rate)),
    dist.Gamma: lambda conc, rate: osp.gamma(conc, scale=1. / rate),
    dist.GeometricProbs: lambda probs: osp.geom(p=probs, loc=-1),
    dist.GeometricLogits: lambda logits: osp.geom(p=_to_probs_bernoulli(logits), loc=-1),
    dist.Gumbel: lambda loc, scale: osp.gumbel_r(loc=loc, scale=scale),
    dist.HalfCauchy: lambda scale: osp.halfcauchy(scale=scale),
    dist.HalfNormal: lambda scale: osp.halfnorm(scale=scale),
    dist.InverseGamma: lambda conc, rate: osp.invgamma(conc, scale=rate),
    dist.Laplace: lambda loc, scale: osp.laplace(loc=loc, scale=scale),
    dist.LogNormal: lambda loc, scale: osp.lognorm(s=scale, scale=jnp.exp(loc)),
    dist.MultinomialProbs: lambda probs, total_count: osp.multinomial(n=total_count, p=probs),
    dist.MultinomialLogits: lambda logits, total_count: osp.multinomial(n=total_count,
                                                                        p=_to_probs_multinom(logits)),
    dist.MultivariateNormal: _mvn_to_scipy,
    dist.LowRankMultivariateNormal: _lowrank_mvn_to_scipy,
    dist.Normal: lambda loc, scale: osp.norm(loc=loc, scale=scale),
    dist.Pareto: lambda scale, alpha: osp.pareto(alpha, scale=scale),
    dist.Poisson: lambda rate: osp.poisson(rate),
    dist.StudentT: lambda df, loc, scale: osp.t(df=df, loc=loc, scale=scale),
    dist.Uniform: lambda a, b: osp.uniform(a, b - a),
    dist.Logistic: lambda loc, scale: osp.logistic(loc=loc, scale=scale)
}

CONTINUOUS = [
    T(dist.Beta, 1., 2.),
    T(dist.Beta, 1., jnp.array([2., 2.])),
    T(dist.Beta, 1., jnp.array([[1., 1.], [2., 2.]])),
    T(dist.Chi2, 2.),
    T(dist.Chi2, jnp.array([0.3, 1.3])),
    T(dist.Cauchy, 0., 1.),
    T(dist.Cauchy, 0., jnp.array([1., 2.])),
    T(dist.Cauchy, jnp.array([0., 1.]), jnp.array([[1.], [2.]])),
    T(dist.Dirichlet, jnp.array([1.7])),
    T(dist.Dirichlet, jnp.array([0.2, 1.1])),
    T(dist.Dirichlet, jnp.array([[0.2, 1.1], [2., 2.]])),
    T(dist.Exponential, 2.),
    T(dist.Exponential, jnp.array([4., 2.])),
    T(dist.Gamma, jnp.array([1.7]), jnp.array([[2.], [3.]])),
    T(dist.Gamma, jnp.array([0.5, 1.3]), jnp.array([[1.], [3.]])),
    T(dist.GaussianRandomWalk, 0.1, 10),
    T(dist.GaussianRandomWalk, jnp.array([0.1, 0.3, 0.25]), 10),
    T(dist.Gumbel, 0., 1.),
    T(dist.Gumbel, 0.5, 2.),
    T(dist.Gumbel, jnp.array([0., 0.5]), jnp.array([1., 2.])),
    T(dist.HalfCauchy, 1.),
    T(dist.HalfCauchy, jnp.array([1., 2.])),
    T(dist.HalfNormal, 1.),
    T(dist.HalfNormal, jnp.array([1., 2.])),
    T(_ImproperWrapper, constraints.positive, (), (3,)),
    T(dist.InverseGamma, jnp.array([1.7]), jnp.array([[2.], [3.]])),
    T(dist.InverseGamma, jnp.array([0.5, 1.3]), jnp.array([[1.], [3.]])),
    T(dist.Laplace, 0., 1.),
    T(dist.Laplace, 0.5, jnp.array([1., 2.5])),
    T(dist.Laplace, jnp.array([1., -0.5]), jnp.array([2.3, 3.])),
    T(dist.LKJ, 2, 0.5, "onion"),
    T(dist.LKJ, 5, jnp.array([0.5, 1., 2.]), "cvine"),
    T(dist.LKJCholesky, 2, 0.5, "onion"),
    T(dist.LKJCholesky, 2, 0.5, "cvine"),
    T(dist.LKJCholesky, 5, jnp.array([0.5, 1., 2.]), "onion"),
    pytest.param(*T(dist.LKJCholesky, 5, jnp.array([0.5, 1., 2.]), "cvine"),
                 marks=pytest.mark.skipif('CI' in os.environ, reason="reduce time for Travis")),
    pytest.param(*T(dist.LKJCholesky, 3, jnp.array([[3., 0.6], [0.2, 5.]]), "onion"),
                 marks=pytest.mark.skipif('CI' in os.environ, reason="reduce time for Travis")),
    T(dist.LKJCholesky, 3, jnp.array([[3., 0.6], [0.2, 5.]]), "cvine"),
    T(dist.Logistic, 0., 1.),
    T(dist.Logistic, 1., jnp.array([1., 2.])),
    T(dist.Logistic, jnp.array([0., 1.]), jnp.array([[1.], [2.]])),
    T(dist.LogNormal, 1., 0.2),
    T(dist.LogNormal, -1., jnp.array([0.5, 1.3])),
    T(dist.LogNormal, jnp.array([0.5, -0.7]), jnp.array([[0.1, 0.4], [0.5, 0.1]])),
    T(dist.MultivariateNormal, 0., jnp.array([[1., 0.5], [0.5, 1.]]), None, None),
    T(dist.MultivariateNormal, jnp.array([1., 3.]), None, jnp.array([[1., 0.5], [0.5, 1.]]), None),
    T(dist.MultivariateNormal, jnp.array([1., 3.]), None, jnp.array([[[1., 0.5], [0.5, 1.]]]), None),
    T(dist.MultivariateNormal, jnp.array([2.]), None, None, jnp.array([[1., 0.], [0.5, 1.]])),
    T(dist.MultivariateNormal, jnp.arange(6, dtype=jnp.float32).reshape((3, 2)), None, None,
      jnp.array([[1., 0.], [0., 1.]])),
    T(dist.MultivariateNormal, 0., None, jnp.broadcast_to(jnp.identity(3), (2, 3, 3)), None),
    T(dist.LowRankMultivariateNormal, jnp.zeros(2), jnp.array([[1], [0]]), jnp.array([1, 1])),
    T(dist.LowRankMultivariateNormal, jnp.arange(6, dtype=jnp.float32).reshape((2, 3)),
      jnp.arange(6, dtype=jnp.float32).reshape((3, 2)), jnp.array([1, 2, 3])),
    T(dist.Normal, 0., 1.),
    T(dist.Normal, 1., jnp.array([1., 2.])),
    T(dist.Normal, jnp.array([0., 1.]), jnp.array([[1.], [2.]])),
    T(dist.Pareto, 1., 2.),
    T(dist.Pareto, jnp.array([1., 0.5]), jnp.array([0.3, 2.])),
    T(dist.Pareto, jnp.array([[1.], [3.]]), jnp.array([1., 0.5])),
    T(dist.StudentT, 1., 1., 0.5),
    T(dist.StudentT, 2., jnp.array([1., 2.]), 2.),
    T(dist.StudentT, jnp.array([3, 5]), jnp.array([[1.], [2.]]), 2.),
    T(dist.TruncatedCauchy, -1., 0., 1.),
    T(dist.TruncatedCauchy, 1., 0., jnp.array([1., 2.])),
    T(dist.TruncatedCauchy, jnp.array([-2., 2.]), jnp.array([0., 1.]), jnp.array([[1.], [2.]])),
    T(dist.TruncatedNormal, -1., 0., 1.),
    T(dist.TruncatedNormal, 1., -1., jnp.array([1., 2.])),
    T(dist.TruncatedNormal, jnp.array([-2., 2.]), jnp.array([0., 1.]), jnp.array([[1.], [2.]])),
    T(dist.Uniform, 0., 2.),
    T(dist.Uniform, 1., jnp.array([2., 3.])),
    T(dist.Uniform, jnp.array([0., 0.]), jnp.array([[2.], [3.]])),
]

DIRECTIONAL = [
    T(dist.VonMises, 2., 10.),
    T(dist.VonMises, 2., jnp.array([150., 10.])),
    T(dist.VonMises, jnp.array([1 / 3 * jnp.pi, -1.]), jnp.array([20., 30.])),
]

DISCRETE = [
    T(dist.BetaBinomial, 2., 5., 10),
    T(dist.BetaBinomial, jnp.array([2., 4.]), jnp.array([5., 3.]), jnp.array([10, 12])),
    T(dist.BernoulliProbs, 0.2),
    T(dist.BernoulliProbs, jnp.array([0.2, 0.7])),
    T(dist.BernoulliLogits, jnp.array([-1., 3.])),
    T(dist.BinomialProbs, jnp.array([0.2, 0.7]), jnp.array([10, 2])),
    T(dist.BinomialProbs, jnp.array([0.2, 0.7]), jnp.array([5, 8])),
    T(dist.BinomialLogits, jnp.array([-1., 3.]), jnp.array([5, 8])),
    T(dist.CategoricalProbs, jnp.array([1.])),
    T(dist.CategoricalProbs, jnp.array([0.1, 0.5, 0.4])),
    T(dist.CategoricalProbs, jnp.array([[0.1, 0.5, 0.4], [0.4, 0.4, 0.2]])),
    T(dist.CategoricalLogits, jnp.array([-5.])),
    T(dist.CategoricalLogits, jnp.array([1., 2., -2.])),
    T(dist.CategoricalLogits, jnp.array([[-1, 2., 3.], [3., -4., -2.]])),
    T(dist.Delta, 1),
    T(dist.Delta, jnp.array([0., 2.])),
    T(dist.Delta, jnp.array([0., 2.]), jnp.array([-2., -4.])),
    T(dist.DirichletMultinomial, jnp.array([1.0, 2.0, 3.9]), 10),
    T(dist.DirichletMultinomial, jnp.array([0.2, 0.7, 1.1]), jnp.array([5, 5])),
    T(dist.GammaPoisson, 2., 2.),
    T(dist.GammaPoisson, jnp.array([6., 2]), jnp.array([2., 8.])),
    T(dist.GeometricProbs, 0.2),
    T(dist.GeometricProbs, jnp.array([0.2, 0.7])),
    T(dist.GeometricLogits, jnp.array([-1., 3.])),
    T(dist.MultinomialProbs, jnp.array([0.2, 0.7, 0.1]), 10),
    T(dist.MultinomialProbs, jnp.array([0.2, 0.7, 0.1]), jnp.array([5, 8])),
    T(dist.MultinomialLogits, jnp.array([-1., 3.]), jnp.array([[5], [8]])),
    T(dist.OrderedLogistic, -2, jnp.array([-10., 4., 9.])),
    T(dist.OrderedLogistic, jnp.array([-4, 3, 4, 5]), jnp.array([-1.5])),
    T(dist.Poisson, 2.),
    T(dist.Poisson, jnp.array([2., 3., 5.])),
    T(dist.ZeroInflatedPoisson, 0.6, 2.),
    T(dist.ZeroInflatedPoisson, jnp.array([0.2, 0.7, 0.3]), jnp.array([2., 3., 5.])),
]


def _is_batched_multivariate(jax_dist):
    return len(jax_dist.event_shape) > 0 and len(jax_dist.batch_shape) > 0


def gen_values_within_bounds(constraint, size, key=random.PRNGKey(11)):
    eps = 1e-6

    if isinstance(constraint, constraints._Boolean):
        return random.bernoulli(key, shape=size)
    elif isinstance(constraint, constraints._GreaterThan):
        return jnp.exp(random.normal(key, size)) + constraint.lower_bound + eps
    elif isinstance(constraint, constraints._IntegerInterval):
        lower_bound = jnp.broadcast_to(constraint.lower_bound, size)
        upper_bound = jnp.broadcast_to(constraint.upper_bound, size)
        return random.randint(key, size, lower_bound, upper_bound + 1)
    elif isinstance(constraint, constraints._IntegerGreaterThan):
        return constraint.lower_bound + random.poisson(key, np.array(5), shape=size)
    elif isinstance(constraint, constraints._Interval):
        lower_bound = jnp.broadcast_to(constraint.lower_bound, size)
        upper_bound = jnp.broadcast_to(constraint.upper_bound, size)
        return random.uniform(key, size, minval=lower_bound, maxval=upper_bound)
    elif isinstance(constraint, (constraints._Real, constraints._RealVector)):
        return random.normal(key, size)
    elif isinstance(constraint, constraints._Simplex):
        return osp.dirichlet.rvs(alpha=jnp.ones((size[-1],)), size=size[:-1])
    elif isinstance(constraint, constraints._Multinomial):
        n = size[-1]
        return multinomial(key, p=jnp.ones((n,)) / n, n=constraint.upper_bound, shape=size[:-1])
    elif isinstance(constraint, constraints._CorrCholesky):
        return signed_stick_breaking_tril(
            random.uniform(key, size[:-2] + (size[-1] * (size[-1] - 1) // 2,), minval=-1, maxval=1))
    elif isinstance(constraint, constraints._CorrMatrix):
        cholesky = signed_stick_breaking_tril(
            random.uniform(key, size[:-2] + (size[-1] * (size[-1] - 1) // 2,), minval=-1, maxval=1))
        return jnp.matmul(cholesky, jnp.swapaxes(cholesky, -2, -1))
    elif isinstance(constraint, constraints._LowerCholesky):
        return jnp.tril(random.uniform(key, size))
    elif isinstance(constraint, constraints._PositiveDefinite):
        x = random.normal(key, size)
        return jnp.matmul(x, jnp.swapaxes(x, -2, -1))
    elif isinstance(constraint, constraints._OrderedVector):
        x = jnp.cumsum(random.exponential(key, size), -1)
        return x - random.normal(key, size[:-1])
    else:
        raise NotImplementedError('{} not implemented.'.format(constraint))


def gen_values_outside_bounds(constraint, size, key=random.PRNGKey(11)):
    if isinstance(constraint, constraints._Boolean):
        return random.bernoulli(key, shape=size) - 2
    elif isinstance(constraint, constraints._GreaterThan):
        return constraint.lower_bound - jnp.exp(random.normal(key, size))
    elif isinstance(constraint, constraints._IntegerInterval):
        lower_bound = jnp.broadcast_to(constraint.lower_bound, size)
        return random.randint(key, size, lower_bound - 1, lower_bound)
    elif isinstance(constraint, constraints._IntegerGreaterThan):
        return constraint.lower_bound - random.poisson(key, np.array(5), shape=size)
    elif isinstance(constraint, constraints._Interval):
        upper_bound = jnp.broadcast_to(constraint.upper_bound, size)
        return random.uniform(key, size, minval=upper_bound, maxval=upper_bound + 1.)
    elif isinstance(constraint, (constraints._Real, constraints._RealVector)):
        return lax.full(size, jnp.nan)
    elif isinstance(constraint, constraints._Simplex):
        return osp.dirichlet.rvs(alpha=jnp.ones((size[-1],)), size=size[:-1]) + 1e-2
    elif isinstance(constraint, constraints._Multinomial):
        n = size[-1]
        return multinomial(key, p=jnp.ones((n,)) / n, n=constraint.upper_bound, shape=size[:-1]) + 1
    elif isinstance(constraint, constraints._CorrCholesky):
        return signed_stick_breaking_tril(
            random.uniform(key, size[:-2] + (size[-1] * (size[-1] - 1) // 2,),
                           minval=-1, maxval=1)) + 1e-2
    elif isinstance(constraint, constraints._CorrMatrix):
        cholesky = 1e-2 + signed_stick_breaking_tril(
            random.uniform(key, size[:-2] + (size[-1] * (size[-1] - 1) // 2,), minval=-1, maxval=1))
        return jnp.matmul(cholesky, jnp.swapaxes(cholesky, -2, -1))
    elif isinstance(constraint, constraints._LowerCholesky):
        return random.uniform(key, size)
    elif isinstance(constraint, constraints._PositiveDefinite):
        return random.normal(key, size)
    elif isinstance(constraint, constraints._OrderedVector):
        x = jnp.cumsum(random.exponential(key, size), -1)
        return x[..., ::-1]
    else:
        raise NotImplementedError('{} not implemented.'.format(constraint))


@pytest.mark.parametrize('jax_dist, sp_dist, params', CONTINUOUS + DISCRETE + DIRECTIONAL)
@pytest.mark.parametrize('prepend_shape', [
    (),
    (2,),
    (2, 3),
])
def test_dist_shape(jax_dist, sp_dist, params, prepend_shape):
    jax_dist = jax_dist(*params)
    rng_key = random.PRNGKey(0)
    expected_shape = prepend_shape + jax_dist.batch_shape + jax_dist.event_shape
    samples = jax_dist.sample(key=rng_key, sample_shape=prepend_shape)
    assert isinstance(samples, jax.interpreters.xla.DeviceArray)
    assert jnp.shape(samples) == expected_shape
    if sp_dist and not _is_batched_multivariate(jax_dist):
        sp_dist = sp_dist(*params)
        sp_samples = sp_dist.rvs(size=prepend_shape + jax_dist.batch_shape)
        assert jnp.shape(sp_samples) == expected_shape
    if isinstance(jax_dist, dist.MultivariateNormal):
        assert jax_dist.covariance_matrix.ndim == len(jax_dist.batch_shape) + 2
        assert_allclose(jax_dist.precision_matrix, jnp.linalg.inv(jax_dist.covariance_matrix), rtol=1e-6)


@pytest.mark.parametrize('batch_shape', [(), (4,), (3, 2)])
def test_unit(batch_shape):
    log_factor = random.normal(random.PRNGKey(0), batch_shape)

    d = dist.Unit(log_factor=log_factor)
    x = d.sample(random.PRNGKey(1))
    assert x.shape == batch_shape + (0,)
    assert (d.log_prob(x) == log_factor).all()


@pytest.mark.parametrize('jax_dist, sp_dist, params', CONTINUOUS)
def test_sample_gradient(jax_dist, sp_dist, params):
    if not jax_dist.reparametrized_params:
        pytest.skip('{} not reparametrized.'.format(jax_dist.__name__))

    dist_args = [p.name for p in inspect.signature(jax_dist).parameters.values()]
    params_dict = dict(zip(dist_args[:len(params)], params))
    nonrepara_params_dict = {k: v for k, v in params_dict.items()
                             if k not in jax_dist.reparametrized_params}
    repara_params = tuple(v for k, v in params_dict.items()
                          if k in jax_dist.reparametrized_params)

    rng_key = random.PRNGKey(0)

    def fn(args):
        args_dict = dict(zip(jax_dist.reparametrized_params, args))
        return jnp.sum(jax_dist(**args_dict, **nonrepara_params_dict).sample(key=rng_key))

    actual_grad = jax.grad(fn)(repara_params)
    assert len(actual_grad) == len(repara_params)

    eps = 1e-3
    for i in range(len(repara_params)):
        if repara_params[i] is None:
            continue
        args_lhs = [p if j != i else p - eps for j, p in enumerate(repara_params)]
        args_rhs = [p if j != i else p + eps for j, p in enumerate(repara_params)]
        fn_lhs = fn(args_lhs)
        fn_rhs = fn(args_rhs)
        # finite diff approximation
        expected_grad = (fn_rhs - fn_lhs) / (2. * eps)
        assert jnp.shape(actual_grad[i]) == jnp.shape(repara_params[i])
        assert_allclose(jnp.sum(actual_grad[i]), expected_grad, rtol=0.02)


@pytest.mark.parametrize('jax_dist, sp_dist, params', [
    (dist.Gamma, osp.gamma, (1.,)),
    (dist.Gamma, osp.gamma, (0.1,)),
    (dist.Gamma, osp.gamma, (10.,)),
    # TODO: add more test cases for Beta/StudentT (and Dirichlet too) when
    # their pathwise grad (independent of standard_gamma grad) is implemented.
    pytest.param(dist.Beta, osp.beta, (1., 1.), marks=pytest.mark.xfail(
        reason='currently, variance of grad of beta sampler is large')),
    pytest.param(dist.StudentT, osp.t, (1.,), marks=pytest.mark.xfail(
        reason='currently, variance of grad of t sampler is large')),
])
def test_pathwise_gradient(jax_dist, sp_dist, params):
    rng_key = random.PRNGKey(0)
    N = 100
    z = jax_dist(*params).sample(key=rng_key, sample_shape=(N,))
    actual_grad = jacfwd(lambda x: jax_dist(*x).sample(key=rng_key, sample_shape=(N,)))(params)
    eps = 1e-3
    for i in range(len(params)):
        args_lhs = [p if j != i else p - eps for j, p in enumerate(params)]
        args_rhs = [p if j != i else p + eps for j, p in enumerate(params)]
        cdf_dot = (sp_dist(*args_rhs).cdf(z) - sp_dist(*args_lhs).cdf(z)) / (2 * eps)
        expected_grad = -cdf_dot / sp_dist(*params).pdf(z)
        assert_allclose(actual_grad[i], expected_grad, rtol=0.005)


@pytest.mark.parametrize('jax_dist, sp_dist, params', CONTINUOUS + DISCRETE + DIRECTIONAL)
@pytest.mark.parametrize('prepend_shape', [
    (),
    (2,),
    (2, 3),
])
@pytest.mark.parametrize('jit', [False, True])
def test_log_prob(jax_dist, sp_dist, params, prepend_shape, jit):
    jit_fn = _identity if not jit else jax.jit
    jax_dist = jax_dist(*params)
    rng_key = random.PRNGKey(0)
    samples = jax_dist.sample(key=rng_key, sample_shape=prepend_shape)
    assert jax_dist.log_prob(samples).shape == prepend_shape + jax_dist.batch_shape
    if not sp_dist:
        if isinstance(jax_dist, dist.TruncatedCauchy) or isinstance(jax_dist, dist.TruncatedNormal):
            low, loc, scale = params
            high = jnp.inf
            sp_dist = osp.cauchy if isinstance(jax_dist, dist.TruncatedCauchy) else osp.norm
            sp_dist = sp_dist(loc, scale)
            expected = sp_dist.logpdf(samples) - jnp.log(sp_dist.cdf(high) - sp_dist.cdf(low))
            assert_allclose(jit_fn(jax_dist.log_prob)(samples), expected, atol=1e-5)
            return
        pytest.skip('no corresponding scipy distn.')
    if _is_batched_multivariate(jax_dist):
        pytest.skip('batching not allowed in multivariate distns.')
    if jax_dist.event_shape and prepend_shape:
        # >>> d = sp.dirichlet([1.1, 1.1])
        # >>> samples = d.rvs(size=(2,))
        # >>> d.logpdf(samples)
        # ValueError: The input vector 'x' must lie within the normal simplex ...
        pytest.skip('batched samples cannot be scored by multivariate distributions.')
    sp_dist = sp_dist(*params)
    try:
        expected = sp_dist.logpdf(samples)
    except AttributeError:
        expected = sp_dist.logpmf(samples)
    except ValueError as e:
        # precision issue: jnp.sum(x / jnp.sum(x)) = 0.99999994 != 1
        if "The input vector 'x' must lie within the normal simplex." in str(e):
            samples = samples.copy().astype('float64')
            samples = samples / samples.sum(axis=-1, keepdims=True)
            expected = sp_dist.logpdf(samples)
        else:
            raise e
    assert_allclose(jit_fn(jax_dist.log_prob)(samples), expected, atol=1e-5)


@pytest.mark.parametrize('jax_dist, sp_dist, params', CONTINUOUS + DISCRETE)
def test_independent_shape(jax_dist, sp_dist, params):
    d = jax_dist(*params)
    batch_shape, event_shape = d.batch_shape, d.event_shape
    shape = batch_shape + event_shape
    for i in range(len(batch_shape)):
        indep = dist.Independent(d, reinterpreted_batch_ndims=i)
        sample = indep.sample(random.PRNGKey(0))
        event_boundary = len(shape) - len(event_shape) - i
        assert indep.batch_shape == shape[:event_boundary]
        assert indep.event_shape == shape[event_boundary:]
        assert jnp.shape(indep.log_prob(sample)) == shape[:event_boundary]


def _tril_cholesky_to_tril_corr(x):
    w = vec_to_tril_matrix(x, diagonal=-1)
    diag = jnp.sqrt(1 - jnp.sum(w ** 2, axis=-1))
    cholesky = w + jnp.expand_dims(diag, axis=-1) * jnp.identity(w.shape[-1])
    corr = jnp.matmul(cholesky, cholesky.T)
    return matrix_to_tril_vec(corr, diagonal=-1)


@pytest.mark.parametrize('dimension', [2, 3, 5])
def test_log_prob_LKJCholesky_uniform(dimension):
    # When concentration=1, the distribution of correlation matrices is uniform.
    # We will test that fact here.
    d = dist.LKJCholesky(dimension=dimension, concentration=1)
    N = 5
    corr_log_prob = []
    for i in range(N):
        sample = d.sample(random.PRNGKey(i))
        log_prob = d.log_prob(sample)
        sample_tril = matrix_to_tril_vec(sample, diagonal=-1)
        cholesky_to_corr_jac = np.linalg.slogdet(
            jax.jacobian(_tril_cholesky_to_tril_corr)(sample_tril))[1]
        corr_log_prob.append(log_prob - cholesky_to_corr_jac)

    corr_log_prob = jnp.array(corr_log_prob)
    # test if they are constant
    assert_allclose(corr_log_prob, jnp.broadcast_to(corr_log_prob[0], corr_log_prob.shape),
                    rtol=1e-6)

    if dimension == 2:
        # when concentration = 1, LKJ gives a uniform distribution over correlation matrix,
        # hence for the case dimension = 2,
        # density of a correlation matrix will be Uniform(-1, 1) = 0.5.
        # In addition, jacobian of the transformation from cholesky -> corr is 1 (hence its
        # log value is 0) because the off-diagonal lower triangular element does not change
        # in the transform.
        # So target_log_prob = log(0.5)
        assert_allclose(corr_log_prob[0], jnp.log(0.5), rtol=1e-6)


@pytest.mark.parametrize("dimension", [2, 3, 5])
@pytest.mark.parametrize("concentration", [0.6, 2.2])
def test_log_prob_LKJCholesky(dimension, concentration):
    # We will test against the fact that LKJCorrCholesky can be seen as a
    # TransformedDistribution with base distribution is a distribution of partial
    # correlations in C-vine method (modulo an affine transform to change domain from (0, 1)
    # to (1, 0)) and transform is a signed stick-breaking process.
    d = dist.LKJCholesky(dimension, concentration, sample_method="cvine")

    beta_sample = d._beta.sample(random.PRNGKey(0))
    beta_log_prob = jnp.sum(d._beta.log_prob(beta_sample))
    partial_correlation = 2 * beta_sample - 1
    affine_logdet = beta_sample.shape[-1] * jnp.log(2)
    sample = signed_stick_breaking_tril(partial_correlation)

    # compute signed stick breaking logdet
    inv_tanh = lambda t: jnp.log((1 + t) / (1 - t)) / 2  # noqa: E731
    inv_tanh_logdet = jnp.sum(jnp.log(vmap(grad(inv_tanh))(partial_correlation)))
    unconstrained = inv_tanh(partial_correlation)
    corr_cholesky_logdet = biject_to(constraints.corr_cholesky).log_abs_det_jacobian(
        unconstrained,
        sample,
    )
    signed_stick_breaking_logdet = corr_cholesky_logdet + inv_tanh_logdet

    actual_log_prob = d.log_prob(sample)
    expected_log_prob = beta_log_prob - affine_logdet - signed_stick_breaking_logdet
    assert_allclose(actual_log_prob, expected_log_prob, rtol=2e-5)

    assert_allclose(jax.jit(d.log_prob)(sample), d.log_prob(sample), atol=2e-6)


@pytest.mark.parametrize('rate', [0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 10.0])
def test_ZIP_log_prob(rate):
    # if gate is 0 ZIP is Poisson
    zip_ = dist.ZeroInflatedPoisson(0., rate)
    pois = dist.Poisson(rate)
    s = zip_.sample(random.PRNGKey(0), (20,))
    zip_prob = zip_.log_prob(s)
    pois_prob = pois.log_prob(s)
    assert_allclose(zip_prob, pois_prob)

    # if gate is 1 ZIP is Delta(0)
    zip_ = dist.ZeroInflatedPoisson(1., rate)
    delta = dist.Delta(0.)
    s = jnp.array([0., 1.])
    zip_prob = zip_.log_prob(s)
    delta_prob = delta.log_prob(s)
    assert_allclose(zip_prob, delta_prob)


@pytest.mark.parametrize("total_count", [1, 2, 3, 10])
@pytest.mark.parametrize("shape", [(1,), (3, 1), (2, 3, 1)])
def test_beta_binomial_log_prob(total_count, shape):
    concentration0 = np.exp(np.random.normal(size=shape))
    concentration1 = np.exp(np.random.normal(size=shape))
    value = jnp.arange(1 + total_count)

    num_samples = 100000
    probs = np.random.beta(concentration1, concentration0, size=(num_samples,) + shape)
    log_probs = dist.Binomial(total_count, probs).log_prob(value)
    expected = logsumexp(log_probs, 0) - jnp.log(num_samples)

    actual = dist.BetaBinomial(concentration1, concentration0, total_count).log_prob(value)
    assert_allclose(actual, expected, rtol=0.02)


@pytest.mark.parametrize("total_count", [1, 2, 3, 10])
@pytest.mark.parametrize("batch_shape", [(1,), (3, 1), (2, 3, 1)])
def test_dirichlet_multinomial_log_prob(total_count, batch_shape):
    event_shape = (3,)
    concentration = np.exp(np.random.normal(size=batch_shape + event_shape))
    # test on one-hots
    value = total_count * jnp.eye(event_shape[-1]).reshape(event_shape + (1,) * len(batch_shape) + event_shape)

    num_samples = 100000
    probs = dist.Dirichlet(concentration).sample(random.PRNGKey(0), (num_samples, 1))
    log_probs = dist.Multinomial(total_count, probs).log_prob(value)
    expected = logsumexp(log_probs, 0) - jnp.log(num_samples)

    actual = dist.DirichletMultinomial(concentration, total_count).log_prob(value)
    assert_allclose(actual, expected, rtol=0.05)


@pytest.mark.parametrize("shape", [(1,), (3, 1), (2, 3, 1)])
def test_gamma_poisson_log_prob(shape):
    gamma_conc = np.exp(np.random.normal(size=shape))
    gamma_rate = np.exp(np.random.normal(size=shape))
    value = jnp.arange(15)

    num_samples = 300000
    poisson_rate = np.random.gamma(gamma_conc, 1 / gamma_rate, size=(num_samples,) + shape)
    log_probs = dist.Poisson(poisson_rate).log_prob(value)
    expected = logsumexp(log_probs, 0) - jnp.log(num_samples)
    actual = dist.GammaPoisson(gamma_conc, gamma_rate).log_prob(value)
    assert_allclose(actual, expected, rtol=0.05)


@pytest.mark.parametrize('jax_dist, sp_dist, params', CONTINUOUS + DISCRETE + DIRECTIONAL)
def test_log_prob_gradient(jax_dist, sp_dist, params):
    if jax_dist in [dist.LKJ, dist.LKJCholesky]:
        pytest.skip('we have separated tests for LKJCholesky distribution')
    if jax_dist is _ImproperWrapper:
        pytest.skip('no param for ImproperUniform to test for log_prob gradient')

    rng_key = random.PRNGKey(0)
    value = jax_dist(*params).sample(rng_key)

    def fn(*args):
        return jnp.sum(jax_dist(*args).log_prob(value))

    eps = 1e-3
    for i in range(len(params)):
        if params[i] is None or jnp.result_type(params[i]) in (jnp.int32, jnp.int64):
            continue
        actual_grad = jax.grad(fn, i)(*params)
        args_lhs = [p if j != i else p - eps for j, p in enumerate(params)]
        args_rhs = [p if j != i else p + eps for j, p in enumerate(params)]
        fn_lhs = fn(*args_lhs)
        fn_rhs = fn(*args_rhs)
        # finite diff approximation
        expected_grad = (fn_rhs - fn_lhs) / (2. * eps)
        assert jnp.shape(actual_grad) == jnp.shape(params[i])
        if i == 0 and jax_dist is dist.Delta:
            # grad w.r.t. `value` of Delta distribution will be 0
            # but numerical value will give nan (= inf - inf)
            expected_grad = 0.
        assert_allclose(jnp.sum(actual_grad), expected_grad, rtol=0.01, atol=0.01)


@pytest.mark.parametrize('jax_dist, sp_dist, params', CONTINUOUS + DISCRETE + DIRECTIONAL)
def test_mean_var(jax_dist, sp_dist, params):
    if jax_dist is _ImproperWrapper:
        pytest.skip("Improper distribution does not has mean/var implemented")

    n = 20000 if jax_dist in [dist.LKJ, dist.LKJCholesky] else 200000
    d_jax = jax_dist(*params)
    k = random.PRNGKey(0)
    samples = d_jax.sample(k, sample_shape=(n,))
    # check with suitable scipy implementation if available
    if sp_dist and not _is_batched_multivariate(d_jax):
        d_sp = sp_dist(*params)
        try:
            sp_mean = d_sp.mean()
        except TypeError:  # mvn does not have .mean() method
            sp_mean = d_sp.mean
        # for multivariate distns try .cov first
        if d_jax.event_shape:
            try:
                sp_var = jnp.diag(d_sp.cov())
            except TypeError:  # mvn does not have .cov() method
                sp_var = jnp.diag(d_sp.cov)
            except AttributeError:
                sp_var = d_sp.var()
        else:
            sp_var = d_sp.var()
        assert_allclose(d_jax.mean, sp_mean, rtol=0.01, atol=1e-7)
        assert_allclose(d_jax.variance, sp_var, rtol=0.01, atol=1e-7)
        if jnp.all(jnp.isfinite(sp_mean)):
            assert_allclose(jnp.mean(samples, 0), d_jax.mean, rtol=0.05, atol=1e-2)
        if jnp.all(jnp.isfinite(sp_var)):
            assert_allclose(jnp.std(samples, 0), jnp.sqrt(d_jax.variance), rtol=0.05, atol=1e-2)
    elif jax_dist in [dist.LKJ, dist.LKJCholesky]:
        if jax_dist is dist.LKJCholesky:
            corr_samples = jnp.matmul(samples, jnp.swapaxes(samples, -2, -1))
        else:
            corr_samples = samples
        dimension, concentration, _ = params
        # marginal of off-diagonal entries
        marginal = dist.Beta(concentration + 0.5 * (dimension - 2),
                             concentration + 0.5 * (dimension - 2))
        # scale statistics due to linear mapping
        marginal_mean = 2 * marginal.mean - 1
        marginal_std = 2 * jnp.sqrt(marginal.variance)
        expected_mean = jnp.broadcast_to(jnp.reshape(marginal_mean, jnp.shape(marginal_mean) + (1, 1)),
                                         jnp.shape(marginal_mean) + d_jax.event_shape)
        expected_std = jnp.broadcast_to(jnp.reshape(marginal_std, jnp.shape(marginal_std) + (1, 1)),
                                        jnp.shape(marginal_std) + d_jax.event_shape)
        # diagonal elements of correlation matrices are 1
        expected_mean = expected_mean * (1 - jnp.identity(dimension)) + jnp.identity(dimension)
        expected_std = expected_std * (1 - jnp.identity(dimension))

        assert_allclose(jnp.mean(corr_samples, axis=0), expected_mean, atol=0.01)
        assert_allclose(jnp.std(corr_samples, axis=0), expected_std, atol=0.01)
    elif jax_dist in [dist.VonMises]:
        # circular mean = sample mean
        assert_allclose(d_jax.mean, jnp.mean(samples, 0), rtol=0.05, atol=1e-2)

        # circular variance
        x, y = jnp.mean(jnp.cos(samples), 0), jnp.mean(jnp.sin(samples), 0)

        expected_variance = 1 - jnp.sqrt(x ** 2 + y ** 2)
        assert_allclose(d_jax.variance, expected_variance, rtol=0.05, atol=1e-2)
    else:
        if jnp.all(jnp.isfinite(d_jax.mean)):
            assert_allclose(jnp.mean(samples, 0), d_jax.mean, rtol=0.05, atol=1e-2)
        if jnp.all(jnp.isfinite(d_jax.variance)):
            assert_allclose(jnp.std(samples, 0), jnp.sqrt(d_jax.variance), rtol=0.05, atol=1e-2)


@pytest.mark.parametrize('jax_dist, sp_dist, params', CONTINUOUS + DISCRETE + DIRECTIONAL)
@pytest.mark.parametrize('prepend_shape', [
    (),
    (2,),
    (2, 3),
])
def test_distribution_constraints(jax_dist, sp_dist, params, prepend_shape):
    dist_args = [p.name for p in inspect.signature(jax_dist).parameters.values()]

    valid_params, oob_params = list(params), list(params)
    key = random.PRNGKey(1)
    dependent_constraint = False
    for i in range(len(params)):
        if jax_dist in (_ImproperWrapper, dist.LKJ, dist.LKJCholesky) and dist_args[i] != "concentration":
            continue
        if params[i] is None:
            oob_params[i] = None
            valid_params[i] = None
            continue
        constraint = jax_dist.arg_constraints[dist_args[i]]
        if isinstance(constraint, constraints._Dependent):
            dependent_constraint = True
            break
        key, key_gen = random.split(key)
        oob_params[i] = gen_values_outside_bounds(constraint, jnp.shape(params[i]), key_gen)
        valid_params[i] = gen_values_within_bounds(constraint, jnp.shape(params[i]), key_gen)

    assert jax_dist(*oob_params)

    # Invalid parameter values throw ValueError
    if not dependent_constraint and jax_dist is not _ImproperWrapper:
        with pytest.raises(ValueError):
            jax_dist(*oob_params, validate_args=True)

    d = jax_dist(*valid_params, validate_args=True)

    # Test agreement of log density evaluation on randomly generated samples
    # with scipy's implementation when available.
    if sp_dist and \
            not _is_batched_multivariate(d) and \
            not (d.event_shape and prepend_shape):
        valid_samples = gen_values_within_bounds(d.support, size=prepend_shape + d.batch_shape + d.event_shape)
        try:
            expected = sp_dist(*valid_params).logpdf(valid_samples)
        except AttributeError:
            expected = sp_dist(*valid_params).logpmf(valid_samples)
        assert_allclose(d.log_prob(valid_samples), expected, atol=1e-5, rtol=1e-5)

    # Out of support samples throw ValueError
    oob_samples = gen_values_outside_bounds(d.support, size=prepend_shape + d.batch_shape + d.event_shape)
    with pytest.warns(UserWarning):
        d.log_prob(oob_samples)


def test_categorical_log_prob_grad():
    data = jnp.repeat(jnp.arange(3), 10)

    def f(x):
        return dist.Categorical(jax.nn.softmax(x * jnp.arange(1, 4))).log_prob(data).sum()

    def g(x):
        return dist.Categorical(logits=x * jnp.arange(1, 4)).log_prob(data).sum()

    x = 0.5
    fx, grad_fx = jax.value_and_grad(f)(x)
    gx, grad_gx = jax.value_and_grad(g)(x)
    assert_allclose(fx, gx)
    assert_allclose(grad_fx, grad_gx, atol=1e-4)


########################################
# Tests for constraints and transforms #
########################################


@pytest.mark.parametrize('constraint, x, expected', [
    (constraints.boolean, jnp.array([True, False]), jnp.array([True, True])),
    (constraints.boolean, jnp.array([1, 1]), jnp.array([True, True])),
    (constraints.boolean, jnp.array([-1, 1]), jnp.array([False, True])),
    (constraints.corr_cholesky, jnp.array([[[1, 0], [0, 1]], [[1, 0.1], [0, 1]]]),
     jnp.array([True, False])),  # NB: not lower_triangular
    (constraints.corr_cholesky, jnp.array([[[1, 0], [1, 0]], [[1, 0], [0.5, 0.5]]]),
     jnp.array([False, False])),  # NB: not positive_diagonal & not unit_norm_row
    (constraints.corr_matrix, jnp.array([[[1, 0], [0, 1]], [[1, 0.1], [0, 1]]]),
     jnp.array([True, False])),  # NB: not lower_triangular
    (constraints.corr_matrix, jnp.array([[[1, 0], [1, 0]], [[1, 0], [0.5, 0.5]]]),
     jnp.array([False, False])),  # NB: not unit diagonal
    (constraints.greater_than(1), 3, True),
    (constraints.greater_than(1), jnp.array([-1, 1, 5]), jnp.array([False, False, True])),
    (constraints.integer_interval(-3, 5), 0, True),
    (constraints.integer_interval(-3, 5), jnp.array([-5, -3, 0, 1.1, 5, 7]),
     jnp.array([False, True, True, False, True, False])),
    (constraints.interval(-3, 5), 0, True),
    (constraints.interval(-3, 5), jnp.array([-5, -3, 0, 5, 7]),
     jnp.array([False, True, True, True, False])),
    (constraints.less_than(1), -2, True),
    (constraints.less_than(1), jnp.array([-1, 1, 5]), jnp.array([True, False, False])),
    (constraints.lower_cholesky, jnp.array([[1., 0.], [-2., 0.1]]), True),
    (constraints.lower_cholesky, jnp.array([[[1., 0.], [-2., -0.1]], [[1., 0.1], [2., 0.2]]]),
     jnp.array([False, False])),
    (constraints.nonnegative_integer, 3, True),
    (constraints.nonnegative_integer, jnp.array([-1., 0., 5.]), jnp.array([False, True, True])),
    (constraints.positive, 3, True),
    (constraints.positive, jnp.array([-1, 0, 5]), jnp.array([False, False, True])),
    (constraints.positive_definite, jnp.array([[1., 0.3], [0.3, 1.]]), True),
    (constraints.positive_definite, jnp.array([[[2., 0.4], [0.3, 2.]], [[1., 0.1], [0.1, 0.]]]),
     jnp.array([False, False])),
    (constraints.positive_integer, 3, True),
    (constraints.positive_integer, jnp.array([-1., 0., 5.]), jnp.array([False, False, True])),
    (constraints.real, -1, True),
    (constraints.real, jnp.array([jnp.inf, jnp.NINF, jnp.nan, jnp.pi]),
     jnp.array([False, False, False, True])),
    (constraints.simplex, jnp.array([0.1, 0.3, 0.6]), True),
    (constraints.simplex, jnp.array([[0.1, 0.3, 0.6], [-0.1, 0.6, 0.5], [0.1, 0.6, 0.5]]),
     jnp.array([True, False, False])),
    (constraints.unit_interval, 0.1, True),
    (constraints.unit_interval, jnp.array([-5, 0, 0.5, 1, 7]),
     jnp.array([False, True, True, True, False])),
])
def test_constraints(constraint, x, expected):
    assert_array_equal(constraint(x), expected)


@pytest.mark.parametrize('constraint', [
    constraints.corr_cholesky,
    constraints.corr_matrix,
    constraints.greater_than(2),
    constraints.interval(-3, 5),
    constraints.less_than(1),
    constraints.lower_cholesky,
    constraints.ordered_vector,
    constraints.positive,
    constraints.positive_definite,
    constraints.real,
    constraints.simplex,
    constraints.unit_interval,
], ids=lambda x: x.__class__)
@pytest.mark.parametrize('shape', [(), (1,), (3,), (6,), (3, 1), (1, 3), (5, 3)])
def test_biject_to(constraint, shape):
    transform = biject_to(constraint)
    if transform.event_dim == 2:
        event_dim = 1  # actual dim of unconstrained domain
    else:
        event_dim = transform.event_dim
    if isinstance(constraint, constraints._Interval):
        assert transform.codomain.upper_bound == constraint.upper_bound
        assert transform.codomain.lower_bound == constraint.lower_bound
    elif isinstance(constraint, constraints._GreaterThan):
        assert transform.codomain.lower_bound == constraint.lower_bound
    elif isinstance(constraint, constraints._LessThan):
        assert transform.codomain.upper_bound == constraint.upper_bound
    if len(shape) < event_dim:
        return
    rng_key = random.PRNGKey(0)
    x = random.normal(rng_key, shape)
    y = transform(x)

    # test inv work for NaN arrays:
    x_nan = transform.inv(jnp.full(jnp.shape(y), jnp.nan))
    assert (x_nan.shape == x.shape)

    # test codomain
    batch_shape = shape if event_dim == 0 else shape[:-1]
    assert_array_equal(transform.codomain(y), jnp.ones(batch_shape, dtype=jnp.bool_))

    # test inv
    z = transform.inv(y)
    assert_allclose(x, z, atol=1e-6, rtol=1e-6)

    # test domain, currently all is constraints.real or constraints.real_vector
    assert_array_equal(transform.domain(z), jnp.ones(batch_shape))

    # test log_abs_det_jacobian
    actual = transform.log_abs_det_jacobian(x, y)
    assert jnp.shape(actual) == batch_shape
    if len(shape) == event_dim:
        if constraint is constraints.simplex:
            expected = np.linalg.slogdet(jax.jacobian(transform)(x)[:-1, :])[1]
            inv_expected = np.linalg.slogdet(jax.jacobian(transform.inv)(y)[:, :-1])[1]
        elif constraint is constraints.ordered_vector:
            expected = np.linalg.slogdet(jax.jacobian(transform)(x))[1]
            inv_expected = np.linalg.slogdet(jax.jacobian(transform.inv)(y))[1]
        elif constraint in [constraints.corr_cholesky, constraints.corr_matrix]:
            vec_transform = lambda x: matrix_to_tril_vec(transform(x), diagonal=-1)  # noqa: E731
            y_tril = matrix_to_tril_vec(y, diagonal=-1)

            def inv_vec_transform(y):
                matrix = vec_to_tril_matrix(y, diagonal=-1)
                if constraint is constraints.corr_matrix:
                    # fill the upper triangular part
                    matrix = matrix + jnp.swapaxes(matrix, -2, -1) + jnp.identity(matrix.shape[-1])
                return transform.inv(matrix)

            expected = np.linalg.slogdet(jax.jacobian(vec_transform)(x))[1]
            inv_expected = np.linalg.slogdet(jax.jacobian(inv_vec_transform)(y_tril))[1]
        elif constraint in [constraints.lower_cholesky, constraints.positive_definite]:
            vec_transform = lambda x: matrix_to_tril_vec(transform(x))  # noqa: E731
            y_tril = matrix_to_tril_vec(y)

            def inv_vec_transform(y):
                matrix = vec_to_tril_matrix(y)
                if constraint is constraints.positive_definite:
                    # fill the upper triangular part
                    matrix = matrix + jnp.swapaxes(matrix, -2, -1) - jnp.diag(jnp.diag(matrix))
                return transform.inv(matrix)

            expected = np.linalg.slogdet(jax.jacobian(vec_transform)(x))[1]
            inv_expected = np.linalg.slogdet(jax.jacobian(inv_vec_transform)(y_tril))[1]
        else:
            expected = jnp.log(jnp.abs(grad(transform)(x)))
            inv_expected = jnp.log(jnp.abs(grad(transform.inv)(y)))

        assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
        assert_allclose(actual, -inv_expected, atol=1e-6, rtol=1e-6)


# NB: skip transforms which are tested in `test_biject_to`
@pytest.mark.parametrize('transform, event_shape', [
    (PermuteTransform(jnp.array([3, 0, 4, 1, 2])), (5,)),
    (PowerTransform(2.), ()),
    (LowerCholeskyAffine(jnp.array([1., 2.]), jnp.array([[0.6, 0.], [1.5, 0.4]])), (2,))
])
@pytest.mark.parametrize('batch_shape', [(), (1,), (3,), (6,), (3, 1), (1, 3), (5, 3)])
def test_bijective_transforms(transform, event_shape, batch_shape):
    shape = batch_shape + event_shape
    rng_key = random.PRNGKey(0)
    x = biject_to(transform.domain)(random.normal(rng_key, shape))
    y = transform(x)

    # test codomain
    assert_array_equal(transform.codomain(y), jnp.ones(batch_shape))

    # test inv
    z = transform.inv(y)
    assert_allclose(x, z, atol=1e-6, rtol=1e-6)

    # test domain
    assert_array_equal(transform.domain(z), jnp.ones(batch_shape))

    # test log_abs_det_jacobian
    actual = transform.log_abs_det_jacobian(x, y)
    assert jnp.shape(actual) == batch_shape
    if len(shape) == transform.event_dim:
        if len(event_shape) == 1:
            expected = np.linalg.slogdet(jax.jacobian(transform)(x))[1]
            inv_expected = np.linalg.slogdet(jax.jacobian(transform.inv)(y))[1]
        else:
            expected = jnp.log(jnp.abs(grad(transform)(x)))
            inv_expected = jnp.log(jnp.abs(grad(transform.inv)(y)))

        assert_allclose(actual, expected, atol=1e-6)
        assert_allclose(actual, -inv_expected, atol=1e-6)


@pytest.mark.parametrize('transformed_dist', [
    dist.TransformedDistribution(dist.Normal(jnp.array([2., 3.]), 1.), transforms.ExpTransform()),
    dist.TransformedDistribution(dist.Exponential(jnp.ones(2)), [
        transforms.PowerTransform(0.7),
        transforms.AffineTransform(0., jnp.ones(2) * 3)
    ]),
])
def test_transformed_distribution_intermediates(transformed_dist):
    sample, intermediates = transformed_dist.sample_with_intermediates(random.PRNGKey(1))
    assert_allclose(transformed_dist.log_prob(sample, intermediates), transformed_dist.log_prob(sample))


def test_transformed_transformed_distribution():
    loc, scale = -2, 3
    dist1 = dist.TransformedDistribution(dist.Normal(2, 3), transforms.PowerTransform(2.))
    dist2 = dist.TransformedDistribution(dist1, transforms.AffineTransform(-2, 3))
    assert isinstance(dist2.base_dist, dist.Normal)
    assert len(dist2.transforms) == 2
    assert isinstance(dist2.transforms[0], transforms.PowerTransform)
    assert isinstance(dist2.transforms[1], transforms.AffineTransform)

    rng_key = random.PRNGKey(0)
    assert_allclose(loc + scale * dist1.sample(rng_key), dist2.sample(rng_key))
    intermediates = dist2.sample_with_intermediates(rng_key)
    assert len(intermediates) == 2


def _make_iaf(input_dim, hidden_dims, rng_key):
    arn_init, arn = AutoregressiveNN(input_dim, hidden_dims, param_dims=[1, 1])
    _, init_params = arn_init(rng_key, (input_dim,))
    return InverseAutoregressiveTransform(partial(arn, init_params))


@pytest.mark.parametrize('ts', [
    [transforms.PowerTransform(0.7), transforms.AffineTransform(2., 3.)],
    [transforms.ExpTransform()],
    [transforms.ComposeTransform([transforms.AffineTransform(-2, 3),
                                  transforms.ExpTransform()]),
     transforms.PowerTransform(3.)],
    [_make_iaf(5, hidden_dims=[10], rng_key=random.PRNGKey(0)),
     transforms.PermuteTransform(jnp.arange(5)[::-1]),
     _make_iaf(5, hidden_dims=[10], rng_key=random.PRNGKey(1))]
])
def test_compose_transform_with_intermediates(ts):
    transform = transforms.ComposeTransform(ts)
    x = random.normal(random.PRNGKey(2), (7, 5))
    y, intermediates = transform.call_with_intermediates(x)
    logdet = transform.log_abs_det_jacobian(x, y, intermediates)
    assert_allclose(y, transform(x))
    assert_allclose(logdet, transform.log_abs_det_jacobian(x, y))


@pytest.mark.parametrize('x_dim, y_dim', [(3, 3), (3, 4)])
def test_unpack_transform(x_dim, y_dim):
    xy = np.random.randn(x_dim + y_dim)
    unpack_fn = lambda xy: {'x': xy[:x_dim], 'y': xy[x_dim:]}  # noqa: E731
    transform = transforms.UnpackTransform(unpack_fn)
    z = transform(xy)
    if x_dim == y_dim:
        with pytest.warns(UserWarning, match="UnpackTransform.inv"):
            t = transform.inv(z)
    else:
        t = transform.inv(z)

    assert_allclose(t, xy)


@pytest.mark.parametrize('jax_dist, sp_dist, params', CONTINUOUS)
def test_generated_sample_distribution(jax_dist, sp_dist, params,
                                       N_sample=100_000,
                                       key=random.PRNGKey(11)):
    """ On samplers that we do not get directly from JAX, (e.g. we only get
    Gumbel(0,1) but also provide samplers for Gumbel(loc, scale)), also test
    agreement in the empirical distribution of generated samples between our
    samplers and those from SciPy.
    """

    if jax_dist not in [dist.Gumbel]:
        pytest.skip("{} sampling method taken from upstream, no need to"
                    "test generated samples.".format(jax_dist.__name__))

    jax_dist = jax_dist(*params)
    if sp_dist and not jax_dist.event_shape and not jax_dist.batch_shape:
        our_samples = jax_dist.sample(key, (N_sample,))
        ks_result = osp.kstest(our_samples, sp_dist(*params).cdf)
        assert ks_result.pvalue > 0.05


@pytest.mark.parametrize('jax_dist, params, support', [
    (dist.BernoulliLogits, (5.,), jnp.arange(2)),
    (dist.BernoulliProbs, (0.5,), jnp.arange(2)),
    (dist.BinomialLogits, (4.5, 10), jnp.arange(11)),
    (dist.BinomialProbs, (0.5, 11), jnp.arange(12)),
    (dist.BetaBinomial, (2., 0.5, 12), jnp.arange(13)),
    (dist.CategoricalLogits, (jnp.array([3., 4., 5.]),), jnp.arange(3)),
    (dist.CategoricalProbs, (jnp.array([0.1, 0.5, 0.4]),), jnp.arange(3)),
])
@pytest.mark.parametrize('batch_shape', [(5,), ()])
@pytest.mark.parametrize('expand', [False, True])
def test_enumerate_support_smoke(jax_dist, params, support, batch_shape, expand):
    p0 = jnp.broadcast_to(params[0], batch_shape + jnp.shape(params[0]))
    actual = jax_dist(p0, *params[1:]).enumerate_support(expand=expand)
    expected = support.reshape((-1,) + (1,) * len(batch_shape))
    if expand:
        expected = jnp.broadcast_to(expected, support.shape + batch_shape)
    assert_allclose(actual, expected)


@pytest.mark.parametrize('jax_dist, sp_dist, params', CONTINUOUS + DISCRETE)
@pytest.mark.parametrize('prepend_shape', [
    (),
    (2, 3),
])
@pytest.mark.parametrize('sample_shape', [
    (),
    (4,),
])
def test_expand(jax_dist, sp_dist, params, prepend_shape, sample_shape):
    jax_dist = jax_dist(*params)
    new_batch_shape = prepend_shape + jax_dist.batch_shape
    expanded_dist = jax_dist.expand(new_batch_shape)
    rng_key = random.PRNGKey(0)
    samples = expanded_dist.sample(rng_key, sample_shape)
    assert expanded_dist.batch_shape == new_batch_shape
    assert samples.shape == sample_shape + new_batch_shape + jax_dist.event_shape
    assert expanded_dist.log_prob(samples).shape == sample_shape + new_batch_shape
    # test expand of expand
    assert expanded_dist.expand((3,) + new_batch_shape).batch_shape == (3,) + new_batch_shape
    # test expand error
    if prepend_shape:
        with pytest.raises(ValueError, match="Cannot broadcast distribution of shape"):
            assert expanded_dist.expand((3,) + jax_dist.batch_shape)


@pytest.mark.parametrize('batch_shape', [
    (),
    (4,),
])
def test_polya_gamma(batch_shape, num_points=20000):
    d = dist.TruncatedPolyaGamma(batch_shape=batch_shape)
    rng_key = random.PRNGKey(0)

    # test density approximately normalized
    x = jnp.linspace(1.0e-6, d.truncation_point, num_points)
    prob = (d.truncation_point / num_points) * jnp.exp(logsumexp(d.log_prob(x), axis=-1))
    assert_allclose(prob, jnp.ones(batch_shape), rtol=1.0e-4)

    # test mean of approximate sampler
    z = d.sample(rng_key, sample_shape=(3000,))
    mean = jnp.mean(z, axis=-1)
    assert_allclose(mean, 0.25 * jnp.ones(batch_shape), rtol=0.07)


@pytest.mark.parametrize("extra_event_dims,expand_shape", [
    (0, (4, 3, 2, 1)),
    (0, (4, 3, 2, 2)),
    (1, (5, 4, 3, 2)),
    (2, (5, 4, 3)),
])
def test_expand_reshaped_distribution(extra_event_dims, expand_shape):
    loc = jnp.zeros((1, 6))
    scale_tril = jnp.eye(6)
    d = dist.MultivariateNormal(loc, scale_tril=scale_tril)
    full_shape = (4, 1, 1, 1, 6)
    reshaped_dist = d.expand([4, 1, 1, 1]).to_event(extra_event_dims)
    cut = 4 - extra_event_dims
    batch_shape, event_shape = full_shape[:cut], full_shape[cut:]
    assert reshaped_dist.batch_shape == batch_shape
    assert reshaped_dist.event_shape == event_shape
    large = reshaped_dist.expand(expand_shape)
    assert large.batch_shape == expand_shape
    assert large.event_shape == event_shape

    # Throws error when batch shape cannot be broadcasted
    with pytest.raises((RuntimeError, ValueError)):
        reshaped_dist.expand(expand_shape + (3,))

    # Throws error when trying to shrink existing batch shape
    with pytest.raises((RuntimeError, ValueError)):
        large.expand(expand_shape[1:])


@pytest.mark.parametrize('batch_shape, mask_shape', [
    ((), ()),
    ((2,), ()),
    ((), (2,)),
    ((2,), (2,)),
    ((4, 2), (1, 2)),
    ((2,), (4, 2)),
])
@pytest.mark.parametrize('event_shape', [
    (),
    (3,)
])
def test_mask(batch_shape, event_shape, mask_shape):
    jax_dist = dist.Normal().expand(batch_shape + event_shape).to_event(len(event_shape))
    mask = dist.Bernoulli(0.5).sample(random.PRNGKey(0), mask_shape)
    if mask_shape == ():
        mask = bool(mask)
    samples = jax_dist.sample(random.PRNGKey(1))
    actual = jax_dist.mask(mask).log_prob(samples)
    assert_allclose(actual != 0, jnp.broadcast_to(mask, lax.broadcast_shapes(batch_shape, mask_shape)))


@pytest.mark.parametrize('jax_dist, sp_dist, params', CONTINUOUS + DISCRETE + DIRECTIONAL)
def test_dist_pytree(jax_dist, sp_dist, params):
    def f(x):
        return jax_dist(*params)

    if jax_dist is _ImproperWrapper:
        pytest.skip('Cannot flattening ImproperUniform')
    jax.jit(f)(0)  # this test for flatten/unflatten
    lax.map(f, np.ones(3))  # this test for compatibility w.r.t. scan


@pytest.mark.parametrize('method, arg', [
    ('to_event', 1),
    ('mask', False),
    ('expand', [5]),
])
def test_special_dist_pytree(method, arg):
    def f(x):
        d = dist.Normal(np.zeros(1), np.ones(1))
        return getattr(d, method)(arg)

    jax.jit(f)(0)
    lax.map(f, np.ones(3))


def test_expand_pytree():
    def g(x):
        return dist.Normal(x, 1).expand([10, 3])

    assert lax.map(g, jnp.ones((5, 3))).batch_shape == (5, 10, 3)
    assert jax.tree_map(lambda x: x[None], g(0)).batch_shape == (1, 10, 3)


@pytest.mark.parametrize('batch_shape', [(), (4,), (2, 3)], ids=str)
def test_kl_delta_normal_shape(batch_shape):
    v = np.random.normal(size=batch_shape)
    loc = np.random.normal(size=batch_shape)
    scale = np.exp(np.random.normal(size=batch_shape))
    p = dist.Delta(v)
    q = dist.Normal(loc, scale)
    assert kl_divergence(p, q).shape == batch_shape


@pytest.mark.parametrize('batch_shape', [(), (4,), (2, 3)], ids=str)
@pytest.mark.parametrize('event_shape', [(), (4,), (2, 3)], ids=str)
def test_kl_independent_normal(batch_shape, event_shape):
    shape = batch_shape + event_shape
    p = dist.Normal(np.random.normal(size=shape), np.exp(np.random.normal(size=shape)))
    q = dist.Normal(np.random.normal(size=shape), np.exp(np.random.normal(size=shape)))
    actual = kl_divergence(dist.Independent(p, len(event_shape)),
                           dist.Independent(q, len(event_shape)))
    expected = sum_rightmost(kl_divergence(p, q), len(event_shape))
    assert_allclose(actual, expected)


@pytest.mark.parametrize('batch_shape', [(), (4,), (2, 3)], ids=str)
@pytest.mark.parametrize('event_shape', [(), (4,), (2, 3)], ids=str)
def test_kl_expanded_normal(batch_shape, event_shape):
    shape = batch_shape + event_shape
    p = dist.Normal(np.random.normal(), np.exp(np.random.normal())).expand(shape)
    q = dist.Normal(np.random.normal(), np.exp(np.random.normal())).expand(shape)
    actual = kl_divergence(dist.Independent(p, len(event_shape)),
                           dist.Independent(q, len(event_shape)))
    expected = sum_rightmost(kl_divergence(p, q), len(event_shape))
    assert_allclose(actual, expected)


@pytest.mark.parametrize('shape', [(), (4,), (2, 3)], ids=str)
def test_kl_normal_normal(shape):
    p = dist.Normal(np.random.normal(size=shape), np.exp(np.random.normal(size=shape)))
    q = dist.Normal(np.random.normal(size=shape), np.exp(np.random.normal(size=shape)))
    actual = kl_divergence(p, q)
    x = p.sample(random.PRNGKey(0), (10000,)).copy()
    expected = jnp.mean((p.log_prob(x) - q.log_prob(x)), 0)
    assert_allclose(actual, expected, rtol=0.05)
