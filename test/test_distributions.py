from collections import namedtuple
from functools import partial
import inspect

import numpy as onp
from numpy.testing import assert_allclose, assert_array_equal
import pytest
import scipy.stats as osp

import jax
from jax import grad, jacfwd, lax, vmap
import jax.numpy as np
import jax.random as random

from numpyro.contrib.nn import AutoregressiveNN
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
from numpyro.distributions.constraints import MultivariateAffineTransform, PermuteTransform, PowerTransform, biject_to
from numpyro.distributions.discrete import _to_probs_bernoulli, _to_probs_multinom
from numpyro.distributions.flows import InverseAutoregressiveTransform
from numpyro.distributions.util import (
    matrix_to_tril_vec,
    multinomial,
    poisson,
    signed_stick_breaking_tril,
    vec_to_tril_matrix
)


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


_DIST_MAP = {
    dist.BernoulliProbs: lambda probs: osp.bernoulli(p=probs),
    dist.BernoulliLogits: lambda logits: osp.bernoulli(p=_to_probs_bernoulli(logits)),
    dist.Beta: lambda con1, con0: osp.beta(con1, con0),
    dist.BinomialProbs: lambda probs, total_count: osp.binom(n=total_count, p=probs),
    dist.BinomialLogits: lambda logits, total_count: osp.binom(n=total_count, p=_to_probs_bernoulli(logits)),
    dist.Cauchy: lambda loc, scale: osp.cauchy(loc=loc, scale=scale),
    dist.Chi2: lambda df: osp.chi2(df),
    dist.Dirichlet: lambda conc: osp.dirichlet(conc),
    dist.Exponential: lambda rate: osp.expon(scale=np.reciprocal(rate)),
    dist.Gamma: lambda conc, rate: osp.gamma(conc, scale=1./rate),
    dist.HalfCauchy: lambda scale: osp.halfcauchy(scale=scale),
    dist.HalfNormal: lambda scale: osp.halfnorm(scale=scale),
    dist.InverseGamma: lambda conc, rate: osp.invgamma(conc, scale=rate),
    dist.LogNormal: lambda loc, scale: osp.lognorm(s=scale, scale=np.exp(loc)),
    dist.MultinomialProbs: lambda probs, total_count: osp.multinomial(n=total_count, p=probs),
    dist.MultinomialLogits: lambda logits, total_count: osp.multinomial(n=total_count,
                                                                        p=_to_probs_multinom(logits)),
    dist.MultivariateNormal: _mvn_to_scipy,
    dist.Normal: lambda loc, scale: osp.norm(loc=loc, scale=scale),
    dist.Pareto: lambda alpha, scale: osp.pareto(alpha, scale=scale),
    dist.Poisson: lambda rate: osp.poisson(rate),
    dist.StudentT: lambda df, loc, scale: osp.t(df=df, loc=loc, scale=scale),
    dist.Uniform: lambda a, b: osp.uniform(a, b - a),
}


CONTINUOUS = [
    T(dist.Beta, 1., 2.),
    T(dist.Beta, 1., np.array([2., 2.])),
    T(dist.Beta, 1., np.array([[1., 1.], [2., 2.]])),
    T(dist.Chi2, 2.),
    T(dist.Chi2, np.array([0.3, 1.3])),
    T(dist.Cauchy, 0., 1.),
    T(dist.Cauchy, 0., np.array([1., 2.])),
    T(dist.Cauchy, np.array([0., 1.]), np.array([[1.], [2.]])),
    T(dist.Dirichlet, np.array([1.7])),
    T(dist.Dirichlet, np.array([0.2, 1.1])),
    T(dist.Dirichlet, np.array([[0.2, 1.1], [2., 2.]])),
    T(dist.Exponential, 2.),
    T(dist.Exponential, np.array([4., 2.])),
    T(dist.Gamma, np.array([1.7]), np.array([[2.], [3.]])),
    T(dist.Gamma, np.array([0.5, 1.3]), np.array([[1.], [3.]])),
    T(dist.GaussianRandomWalk, 0.1, 10),
    T(dist.GaussianRandomWalk, np.array([0.1, 0.3, 0.25]), 10),
    T(dist.HalfCauchy, 1.),
    T(dist.HalfCauchy, np.array([1., 2.])),
    T(dist.HalfNormal, 1.),
    T(dist.HalfNormal, np.array([1., 2.])),
    T(dist.InverseGamma, np.array([1.7]), np.array([[2.], [3.]])),
    T(dist.InverseGamma, np.array([0.5, 1.3]), np.array([[1.], [3.]])),
    T(dist.LKJ, 2, 0.5, "onion"),
    T(dist.LKJ, 5, np.array([0.5, 1., 2.]), "cvine"),
    T(dist.LKJCholesky, 2, 0.5, "onion"),
    T(dist.LKJCholesky, 2, 0.5, "cvine"),
    T(dist.LKJCholesky, 5, np.array([0.5, 1., 2.]), "onion"),
    T(dist.LKJCholesky, 5, np.array([0.5, 1., 2.]), "cvine"),
    T(dist.LKJCholesky, 3, np.array([[3., 0.6], [0.2, 5.]]), "onion"),
    T(dist.LKJCholesky, 3, np.array([[3., 0.6], [0.2, 5.]]), "cvine"),
    T(dist.LogNormal, 1., 0.2),
    T(dist.LogNormal, -1., np.array([0.5, 1.3])),
    T(dist.LogNormal, np.array([0.5, -0.7]), np.array([[0.1, 0.4], [0.5, 0.1]])),
    T(dist.MultivariateNormal, 0., np.array([[1., 0.5], [0.5, 1.]]), None, None),
    T(dist.MultivariateNormal, np.array([1., 3.]), None, np.array([[1., 0.5], [0.5, 1.]]), None),
    T(dist.MultivariateNormal, np.array([2.]), None, None, np.array([[1., 0.], [0.5, 1.]])),
    T(dist.MultivariateNormal, np.arange(6, dtype=np.float32).reshape((3, 2)), None, None,
      np.array([[1., 0.], [0., 1.]])),
    T(dist.MultivariateNormal, 0., None, np.broadcast_to(np.identity(3), (2, 3, 3)), None),
    T(dist.Normal, 0., 1.),
    T(dist.Normal, 1., np.array([1., 2.])),
    T(dist.Normal, np.array([0., 1.]), np.array([[1.], [2.]])),
    T(dist.Pareto, 2., 1.),
    T(dist.Pareto, np.array([0.3, 2.]), np.array([1., 0.5])),
    T(dist.Pareto, np.array([1., 0.5]), np.array([[1.], [3.]])),
    T(dist.StudentT, 1., 1., 0.5),
    T(dist.StudentT, 2., np.array([1., 2.]), 2.),
    T(dist.StudentT, np.array([3, 5]), np.array([[1.], [2.]]), 2.),
    T(dist.TruncatedCauchy, -1., 0., 1.),
    T(dist.TruncatedCauchy, 1., 0., np.array([1., 2.])),
    T(dist.TruncatedCauchy, np.array([-2., 2.]), np.array([0., 1.]), np.array([[1.], [2.]])),
    T(dist.TruncatedNormal, -1., 0., 1.),
    T(dist.TruncatedNormal, 1., -1., np.array([1., 2.])),
    T(dist.TruncatedNormal, np.array([-2., 2.]), np.array([0., 1.]), np.array([[1.], [2.]])),
    T(dist.Uniform, 0., 2.),
    T(dist.Uniform, 1., np.array([2., 3.])),
    T(dist.Uniform, np.array([0., 0.]), np.array([[2.], [3.]])),
]


DISCRETE = [
    T(dist.BernoulliProbs, 0.2),
    T(dist.BernoulliProbs, np.array([0.2, 0.7])),
    T(dist.BernoulliLogits, np.array([-1., 3.])),
    T(dist.BinomialProbs, np.array([0.2, 0.7]), np.array([10, 2])),
    T(dist.BinomialProbs, np.array([0.2, 0.7]), np.array([5, 8])),
    T(dist.BinomialLogits, np.array([-1., 3.]), np.array([5, 8])),
    T(dist.CategoricalProbs, np.array([1.])),
    T(dist.CategoricalProbs, np.array([0.1, 0.5, 0.4])),
    T(dist.CategoricalProbs, np.array([[0.1, 0.5, 0.4], [0.4, 0.4, 0.2]])),
    T(dist.CategoricalLogits, np.array([-5.])),
    T(dist.CategoricalLogits, np.array([1., 2., -2.])),
    T(dist.Delta, 1),
    T(dist.Delta, np.array([0., 2.])),
    T(dist.Delta, np.array([0., 2.]), np.array([-2., -4.])),
    T(dist.CategoricalLogits, np.array([[-1, 2., 3.], [3., -4., -2.]])),
    T(dist.MultinomialProbs, np.array([0.2, 0.7, 0.1]), 10),
    T(dist.MultinomialProbs, np.array([0.2, 0.7, 0.1]), np.array([5, 8])),
    T(dist.MultinomialLogits, np.array([-1., 3.]), np.array([[5], [8]])),
    T(dist.Poisson, 2.),
    T(dist.Poisson, np.array([2., 3., 5.])),
]


def _is_batched_multivariate(jax_dist):
    return len(jax_dist.event_shape) > 0 and len(jax_dist.batch_shape) > 0


def gen_values_within_bounds(constraint, size, key=random.PRNGKey(11)):
    eps = 1e-6

    if isinstance(constraint, constraints._Boolean):
        return random.bernoulli(key, shape=size)
    elif isinstance(constraint, constraints._GreaterThan):
        return np.exp(random.normal(key, size)) + constraint.lower_bound + eps
    elif isinstance(constraint, constraints._IntegerInterval):
        lower_bound = np.broadcast_to(constraint.lower_bound, size)
        upper_bound = np.broadcast_to(constraint.upper_bound, size)
        return random.randint(key, size, lower_bound, upper_bound + 1)
    elif isinstance(constraint, constraints._IntegerGreaterThan):
        return constraint.lower_bound + poisson(key, 5, shape=size)
    elif isinstance(constraint, constraints._Interval):
        lower_bound = np.broadcast_to(constraint.lower_bound, size)
        upper_bound = np.broadcast_to(constraint.upper_bound, size)
        return random.uniform(key, size, minval=lower_bound, maxval=upper_bound)
    elif isinstance(constraint, (constraints._Real, constraints._RealVector)):
        return random.normal(key, size)
    elif isinstance(constraint, constraints._Simplex):
        return osp.dirichlet.rvs(alpha=np.ones((size[-1],)), size=size[:-1])
    elif isinstance(constraint, constraints._Multinomial):
        n = size[-1]
        return multinomial(key, p=np.ones((n,)) / n, n=constraint.upper_bound, shape=size[:-1])
    elif isinstance(constraint, constraints._CorrCholesky):
        return signed_stick_breaking_tril(
            random.uniform(key, size[:-2] + (size[-1] * (size[-1] - 1) // 2,), minval=-1, maxval=1))
    elif isinstance(constraint, constraints._CorrMatrix):
        cholesky = signed_stick_breaking_tril(
            random.uniform(key, size[:-2] + (size[-1] * (size[-1] - 1) // 2,), minval=-1, maxval=1))
        return np.matmul(cholesky, np.swapaxes(cholesky, -2, -1))
    elif isinstance(constraint, constraints._LowerCholesky):
        return np.tril(random.uniform(key, size))
    elif isinstance(constraint, constraints._PositiveDefinite):
        x = random.normal(key, size)
        return np.matmul(x, np.swapaxes(x, -2, -1))
    else:
        raise NotImplementedError('{} not implemented.'.format(constraint))


def gen_values_outside_bounds(constraint, size, key=random.PRNGKey(11)):
    if isinstance(constraint, constraints._Boolean):
        return random.bernoulli(key, shape=size) - 2
    elif isinstance(constraint, constraints._GreaterThan):
        return constraint.lower_bound - np.exp(random.normal(key, size))
    elif isinstance(constraint, constraints._IntegerInterval):
        lower_bound = np.broadcast_to(constraint.lower_bound, size)
        return random.randint(key, size, lower_bound - 1, lower_bound)
    elif isinstance(constraint, constraints._IntegerGreaterThan):
        return constraint.lower_bound - poisson(key, 5, shape=size)
    elif isinstance(constraint, constraints._Interval):
        upper_bound = np.broadcast_to(constraint.upper_bound, size)
        return random.uniform(key, size, minval=upper_bound, maxval=upper_bound + 1.)
    elif isinstance(constraint, (constraints._Real, constraints._RealVector)):
        return lax.full(size, np.nan)
    elif isinstance(constraint, constraints._Simplex):
        return osp.dirichlet.rvs(alpha=np.ones((size[-1],)), size=size[:-1]) + 1e-2
    elif isinstance(constraint, constraints._Multinomial):
        n = size[-1]
        return multinomial(key, p=np.ones((n,)) / n, n=constraint.upper_bound, shape=size[:-1]) + 1
    elif isinstance(constraint, constraints._CorrCholesky):
        return signed_stick_breaking_tril(
            random.uniform(key, size[:-2] + (size[-1] * (size[-1] - 1) // 2,),
                           minval=-1, maxval=1)) + 1e-2
    elif isinstance(constraint, constraints._CorrMatrix):
        cholesky = 1e-2 + signed_stick_breaking_tril(
            random.uniform(key, size[:-2] + (size[-1] * (size[-1] - 1) // 2,), minval=-1, maxval=1))
        return np.matmul(cholesky, np.swapaxes(cholesky, -2, -1))
    elif isinstance(constraint, constraints._LowerCholesky):
        return random.uniform(key, size)
    elif isinstance(constraint, constraints._PositiveDefinite):
        return random.normal(key, size)
    else:
        raise NotImplementedError('{} not implemented.'.format(constraint))


@pytest.mark.parametrize('jax_dist, sp_dist, params', CONTINUOUS + DISCRETE)
@pytest.mark.parametrize('prepend_shape', [
    (),
    (2,),
    (2, 3),
])
def test_dist_shape(jax_dist, sp_dist, params, prepend_shape):
    jax_dist = jax_dist(*params)
    rng = random.PRNGKey(0)
    expected_shape = prepend_shape + jax_dist.batch_shape + jax_dist.event_shape
    samples = jax_dist.sample(key=rng, sample_shape=prepend_shape)
    assert isinstance(samples, jax.interpreters.xla.DeviceArray)
    assert np.shape(samples) == expected_shape
    if sp_dist and not _is_batched_multivariate(jax_dist):
        sp_dist = sp_dist(*params)
        sp_samples = sp_dist.rvs(size=prepend_shape + jax_dist.batch_shape)
        assert np.shape(sp_samples) == expected_shape


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

    rng = random.PRNGKey(0)

    def fn(args):
        args_dict = dict(zip(jax_dist.reparametrized_params, args))
        return np.sum(jax_dist(**args_dict, **nonrepara_params_dict).sample(key=rng))

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
        assert np.shape(actual_grad[i]) == np.shape(repara_params[i])
        assert_allclose(np.sum(actual_grad[i]), expected_grad, rtol=0.02)


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
    rng = random.PRNGKey(0)
    N = 100
    z = jax_dist(*params).sample(key=rng, sample_shape=(N,))
    actual_grad = jacfwd(lambda x: jax_dist(*x).sample(key=rng, sample_shape=(N,)))(params)
    eps = 1e-3
    for i in range(len(params)):
        args_lhs = [p if j != i else p - eps for j, p in enumerate(params)]
        args_rhs = [p if j != i else p + eps for j, p in enumerate(params)]
        cdf_dot = (sp_dist(*args_rhs).cdf(z) - sp_dist(*args_lhs).cdf(z)) / (2 * eps)
        expected_grad = -cdf_dot / sp_dist(*params).pdf(z)
        assert_allclose(actual_grad[i], expected_grad, rtol=0.005)


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
    rng = random.PRNGKey(0)
    samples = jax_dist.sample(key=rng, sample_shape=prepend_shape)
    assert jax_dist.log_prob(samples).shape == prepend_shape + jax_dist.batch_shape
    if not sp_dist:
        if isinstance(jax_dist, dist.TruncatedCauchy) or isinstance(jax_dist, dist.TruncatedNormal):
            low, loc, scale = params
            high = np.inf
            sp_dist = osp.cauchy if isinstance(jax_dist, dist.TruncatedCauchy) else osp.norm
            sp_dist = sp_dist(loc, scale)
            expected = sp_dist.logpdf(samples) - np.log(sp_dist.cdf(high) - sp_dist.cdf(low))
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
        # precision issue: np.sum(x / np.sum(x)) = 0.99999994 != 1
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
        assert np.shape(indep.log_prob(sample)) == shape[:event_boundary]


def _tril_cholesky_to_tril_corr(x):
    w = vec_to_tril_matrix(x, diagonal=-1)
    diag = np.sqrt(1 - np.sum(w ** 2, axis=-1))
    cholesky = w + np.expand_dims(diag, axis=-1) * np.identity(w.shape[-1])
    corr = np.matmul(cholesky, cholesky.T)
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
        cholesky_to_corr_jac = onp.linalg.slogdet(
            jax.jacobian(_tril_cholesky_to_tril_corr)(sample_tril))[1]
        corr_log_prob.append(log_prob - cholesky_to_corr_jac)

    corr_log_prob = np.array(corr_log_prob)
    # test if they are constant
    assert_allclose(corr_log_prob, np.broadcast_to(corr_log_prob[0], corr_log_prob.shape),
                    rtol=1e-6)

    if dimension == 2:
        # when concentration = 1, LKJ gives a uniform distribution over correlation matrix,
        # hence for the case dimension = 2,
        # density of a correlation matrix will be Uniform(-1, 1) = 0.5.
        # In addition, jacobian of the transformation from cholesky -> corr is 1 (hence its
        # log value is 0) because the off-diagonal lower triangular element does not change
        # in the transform.
        # So target_log_prob = log(0.5)
        assert_allclose(corr_log_prob[0], np.log(0.5), rtol=1e-6)


@pytest.mark.parametrize("dimension", [2, 3, 5])
@pytest.mark.parametrize("concentration", [0.6, 2.2])
def test_log_prob_LKJCholesky(dimension, concentration):
    # We will test against the fact that LKJCorrCholesky can be seen as a
    # TransformedDistribution with base distribution is a distribution of partial
    # correlations in C-vine method (modulo an affine transform to change domain from (0, 1)
    # to (1, 0)) and transform is a signed stick-breaking process.
    d = dist.LKJCholesky(dimension, concentration, sample_method="cvine")

    beta_sample = d._beta.sample(random.PRNGKey(0))
    beta_log_prob = np.sum(d._beta.log_prob(beta_sample))
    partial_correlation = 2 * beta_sample - 1
    affine_logdet = beta_sample.shape[-1] * np.log(2)
    sample = signed_stick_breaking_tril(partial_correlation)

    # compute signed stick breaking logdet
    inv_tanh = lambda t: np.log((1 + t) / (1 - t)) / 2  # noqa: E731
    inv_tanh_logdet = np.sum(np.log(vmap(grad(inv_tanh))(partial_correlation)))
    unconstrained = inv_tanh(partial_correlation)
    corr_cholesky_logdet = biject_to(constraints.corr_cholesky).log_abs_det_jacobian(
        unconstrained,
        sample,
    )
    signed_stick_breaking_logdet = corr_cholesky_logdet + inv_tanh_logdet

    actual_log_prob = d.log_prob(sample)
    expected_log_prob = beta_log_prob - affine_logdet - signed_stick_breaking_logdet
    assert_allclose(actual_log_prob, expected_log_prob, rtol=1e-5)

    assert_allclose(jax.jit(d.log_prob)(sample), d.log_prob(sample), atol=1e-7)


@pytest.mark.parametrize('jax_dist, sp_dist, params', CONTINUOUS + DISCRETE)
def test_log_prob_gradient(jax_dist, sp_dist, params):
    if jax_dist in [dist.LKJ, dist.LKJCholesky]:
        pytest.skip('we have separated tests for LKJCholesky distribution')
    rng = random.PRNGKey(0)

    value = jax_dist(*params).sample(rng)

    def fn(*args):
        return np.sum(jax_dist(*args).log_prob(value))

    eps = 1e-3
    for i in range(len(params)):
        if params[i] is None or np.result_type(params[i]) in (np.int32, np.int64):
            continue
        actual_grad = jax.grad(fn, i)(*params)
        args_lhs = [p if j != i else p - eps for j, p in enumerate(params)]
        args_rhs = [p if j != i else p + eps for j, p in enumerate(params)]
        fn_lhs = fn(*args_lhs)
        fn_rhs = fn(*args_rhs)
        # finite diff approximation
        expected_grad = (fn_rhs - fn_lhs) / (2. * eps)
        assert np.shape(actual_grad) == np.shape(params[i])
        if i == 0 and jax_dist is dist.Delta:
            # grad w.r.t. `value` of Delta distribution will be 0
            # but numerical value will give nan (= inf - inf)
            expected_grad = 0.
        assert_allclose(np.sum(actual_grad), expected_grad, rtol=0.01, atol=1e-3)


@pytest.mark.parametrize('jax_dist, sp_dist, params', CONTINUOUS + DISCRETE)
def test_mean_var(jax_dist, sp_dist, params):
    n = 200000
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
                sp_var = np.diag(d_sp.cov())
            except TypeError:  # mvn does not have .cov() method
                sp_var = np.diag(d_sp.cov)
            except AttributeError:
                sp_var = d_sp.var()
        else:
            sp_var = d_sp.var()
        assert_allclose(d_jax.mean, sp_mean, rtol=0.01, atol=1e-7)
        assert_allclose(d_jax.variance, sp_var, rtol=0.01, atol=1e-7)
        if np.all(np.isfinite(sp_mean)):
            assert_allclose(np.mean(samples, 0), d_jax.mean, rtol=0.05, atol=1e-2)
        if np.all(np.isfinite(sp_var)):
            assert_allclose(np.std(samples, 0), np.sqrt(d_jax.variance), rtol=0.05, atol=1e-2)
    elif jax_dist in [dist.LKJ, dist.LKJCholesky]:
        if jax_dist is dist.LKJCholesky:
            corr_samples = np.matmul(samples, np.swapaxes(samples, -2, -1))
        else:
            corr_samples = samples
        dimension, concentration, _ = params
        # marginal of off-diagonal entries
        marginal = dist.Beta(concentration + 0.5 * (dimension - 2),
                             concentration + 0.5 * (dimension - 2))
        # scale statistics due to linear mapping
        marginal_mean = 2 * marginal.mean - 1
        marginal_std = 2 * np.sqrt(marginal.variance)
        expected_mean = np.broadcast_to(np.reshape(marginal_mean, np.shape(marginal_mean) + (1, 1)),
                                        np.shape(marginal_mean) + d_jax.event_shape)
        expected_std = np.broadcast_to(np.reshape(marginal_std, np.shape(marginal_std) + (1, 1)),
                                       np.shape(marginal_std) + d_jax.event_shape)
        # diagonal elements of correlation matrices are 1
        expected_mean = expected_mean * (1 - np.identity(dimension)) + np.identity(dimension)
        expected_std = expected_std * (1 - np.identity(dimension))

        assert_allclose(np.mean(corr_samples, axis=0), expected_mean, atol=0.005)
        assert_allclose(np.std(corr_samples, axis=0), expected_std, atol=0.005)
    else:
        if np.all(np.isfinite(d_jax.mean)):
            assert_allclose(np.mean(samples, 0), d_jax.mean, rtol=0.05, atol=1e-2)
        if np.all(np.isfinite(d_jax.variance)):
            assert_allclose(np.std(samples, 0), np.sqrt(d_jax.variance), rtol=0.05, atol=1e-2)


@pytest.mark.parametrize('jax_dist, sp_dist, params', CONTINUOUS + DISCRETE)
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
        if jax_dist in (dist.LKJ, dist.LKJCholesky) and dist_args[i] != "concentration":
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
        oob_params[i] = gen_values_outside_bounds(constraint, np.shape(params[i]), key_gen)
        valid_params[i] = gen_values_within_bounds(constraint, np.shape(params[i]), key_gen)

    assert jax_dist(*oob_params)

    # Invalid parameter values throw ValueError
    if not dependent_constraint:
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
    with pytest.raises(ValueError):
        d.log_prob(oob_samples)


########################################
# Tests for constraints and transforms #
########################################


@pytest.mark.parametrize('constraint, x, expected', [
    (constraints.boolean, np.array([True, False]), np.array([True, True])),
    (constraints.boolean, np.array([1, 1]), np.array([True, True])),
    (constraints.boolean, np.array([-1, 1]), np.array([False, True])),
    (constraints.corr_cholesky, np.array([[[1, 0], [0, 1]], [[1, 0.1], [0, 1]]]),
     np.array([True, False])),  # NB: not lower_triangular
    (constraints.corr_cholesky, np.array([[[1, 0], [1, 0]], [[1, 0], [0.5, 0.5]]]),
     np.array([False, False])),  # NB: not positive_diagonal & not unit_norm_row
    (constraints.corr_matrix, np.array([[[1, 0], [0, 1]], [[1, 0.1], [0, 1]]]),
     np.array([True, False])),  # NB: not lower_triangular
    (constraints.corr_matrix, np.array([[[1, 0], [1, 0]], [[1, 0], [0.5, 0.5]]]),
     np.array([False, False])),  # NB: not unit diagonal
    (constraints.greater_than(1), 3, True),
    (constraints.greater_than(1), np.array([-1, 1, 5]), np.array([False, False, True])),
    (constraints.integer_interval(-3, 5), 0, True),
    (constraints.integer_interval(-3, 5), np.array([-5, -3, 0, 1.1, 5, 7]),
     np.array([False, True, True, False, True, False])),
    (constraints.interval(-3, 5), 0, True),
    (constraints.interval(-3, 5), np.array([-5, -3, 0, 5, 7]),
     np.array([False, False, True, False, False])),
    (constraints.lower_cholesky, np.array([[1., 0.], [-2., 0.1]]), True),
    (constraints.lower_cholesky, np.array([[[1., 0.], [-2., -0.1]], [[1., 0.1], [2., 0.2]]]),
     np.array([False, False])),
    (constraints.nonnegative_integer, 3, True),
    (constraints.nonnegative_integer, np.array([-1., 0., 5.]), np.array([False, True, True])),
    (constraints.positive, 3, True),
    (constraints.positive, np.array([-1, 0, 5]), np.array([False, False, True])),
    (constraints.positive_definite, np.array([[1., 0.3], [0.3, 1.]]), True),
    (constraints.positive_definite, np.array([[[2., 0.4], [0.3, 2.]], [[1., 0.1], [0.1, 0.]]]),
     np.array([False, False])),
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
])
def test_constraints(constraint, x, expected):
    assert_array_equal(constraint(x), expected)


@pytest.mark.parametrize('constraint', [
    constraints.corr_cholesky,
    constraints.corr_matrix,
    constraints.greater_than(2),
    constraints.interval(-3, 5),
    constraints.lower_cholesky,
    constraints.positive,
    constraints.positive_definite,
    constraints.real,
    constraints.simplex,
    constraints.unit_interval,
])
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
    if len(shape) < event_dim:
        return
    rng = random.PRNGKey(0)
    x = random.normal(rng, shape)
    y = transform(x)

    # test codomain
    batch_shape = shape if event_dim == 0 else shape[:-1]
    assert_array_equal(transform.codomain(y), np.ones(batch_shape, dtype=np.bool_))

    # test inv
    z = transform.inv(y)
    assert_allclose(x, z, atol=1e-6, rtol=1e-6)

    # test domain, currently all is constraints.real or constraints.real_vector
    assert_array_equal(transform.domain(z), np.ones(batch_shape))

    # test log_abs_det_jacobian
    actual = transform.log_abs_det_jacobian(x, y)
    assert np.shape(actual) == batch_shape
    if len(shape) == event_dim:
        if constraint is constraints.simplex:
            expected = onp.linalg.slogdet(jax.jacobian(transform)(x)[:-1, :])[1]
            inv_expected = onp.linalg.slogdet(jax.jacobian(transform.inv)(y)[:, :-1])[1]
        elif constraint in [constraints.corr_cholesky, constraints.corr_matrix]:
            vec_transform = lambda x: matrix_to_tril_vec(transform(x), diagonal=-1)  # noqa: E731
            y_tril = matrix_to_tril_vec(y, diagonal=-1)

            def inv_vec_transform(y):
                matrix = vec_to_tril_matrix(y, diagonal=-1)
                if constraint is constraints.corr_matrix:
                    # fill the upper triangular part
                    matrix = matrix + np.swapaxes(matrix, -2, -1) + np.identity(matrix.shape[-1])
                return transform.inv(matrix)

            expected = onp.linalg.slogdet(jax.jacobian(vec_transform)(x))[1]
            inv_expected = onp.linalg.slogdet(jax.jacobian(inv_vec_transform)(y_tril))[1]
        elif constraint in [constraints.lower_cholesky, constraints.positive_definite]:
            vec_transform = lambda x: matrix_to_tril_vec(transform(x))  # noqa: E731
            y_tril = matrix_to_tril_vec(y)

            def inv_vec_transform(y):
                matrix = vec_to_tril_matrix(y)
                if constraint is constraints.positive_definite:
                    # fill the upper triangular part
                    matrix = matrix + np.swapaxes(matrix, -2, -1) - np.diag(np.diag(matrix))
                return transform.inv(matrix)

            expected = onp.linalg.slogdet(jax.jacobian(vec_transform)(x))[1]
            inv_expected = onp.linalg.slogdet(jax.jacobian(inv_vec_transform)(y_tril))[1]
        else:
            expected = np.log(np.abs(grad(transform)(x)))
            inv_expected = np.log(np.abs(grad(transform.inv)(y)))

        assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
        assert_allclose(actual, -inv_expected, atol=1e-6, rtol=1e-6)


# NB: skip transforms which are tested in `test_biject_to`
@pytest.mark.parametrize('transform, event_shape', [
    (PermuteTransform(np.array([3, 0, 4, 1, 2])), (5,)),
    (PowerTransform(2.), ()),
    (MultivariateAffineTransform(np.array([1., 2.]), np.array([[0.6, 0.], [1.5, 0.4]])), (2,))
])
@pytest.mark.parametrize('batch_shape', [(), (1,), (3,), (6,), (3, 1), (1, 3), (5, 3)])
def test_bijective_transforms(transform, event_shape, batch_shape):
    shape = batch_shape + event_shape
    rng = random.PRNGKey(0)
    x = biject_to(transform.domain)(random.normal(rng, shape))
    y = transform(x)

    # test codomain
    assert_array_equal(transform.codomain(y), np.ones(batch_shape))

    # test inv
    z = transform.inv(y)
    assert_allclose(x, z, atol=1e-6, rtol=1e-6)

    # test domain
    assert_array_equal(transform.domain(z), np.ones(batch_shape))

    # test log_abs_det_jacobian
    actual = transform.log_abs_det_jacobian(x, y)
    assert np.shape(actual) == batch_shape
    if len(shape) == transform.event_dim:
        if len(event_shape) == 1:
            expected = onp.linalg.slogdet(jax.jacobian(transform)(x))[1]
            inv_expected = onp.linalg.slogdet(jax.jacobian(transform.inv)(y))[1]
        else:
            expected = np.log(np.abs(grad(transform)(x)))
            inv_expected = np.log(np.abs(grad(transform.inv)(y)))

        assert_allclose(actual, expected, atol=1e-6)
        assert_allclose(actual, -inv_expected, atol=1e-6)


@pytest.mark.parametrize('transformed_dist', [
    dist.TransformedDistribution(dist.Normal(np.array([2., 3.]), 1.), constraints.ExpTransform()),
    dist.TransformedDistribution(dist.Exponential(np.ones(2)), [
        constraints.PowerTransform(0.7),
        constraints.AffineTransform(0., np.ones(2) * 3)
    ]),
])
def test_transformed_distribution_intermediates(transformed_dist):
    sample, intermediates = transformed_dist.sample_with_intermediates(random.PRNGKey(1))
    assert_allclose(transformed_dist.log_prob(sample, intermediates), transformed_dist.log_prob(sample))


def test_transformed_transformed_distribution():
    loc, scale = -2, 3
    dist1 = dist.TransformedDistribution(dist.Normal(2, 3), constraints.PowerTransform(2.))
    dist2 = dist.TransformedDistribution(dist1, constraints.AffineTransform(-2, 3))
    assert isinstance(dist2.base_dist, dist.Normal)
    assert len(dist2.transforms) == 2
    assert isinstance(dist2.transforms[0], constraints.PowerTransform)
    assert isinstance(dist2.transforms[1], constraints.AffineTransform)

    rng = random.PRNGKey(0)
    assert_allclose(loc + scale * dist1.sample(rng), dist2.sample(rng))
    intermediates = dist2.sample_with_intermediates(rng)
    assert len(intermediates) == 2


def _make_iaf(input_dim, hidden_dims, rng):
    arn_init, arn = AutoregressiveNN(input_dim, hidden_dims, param_dims=[1, 1])
    _, init_params = arn_init(rng, (input_dim,))
    return InverseAutoregressiveTransform(partial(arn, init_params))


@pytest.mark.parametrize('transforms', [
    [constraints.PowerTransform(0.7), constraints.AffineTransform(2., 3.)],
    [constraints.ExpTransform()],
    [constraints.ComposeTransform([constraints.AffineTransform(-2, 3),
                                   constraints.ExpTransform()]),
     constraints.PowerTransform(3.)],
    [_make_iaf(5, hidden_dims=[10], rng=random.PRNGKey(0)),
     constraints.PermuteTransform(np.arange(5)[::-1]),
     _make_iaf(5, hidden_dims=[10], rng=random.PRNGKey(1))]
])
def test_compose_transform_with_intermediates(transforms):
    transform = constraints.ComposeTransform(transforms)
    x = random.normal(random.PRNGKey(2), (7, 5))
    y, intermediates = transform.call_with_intermediates(x)
    logdet = transform.log_abs_det_jacobian(x, y, intermediates)
    assert_allclose(y, transform(x))
    assert_allclose(logdet, transform.log_abs_det_jacobian(x, y))


def test_unpack_transform():
    x = np.ones(3)
    unpack_fn = lambda x: {'key': x}  # noqa: E731
    transform = constraints.UnpackTransform(unpack_fn)
    y = transform(x)
    z = transform.inv(y)
    assert_allclose(y['key'], x)
    assert_allclose(z, x)
