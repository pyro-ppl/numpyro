import inspect
from collections import namedtuple

import pytest
import scipy.stats as osp
from numpy.testing import assert_allclose

import jax
import jax.numpy as np
import jax.random as random

import numpyro.contrib.distributions as dist
from numpyro.contrib.distributions.discrete import _to_probs_bernoulli, _to_probs_multinom


def _identity(x): return x


class T(namedtuple('TestCase', ['jax_dist', 'sp_dist', 'params'])):
    def __new__(cls, jax_dist, *params):
        sp_dist = None
        if jax_dist in _DIST_MAP:
            sp_dist = _DIST_MAP[jax_dist]
        return super(cls, T).__new__(cls, jax_dist, sp_dist, params)


_DIST_MAP = {
    dist.Bernoulli: lambda probs: osp.bernoulli(p=probs),
    dist.BernoulliWithLogits: lambda logits: osp.bernoulli(p=_to_probs_bernoulli(logits)),
    dist.Beta: lambda con1, con0: osp.beta(con1, con0),
    dist.Binomial: lambda probs, total_count: osp.binom(n=total_count, p=probs),
    dist.BinomialWithLogits: lambda logits, total_count: osp.binom(n=total_count, p=_to_probs_bernoulli(logits)),
    dist.Cauchy: lambda loc, scale: osp.cauchy(loc=loc, scale=scale),
    dist.Chi2: lambda df: osp.chi2(df),
    dist.Dirichlet: lambda conc: osp.dirichlet(conc),
    dist.Exponential: lambda rate: osp.expon(scale=np.reciprocal(rate)),
    dist.Gamma: lambda conc, rate: osp.gamma(conc, scale=1./rate),
    dist.HalfCauchy: lambda scale: osp.halfcauchy(scale=scale),
    dist.LogNormal: lambda loc, scale: osp.lognorm(s=scale, scale=np.exp(loc)),
    dist.Multinomial: lambda probs, total_count: osp.multinomial(n=total_count, p=probs),
    dist.MultinomialWithLogits: lambda logits, total_count: osp.multinomial(n=total_count,
                                                                            p=_to_probs_multinom(logits)),
    dist.Normal: lambda loc, scale: osp.norm(loc=loc, scale=scale),
    dist.Pareto: lambda scale, alpha: osp.pareto(alpha, scale=scale),
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
    T(dist.HalfCauchy, 1.),
    T(dist.HalfCauchy, np.array([1., 2.])),
    T(dist.LogNormal, 1., 0.2),
    T(dist.LogNormal, -1., np.array([0.5, 1.3])),
    T(dist.LogNormal, np.array([0.5, -0.7]), np.array([[0.1, 0.4], [0.5, 0.1]])),
    T(dist.Normal, 0., 1.),
    T(dist.Normal, 1., np.array([1., 2.])),
    T(dist.Normal, np.array([0., 1.]), np.array([[1.], [2.]])),
    T(dist.Pareto, 2., 1.),
    T(dist.Pareto, np.array([0.3, 2.]), np.array([1., 0.5])),
    T(dist.Pareto, np.array([1., 0.5]), np.array([[1.], [3.]])),
    T(dist.StudentT, 1., 1., 0.5),
    T(dist.StudentT, 1.5, np.array([1., 2.]), 2.),
    T(dist.StudentT, np.array([3, 5]), np.array([[1.], [2.]]), 2.),
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
    T(dist.Categorical, np.array([1.])),
    T(dist.Categorical, np.array([0.1, 0.5, 0.4])),
    T(dist.Categorical, np.array([[0.1, 0.5, 0.4], [0.4, 0.4, 0.2]])),
    T(dist.CategoricalWithLogits, np.array([-5.])),
    T(dist.CategoricalWithLogits, np.array([1., 2., -2.])),
    T(dist.CategoricalWithLogits, np.array([[-1, 2., 3.], [3., -4., -2.]])),
    T(dist.Multinomial, np.array([0.2, 0.7, 0.1]), 10),
    T(dist.Multinomial, np.array([0.2, 0.7, 0.1]), np.array([5, 8])),
    T(dist.MultinomialWithLogits, np.array([-1., 3.]), np.array([[5], [8]])),
    T(dist.Poisson, 2.),
    T(dist.Poisson, np.array([2., 3., 5.])),
]


def _is_batched_multivariate(jax_dist):
    return len(jax_dist.event_shape) > 0 and len(jax_dist.batch_shape) > 0


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
    samples = jax_dist.sample(key=rng, size=prepend_shape)
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

    rng = random.PRNGKey(0)

    def fn(args):
        return np.sum(jax_dist(*args).sample(key=rng))

    actual_grad = jax.grad(fn)(params)
    assert len(actual_grad) == len(params)

    eps = 1e-5
    for i in range(len(params)):
        if np.result_type(params[i]) in (np.int32, np.int64) or \
                dist_args[i] not in jax_dist.reparametrized_params:
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
    rng = random.PRNGKey(0)
    samples = jax_dist.sample(key=rng, size=prepend_shape)
    if not sp_dist:
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
    assert_allclose(jit_fn(jax_dist.log_prob)(samples), expected, atol=1e-5)


@pytest.mark.parametrize('jax_dist, sp_dist, params', CONTINUOUS + DISCRETE)
def test_log_prob_gradient(jax_dist, sp_dist, params):
    rng = random.PRNGKey(0)

    def fn(args, value):
        return np.sum(jax_dist(*args).log_prob(value))

    value = jax_dist(*params).sample(rng)
    actual_grad = jax.grad(fn)(params, value)
    assert len(actual_grad) == len(params)

    eps = 1e-4
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
        assert_allclose(np.sum(actual_grad[i]), expected_grad, rtol=0.10, atol=1e-3)


@pytest.mark.parametrize('jax_dist, sp_dist, params', CONTINUOUS + DISCRETE)
def test_mean_var(jax_dist, sp_dist, params):
    n = 200000
    d_jax = jax_dist(*params)
    k = random.PRNGKey(0)
    samples = d_jax.sample(k, size=(n,))
    # check with suitable scipy implementation if available
    if sp_dist and not _is_batched_multivariate(d_jax):
        d_sp = sp_dist(*params)
        sp_mean = d_sp.mean()
        # for multivariate distns try .cov first
        if d_jax.event_shape:
            try:
                sp_var = np.diag(d_sp.cov())
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
    else:
        if np.all(np.isfinite(d_jax.mean)):
            assert_allclose(np.mean(samples, 0), d_jax.mean, rtol=0.05, atol=1e-2)
        if np.all(np.isfinite(d_jax.variance)):
            assert_allclose(np.std(samples, 0), np.sqrt(d_jax.variance), rtol=0.05, atol=1e-2)
