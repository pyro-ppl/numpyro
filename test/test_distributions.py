# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from functools import partial
import inspect
from itertools import product
import math
import os
from typing import Callable

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
import scipy
from scipy.sparse import csr_matrix
import scipy.stats as osp

import jax
from jax import grad, lax, vmap
import jax.numpy as jnp
import jax.random as random
from jax.scipy.special import expit, logsumexp
from jax.scipy.stats import norm as jax_norm, truncnorm as jax_truncnorm

import numpyro.distributions as dist
from numpyro.distributions import (
    SineBivariateVonMises,
    constraints,
    kl_divergence,
    transforms,
)
from numpyro.distributions.batch_util import vmap_over
from numpyro.distributions.discrete import _to_probs_bernoulli, _to_probs_multinom
from numpyro.distributions.flows import InverseAutoregressiveTransform
from numpyro.distributions.gof import InvalidTest, auto_goodness_of_fit
from numpyro.distributions.transforms import (
    LowerCholeskyAffine,
    PermuteTransform,
    PowerTransform,
    SimplexToOrderedTransform,
    SoftplusTransform,
    biject_to,
)
from numpyro.distributions.util import (
    matrix_to_tril_vec,
    multinomial,
    signed_stick_breaking_tril,
    sum_rightmost,
    vec_to_tril_matrix,
)
from numpyro.nn import AutoregressiveNN

TEST_FAILURE_RATE = 2e-5  # For all goodness-of-fit tests.


def my_kron(A, B):
    D = A[..., :, None, :, None] * B[..., None, :, None, :]
    ds = D.shape
    newshape = (*ds[:-4], ds[-4] * ds[-3], ds[-2] * ds[-1])
    return D.reshape(newshape)


def _identity(x):
    return x


def _circ_mean(angles):
    return jnp.arctan2(
        jnp.mean(jnp.sin(angles), axis=0), jnp.mean(jnp.cos(angles), axis=0)
    )


def sde_fn1(x, _):
    lam = 0.1
    sigma2 = 0.1
    return lam * x, sigma2


def sde_fn2(xy, _):
    tau, a = 2.0, 1.1
    x, y = xy[0], xy[1]
    dx = tau * (x - x**3.0 / 3.0 + y)
    dy = (1.0 / tau) * (a - x)
    dxy = jnp.vstack([dx, dy]).reshape(xy.shape)

    sigma2 = 0.1
    return dxy, sigma2


class T(namedtuple("TestCase", ["jax_dist", "sp_dist", "params"])):
    def __new__(cls, jax_dist, *params):
        sp_dist = get_sp_dist(jax_dist)
        return super(cls, T).__new__(cls, jax_dist, sp_dist, params)


def _mvn_to_scipy(loc, cov, prec, tril):
    jax_dist = dist.MultivariateNormal(loc, cov, prec, tril)
    mean = jax_dist.mean
    cov = jax_dist.covariance_matrix
    return osp.multivariate_normal(mean=mean, cov=cov)


def _multivariate_t_to_scipy(df, loc, tril):
    if scipy.__version__ < "1.6.0":
        pytest.skip(
            "Multivariate Student-T distribution is not available in scipy < 1.6"
        )
    jax_dist = dist.MultivariateStudentT(df, loc, tril)
    mean = jax_dist.mean
    cov = jax_dist.covariance_matrix
    return osp.multivariate_t(loc=mean, shape=cov, df=df)


def _lowrank_mvn_to_scipy(loc, cov_fac, cov_diag):
    jax_dist = dist.LowRankMultivariateNormal(loc, cov_fac, cov_diag)
    mean = jax_dist.mean
    cov = jax_dist.covariance_matrix
    return osp.multivariate_normal(mean=mean, cov=cov)


def _truncnorm_to_scipy(loc, scale, low, high):
    if low is None:
        a = -np.inf
    else:
        a = (low - loc) / scale
    if high is None:
        b = np.inf
    else:
        b = (high - loc) / scale
    return osp.truncnorm(a, b, loc=loc, scale=scale)


def _wishart_to_scipy(conc, scale, rate, tril):
    jax_dist = dist.Wishart(conc, scale, rate, tril)
    if not jnp.isscalar(jax_dist.concentration):
        pytest.skip("scipy Wishart only supports a single scalar concentration")
    # Cast to float explicitly because np.isscalar returns False on scalar jax arrays.
    return osp.wishart(float(jax_dist.concentration), jax_dist.scale_matrix)


def _TruncatedNormal(loc, scale, low, high):
    return dist.TruncatedNormal(loc=loc, scale=scale, low=low, high=high)


def _TruncatedCauchy(loc, scale, low, high):
    return dist.TruncatedCauchy(loc=loc, scale=scale, low=low, high=high)


_TruncatedNormal.arg_constraints = {}
_TruncatedNormal.reparametrized_params = []
_TruncatedNormal.infer_shapes = lambda *args: (lax.broadcast_shapes(*args), ())


class SineSkewedUniform(dist.SineSkewed):
    def __init__(self, skewness, **kwargs):
        lower, upper = (np.array([-math.pi, -math.pi]), np.array([math.pi, math.pi]))
        base_dist = dist.Uniform(lower, upper, **kwargs).to_event(lower.ndim)
        super().__init__(base_dist, skewness, **kwargs)


@vmap_over.register
def _vmap_over_sine_skewed_uniform(self: SineSkewedUniform, skewness=None):
    return vmap_over.dispatch(dist.SineSkewed)(self, base_dist=None, skewness=skewness)


class SineSkewedVonMises(dist.SineSkewed):
    def __init__(self, skewness, **kwargs):
        von_loc, von_conc = (np.array([0.0]), np.array([1.0]))
        base_dist = dist.VonMises(von_loc, von_conc, **kwargs).to_event(von_loc.ndim)
        super().__init__(base_dist, skewness, **kwargs)


@vmap_over.register
def _vmap_over_sine_skewed_von_mises(self: SineSkewedVonMises, skewness=None):
    return vmap_over.dispatch(dist.SineSkewed)(self, base_dist=None, skewness=skewness)


class SineSkewedVonMisesBatched(dist.SineSkewed):
    def __init__(self, skewness, **kwargs):
        von_loc, von_conc = (np.array([0.0, -1.234]), np.array([1.0, 10.0]))
        base_dist = dist.VonMises(von_loc, von_conc, **kwargs).to_event(von_loc.ndim)
        super().__init__(base_dist, skewness, **kwargs)


@vmap_over.register
def _vmap_over_sine_skewed_von_mises_batched(
    self: SineSkewedVonMisesBatched, skewness=None
):
    return vmap_over.dispatch(dist.SineSkewed)(self, base_dist=None, skewness=skewness)


class _GaussianMixture(dist.MixtureSameFamily):
    arg_constraints = {}
    reparametrized_params = []

    def __init__(self, mixing_probs, loc, scale):
        component_dist = dist.Normal(loc=loc, scale=scale)
        mixing_distribution = dist.Categorical(probs=mixing_probs)
        super().__init__(
            mixing_distribution=mixing_distribution,
            component_distribution=component_dist,
        )

    @property
    def loc(self):
        return self.component_distribution.loc

    @property
    def scale(self):
        return self.component_distribution.scale


@vmap_over.register
def _vmap_over_gaussian_mixture(self: _GaussianMixture, loc=None, scale=None):
    component_distribution = vmap_over(
        self.component_distribution, loc=loc, scale=scale
    )
    return vmap_over.dispatch(dist.MixtureSameFamily)(
        self, _component_distribution=component_distribution
    )


class _Gaussian2DMixture(dist.MixtureSameFamily):
    arg_constraints = {}
    reparametrized_params = []

    def __init__(self, mixing_probs, loc, covariance_matrix):
        component_dist = dist.MultivariateNormal(
            loc=loc, covariance_matrix=covariance_matrix
        )
        mixing_distribution = dist.Categorical(probs=mixing_probs)
        super().__init__(
            mixing_distribution=mixing_distribution,
            component_distribution=component_dist,
        )

    @property
    def loc(self):
        return self.component_distribution.loc

    @property
    def covariance_matrix(self):
        return self.component_distribution.covariance_matrix


@vmap_over.register
def _vmap_over_gaussian_2d_mixture(self: _Gaussian2DMixture, loc=None):
    component_distribution = vmap_over(self.component_distribution, loc=loc)
    return vmap_over.dispatch(dist.MixtureSameFamily)(
        self, _component_distribution=component_distribution
    )


class _GeneralMixture(dist.MixtureGeneral):
    arg_constraints = {}
    reparametrized_params = []

    def __init__(self, mixing_probs, locs, scales):
        component_dists = [
            dist.Normal(loc=loc_, scale=scale_) for loc_, scale_ in zip(locs, scales)
        ]
        mixing_distribution = dist.Categorical(probs=mixing_probs)
        return super().__init__(
            mixing_distribution=mixing_distribution,
            component_distributions=component_dists,
        )

    @property
    def locs(self):
        # hotfix for vmapping tests, which cannot easily check non-array attributes
        return self.component_distributions[0].loc

    @property
    def scales(self):
        return self.component_distributions[0].scale


@vmap_over.register
def _vmap_over_general_mixture(self: _GeneralMixture, locs=None, scales=None):
    component_distributions = [
        vmap_over(d, loc=locs, scale=scales) for d in self.component_distributions
    ]
    return vmap_over.dispatch(dist.MixtureGeneral)(
        self, _component_distributions=component_distributions
    )


class _General2DMixture(dist.MixtureGeneral):
    arg_constraints = {}
    reparametrized_params = []

    def __init__(self, mixing_probs, locs, covariance_matrices):
        component_dists = [
            dist.MultivariateNormal(loc=loc_, covariance_matrix=covariance_matrix)
            for loc_, covariance_matrix in zip(locs, covariance_matrices)
        ]
        mixing_distribution = dist.Categorical(probs=mixing_probs)
        return super().__init__(
            mixing_distribution=mixing_distribution,
            component_distributions=component_dists,
        )

    @property
    def locs(self):
        # hotfix for vmapping tests, which cannot easily check non-array attributes
        return self.component_distributions[0].loc

    @property
    def covariance_matrices(self):
        return self.component_distributions[0].covariance_matrix


@vmap_over.register
def _vmap_over_general_2d_mixture(self: _General2DMixture, locs=None):
    component_distributions = [
        vmap_over(d, loc=locs) for d in self.component_distributions
    ]
    return vmap_over.dispatch(dist.MixtureGeneral)(
        self, _component_distributions=component_distributions
    )


class _ImproperWrapper(dist.ImproperUniform):
    def sample(self, key, sample_shape=()):
        transform = biject_to(self.support)
        prototype_value = jnp.zeros(self.event_shape)
        unconstrained_event_shape = jnp.shape(transform.inv(prototype_value))
        shape = sample_shape + self.batch_shape + unconstrained_event_shape
        unconstrained_samples = random.uniform(key, shape, minval=-2, maxval=2)
        return transform(unconstrained_samples)


class ZeroInflatedPoissonLogits(dist.discrete.ZeroInflatedLogits):
    arg_constraints = {"rate": constraints.positive, "gate_logits": constraints.real}
    pytree_data_fields = ("rate",)

    def __init__(self, rate, gate_logits, *, validate_args=None):
        self.rate = rate
        super().__init__(dist.Poisson(rate), gate_logits, validate_args=validate_args)


@vmap_over.register
def _vmap_over_zero_inflated_poisson_logits(
    self: ZeroInflatedPoissonLogits, rate=None, gate_logits=None
):
    dist_axes = vmap_over.dispatch(dist.discrete.ZeroInflatedLogits)(
        self,
        base_dist=vmap_over(self.base_dist, rate=rate),
        gate_logits=gate_logits,
        gate=gate_logits,
    )
    dist_axes.rate = rate
    return dist_axes


class SparsePoisson(dist.Poisson):
    def __init__(self, rate, *, validate_args=None):
        super().__init__(rate, is_sparse=True, validate_args=validate_args)


class FoldedNormal(dist.FoldedDistribution):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}

    def __init__(self, loc, scale, validate_args=None):
        self.loc = loc
        self.scale = scale
        super().__init__(dist.Normal(loc, scale), validate_args=validate_args)


@vmap_over.register
def _vmap_over_folded_normal(self: "FoldedNormal", loc=None, scale=None):
    d = vmap_over.dispatch(dist.FoldedDistribution)(
        self, base_dist=vmap_over(self.base_dist, loc=loc, scale=scale)
    )
    d.loc = loc
    d.scale = scale
    return d


class _SparseCAR(dist.CAR):
    reparametrized_params = ["loc", "correlation", "conditional_precision"]

    def __init__(
        self,
        loc,
        correlation,
        conditional_precision,
        adj_matrix,
        *,
        is_sparse=True,
        validate_args=None,
    ):
        super().__init__(
            loc,
            correlation,
            conditional_precision,
            adj_matrix,
            is_sparse=True,
            validate_args=validate_args,
        )


_DIST_MAP = {
    dist.AsymmetricLaplace: lambda loc, scale, asymmetry: osp.laplace_asymmetric(
        asymmetry, loc=loc, scale=scale
    ),
    dist.BernoulliProbs: lambda probs: osp.bernoulli(p=probs),
    dist.BernoulliLogits: lambda logits: osp.bernoulli(p=_to_probs_bernoulli(logits)),
    dist.Beta: lambda con1, con0: osp.beta(con1, con0),
    dist.BetaProportion: lambda mu, kappa: osp.beta(mu * kappa, (1 - mu) * kappa),
    dist.BinomialProbs: lambda probs, total_count: osp.binom(n=total_count, p=probs),
    dist.BinomialLogits: lambda logits, total_count: osp.binom(
        n=total_count, p=_to_probs_bernoulli(logits)
    ),
    dist.Cauchy: lambda loc, scale: osp.cauchy(loc=loc, scale=scale),
    dist.Chi2: lambda df: osp.chi2(df),
    dist.Dirichlet: lambda conc: osp.dirichlet(conc),
    dist.DiscreteUniform: lambda low, high: osp.randint(low, high + 1),
    dist.Exponential: lambda rate: osp.expon(scale=jnp.reciprocal(rate)),
    dist.Gamma: lambda conc, rate: osp.gamma(conc, scale=1.0 / rate),
    dist.GeometricProbs: lambda probs: osp.geom(p=probs, loc=-1),
    dist.GeometricLogits: lambda logits: osp.geom(
        p=_to_probs_bernoulli(logits), loc=-1
    ),
    dist.Gumbel: lambda loc, scale: osp.gumbel_r(loc=loc, scale=scale),
    dist.HalfCauchy: lambda scale: osp.halfcauchy(scale=scale),
    dist.HalfNormal: lambda scale: osp.halfnorm(scale=scale),
    dist.InverseGamma: lambda conc, rate: osp.invgamma(conc, scale=rate),
    dist.Laplace: lambda loc, scale: osp.laplace(loc=loc, scale=scale),
    dist.LogNormal: lambda loc, scale: osp.lognorm(s=scale, scale=jnp.exp(loc)),
    dist.LogUniform: lambda a, b: osp.loguniform(a, b),
    dist.MultinomialProbs: lambda probs, total_count: osp.multinomial(
        n=total_count, p=probs
    ),
    dist.MultinomialLogits: lambda logits, total_count: osp.multinomial(
        n=total_count, p=_to_probs_multinom(logits)
    ),
    dist.MultivariateNormal: _mvn_to_scipy,
    dist.MultivariateStudentT: _multivariate_t_to_scipy,
    dist.LowRankMultivariateNormal: _lowrank_mvn_to_scipy,
    dist.Normal: lambda loc, scale: osp.norm(loc=loc, scale=scale),
    dist.Pareto: lambda scale, alpha: osp.pareto(alpha, scale=scale),
    dist.Poisson: lambda rate: osp.poisson(rate),
    dist.StudentT: lambda df, loc, scale: osp.t(df=df, loc=loc, scale=scale),
    dist.Uniform: lambda a, b: osp.uniform(a, b - a),
    dist.Logistic: lambda loc, scale: osp.logistic(loc=loc, scale=scale),
    dist.VonMises: lambda loc, conc: osp.vonmises(
        loc=np.array(loc, dtype=np.float64), kappa=np.array(conc, dtype=np.float64)
    ),
    dist.Weibull: lambda scale, conc: osp.weibull_min(
        c=conc,
        scale=scale,
    ),
    dist.Wishart: _wishart_to_scipy,
    _TruncatedNormal: _truncnorm_to_scipy,
}


def get_sp_dist(jax_dist):
    classes = jax_dist.mro() if isinstance(jax_dist, type) else [jax_dist]
    for cls in classes:
        if cls in _DIST_MAP:
            return _DIST_MAP[cls]


CONTINUOUS = [
    T(dist.AsymmetricLaplace, 1.0, 0.5, 1.0),
    T(dist.AsymmetricLaplace, np.array([1.0, 2.0]), 2.0, 2.0),
    T(dist.AsymmetricLaplace, np.array([[1.0], [2.0]]), 2.0, np.array([3.0, 5.0])),
    T(dist.AsymmetricLaplaceQuantile, 0.0, 1.0, 0.5),
    T(dist.AsymmetricLaplaceQuantile, np.array([1.0, 2.0]), 2.0, 0.7),
    T(
        dist.AsymmetricLaplaceQuantile,
        np.array([[1.0], [2.0]]),
        2.0,
        np.array([0.2, 0.8]),
    ),
    T(dist.Beta, 0.2, 1.1),
    T(dist.Beta, 1.0, np.array([2.0, 2.0])),
    T(dist.Beta, 1.0, np.array([[1.0, 1.0], [2.0, 2.0]])),
    T(dist.BetaProportion, 0.2, 10.0),
    T(dist.BetaProportion, 0.51, np.array([2.0, 1.0])),
    T(dist.BetaProportion, 0.5, np.array([[4.0, 4.0], [2.0, 2.0]])),
    T(dist.Chi2, 2.0),
    T(dist.Chi2, np.array([0.3, 1.3])),
    T(dist.Cauchy, 0.0, 1.0),
    T(dist.Cauchy, 0.0, np.array([1.0, 2.0])),
    T(dist.Cauchy, np.array([0.0, 1.0]), np.array([[1.0], [2.0]])),
    T(dist.Dirichlet, np.array([1.7])),
    T(dist.Dirichlet, np.array([0.2, 1.1])),
    T(dist.Dirichlet, np.array([[0.2, 1.1], [2.0, 2.0]])),
    T(
        dist.EulerMaruyama,
        np.array([0.0, 0.1, 0.2]),
        sde_fn1,
        dist.Normal(0.1, 1.0),
    ),
    T(
        dist.EulerMaruyama,
        np.array([0.0, 0.1, 0.2]),
        sde_fn2,
        dist.Normal(jnp.array([0.0, 1.0]), 1e-3).to_event(1),
    ),
    T(
        dist.EulerMaruyama,
        np.array([[0.0, 0.1, 0.2], [10.0, 10.1, 10.2]]),
        sde_fn2,
        dist.Normal(jnp.array([0.0, 1.0]), 1e-3).to_event(1),
    ),
    T(
        dist.EulerMaruyama,
        np.array([[0.0, 0.1, 0.2], [10.0, 10.1, 10.2]]),
        sde_fn2,
        dist.Normal(jnp.array([[0.0, 1.0], [2.0, 3.0]]), 1e-2).to_event(1),
    ),
    T(dist.Exponential, 2.0),
    T(dist.Exponential, np.array([4.0, 2.0])),
    T(dist.Gamma, np.array([1.7]), np.array([[2.0], [3.0]])),
    T(dist.Gamma, np.array([0.5, 1.3]), np.array([[1.0], [3.0]])),
    T(dist.GaussianRandomWalk, 0.1, 10),
    T(dist.GaussianRandomWalk, np.array([0.1, 0.3, 0.25]), 10),
    T(
        dist.GaussianCopulaBeta,
        np.array([7.0, 2.0]),
        np.array([4.0, 10.0]),
        np.array([[1.0, 0.75], [0.75, 1.0]]),
    ),
    T(dist.GaussianCopulaBeta, 2.0, 1.5, np.eye(3)),
    T(dist.GaussianCopulaBeta, 2.0, 1.5, np.full((5, 3, 3), np.eye(3))),
    T(dist.Gompertz, np.array([1.7]), np.array([[2.0], [3.0]])),
    T(dist.Gompertz, np.array([0.5, 1.3]), np.array([[1.0], [3.0]])),
    T(dist.Gumbel, 0.0, 1.0),
    T(dist.Gumbel, 0.5, 2.0),
    T(dist.Gumbel, np.array([0.0, 0.5]), np.array([1.0, 2.0])),
    T(FoldedNormal, 2.0, 4.0),
    T(FoldedNormal, np.array([2.0, 50.0]), np.array([4.0, 100.0])),
    T(dist.HalfCauchy, 1.0),
    T(dist.HalfCauchy, np.array([1.0, 2.0])),
    T(dist.HalfNormal, 1.0),
    T(dist.HalfNormal, np.array([1.0, 2.0])),
    T(_ImproperWrapper, constraints.positive, (), (3,)),
    T(dist.InverseGamma, np.array([1.7]), np.array([[2.0], [3.0]])),
    T(dist.InverseGamma, np.array([0.5, 1.3]), np.array([[1.0], [3.0]])),
    T(dist.Kumaraswamy, 10.0, np.array([2.0, 3.0])),
    T(dist.Kumaraswamy, np.array([1.7]), np.array([[2.0], [3.0]])),
    T(dist.Kumaraswamy, 0.6, 0.5),
    T(dist.Laplace, 0.0, 1.0),
    T(dist.Laplace, 0.5, np.array([1.0, 2.5])),
    T(dist.Laplace, np.array([1.0, -0.5]), np.array([2.3, 3.0])),
    T(dist.LKJ, 2, 0.5, "onion"),
    T(dist.LKJ, 5, np.array([0.5, 1.0, 2.0]), "cvine"),
    T(dist.LKJCholesky, 2, 0.5, "onion"),
    T(dist.LKJCholesky, 2, 0.5, "cvine"),
    T(dist.LKJCholesky, 5, np.array([0.5, 1.0, 2.0]), "onion"),
    pytest.param(
        *T(dist.LKJCholesky, 5, np.array([0.5, 1.0, 2.0]), "cvine"),
        marks=pytest.mark.skipif("CI" in os.environ, reason="reduce time for CI"),
    ),
    pytest.param(
        *T(dist.LKJCholesky, 3, np.array([[3.0, 0.6], [0.2, 5.0]]), "onion"),
        marks=pytest.mark.skipif("CI" in os.environ, reason="reduce time for CI"),
    ),
    T(dist.LKJCholesky, 3, np.array([[3.0, 0.6], [0.2, 5.0]]), "cvine"),
    T(dist.Logistic, 0.0, 1.0),
    T(dist.Logistic, 1.0, np.array([1.0, 2.0])),
    T(dist.Logistic, np.array([0.0, 1.0]), np.array([[1.0], [2.0]])),
    T(dist.LogNormal, 1.0, 0.2),
    T(dist.LogNormal, -1.0, np.array([0.5, 1.3])),
    T(dist.LogNormal, np.array([0.5, -0.7]), np.array([[0.1, 0.4], [0.5, 0.1]])),
    T(dist.LogUniform, 1.0, 2.0),
    T(dist.LogUniform, 1.0, np.array([2.0, 3.0])),
    T(dist.LogUniform, np.array([1.0, 2.0]), np.array([[3.0], [4.0]])),
    T(
        dist.MatrixNormal,
        1.0 * np.arange(6).reshape(3, 2),
        np.array([[1.0, 0, 0], [0.3, 0.36, 0], [0.4, 0.49, 4]]),
        np.array([[1.0, 0], [0.4, 1]]),
    ),
    T(
        dist.MatrixNormal,
        1.0 * np.arange(12).reshape((2, 3, 2)),
        np.array([[1.0, 0, 0], [0.3, 0.36, 0], [0.4, 0.49, 4]]) * np.ones((2, 3, 3)),
        np.array([[1.0, 0], [0.4, 0.5]]) * np.ones((2, 2, 2)),
    ),
    T(
        dist.MatrixNormal,
        1.0 * np.arange(36).reshape((2, 3, 3, 2)),
        np.identity(3),
        np.identity(2),
    ),
    T(dist.MultivariateNormal, 0.0, np.array([[1.0, 0.5], [0.5, 1.0]]), None, None),
    T(
        dist.MultivariateNormal,
        np.array([1.0, 3.0]),
        None,
        np.array([[1.0, 0.5], [0.5, 1.0]]),
        None,
    ),
    T(
        dist.MultivariateNormal,
        np.array([1.0, 3.0]),
        None,
        np.array([[[1.0, 0.5], [0.5, 1.0]]]),
        None,
    ),
    T(
        dist.MultivariateNormal,
        np.array([2.0]),
        None,
        None,
        np.array([[1.0, 0.0], [0.5, 1.0]]),
    ),
    T(
        dist.MultivariateNormal,
        np.arange(6, dtype=np.float32).reshape((3, 2)),
        None,
        None,
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    ),
    T(
        dist.MultivariateNormal,
        0.0,
        None,
        np.broadcast_to(np.identity(3), (2, 3, 3)),
        None,
    ),
    T(
        dist.CAR,
        1.2,
        np.array([-0.2, 0.3]),
        0.1,
        np.array(
            [
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
            ]
        ),
    ),
    T(
        dist.CAR,
        np.array([0.0, 1.0, 3.0, 4.0]),
        0.1,
        np.array([0.3, 0.7]),
        np.array(
            [
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
            ]
        ),
    ),
    T(
        _SparseCAR,
        np.array([[0.0, 1.0, 3.0, 4.0], [2.0, -1.0, -3.0, 2.0]]),
        0.0,
        0.1,
        np.array(
            [
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
            ]
        ),
    ),
    T(
        dist.MultivariateStudentT,
        15.0,
        0.0,
        np.array([[1.0, 0.0], [0.5, 1.0]]),
    ),
    T(
        dist.MultivariateStudentT,
        15.0,
        np.array([1.0, 3.0]),
        np.array([[1.0, 0.0], [0.5, 1.0]]),
    ),
    T(
        dist.MultivariateStudentT,
        15.0,
        np.array([1.0, 3.0]),
        np.array([[[1.0, 0.0], [0.5, 1.0]]]),
    ),
    T(
        dist.MultivariateStudentT,
        15.0,
        np.array([3.0]),
        np.array([[1.0, 0.0], [0.5, 1.0]]),
    ),
    T(
        dist.MultivariateStudentT,
        15.0,
        np.arange(6, dtype=np.float32).reshape((3, 2)),
        np.array([[1.0, 0.0], [0.5, 1.0]]),
    ),
    T(
        dist.MultivariateStudentT,
        15.0,
        np.ones(3),
        np.broadcast_to(np.identity(3), (2, 3, 3)),
    ),
    T(
        dist.MultivariateStudentT,
        np.array(7.0),
        np.array([1.0, 3.0]),
        np.array([[1.0, 0.0], [0.5, 1.0]]),
    ),
    T(
        dist.MultivariateStudentT,
        np.arange(20, 22, dtype=jnp.float32),
        np.ones(3),
        np.broadcast_to(jnp.identity(3), (2, 3, 3)),
    ),
    T(
        dist.MultivariateStudentT,
        np.arange(20, 26, dtype=jnp.float32).reshape((3, 2)),
        np.ones(2),
        np.array([[1.0, 0.0], [0.5, 1.0]]),
    ),
    T(
        dist.LowRankMultivariateNormal,
        np.zeros(2),
        np.array([[1.0], [0.0]]),
        np.array([1.0, 1.0]),
    ),
    T(
        dist.LowRankMultivariateNormal,
        np.arange(6, dtype=jnp.float32).reshape((2, 3)),
        np.arange(6, dtype=jnp.float32).reshape((3, 2)),
        np.array([1.0, 2.0, 3.0]),
    ),
    T(dist.Normal, 0.0, 1.0),
    T(dist.Normal, 1.0, np.array([1.0, 2.0])),
    T(dist.Normal, np.array([0.0, 1.0]), np.array([[1.0], [2.0]])),
    T(dist.Pareto, 1.0, 2.0),
    T(dist.Pareto, np.array([1.0, 0.5]), np.array([0.3, 2.0])),
    T(dist.Pareto, np.array([[1.0], [3.0]]), np.array([1.0, 0.5])),
    T(dist.RelaxedBernoulliLogits, 2.0, -10.0),
    T(dist.RelaxedBernoulliLogits, np.array([1.0, 3.0]), np.array([3.0, 8.0])),
    T(dist.SoftLaplace, 1.0, 1.0),
    T(dist.SoftLaplace, np.array([-1.0, 50.0]), np.array([4.0, 100.0])),
    T(dist.StudentT, 1.0, 1.0, 0.5),
    T(dist.StudentT, 2.0, np.array([1.0, 2.0]), 2.0),
    T(dist.StudentT, np.array([3.0, 5.0]), np.array([[1.0], [2.0]]), 2.0),
    T(_TruncatedCauchy, 0.0, 1.0, -1.0, None),
    T(_TruncatedCauchy, 0.0, np.array([1.0, 2.0]), 1.0, None),
    T(
        _TruncatedCauchy,
        np.array([0.0, 1.0]),
        np.array([[1.0], [2.0]]),
        np.array([-2.0, 2.0]),
        None,
    ),
    T(_TruncatedCauchy, 0.0, 1.0, None, 1.0),
    T(_TruncatedCauchy, 0.0, 1.0, -1.0, 1.0),
    T(_TruncatedNormal, 0.0, 1.0, -1.0, None),
    T(_TruncatedNormal, -1.0, np.array([1.0, 2.0]), 1.0, None),
    T(
        _TruncatedNormal,
        np.array([0.0, 1.0]),
        np.array([[1.0], [2.0]]),
        np.array([-2.0, 2.0]),
        None,
    ),
    T(_TruncatedNormal, -1.0, 2.0, 1.0, 5.0),
    T(_TruncatedNormal, np.array([-1.0, 4.0]), 2.0, None, 5.0),
    T(_TruncatedNormal, -1.0, np.array([2.0, 3.0]), 1.0, None),
    T(_TruncatedNormal, -1.0, 2.0, np.array([-6.0, 4.0]), np.array([-4.0, 6.0])),
    T(
        _TruncatedNormal,
        np.array([0.0, 1.0]),
        np.array([[1.0], [2.0]]),
        None,
        np.array([-2.0, 2.0]),
    ),
    T(dist.TwoSidedTruncatedDistribution, dist.Laplace(0.0, 1.0), -2.0, 3.0),
    T(dist.Uniform, 0.0, 2.0),
    T(dist.Uniform, 1.0, np.array([2.0, 3.0])),
    T(dist.Uniform, np.array([0.0, 0.0]), np.array([[2.0], [3.0]])),
    T(dist.Weibull, 0.2, 1.1),
    T(dist.Weibull, 2.8, np.array([2.0, 2.0])),
    T(dist.Weibull, 1.8, np.array([[1.0, 1.0], [2.0, 2.0]])),
    T(dist.Wishart, 3, 2 * np.eye(2) + 0.1, None, None),
    T(
        dist.Wishart,
        3.0,
        None,
        np.array([[1.0, 0.5], [0.5, 1.0]]),
        None,
    ),
    T(
        dist.Wishart,
        np.array([4.0, 5.0]),
        None,
        np.array([[[1.0, 0.5], [0.5, 1.0]]]),
        None,
    ),
    T(
        dist.Wishart,
        np.array([3.0]),
        None,
        None,
        np.array([[1.0, 0.0], [0.5, 1.0]]),
    ),
    T(
        dist.Wishart,
        np.arange(3, 9, dtype=np.float32).reshape((3, 2)),
        None,
        None,
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    ),
    T(
        dist.Wishart,
        9.0,
        None,
        np.broadcast_to(np.identity(3), (2, 3, 3)),
        None,
    ),
    T(dist.WishartCholesky, 3, 2 * np.eye(2) + 0.1, None, None),
    T(
        dist.WishartCholesky,
        3.0,
        None,
        np.array([[1.0, 0.5], [0.5, 1.0]]),
        None,
    ),
    T(
        dist.WishartCholesky,
        np.array([4.0, 5.0]),
        None,
        np.array([[[1.0, 0.5], [0.5, 1.0]]]),
        None,
    ),
    T(
        dist.WishartCholesky,
        np.array([3.0]),
        None,
        None,
        np.array([[1.0, 0.0], [0.5, 1.0]]),
    ),
    T(
        dist.WishartCholesky,
        np.arange(3, 9, dtype=np.float32).reshape((3, 2)),
        None,
        None,
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    ),
    T(
        dist.WishartCholesky,
        9.0,
        None,
        np.broadcast_to(np.identity(3), (2, 3, 3)),
        None,
    ),
    T(dist.ZeroSumNormal, 1.0, (5,)),
    T(dist.ZeroSumNormal, np.array([2.0]), (5,)),
    T(dist.ZeroSumNormal, 1.0, (4, 5)),
    T(
        _GaussianMixture,
        np.ones(3) / 3.0,
        np.array([0.0, 7.7, 2.1]),
        np.array([4.2, 7.7, 2.1]),
    ),
    T(
        _Gaussian2DMixture,
        np.array([0.2, 0.5, 0.3]),
        np.array([[-1.2, 1.5], [2.0, 2.0], [-1, 4.0]]),  # Mean
        np.array(
            [
                [
                    [0.1, -0.2],
                    [-0.2, 1.0],
                ],
                [
                    [0.75, 0.0],
                    [0.0, 0.75],
                ],
                [
                    [1.0, 0.5],
                    [0.5, 0.27],
                ],
            ]
        ),  # Covariance
    ),
    T(
        _GeneralMixture,
        np.array([0.2, 0.3, 0.5]),
        np.array([0.0, 7.7, 2.1]),
        np.array([4.2, 1.7, 2.1]),
    ),
    T(
        _General2DMixture,
        np.array([0.2, 0.5, 0.3]),
        np.array([[-1.2, 1.5], [2.0, 2.0], [-1, 4.0]]),  # Mean
        np.array(
            [
                [
                    [0.1, -0.2],
                    [-0.2, 1.0],
                ],
                [
                    [0.75, 0.0],
                    [0.0, 0.75],
                ],
                [
                    [1.0, 0.5],
                    [0.5, 0.27],
                ],
            ]
        ),  # Covariance
    ),
]

DIRECTIONAL = [
    T(dist.VonMises, 2.0, 10.0),
    T(dist.VonMises, 2.0, np.array([150.0, 10.0])),
    T(dist.VonMises, np.array([1 / 3 * np.pi, -1.0]), np.array([20.0, 30.0])),
    pytest.param(
        *T(
            dist.SineBivariateVonMises,
            0.0,
            0.0,
            5.0,
            6.0,
            2.0,
        ),
        marks=pytest.mark.skipif("CI" in os.environ, reason="reduce time for CI"),
    ),
    T(
        dist.SineBivariateVonMises,
        3.003,
        -1.343,
        5.0,
        6.0,
        2.0,
    ),
    pytest.param(
        *T(
            dist.SineBivariateVonMises,
            -1.232,
            -1.3430,
            3.4,
            2.0,
            1.0,
        ),
        marks=pytest.mark.skipif("CI" in os.environ, reason="reduce time for CI"),
    ),
    pytest.param(
        *T(
            dist.SineBivariateVonMises,
            np.array([math.pi - 0.2, 1.0]),
            np.array([0.0, 1.0]),
            np.array([5.0, 5.0]),
            np.array([7.0, 0.5]),
            None,
            np.array([0.5, 0.1]),
        ),
        marks=pytest.mark.skipif("CI" in os.environ, reason="reduce time for CI"),
    ),
    T(dist.ProjectedNormal, np.array([0.0, 0.0])),
    T(dist.ProjectedNormal, np.array([[2.0, 3.0]])),
    T(dist.ProjectedNormal, np.array([0.0, 0.0, 0.0])),
    T(dist.ProjectedNormal, np.array([[-1.0, 2.0, 3.0]])),
    T(SineSkewedUniform, np.array([-math.pi / 4, 0.1])),
    T(SineSkewedVonMises, np.array([0.342355])),
    T(SineSkewedVonMisesBatched, np.array([[0.342355, -0.0001], [0.91, 0.09]])),
]

DISCRETE = [
    T(dist.BetaBinomial, 2.0, 5.0, 10),
    T(
        dist.BetaBinomial,
        np.array([2.0, 4.0]),
        np.array([5.0, 3.0]),
        np.array([10, 12]),
    ),
    T(dist.BernoulliProbs, 0.2),
    T(dist.BernoulliProbs, np.array([0.2, 0.7])),
    T(dist.BernoulliLogits, np.array([-1.0, 3.0])),
    T(dist.BinomialProbs, np.array([0.2, 0.7]), np.array([10, 2])),
    T(dist.BinomialProbs, np.array([0.2, 0.7]), np.array([5, 8])),
    T(dist.BinomialLogits, np.array([-1.0, 3.0]), np.array([5, 8])),
    T(dist.CategoricalProbs, np.array([1.0])),
    T(dist.CategoricalProbs, np.array([0.1, 0.5, 0.4])),
    T(dist.CategoricalProbs, np.array([[0.1, 0.5, 0.4], [0.4, 0.4, 0.2]])),
    T(dist.CategoricalLogits, np.array([-5.0])),
    T(dist.CategoricalLogits, np.array([1.0, 2.0, -2.0])),
    T(dist.CategoricalLogits, np.array([[-1, 2.0, 3.0], [3.0, -4.0, -2.0]])),
    T(dist.Delta, 1),
    T(dist.Delta, np.array([0.0, 2.0])),
    T(dist.Delta, np.array([0.0, 2.0]), np.array([-2.0, -4.0])),
    T(dist.DirichletMultinomial, np.array([1.0, 2.0, 3.9]), 10),
    T(dist.DirichletMultinomial, np.array([0.2, 0.7, 1.1]), np.array([5, 5])),
    T(dist.GammaPoisson, 2.0, 2.0),
    T(dist.GammaPoisson, np.array([6.0, 2]), np.array([2.0, 8.0])),
    T(dist.GeometricProbs, 0.2),
    T(dist.GeometricProbs, np.array([0.2, 0.7])),
    T(dist.GeometricLogits, np.array([-1.0, 3.0])),
    T(dist.MultinomialProbs, np.array([0.2, 0.7, 0.1]), 10),
    T(dist.MultinomialProbs, np.array([0.2, 0.7, 0.1]), np.array([5, 8])),
    T(dist.MultinomialLogits, np.array([-1.0, 3.0]), np.array([[5], [8]])),
    T(dist.NegativeBinomialProbs, 10, 0.2),
    T(dist.NegativeBinomialProbs, 10, np.array([0.2, 0.6])),
    T(dist.NegativeBinomialProbs, np.array([4.2, 10.7, 2.1]), 0.2),
    T(
        dist.NegativeBinomialProbs,
        np.array([4.2, 10.7, 2.1]),
        np.array([0.2, 0.6, 0.5]),
    ),
    T(dist.NegativeBinomialLogits, 10, -2.1),
    T(dist.NegativeBinomialLogits, 10, np.array([-5.2, 2.1])),
    T(dist.NegativeBinomialLogits, np.array([4.2, 10.7, 2.1]), -5.2),
    T(
        dist.NegativeBinomialLogits,
        np.array([4.2, 7.7, 2.1]),
        np.array([4.2, 0.7, 2.1]),
    ),
    T(dist.NegativeBinomial2, 0.3, 10),
    T(dist.NegativeBinomial2, np.array([10.2, 7, 31]), 10),
    T(dist.NegativeBinomial2, np.array([10.2, 7, 31]), np.array([10.2, 20.7, 2.1])),
    T(dist.OrderedLogistic, -2, np.array([-10.0, 4.0, 9.0])),
    T(dist.OrderedLogistic, np.array([-4, 3, 4, 5]), np.array([-1.5])),
    T(dist.DiscreteUniform, -2, np.array([-1.0, 4.0, 9.0])),
    T(dist.DiscreteUniform, np.array([-4, 3, 4, 5]), np.array([6])),
    T(dist.Poisson, 2.0),
    T(dist.Poisson, np.array([2.0, 3.0, 5.0])),
    T(SparsePoisson, 2.0),
    T(SparsePoisson, np.array([2.0, 3.0, 5.0])),
    T(SparsePoisson, 2),
    T(dist.ZeroInflatedPoisson, 0.6, 2.0),
    T(dist.ZeroInflatedPoisson, np.array([0.2, 0.7, 0.3]), np.array([2.0, 3.0, 5.0])),
    T(ZeroInflatedPoissonLogits, 2.0, 3.0),
    T(
        ZeroInflatedPoissonLogits,
        np.array([0.2, 4.0, 0.3]),
        np.array([2.0, -3.0, 5.0]),
    ),
]

BASE = [
    T(lambda *args: dist.Normal(*args).to_event(2), np.arange(24).reshape(3, 4, 2)),
    T(lambda *args: dist.Normal(*args).expand((3, 4, 7)), np.arange(7)),
    T(
        lambda *args: dist.Normal(*args).to_event(2).expand((7, 3)),
        np.arange(24).reshape(3, 4, 2),
    ),
]


def _is_batched_multivariate(jax_dist):
    return len(jax_dist.event_shape) > 0 and len(jax_dist.batch_shape) > 0


def gen_values_within_bounds(constraint, size, key=random.PRNGKey(11)):
    eps = 1e-6

    if constraint is constraints.boolean:
        return random.bernoulli(key, shape=size)
    elif isinstance(constraint, constraints.greater_than):
        return jnp.exp(random.normal(key, size)) + constraint.lower_bound + eps
    elif isinstance(constraint, constraints.integer_interval):
        lower_bound = jnp.broadcast_to(constraint.lower_bound, size)
        upper_bound = jnp.broadcast_to(constraint.upper_bound, size)
        return random.randint(key, size, lower_bound, upper_bound + 1)
    elif isinstance(constraint, constraints.integer_greater_than):
        return constraint.lower_bound + random.poisson(key, np.array(5), shape=size)
    elif isinstance(constraint, constraints.interval):
        lower_bound = jnp.broadcast_to(constraint.lower_bound, size)
        upper_bound = jnp.broadcast_to(constraint.upper_bound, size)
        return random.uniform(key, size, minval=lower_bound, maxval=upper_bound)
    elif constraint in (constraints.real, constraints.real_vector):
        return random.normal(key, size)
    elif constraint is constraints.simplex:
        return osp.dirichlet.rvs(alpha=jnp.ones((size[-1],)), size=size[:-1])
    elif isinstance(constraint, constraints.multinomial):
        n = size[-1]
        return multinomial(
            key, p=jnp.ones((n,)) / n, n=constraint.upper_bound, shape=size[:-1]
        )
    elif constraint is constraints.corr_cholesky:
        return signed_stick_breaking_tril(
            random.uniform(
                key, size[:-2] + (size[-1] * (size[-1] - 1) // 2,), minval=-1, maxval=1
            )
        )
    elif constraint is constraints.corr_matrix:
        cholesky = signed_stick_breaking_tril(
            random.uniform(
                key, size[:-2] + (size[-1] * (size[-1] - 1) // 2,), minval=-1, maxval=1
            )
        )
        return jnp.matmul(cholesky, jnp.swapaxes(cholesky, -2, -1))
    elif constraint is constraints.lower_cholesky:
        return jnp.tril(random.uniform(key, size))
    elif constraint is constraints.positive_definite:
        x = random.normal(key, size)
        return jnp.matmul(x, jnp.swapaxes(x, -2, -1))
    elif constraint is constraints.ordered_vector:
        x = jnp.cumsum(random.exponential(key, size), -1)
        return x - random.normal(key, size[:-1] + (1,))
    elif isinstance(constraint, constraints.independent):
        return gen_values_within_bounds(constraint.base_constraint, size, key)
    elif constraint is constraints.sphere:
        x = random.normal(key, size)
        return x / jnp.linalg.norm(x, axis=-1)
    elif constraint is constraints.l1_ball:
        key1, key2 = random.split(key)
        sign = random.bernoulli(key1)
        bounds = [0, (-1) ** sign * 0.5]
        return random.uniform(key, size, float, *sorted(bounds))
    elif isinstance(constraint, constraints.zero_sum):
        x = random.normal(key, size)
        zero_sum_axes = tuple(i for i in range(-constraint.event_dim, 0))
        for axis in zero_sum_axes:
            x -= x.mean(axis)
        return x

    else:
        raise NotImplementedError("{} not implemented.".format(constraint))


def gen_values_outside_bounds(constraint, size, key=random.PRNGKey(11)):
    if constraint is constraints.boolean:
        return random.bernoulli(key, shape=size) - 2
    elif isinstance(constraint, constraints.greater_than):
        return constraint.lower_bound - jnp.exp(random.normal(key, size))
    elif isinstance(constraint, constraints.integer_interval):
        lower_bound = jnp.broadcast_to(constraint.lower_bound, size)
        return random.randint(key, size, lower_bound - 1, lower_bound)
    elif isinstance(constraint, constraints.integer_greater_than):
        return constraint.lower_bound - random.poisson(key, np.array(5), shape=size)
    elif isinstance(constraint, constraints.interval):
        upper_bound = jnp.broadcast_to(constraint.upper_bound, size)
        return random.uniform(key, size, minval=upper_bound, maxval=upper_bound + 1.0)
    elif constraint in [constraints.real, constraints.real_vector]:
        return lax.full(size, np.nan)
    elif constraint is constraints.simplex:
        return osp.dirichlet.rvs(alpha=jnp.ones((size[-1],)), size=size[:-1]) + 1e-2
    elif isinstance(constraint, constraints.multinomial):
        n = size[-1]
        return (
            multinomial(
                key, p=jnp.ones((n,)) / n, n=constraint.upper_bound, shape=size[:-1]
            )
            + 1
        )
    elif constraint is constraints.corr_cholesky:
        return (
            signed_stick_breaking_tril(
                random.uniform(
                    key,
                    size[:-2] + (size[-1] * (size[-1] - 1) // 2,),
                    minval=-1,
                    maxval=1,
                )
            )
            + 1e-2
        )
    elif constraint is constraints.corr_matrix:
        cholesky = 1e-2 + signed_stick_breaking_tril(
            random.uniform(
                key, size[:-2] + (size[-1] * (size[-1] - 1) // 2,), minval=-1, maxval=1
            )
        )
        return jnp.matmul(cholesky, jnp.swapaxes(cholesky, -2, -1))
    elif constraint is constraints.lower_cholesky:
        return random.uniform(key, size)
    elif constraint is constraints.positive_definite:
        return random.normal(key, size)
    elif constraint is constraints.ordered_vector:
        x = jnp.cumsum(random.exponential(key, size), -1)
        return x[..., ::-1]
    elif isinstance(constraint, constraints.independent):
        return gen_values_outside_bounds(constraint.base_constraint, size, key)
    elif constraint is constraints.sphere:
        x = random.normal(key, size)
        x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
        return 2 * x
    elif constraint is constraints.l1_ball:
        key1, key2 = random.split(key)
        sign = random.bernoulli(key1)
        bounds = [(-1) ** sign * 1.1, (-1) ** sign * 2]
        return random.uniform(key, size, float, *sorted(bounds))
    elif isinstance(constraint, constraints.zero_sum):
        x = random.normal(key, size)
        return x
    else:
        raise NotImplementedError("{} not implemented.".format(constraint))


@pytest.mark.parametrize(
    "jax_dist_cls, sp_dist, params", CONTINUOUS + DISCRETE + DIRECTIONAL
)
@pytest.mark.parametrize("prepend_shape", [(), (2,), (2, 3)])
def test_dist_shape(jax_dist_cls, sp_dist, params, prepend_shape):
    jax_dist = jax_dist_cls(*params)
    rng_key = random.PRNGKey(0)
    expected_shape = prepend_shape + jax_dist.batch_shape + jax_dist.event_shape
    samples = jax_dist.sample(key=rng_key, sample_shape=prepend_shape)
    if jax_dist_cls is not dist.Delta:
        assert isinstance(samples, jnp.ndarray)
    assert jnp.shape(samples) == expected_shape
    if (
        sp_dist
        and not _is_batched_multivariate(jax_dist)
        and not isinstance(jax_dist, dist.MultivariateStudentT)
    ):
        sp_dist = sp_dist(*params)
        size = prepend_shape + jax_dist.batch_shape
        # The scipy implementation of the Wishart distribution cannot handle an empty
        # tuple as the sample size so we replace it by `1` which generates a single
        # sample without any sample shape.
        if isinstance(jax_dist, dist.Wishart):
            size = size or 1
        sp_samples = sp_dist.rvs(size=size)
        assert jnp.shape(sp_samples) == expected_shape
    elif (
        sp_dist
        and not _is_batched_multivariate(jax_dist)
        and isinstance(jax_dist, dist.MultivariateStudentT)
    ):
        sp_dist = sp_dist(*params)
        size_ = prepend_shape + jax_dist.batch_shape
        size = (1) if size_ == () else size_
        try:
            sp_samples = sp_dist.rvs(size=size)
        except ValueError:
            pytest.skip("scipy multivariate t doesn't support size with > 1 element")
        assert jnp.shape(sp_samples) == expected_shape
    if isinstance(jax_dist, (dist.MultivariateNormal, dist.MultivariateStudentT)):
        assert jax_dist.covariance_matrix.ndim == len(jax_dist.batch_shape) + 2
        assert_allclose(
            jax_dist.precision_matrix,
            jnp.linalg.inv(jax_dist.covariance_matrix),
            rtol=1e-6,
        )


@pytest.mark.parametrize(
    "jax_dist, sp_dist, params", CONTINUOUS + DISCRETE + DIRECTIONAL
)
def test_infer_shapes(jax_dist, sp_dist, params):
    shapes = []
    for param in params:
        if param is None:
            shapes.append(None)
            continue
        shape = getattr(param, "shape", ())
        if callable(shape):
            shape = shape()
        shapes.append(shape)
    jax_dist = jax_dist(*params)
    try:
        expected_batch_shape, expected_event_shape = type(jax_dist).infer_shapes(
            *shapes
        )
    except NotImplementedError:
        pytest.skip(f"{type(jax_dist).__name__}.infer_shapes() is not implemented")
    assert jax_dist.batch_shape == expected_batch_shape
    assert jax_dist.event_shape == expected_event_shape


@pytest.mark.parametrize(
    "jax_dist, sp_dist, params", CONTINUOUS + DISCRETE + DIRECTIONAL
)
def test_has_rsample(jax_dist, sp_dist, params):
    jax_dist = jax_dist(*params)
    masked_dist = jax_dist.mask(False)
    indept_dist = jax_dist.expand_by([2]).to_event(1)
    transf_dist = dist.TransformedDistribution(jax_dist, biject_to(constraints.real))
    assert masked_dist.has_rsample == jax_dist.has_rsample
    assert indept_dist.has_rsample == jax_dist.has_rsample
    assert transf_dist.has_rsample == jax_dist.has_rsample

    if jax_dist.has_rsample:
        assert isinstance(jax_dist, dist.Delta) or not jax_dist.is_discrete
        if isinstance(jax_dist, dist.TransformedDistribution):
            assert jax_dist.base_dist.has_rsample
        else:
            assert set(jax_dist.arg_constraints) == set(jax_dist.reparametrized_params)
        jax_dist.rsample(random.PRNGKey(0))
        if isinstance(jax_dist, dist.Normal):
            masked_dist.rsample(random.PRNGKey(0))
            indept_dist.rsample(random.PRNGKey(0))
            transf_dist.rsample(random.PRNGKey(0))
    else:
        with pytest.raises(NotImplementedError):
            jax_dist.rsample(random.PRNGKey(0))
        if isinstance(jax_dist, dist.BernoulliProbs):
            with pytest.raises(NotImplementedError):
                masked_dist.rsample(random.PRNGKey(0))
            with pytest.raises(NotImplementedError):
                indept_dist.rsample(random.PRNGKey(0))
            with pytest.raises(NotImplementedError):
                transf_dist.rsample(random.PRNGKey(0))


@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)])
def test_unit(batch_shape):
    log_factor = random.normal(random.PRNGKey(0), batch_shape)
    d = dist.Unit(log_factor=log_factor)
    x = d.sample(random.PRNGKey(1))
    assert x.shape == batch_shape + (0,)
    assert (d.log_prob(x) == log_factor).all()


@pytest.mark.parametrize("jax_dist, sp_dist, params", CONTINUOUS)
def test_sample_gradient(jax_dist, sp_dist, params):
    # we have pathwise gradient for gamma sampler
    gamma_derived_params = {
        "Gamma": ["concentration"],
        "Beta": ["concentration1", "concentration0"],
        "BetaProportion": ["mean", "concentration"],
        "Chi2": ["df"],
        "Dirichlet": ["concentration"],
        "InverseGamma": ["concentration"],
        "LKJ": ["concentration"],
        "LKJCholesky": ["concentration"],
        "StudentT": ["df"],
    }.get(jax_dist.__name__, [])

    dist_args = [
        p
        for p in (
            inspect.getfullargspec(jax_dist.__init__)[0][1:]
            if inspect.isclass(jax_dist)
            # account the the case jax_dist is a function
            else inspect.getfullargspec(jax_dist)[0]
        )
    ]
    params_dict = dict(zip(dist_args[: len(params)], params))

    jax_class = type(jax_dist(**params_dict))
    reparametrized_params = [
        p for p in jax_class.reparametrized_params if p not in gamma_derived_params
    ]
    if not reparametrized_params:
        pytest.skip("{} not reparametrized.".format(jax_class.__name__))

    nonrepara_params_dict = {
        k: v for k, v in params_dict.items() if k not in reparametrized_params
    }
    repara_params = tuple(
        v for k, v in params_dict.items() if k in reparametrized_params
    )

    rng_key = random.PRNGKey(0)

    def fn(args):
        args_dict = dict(zip(reparametrized_params, args))
        return jnp.sum(
            jax_dist(**args_dict, **nonrepara_params_dict).sample(key=rng_key)
        )

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
        expected_grad = (fn_rhs - fn_lhs) / (2.0 * eps)
        assert jnp.shape(actual_grad[i]) == jnp.shape(repara_params[i])
        assert_allclose(jnp.sum(actual_grad[i]), expected_grad, rtol=0.02, atol=0.03)


@pytest.mark.parametrize(
    "jax_dist, params",
    [
        (dist.Gamma, (1.0,)),
        (dist.Gamma, (0.1,)),
        (dist.Gamma, (10.0,)),
        (dist.Chi2, (1.0,)),
        (dist.Chi2, (0.1,)),
        (dist.Chi2, (10.0,)),
        (dist.Beta, (1.0, 1.0)),
        (dist.StudentT, (5.0, 2.0, 4.0)),
    ],
)
def test_pathwise_gradient(jax_dist, params):
    rng_key = random.PRNGKey(0)
    N = 1000000

    def f(params):
        z = jax_dist(*params).sample(key=rng_key, sample_shape=(N,))
        return (z + z**2).mean(0)

    def g(params):
        d = jax_dist(*params)
        return d.mean + d.variance + d.mean**2

    actual_grad = grad(f)(params)
    expected_grad = grad(g)(params)
    assert_allclose(actual_grad, expected_grad, rtol=0.005)


@pytest.mark.parametrize(
    "jax_dist, sp_dist, params", CONTINUOUS + DISCRETE + DIRECTIONAL
)
def test_jit_log_likelihood(jax_dist, sp_dist, params):
    if jax_dist.__name__ in (
        "EulerMaruyama",
        "GaussianRandomWalk",
        "_ImproperWrapper",
        "LKJ",
        "LKJCholesky",
        "_SparseCAR",
        "ZeroSumNormal",
    ):
        pytest.xfail(reason="non-jittable params")

    rng_key = random.PRNGKey(0)
    samples = jax_dist(*params).sample(key=rng_key, sample_shape=(2, 3))

    def log_likelihood(*params):
        return jax_dist(*params).log_prob(samples)

    expected = log_likelihood(*params)
    actual = jax.jit(log_likelihood)(*params)
    assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize(
    "jax_dist, sp_dist, params", CONTINUOUS + DISCRETE + DIRECTIONAL
)
@pytest.mark.parametrize("prepend_shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("jit", [False, True])
def test_log_prob(jax_dist, sp_dist, params, prepend_shape, jit):
    jit_fn = _identity if not jit else jax.jit
    jax_dist = jax_dist(*params)

    rng_key = random.PRNGKey(0)
    samples = jax_dist.sample(key=rng_key, sample_shape=prepend_shape)
    assert jax_dist.log_prob(samples).shape == prepend_shape + jax_dist.batch_shape
    truncated_dists = (
        dist.LeftTruncatedDistribution,
        dist.RightTruncatedDistribution,
        dist.TwoSidedTruncatedDistribution,
    )
    if sp_dist is None:
        if isinstance(jax_dist, truncated_dists):
            if isinstance(params[0], dist.Distribution):
                # new api
                loc, scale, low, high = (
                    params[0].loc,
                    params[0].scale,
                    params[1],
                    params[2],
                )
            else:
                # old api
                loc, scale, low, high = params
            if low is None:
                low = -np.inf
            if high is None:
                high = np.inf
            sp_dist = get_sp_dist(type(jax_dist.base_dist))(loc, scale)
            expected = sp_dist.logpdf(samples) - jnp.log(
                sp_dist.cdf(high) - sp_dist.cdf(low)
            )
            assert_allclose(jit_fn(jax_dist.log_prob)(samples), expected, atol=1e-5)
            return
        pytest.skip("no corresponding scipy distn.")
    if _is_batched_multivariate(jax_dist):
        pytest.skip("batching not allowed in multivariate distns.")
    if jax_dist.event_shape and prepend_shape:
        # >>> d = sp.dirichlet([1.1, 1.1])
        # >>> samples = d.rvs(size=(2,))
        # >>> d.logpdf(samples)
        # ValueError: The input vector 'x' must lie within the normal simplex ...
        pytest.skip("batched samples cannot be scored by multivariate distributions.")
    sp_dist = sp_dist(*params)
    try:
        expected = sp_dist.logpdf(samples)
    except AttributeError:
        expected = sp_dist.logpmf(samples)
    except ValueError as e:
        # precision issue: jnp.sum(x / jnp.sum(x)) = 0.99999994 != 1
        if "The input vector 'x' must lie within the normal simplex." in str(e):
            samples = jax.device_get(samples).astype("float64")
            samples = samples / samples.sum(axis=-1, keepdims=True)
            expected = sp_dist.logpdf(samples)
        else:
            raise e
    assert_allclose(jit_fn(jax_dist.log_prob)(samples), expected, atol=1e-5)


@pytest.mark.parametrize(
    "jax_dist, sp_dist, params", CONTINUOUS + DISCRETE + DIRECTIONAL
)
def test_entropy_scipy(jax_dist, sp_dist, params):
    jax_dist = jax_dist(*params)

    try:
        actual = jax_dist.entropy()
    except NotImplementedError:
        pytest.skip(reason=f"distribution {jax_dist} does not implement `entropy`")
    if _is_batched_multivariate(jax_dist):
        pytest.skip("batching not allowed in multivariate distns.")
    if sp_dist is None:
        pytest.skip(reason="no corresponding scipy distribution")

    sp_dist = sp_dist(*params)
    expected = sp_dist.entropy()
    assert_allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize(
    "jax_dist, sp_dist, params", CONTINUOUS + DISCRETE + DIRECTIONAL + BASE
)
def test_entropy_samples(jax_dist, sp_dist, params):
    jax_dist = jax_dist(*params)

    try:
        actual = jax_dist.entropy()
    except NotImplementedError:
        pytest.skip(reason=f"distribution {jax_dist} does not implement `entropy`")

    samples = jax_dist.sample(jax.random.key(8), (1000,))
    neg_log_probs = -jax_dist.log_prob(samples)
    mean = neg_log_probs.mean(axis=0)
    stderr = neg_log_probs.std(axis=0) / jnp.sqrt(neg_log_probs.shape[-1] - 1)
    z = (actual - mean) / stderr

    # Check the z-score is small or that all values are close. This happens, for
    # example, for uniform distributions with constant log prob and hence zero stderr.
    assert (jnp.abs(z) < 5).all() or jnp.allclose(actual, neg_log_probs, atol=1e-5)


def test_entropy_categorical():
    # There is no scipy mapping for categorical distributions, but the multinomial with
    # one trial has the same entropy--which we check here.
    logits = jax.random.normal(jax.random.key(9), (7,))
    probs = _to_probs_multinom(logits)
    sp_dist = osp.multinomial(1, probs)
    for jax_dist in [dist.CategoricalLogits(logits), dist.CategoricalProbs(probs)]:
        assert_allclose(jax_dist.entropy(), sp_dist.entropy())


def test_mixture_log_prob():
    gmm = dist.MixtureSameFamily(
        dist.Categorical(logits=np.zeros(2)), dist.Normal(0, 1).expand([2])
    )
    actual = gmm.log_prob(0.0)
    expected = dist.Normal(0, 1).log_prob(0.0)
    assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "jax_dist, sp_dist, params",
    # TODO: add more complete pattern for Discrete.cdf
    CONTINUOUS + [T(dist.Poisson, 2.0), T(dist.Poisson, np.array([2.0, 3.0, 5.0]))],
)
@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
def test_cdf_and_icdf(jax_dist, sp_dist, params):
    d = jax_dist(*params)
    if d.event_dim > 0:
        pytest.skip("skip testing cdf/icdf methods of multivariate distributions")
    samples = d.sample(key=random.PRNGKey(0), sample_shape=(100,))
    quantiles = random.uniform(random.PRNGKey(1), (100,) + d.shape())
    try:
        rtol = 2e-3 if jax_dist in (dist.Gamma, dist.StudentT) else 1e-5
        if d.shape() == () and not d.is_discrete:
            assert_allclose(
                jax.vmap(jax.grad(d.cdf))(samples),
                jnp.exp(d.log_prob(samples)),
                atol=1e-5,
                rtol=rtol,
            )
            assert_allclose(
                jax.vmap(jax.grad(d.icdf))(quantiles),
                jnp.exp(-d.log_prob(d.icdf(quantiles))),
                atol=1e-5,
                rtol=rtol,
            )
        assert_allclose(d.cdf(d.icdf(quantiles)), quantiles, atol=1e-5, rtol=1e-5)
        assert_allclose(d.icdf(d.cdf(samples)), samples, atol=1e-5, rtol=rtol)
    except NotImplementedError:
        pass

    # test against scipy
    if not sp_dist:
        pytest.skip("no corresponding scipy distn.")
    sp_dist = sp_dist(*params)
    try:
        actual_cdf = d.cdf(samples)
        expected_cdf = sp_dist.cdf(samples)
        assert_allclose(actual_cdf, expected_cdf, atol=1e-5, rtol=1e-5)
        actual_icdf = d.icdf(quantiles)
        expected_icdf = sp_dist.ppf(quantiles)
        assert_allclose(actual_icdf, expected_icdf, atol=1e-4, rtol=1e-4)
    except NotImplementedError:
        pass


@pytest.mark.parametrize("jax_dist, sp_dist, params", CONTINUOUS + DIRECTIONAL)
def test_gof(jax_dist, sp_dist, params):
    if "Improper" in jax_dist.__name__:
        pytest.skip("distribution has improper .log_prob()")
    if "LKJ" in jax_dist.__name__ or "Wishart" in jax_dist.__name__:
        pytest.xfail("incorrect submanifold scaling")
    if jax_dist is dist.EulerMaruyama:
        d = jax_dist(*params)
        if d.event_dim > 1:
            pytest.skip("EulerMaruyama skip test when event shape is non-trivial.")
    if jax_dist is dist.ZeroSumNormal:
        pytest.skip("skip gof test for ZeroSumNormal")

    num_samples = 10000
    if "BetaProportion" in jax_dist.__name__:
        num_samples = 20000
    rng_key = random.PRNGKey(0)
    d = jax_dist(*params)
    samples = d.sample(key=rng_key, sample_shape=(num_samples,))
    probs = np.exp(d.log_prob(samples))

    dim = None
    if jax_dist is dist.ProjectedNormal:
        dim = samples.shape[-1] - 1

    # Test each batch independently.
    probs = probs.reshape(num_samples, -1)
    samples = samples.reshape(probs.shape + d.event_shape)
    if "Dirichlet" in jax_dist.__name__:
        # The Dirichlet density is over all but one of the probs.
        samples = samples[..., :-1]
    for b in range(probs.shape[1]):
        try:
            gof = auto_goodness_of_fit(samples[:, b], probs[:, b], dim=dim)
        except InvalidTest:
            pytest.skip("expensive test")
        else:
            assert gof > TEST_FAILURE_RATE


@pytest.mark.parametrize("jax_dist, sp_dist, params", CONTINUOUS + DISCRETE)
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
    diag = jnp.sqrt(1 - jnp.sum(w**2, axis=-1))
    cholesky = w + jnp.expand_dims(diag, axis=-1) * jnp.identity(w.shape[-1])
    corr = jnp.matmul(cholesky, cholesky.T)
    return matrix_to_tril_vec(corr, diagonal=-1)


@pytest.mark.parametrize("dimension", [2, 3, 5])
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
            jax.jacobian(_tril_cholesky_to_tril_corr)(sample_tril)
        )[1]
        corr_log_prob.append(log_prob - cholesky_to_corr_jac)

    corr_log_prob = np.array(corr_log_prob)
    # test if they are constant
    assert_allclose(
        corr_log_prob,
        jnp.broadcast_to(corr_log_prob[0], corr_log_prob.shape),
        rtol=1e-6,
    )

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
        unconstrained, sample
    )
    signed_stick_breaking_logdet = corr_cholesky_logdet + inv_tanh_logdet

    actual_log_prob = d.log_prob(sample)
    expected_log_prob = beta_log_prob - affine_logdet - signed_stick_breaking_logdet
    assert_allclose(actual_log_prob, expected_log_prob, rtol=2e-5)

    assert_allclose(jax.jit(d.log_prob)(sample), d.log_prob(sample), atol=2e-6)


def test_zero_inflated_logits_probs_agree():
    concentration = np.exp(np.random.normal(1))
    rate = np.exp(np.random.normal(1))
    d = dist.GammaPoisson(concentration, rate)
    gate_logits = np.random.normal(0)
    gate_probs = expit(gate_logits)
    zi_logits = dist.ZeroInflatedDistribution(d, gate_logits=gate_logits)
    zi_probs = dist.ZeroInflatedDistribution(d, gate=gate_probs)
    sample = np.random.randint(
        0,
        20,
        (
            1000,
            100,
        ),
    )
    assert_allclose(zi_probs.log_prob(sample), zi_logits.log_prob(sample))


@pytest.mark.parametrize("rate", [0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 10.0])
def test_ZIP_log_prob(rate):
    # if gate is 0 ZIP is Poisson
    zip_ = dist.ZeroInflatedPoisson(0.0, rate)
    pois = dist.Poisson(rate)
    s = zip_.sample(random.PRNGKey(0), (20,))
    zip_prob = zip_.log_prob(s)
    pois_prob = pois.log_prob(s)
    assert_allclose(zip_prob, pois_prob, rtol=1e-6)

    # if gate is 1 ZIP is Delta(0)
    zip_ = dist.ZeroInflatedPoisson(1.0, rate)
    delta = dist.Delta(0.0)
    s = np.array([0.0, 1.0])
    zip_prob = zip_.log_prob(s)
    delta_prob = delta.log_prob(s)
    assert_allclose(zip_prob, delta_prob, rtol=1e-6)


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

    actual = dist.BetaBinomial(concentration1, concentration0, total_count).log_prob(
        value
    )
    assert_allclose(actual, expected, rtol=0.02)


@pytest.mark.parametrize("total_count", [1, 2, 3, 10])
@pytest.mark.parametrize("batch_shape", [(1,), (3, 1), (2, 3, 1)])
def test_dirichlet_multinomial_log_prob(total_count, batch_shape):
    event_shape = (3,)
    concentration = np.exp(np.random.normal(size=batch_shape + event_shape))
    # test on one-hots
    value = total_count * jnp.eye(event_shape[-1]).reshape(
        event_shape + (1,) * len(batch_shape) + event_shape
    )

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
    poisson_rate = np.random.gamma(
        gamma_conc, 1 / gamma_rate, size=(num_samples,) + shape
    )
    log_probs = dist.Poisson(poisson_rate).log_prob(value)
    expected = logsumexp(log_probs, 0) - jnp.log(num_samples)
    actual = dist.GammaPoisson(gamma_conc, gamma_rate).log_prob(value)
    assert_allclose(actual, expected, rtol=0.05)


@pytest.mark.parametrize(
    "jax_dist, sp_dist, params", CONTINUOUS + DISCRETE + DIRECTIONAL
)
def test_log_prob_gradient(jax_dist, sp_dist, params):
    if jax_dist in [dist.LKJ, dist.LKJCholesky]:
        pytest.skip("we have separated tests for LKJCholesky distribution")
    if jax_dist is _ImproperWrapper:
        pytest.skip("no param for ImproperUniform to test for log_prob gradient")

    rng_key = random.PRNGKey(0)
    value = jax_dist(*params).sample(rng_key)

    def fn(*args):
        return jnp.sum(jax_dist(*args).log_prob(value))

    eps = 1e-3
    for i in range(len(params)):
        if jax_dist is dist.EulerMaruyama and i == 1:
            # skip taking grad w.r.t. sde_fn
            continue
        if jax_dist is _SparseCAR and i == 3:
            # skip taking grad w.r.t. adj_matrix
            continue
        if jax_dist is dist.ZeroSumNormal and i != 0:
            # skip taking grad w.r.t. event_shape
            continue
        if isinstance(
            params[i], dist.Distribution
        ):  # skip taking grad w.r.t. base_dist
            continue
        if params[i] is None or jnp.result_type(params[i]) in (jnp.int32, jnp.int64):
            continue
        actual_grad = jax.grad(fn, i)(*params)
        args_lhs = [p if j != i else p - eps for j, p in enumerate(params)]
        args_rhs = [p if j != i else p + eps for j, p in enumerate(params)]
        fn_lhs = fn(*args_lhs)
        fn_rhs = fn(*args_rhs)
        # finite diff approximation
        expected_grad = (fn_rhs - fn_lhs) / (2.0 * eps)
        assert jnp.shape(actual_grad) == jnp.shape(params[i])
        if i == 0 and jax_dist is dist.Delta:
            # grad w.r.t. `value` of Delta distribution will be 0
            # but numerical value will give nan (= inf - inf)
            expected_grad = 0.0
        assert_allclose(jnp.sum(actual_grad), expected_grad, rtol=0.01, atol=0.01)


@pytest.mark.parametrize(
    "jax_dist, sp_dist, params", CONTINUOUS + DISCRETE + DIRECTIONAL
)
def test_mean_var(jax_dist, sp_dist, params):
    if jax_dist is _ImproperWrapper:
        pytest.skip("Improper distribution does not has mean/var implemented")
    if jax_dist is FoldedNormal:
        pytest.skip("Folded distribution does not has mean/var implemented")
    if jax_dist is dist.EulerMaruyama:
        pytest.skip("EulerMaruyama distribution does not has mean/var implemented")
    if jax_dist is dist.RelaxedBernoulliLogits:
        pytest.skip("RelaxedBernoulli distribution does not has mean/var implemented")
    if "SineSkewed" in jax_dist.__name__:
        pytest.skip("Skewed Distribution are not symmetric about location.")
    if jax_dist in (
        _TruncatedNormal,
        _TruncatedCauchy,
        dist.LeftTruncatedDistribution,
        dist.RightTruncatedDistribution,
        dist.TwoSidedTruncatedDistribution,
    ):
        pytest.skip("Truncated distributions do not has mean/var implemented")
    if jax_dist is dist.ProjectedNormal:
        pytest.skip("Mean is defined in submanifold")

    n = (
        20000
        if jax_dist in [dist.LKJ, dist.LKJCholesky, dist.SineBivariateVonMises]
        else 200000
    )
    d_jax = jax_dist(*params)
    k = random.PRNGKey(0)
    samples = d_jax.sample(k, sample_shape=(n,)).astype(np.float32)
    # check with suitable scipy implementation if available
    # XXX: VonMises is already tested below
    if (
        sp_dist
        and not _is_batched_multivariate(d_jax)
        and jax_dist
        not in [dist.VonMises, dist.MultivariateStudentT, dist.MatrixNormal]
    ):
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
            except (AttributeError, ValueError):
                sp_var = d_sp.var()
        else:
            sp_var = d_sp.var()
        assert_allclose(d_jax.mean, sp_mean, rtol=0.01, atol=1e-7)
        assert_allclose(d_jax.variance, sp_var, rtol=0.01, atol=1e-7)
        if jnp.all(jnp.isfinite(sp_mean)):
            assert_allclose(jnp.mean(samples, 0), d_jax.mean, rtol=0.05, atol=1e-2)
        if jnp.all(jnp.isfinite(sp_var)):
            assert_allclose(
                jnp.std(samples, 0), jnp.sqrt(d_jax.variance), rtol=0.05, atol=1e-2
            )
    elif jax_dist in [dist.LKJ, dist.LKJCholesky]:
        if jax_dist is dist.LKJCholesky:
            corr_samples = jnp.matmul(samples, jnp.swapaxes(samples, -2, -1))
        else:
            corr_samples = samples
        dimension, concentration, _ = params
        # marginal of off-diagonal entries
        marginal = dist.Beta(
            concentration + 0.5 * (dimension - 2), concentration + 0.5 * (dimension - 2)
        )
        # scale statistics due to linear mapping
        marginal_mean = 2 * marginal.mean - 1
        marginal_std = 2 * jnp.sqrt(marginal.variance)
        expected_mean = jnp.broadcast_to(
            jnp.reshape(marginal_mean, jnp.shape(marginal_mean) + (1, 1)),
            jnp.shape(marginal_mean) + d_jax.event_shape,
        )
        expected_std = jnp.broadcast_to(
            jnp.reshape(marginal_std, jnp.shape(marginal_std) + (1, 1)),
            jnp.shape(marginal_std) + d_jax.event_shape,
        )
        # diagonal elements of correlation matrices are 1
        expected_mean = expected_mean * (1 - jnp.identity(dimension)) + jnp.identity(
            dimension
        )
        expected_std = expected_std * (1 - jnp.identity(dimension))

        assert_allclose(jnp.mean(corr_samples, axis=0), expected_mean, atol=0.01)
        assert_allclose(jnp.std(corr_samples, axis=0), expected_std, atol=0.01)
    elif jax_dist in [dist.VonMises]:
        # circular mean = sample mean
        assert_allclose(d_jax.mean, jnp.mean(samples, 0), rtol=0.05, atol=1e-2)

        # circular variance
        x, y = jnp.mean(jnp.cos(samples), 0), jnp.mean(jnp.sin(samples), 0)

        expected_variance = 1 - jnp.sqrt(x**2 + y**2)
        assert_allclose(d_jax.variance, expected_variance, rtol=0.05, atol=1e-2)
    elif jax_dist in [dist.SineBivariateVonMises]:
        phi_loc = _circ_mean(samples[..., 0])
        psi_loc = _circ_mean(samples[..., 1])

        assert_allclose(
            d_jax.mean, jnp.stack((phi_loc, psi_loc), axis=-1), rtol=0.05, atol=1e-2
        )
    elif jax_dist in [dist.MatrixNormal]:
        sample_shape = (200_000,)
        # use X ~ MN(loc, U, V) then vec(X) ~ MVN(vec(loc), kron(V, U))
        if len(d_jax.batch_shape) > 0:
            axes = [len(sample_shape) + i for i in range(len(d_jax.batch_shape))]
            axes = tuple(axes)
            samples_re = jnp.moveaxis(samples, axes, jnp.arange(len(axes)))
            subshape = samples_re.shape[: len(axes)]
            ixi = product(*[range(k) for k in subshape])
            for ix in ixi:
                # mean
                def get_min_shape(ix, batch_shape):
                    return min(ix, tuple(map(lambda x: x - 1, batch_shape)))

                ix_loc = get_min_shape(ix, d_jax.loc.shape[: len(ix)])
                jnp.allclose(
                    jnp.mean(samples_re[ix], 0),
                    jnp.squeeze(d_jax.mean[ix_loc]),
                    rtol=0.5,
                    atol=1e-2,
                )
                # cov
                samples_mvn = jnp.squeeze(samples_re[ix]).reshape(
                    sample_shape + (-1,), order="F"
                )
                ix_col = get_min_shape(ix, d_jax.scale_tril_column.shape[: len(ix)])
                ix_row = get_min_shape(ix, d_jax.scale_tril_row.shape[: len(ix)])
                scale_tril = my_kron(
                    d_jax.scale_tril_column[ix_col],
                    d_jax.scale_tril_row[ix_row],
                )
                sample_scale_tril = jnp.linalg.cholesky(jnp.cov(samples_mvn.T))
                jnp.allclose(sample_scale_tril, scale_tril, atol=0.5, rtol=1e-2)
        else:  # unbatched
            # mean
            jnp.allclose(
                jnp.mean(samples, 0),
                jnp.squeeze(d_jax.mean),
                rtol=0.5,
                atol=1e-2,
            )
            # cov
            samples_mvn = jnp.squeeze(samples).reshape(sample_shape + (-1,), order="F")
            scale_tril = my_kron(
                jnp.squeeze(d_jax.scale_tril_column), jnp.squeeze(d_jax.scale_tril_row)
            )
            sample_scale_tril = jnp.linalg.cholesky(jnp.cov(samples_mvn.T))
            jnp.allclose(sample_scale_tril, scale_tril, atol=0.5, rtol=1e-2)
    else:
        if jnp.all(jnp.isfinite(d_jax.mean)):
            assert_allclose(jnp.mean(samples, 0), d_jax.mean, rtol=0.05, atol=1e-2)
        if isinstance(d_jax, dist.CAR):
            pytest.skip("CAR distribution does not have `variance` implemented.")
        if isinstance(d_jax, dist.Gompertz):
            pytest.skip("Gompertz distribution does not have `variance` implemented.")
        if jnp.all(jnp.isfinite(d_jax.variance)):
            assert jnp.allclose(
                jnp.std(samples, 0), jnp.sqrt(d_jax.variance), rtol=0.05, atol=1e-2
            )


@pytest.mark.parametrize(
    "jax_dist, sp_dist, params", CONTINUOUS + DISCRETE + DIRECTIONAL
)
@pytest.mark.parametrize("prepend_shape", [(), (2,), (2, 3)])
def test_distribution_constraints(jax_dist, sp_dist, params, prepend_shape):
    if jax_dist in (
        _TruncatedNormal,
        _TruncatedCauchy,
        _GaussianMixture,
        _Gaussian2DMixture,
        _GeneralMixture,
        _General2DMixture,
    ):
        pytest.skip(f"{jax_dist.__name__} is a function, not a class")
    dist_args = [p for p in inspect.getfullargspec(jax_dist.__init__)[0][1:]]

    valid_params, oob_params = list(params), list(params)
    key = random.PRNGKey(1)
    dependent_constraint = False
    for i in range(len(params)):
        if (
            jax_dist in (_ImproperWrapper, dist.LKJ, dist.LKJCholesky)
            and dist_args[i] != "concentration"
        ):
            continue
        if "SineSkewed" in jax_dist.__name__ and dist_args[i] != "skewness":
            continue
        if jax_dist is dist.EulerMaruyama and dist_args[i] != "t":
            continue
        if (
            jax_dist is dist.TwoSidedTruncatedDistribution
            and dist_args[i] == "base_dist"
        ):
            continue
        if jax_dist is dist.GaussianRandomWalk and dist_args[i] == "num_steps":
            continue
        if jax_dist is dist.ZeroSumNormal and dist_args[i] == "event_shape":
            continue
        if (
            jax_dist is dist.SineBivariateVonMises
            and dist_args[i] == "weighted_correlation"
        ):
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
        oob_params[i] = gen_values_outside_bounds(
            constraint, jnp.shape(params[i]), key_gen
        )
        valid_params[i] = gen_values_within_bounds(
            constraint, jnp.shape(params[i]), key_gen
        )
        if jax_dist is dist.MultivariateStudentT:
            # As mean is only defined for df > 1 & we instantiate
            # scipy.stats.multivariate_t with same mean as jax_dist
            # we need to ensure this is defined, so force df >= 1
            valid_params[0] += 1

        if jax_dist is dist.LogUniform:
            # scipy.stats.loguniform take parameter a and b
            # which is a > 0 and b > a.
            # gen_values_within_bounds() generates just
            # a > 0 and b > 0. Then, make b = a + b.
            valid_params[1] += valid_params[0]

    assert jax_dist(*oob_params)

    # Invalid parameter values throw ValueError
    if not dependent_constraint and (
        jax_dist is not _ImproperWrapper and "SineSkewed" not in jax_dist.__name__
    ):
        with pytest.raises(ValueError):
            jax_dist(*oob_params, validate_args=True)

        with pytest.raises(ValueError):
            # test error raised under jit omnistaging
            oob_params = jax.device_get(oob_params)

            def dist_gen_fn():
                d = jax_dist(*oob_params, validate_args=True)
                return d

            jax.jit(dist_gen_fn)()

    d = jax_dist(*valid_params, validate_args=True)

    # Test agreement of log density evaluation on randomly generated samples
    # with scipy's implementation when available.
    if (
        sp_dist
        and not _is_batched_multivariate(d)
        and not (d.event_shape and prepend_shape)
    ):
        valid_samples = gen_values_within_bounds(
            d.support, size=prepend_shape + d.batch_shape + d.event_shape
        )
        try:
            expected = sp_dist(*valid_params).logpdf(valid_samples)
        except AttributeError:
            expected = sp_dist(*valid_params).logpmf(valid_samples)
        assert_allclose(d.log_prob(valid_samples), expected, atol=1e-5, rtol=1e-5)

    # Out of support samples throw ValueError
    oob_samples = gen_values_outside_bounds(
        d.support, size=prepend_shape + d.batch_shape + d.event_shape
    )
    with pytest.warns(UserWarning, match="Out-of-support"):
        d.log_prob(oob_samples)

    with pytest.warns(UserWarning, match="Out-of-support"):
        # test warning work under jit omnistaging
        oob_samples = jax.device_get(oob_samples)
        valid_params = jax.device_get(valid_params)

        def log_prob_fn():
            d = jax_dist(*valid_params, validate_args=True)
            return d.log_prob(oob_samples)

        jax.jit(log_prob_fn)()


def test_omnistaging_invalid_param():
    def f(x):
        return dist.LogNormal(x, -np.ones(2), validate_args=True).log_prob(0)

    with pytest.raises(ValueError, match="got invalid"):
        jax.jit(f)(0)


def test_omnistaging_invalid_sample():
    def f(x):
        return dist.LogNormal(x, np.ones(2), validate_args=True).log_prob(-1)

    with pytest.warns(UserWarning, match="Out-of-support"):
        jax.jit(f)(0)


def test_categorical_log_prob_grad():
    data = jnp.repeat(jnp.arange(3), 10)

    def f(x):
        return (
            dist.Categorical(jax.nn.softmax(x * jnp.arange(1, 4))).log_prob(data).sum()
        )

    def g(x):
        return dist.Categorical(logits=x * jnp.arange(1, 4)).log_prob(data).sum()

    x = 0.5
    fx, grad_fx = jax.value_and_grad(f)(x)
    gx, grad_gx = jax.value_and_grad(g)(x)
    assert_allclose(fx, gx, rtol=1e-6)
    assert_allclose(grad_fx, grad_gx, atol=1e-4)


def test_beta_proportion_invalid_mean():
    with dist.distribution.validation_enabled(), pytest.raises(
        ValueError, match=r"^BetaProportion distribution got invalid mean parameter\.$"
    ):
        dist.BetaProportion(1.0, 1.0)


########################################
# Tests for constraints and transforms #
########################################


@pytest.mark.parametrize(
    "constraint, x, expected",
    [
        (constraints.boolean, np.array([True, False]), np.array([True, True])),
        (constraints.boolean, np.array([1, 1]), np.array([True, True])),
        (constraints.boolean, np.array([-1, 1]), np.array([False, True])),
        (
            constraints.corr_cholesky,
            np.array([[[1, 0], [0, 1]], [[1, 0.1], [0, 1]]]),
            np.array([True, False]),
        ),  # NB: not lower_triangular
        (
            constraints.corr_cholesky,
            np.array([[[1, 0], [1, 0]], [[1, 0], [0.5, 0.5]]]),
            np.array([False, False]),
        ),  # NB: not positive_diagonal & not unit_norm_row
        (
            constraints.corr_matrix,
            np.array([[[1, 0], [0, 1]], [[1, 0.1], [0, 1]]]),
            np.array([True, False]),
        ),  # NB: not lower_triangular
        (
            constraints.corr_matrix,
            np.array([[[1, 0], [1, 0]], [[1, 0], [0.5, 0.5]]]),
            np.array([False, False]),
        ),  # NB: not unit diagonal
        (constraints.greater_than(1), 3, True),
        (
            constraints.greater_than(1),
            np.array([-1, 1, 5]),
            np.array([False, False, True]),
        ),
        (constraints.integer_interval(-3, 5), 0, True),
        (
            constraints.integer_interval(-3, 5),
            np.array([-5, -3, 0, 1.1, 5, 7]),
            np.array([False, True, True, False, True, False]),
        ),
        (constraints.interval(-3, 5), 0, True),
        (
            constraints.interval(-3, 5),
            np.array([-5, -3, 0, 5, 7]),
            np.array([False, True, True, True, False]),
        ),
        (constraints.less_than(1), -2, True),
        (
            constraints.less_than(1),
            np.array([-1, 1, 5]),
            np.array([True, False, False]),
        ),
        (constraints.lower_cholesky, np.array([[1.0, 0.0], [-2.0, 0.1]]), True),
        (
            constraints.lower_cholesky,
            np.array([[[1.0, 0.0], [-2.0, -0.1]], [[1.0, 0.1], [2.0, 0.2]]]),
            np.array([False, False]),
        ),
        (constraints.nonnegative_integer, 3, True),
        (
            constraints.nonnegative_integer,
            np.array([-1.0, 0.0, 5.0]),
            np.array([False, True, True]),
        ),
        (constraints.positive, 3, True),
        (constraints.positive, np.array([-1, 0, 5]), np.array([False, False, True])),
        (constraints.positive_definite, np.array([[1.0, 0.3], [0.3, 1.0]]), True),
        (
            constraints.positive_definite,
            np.array([[[2.0, 0.4], [0.3, 2.0]], [[1.0, 0.1], [0.1, 0.0]]]),
            np.array([False, False]),
        ),
        (constraints.positive_integer, 3, True),
        (
            constraints.positive_integer,
            np.array([-1.0, 0.0, 5.0]),
            np.array([False, False, True]),
        ),
        (constraints.real, -1, True),
        (
            constraints.real,
            np.array([np.inf, -np.inf, np.nan, np.pi]),
            np.array([False, False, False, True]),
        ),
        (constraints.simplex, np.array([0.1, 0.3, 0.6]), True),
        (
            constraints.simplex,
            np.array([[0.1, 0.3, 0.6], [-0.1, 0.6, 0.5], [0.1, 0.6, 0.5]]),
            np.array([True, False, False]),
        ),
        (constraints.softplus_positive, 3, True),
        (
            constraints.softplus_positive,
            np.array([-1, 0, 5]),
            np.array([False, False, True]),
        ),
        (
            constraints.softplus_lower_cholesky,
            np.array([[1.0, 0.0], [-2.0, 0.1]]),
            True,
        ),
        (
            constraints.softplus_lower_cholesky,
            np.array([[[1.0, 0.0], [-2.0, -0.1]], [[1.0, 0.1], [2.0, 0.2]]]),
            np.array([False, False]),
        ),
        (constraints.unit_interval, 0.1, True),
        (
            constraints.unit_interval,
            np.array([-5, 0, 0.5, 1, 7]),
            np.array([False, True, True, True, False]),
        ),
        (
            constraints.sphere,
            np.array([[1, 0, 0], [0.5, 0.5, 0]]),
            np.array([True, False]),
        ),
        (
            constraints.open_interval(0.0, 1.0),
            np.array([-5, 0, 0.5, 1, 7]),
            np.array([False, False, True, False, False]),
        ),
    ],
)
def test_constraints(constraint, x, expected):
    v = constraint.feasible_like(x)
    if jnp.result_type(v) == "float32" or jnp.result_type(v) == "float64":
        assert not constraint.is_discrete
    assert_array_equal(constraint(x), expected)

    feasible_value = constraint.feasible_like(x)
    assert jnp.shape(feasible_value) == jnp.shape(x)
    assert_allclose(constraint(feasible_value), jnp.full(jnp.shape(expected), True))

    try:
        inverse = biject_to(constraint).inv(feasible_value)
    except NotImplementedError:
        pass
    else:
        assert_allclose(inverse, jnp.zeros_like(inverse), atol=2e-7)


@pytest.mark.parametrize(
    "constraint",
    [
        constraints.corr_cholesky,
        constraints.corr_matrix,
        constraints.greater_than(2),
        constraints.interval(-3, 5),
        constraints.l1_ball,
        constraints.less_than(1),
        constraints.lower_cholesky,
        constraints.scaled_unit_lower_cholesky,
        constraints.ordered_vector,
        constraints.positive,
        constraints.positive_definite,
        constraints.positive_ordered_vector,
        constraints.real,
        constraints.real_vector,
        constraints.simplex,
        constraints.softplus_positive,
        constraints.softplus_lower_cholesky,
        constraints.unit_interval,
        constraints.open_interval(0.0, 1.0),
    ],
    ids=lambda x: x.__class__,
)
@pytest.mark.parametrize("shape", [(), (1,), (3,), (6,), (3, 1), (1, 3), (5, 3)])
def test_biject_to(constraint, shape):
    transform = biject_to(constraint)
    event_dim = transform.domain.event_dim
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

    assert transform.forward_shape(x.shape) == y.shape
    assert transform.inverse_shape(y.shape) == x.shape

    # test inv work for NaN arrays:
    x_nan = transform.inv(jnp.full(jnp.shape(y), np.nan))
    assert x_nan.shape == x.shape

    # test codomain
    batch_shape = shape if event_dim == 0 else shape[:-1]
    assert_array_equal(transform.codomain(y), jnp.ones(batch_shape, dtype=jnp.bool_))

    # test inv
    z = transform.inv(y)
    assert_allclose(x, z, atol=1e-5, rtol=1e-5)

    # test domain, currently all is constraints.real or constraints.real_vector
    assert_array_equal(transform.domain(z), jnp.ones(batch_shape))

    # test log_abs_det_jacobian
    actual = transform.log_abs_det_jacobian(x, y)
    assert jnp.shape(actual) == batch_shape
    if len(shape) == event_dim:
        if constraint is constraints.simplex:
            expected = np.linalg.slogdet(jax.jacobian(transform)(x)[:-1, :])[1]
            inv_expected = np.linalg.slogdet(jax.jacobian(transform.inv)(y)[:, :-1])[1]
        elif constraint in [
            constraints.real_vector,
            constraints.ordered_vector,
            constraints.positive_ordered_vector,
            constraints.l1_ball,
        ]:
            expected = np.linalg.slogdet(jax.jacobian(transform)(x))[1]
            inv_expected = np.linalg.slogdet(jax.jacobian(transform.inv)(y))[1]
        elif constraint in [constraints.corr_cholesky, constraints.corr_matrix]:
            vec_transform = lambda x: matrix_to_tril_vec(  # noqa: E731
                transform(x), diagonal=-1
            )
            y_tril = matrix_to_tril_vec(y, diagonal=-1)

            def inv_vec_transform(y):
                matrix = vec_to_tril_matrix(y, diagonal=-1)
                if constraint is constraints.corr_matrix:
                    # fill the upper triangular part
                    matrix = (
                        matrix
                        + jnp.swapaxes(matrix, -2, -1)
                        + jnp.identity(matrix.shape[-1])
                    )
                return transform.inv(matrix)

            expected = np.linalg.slogdet(jax.jacobian(vec_transform)(x))[1]
            inv_expected = np.linalg.slogdet(jax.jacobian(inv_vec_transform)(y_tril))[1]
        elif constraint in [
            constraints.lower_cholesky,
            constraints.scaled_unit_lower_cholesky,
            constraints.positive_definite,
            constraints.softplus_lower_cholesky,
        ]:
            vec_transform = lambda x: matrix_to_tril_vec(transform(x))  # noqa: E731
            y_tril = matrix_to_tril_vec(y)

            def inv_vec_transform(y):
                matrix = vec_to_tril_matrix(y)
                if constraint is constraints.positive_definite:
                    # fill the upper triangular part
                    matrix = (
                        matrix
                        + jnp.swapaxes(matrix, -2, -1)
                        - jnp.diag(jnp.diag(matrix))
                    )
                return transform.inv(matrix)

            expected = np.linalg.slogdet(jax.jacobian(vec_transform)(x))[1]
            inv_expected = np.linalg.slogdet(jax.jacobian(inv_vec_transform)(y_tril))[1]
        else:
            expected = jnp.log(jnp.abs(grad(transform)(x)))
            inv_expected = jnp.log(jnp.abs(grad(transform.inv)(y)))

        assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
        assert_allclose(actual, -inv_expected, atol=1e-5, rtol=1e-5)


# NB: skip transforms which are tested in `test_biject_to`
@pytest.mark.parametrize(
    "transform, event_shape",
    [
        (PermuteTransform(np.array([3, 0, 4, 1, 2])), (5,)),
        (PowerTransform(2.0), ()),
        (SoftplusTransform(), ()),
        (
            LowerCholeskyAffine(
                np.array([1.0, 2.0]), np.array([[0.6, 0.0], [1.5, 0.4]])
            ),
            (2,),
        ),
        (
            transforms.ComposeTransform(
                [
                    biject_to(constraints.simplex),
                    SimplexToOrderedTransform(0.0),
                    biject_to(constraints.ordered_vector).inv,
                ]
            ),
            (5,),
        ),
    ],
)
@pytest.mark.parametrize(
    "batch_shape",
    [
        (),
        (1,),
        (3,),
        (6,),
        (3, 1),
        (1, 3),
        (5, 3),
    ],
)
def test_bijective_transforms(transform, event_shape, batch_shape):
    shape = batch_shape + event_shape
    rng_key = random.PRNGKey(0)
    x = biject_to(transform.domain)(random.normal(rng_key, shape))
    y = transform(x)

    # test codomain
    assert_array_equal(transform.codomain(y), jnp.ones(batch_shape))

    # test inv
    z = transform.inv(y)
    assert_allclose(x, z, atol=1e-6, rtol=1e-4)
    assert transform.inv.inv is transform
    assert transform.inv is transform.inv
    assert transform.domain is transform.inv.codomain
    assert transform.codomain is transform.inv.domain

    # test domain
    assert_array_equal(transform.domain(z), jnp.ones(batch_shape))

    # test log_abs_det_jacobian
    actual = transform.log_abs_det_jacobian(x, y)
    assert_allclose(actual, -transform.inv.log_abs_det_jacobian(y, x))
    assert jnp.shape(actual) == batch_shape
    if len(shape) == transform.domain.event_dim:
        if len(event_shape) == 1:
            expected = np.linalg.slogdet(jax.jacobian(transform)(x))[1]
            inv_expected = np.linalg.slogdet(jax.jacobian(transform.inv)(y))[1]
        else:
            expected = jnp.log(jnp.abs(grad(transform)(x)))
            inv_expected = jnp.log(jnp.abs(grad(transform.inv)(y)))

        assert_allclose(actual, expected, atol=1e-6)
        assert_allclose(actual, -inv_expected, atol=1e-6)


@pytest.mark.parametrize("batch_shape", [(), (5,)])
def test_composed_transform(batch_shape):
    t1 = transforms.AffineTransform(0, 2)
    t2 = transforms.LowerCholeskyTransform()
    t = transforms.ComposeTransform([t1, t2, t1])
    assert t.domain.event_dim == 1
    assert t.codomain.event_dim == 2

    x = np.random.normal(size=batch_shape + (6,))
    y = t(x)
    log_det = t.log_abs_det_jacobian(x, y)
    assert log_det.shape == batch_shape
    expected_log_det = (
        jnp.log(2) * 6 + t2.log_abs_det_jacobian(x * 2, y / 2) + jnp.log(2) * 9
    )
    assert_allclose(log_det, expected_log_det)


@pytest.mark.parametrize("batch_shape", [(), (5,)])
def test_composed_transform_1(batch_shape):
    t1 = transforms.AffineTransform(0, 2)
    t2 = transforms.LowerCholeskyTransform()
    t = transforms.ComposeTransform([t1, t2, t2])
    assert t.domain.event_dim == 1
    assert t.codomain.event_dim == 3

    x = np.random.normal(size=batch_shape + (6,))
    y = t(x)
    log_det = t.log_abs_det_jacobian(x, y)
    assert log_det.shape == batch_shape
    z = t2(x * 2)
    expected_log_det = (
        jnp.log(2) * 6
        + t2.log_abs_det_jacobian(x * 2, z)
        + t2.log_abs_det_jacobian(z, t2(z)).sum(-1)
    )
    assert_allclose(log_det, expected_log_det)


@pytest.mark.parametrize("batch_shape", [(), (5,)])
def test_simplex_to_order_transform(batch_shape):
    simplex = jnp.arange(5.0) / jnp.arange(5.0).sum()
    simplex = jnp.broadcast_to(simplex, batch_shape + simplex.shape)
    transform = SimplexToOrderedTransform()
    out = transform(simplex)
    assert out.shape == transform.forward_shape(simplex.shape)
    assert simplex.shape == transform.inverse_shape(out.shape)


@pytest.mark.parametrize("batch_shape", [(), (5,)])
@pytest.mark.parametrize("prepend_event_shape", [(), (4,)])
@pytest.mark.parametrize("sample_shape", [(), (7,)])
def test_transformed_distribution(batch_shape, prepend_event_shape, sample_shape):
    base_dist = (
        dist.Normal(0, 1)
        .expand(batch_shape + prepend_event_shape + (6,))
        .to_event(1 + len(prepend_event_shape))
    )
    t1 = transforms.AffineTransform(0, 2)
    t2 = transforms.LowerCholeskyTransform()
    d = dist.TransformedDistribution(base_dist, [t1, t2, t1])
    assert d.event_dim == 2 + len(prepend_event_shape)

    y = d.sample(random.PRNGKey(0), sample_shape)
    t = transforms.ComposeTransform([t1, t2, t1])
    x = t.inv(y)
    assert x.shape == sample_shape + base_dist.shape()
    log_prob = d.log_prob(y)
    assert log_prob.shape == sample_shape + batch_shape
    t_log_det = t.log_abs_det_jacobian(x, y)
    if prepend_event_shape:
        t_log_det = t_log_det.sum(-1)
    expected_log_prob = base_dist.log_prob(x) - t_log_det
    assert_allclose(log_prob, expected_log_prob, atol=1e-5)


@pytest.mark.parametrize(
    "transformed_dist",
    [
        dist.TransformedDistribution(
            dist.Normal(np.array([2.0, 3.0]), 1.0), transforms.ExpTransform()
        ),
        dist.TransformedDistribution(
            dist.Exponential(jnp.ones(2)),
            [
                transforms.PowerTransform(0.7),
                transforms.AffineTransform(0.0, jnp.ones(2) * 3),
            ],
        ),
    ],
)
def test_transformed_distribution_intermediates(transformed_dist):
    sample, intermediates = transformed_dist.sample_with_intermediates(
        random.PRNGKey(1)
    )
    assert_allclose(
        transformed_dist.log_prob(sample, intermediates),
        transformed_dist.log_prob(sample),
    )


def test_transformed_transformed_distribution():
    loc, scale = -2, 3
    dist1 = dist.TransformedDistribution(
        dist.Normal(2, 3), transforms.PowerTransform(2.0)
    )
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


@pytest.mark.parametrize(
    "ts",
    [
        [transforms.PowerTransform(0.7), transforms.AffineTransform(2.0, 3.0)],
        [transforms.ExpTransform()],
        [
            transforms.ComposeTransform(
                [transforms.AffineTransform(-2, 3), transforms.ExpTransform()]
            ),
            transforms.PowerTransform(3.0),
        ],
        [
            _make_iaf(5, hidden_dims=[10], rng_key=random.PRNGKey(0)),
            transforms.PermuteTransform(jnp.arange(5)[::-1]),
            _make_iaf(5, hidden_dims=[10], rng_key=random.PRNGKey(1)),
        ],
    ],
)
def test_compose_transform_with_intermediates(ts):
    transform = transforms.ComposeTransform(ts)
    x = random.normal(random.PRNGKey(2), (7, 5))
    y, intermediates = transform.call_with_intermediates(x)
    logdet = transform.log_abs_det_jacobian(x, y, intermediates)
    assert_allclose(y, transform(x))
    assert_allclose(logdet, transform.log_abs_det_jacobian(x, y))


@pytest.mark.parametrize("x_dim, y_dim", [(3, 3), (3, 4)])
def test_unpack_transform(x_dim, y_dim):
    xy = np.random.randn(x_dim + y_dim)
    unpack_fn = lambda xy: {"x": xy[:x_dim], "y": xy[x_dim:]}  # noqa: E731
    pack_fn = lambda d: jnp.concatenate([d["x"], d["y"]], axis=-1)  # noqa: E731
    transform = transforms.UnpackTransform(unpack_fn, pack_fn)
    z = transform(xy)
    if x_dim == y_dim:
        with pytest.warns(UserWarning, match="UnpackTransform.inv"):
            t = transform.inv(z)
    else:
        t = transform.inv(z)

    assert_allclose(t, xy)


@pytest.mark.parametrize("jax_dist, sp_dist, params", CONTINUOUS)
def test_generated_sample_distribution(
    jax_dist, sp_dist, params, N_sample=100_000, key=random.PRNGKey(11)
):
    """On samplers that we do not get directly from JAX, (e.g. we only get
    Gumbel(0,1) but also provide samplers for Gumbel(loc, scale)), also test
    agreement in the empirical distribution of generated samples between our
    samplers and those from SciPy.
    """

    if jax_dist not in [dist.Gumbel]:
        pytest.skip(
            "{} sampling method taken from upstream, no need to"
            "test generated samples.".format(jax_dist.__name__)
        )

    jax_dist = jax_dist(*params)
    if sp_dist and not jax_dist.event_shape and not jax_dist.batch_shape:
        our_samples = jax_dist.sample(key, (N_sample,))
        ks_result = osp.kstest(our_samples, sp_dist(*params).cdf)
        assert ks_result.pvalue > 0.05


@pytest.mark.parametrize(
    "jax_dist, params, support",
    [
        (dist.BernoulliLogits, (5.0,), jnp.arange(2)),
        (dist.BernoulliProbs, (0.5,), jnp.arange(2)),
        (dist.BinomialLogits, (4.5, 10), jnp.arange(11)),
        (dist.BinomialProbs, (0.5, 11), jnp.arange(12)),
        (dist.BetaBinomial, (2.0, 0.5, 12), jnp.arange(13)),
        (dist.CategoricalLogits, (np.array([3.0, 4.0, 5.0]),), jnp.arange(3)),
        (dist.CategoricalProbs, (np.array([0.1, 0.5, 0.4]),), jnp.arange(3)),
    ],
)
@pytest.mark.parametrize("batch_shape", [(5,), ()])
@pytest.mark.parametrize("expand", [False, True])
def test_enumerate_support_smoke(jax_dist, params, support, batch_shape, expand):
    p0 = jnp.broadcast_to(params[0], batch_shape + jnp.shape(params[0]))
    actual = jax_dist(p0, *params[1:]).enumerate_support(expand=expand)
    expected = support.reshape((-1,) + (1,) * len(batch_shape))
    if expand:
        expected = jnp.broadcast_to(expected, support.shape + batch_shape)
    assert_allclose(actual, expected)


def test_zero_inflated_enumerate_support():
    base_dist = dist.Bernoulli(0.5)
    d = dist.ZeroInflatedDistribution(base_dist, gate=0.5)
    assert d.has_enumerate_support
    assert_allclose(d.enumerate_support(), base_dist.enumerate_support())


@pytest.mark.parametrize("jax_dist, sp_dist, params", CONTINUOUS + DISCRETE)
@pytest.mark.parametrize("prepend_shape", [(), (2, 3)])
@pytest.mark.parametrize("sample_shape", [(), (4,)])
def test_expand(jax_dist, sp_dist, params, prepend_shape, sample_shape):
    jax_dist = jax_dist(*params)
    new_batch_shape = prepend_shape + jax_dist.batch_shape
    expanded_dist = jax_dist.expand(new_batch_shape)
    rng_key = random.PRNGKey(0)
    samples = expanded_dist.sample(rng_key, sample_shape)
    assert expanded_dist.batch_shape == new_batch_shape
    assert jnp.shape(samples) == sample_shape + new_batch_shape + jax_dist.event_shape
    assert expanded_dist.log_prob(samples).shape == sample_shape + new_batch_shape
    # test expand of expand
    assert (
        expanded_dist.expand((3,) + new_batch_shape).batch_shape
        == (3,) + new_batch_shape
    )
    # test expand error
    if prepend_shape:
        with pytest.raises(ValueError, match="Cannot broadcast distribution of shape"):
            assert expanded_dist.expand((3,) + jax_dist.batch_shape)


@pytest.mark.parametrize("base_shape", [(2, 1, 5), (3, 1), (2, 1, 1), (1, 1, 5)])
@pytest.mark.parametrize("event_dim", [0, 1, 2, 3])
@pytest.mark.parametrize("sample_shape", [(1000,), (1000, 7, 1), (1000, 1, 7)])
def test_expand_shuffle_regression(base_shape, event_dim, sample_shape):
    expand_shape = (2, 3, 5)
    event_dim = min(event_dim, len(base_shape))
    loc = random.normal(random.PRNGKey(0), base_shape) * 10
    base_dist = dist.Normal(loc, 0.1).to_event(event_dim)
    expanded_dist = base_dist.expand(expand_shape[: len(expand_shape) - event_dim])
    samples = expanded_dist.sample(random.PRNGKey(1), sample_shape)
    expected_mean = jnp.broadcast_to(loc, sample_shape[1:] + expanded_dist.shape())
    assert_allclose(samples.mean(0), expected_mean, atol=0.1)


@pytest.mark.parametrize("batch_shape", [(), (4,), (10, 3)])
def test_sine_bivariate_von_mises_batch_shape(batch_shape):
    phi_loc = jnp.broadcast_to(jnp.array(0.0), batch_shape)
    psi_loc = jnp.array(0.0)
    phi_conc = jnp.array(1.0)
    psi_conc = jnp.array(1.0)
    corr = jnp.array(0.1)

    sine = SineBivariateVonMises(phi_loc, psi_loc, phi_conc, psi_conc, corr)
    assert sine.batch_shape == batch_shape

    samples = sine.sample(random.PRNGKey(0))
    assert samples.shape == (*batch_shape, 2)


def test_sine_bivariate_von_mises_sample_mean():
    loc = jnp.array([[2.0, -1.0], [-2, 1.0]])

    sine = SineBivariateVonMises(*loc, 5000, 5000, 0.0)
    samples = sine.sample(random.PRNGKey(0), (5000,))

    assert_allclose(_circ_mean(samples).T, loc, rtol=5e-3)


@pytest.mark.parametrize("batch_shape", [(), (4,)])
def test_polya_gamma(batch_shape, num_points=20000):
    d = dist.TruncatedPolyaGamma(batch_shape=batch_shape)
    rng_key = random.PRNGKey(0)

    # test density approximately normalized
    x = jnp.linspace(1.0e-6, d.truncation_point, num_points)
    prob = (d.truncation_point / num_points) * jnp.exp(
        logsumexp(d.log_prob(x), axis=-1)
    )
    assert_allclose(prob, jnp.ones(batch_shape), rtol=1.0e-4)

    # test mean of approximate sampler
    z = d.sample(rng_key, sample_shape=(3000,))
    mean = jnp.mean(z, axis=-1)
    assert_allclose(mean, 0.25 * jnp.ones(batch_shape), rtol=0.07)


@pytest.mark.parametrize(
    "extra_event_dims,expand_shape",
    [(0, (4, 3, 2, 1)), (0, (4, 3, 2, 2)), (1, (5, 4, 3, 2)), (2, (5, 4, 3))],
)
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


@pytest.mark.parametrize(
    "batch_shape, mask_shape",
    [((), ()), ((2,), ()), ((), (2,)), ((2,), (2,)), ((4, 2), (1, 2)), ((2,), (4, 2))],
)
@pytest.mark.parametrize("event_shape", [(), (3,)])
def test_mask(batch_shape, event_shape, mask_shape):
    jax_dist = (
        dist.Normal().expand(batch_shape + event_shape).to_event(len(event_shape))
    )
    mask = dist.Bernoulli(0.5).sample(random.PRNGKey(0), mask_shape)
    if mask_shape == ():
        mask = bool(mask)
    samples = jax_dist.sample(random.PRNGKey(1))
    actual = jax_dist.mask(mask).log_prob(samples)
    assert_allclose(
        actual != 0,
        jnp.broadcast_to(mask, lax.broadcast_shapes(batch_shape, mask_shape)),
    )


@pytest.mark.parametrize("event_shape", [(), (4,), (2, 4)])
def test_mask_grad(event_shape):
    def f(x, data):
        base_dist = dist.Beta(jnp.exp(x), jnp.ones(event_shape)).to_event()
        mask = jnp.all(
            jnp.isfinite(data), tuple(-i - 1 for i in range(len(event_shape)))
        )
        log_prob = base_dist.mask(mask).log_prob(data)
        assert log_prob.shape == data.shape[: len(data.shape) - len(event_shape)]
        return log_prob.sum()

    data = np.array([[0.4, np.nan, 0.2, np.nan], [0.5, 0.5, 0.5, 0.5]])
    log_prob, grad = jax.value_and_grad(f)(1.0, data)
    assert jnp.isfinite(grad) and jnp.isfinite(log_prob)


@pytest.mark.parametrize(
    "jax_dist, sp_dist, params", CONTINUOUS + DISCRETE + DIRECTIONAL
)
def test_dist_pytree(jax_dist, sp_dist, params):
    def f(x):
        return jax_dist(*params)

    if jax_dist is _ImproperWrapper:
        pytest.skip("Cannot flattening ImproperUniform")
    if jax_dist is dist.EulerMaruyama:
        pytest.skip("EulerMaruyama doesn't define flatten/unflatten")
    jax.jit(f)(0)  # this test for flatten/unflatten
    lax.map(f, np.ones(3))  # this test for compatibility w.r.t. scan
    # Test that parameters do not change after flattening.
    expected_dist = f(0)
    actual_dist = jax.jit(f)(0)
    for name in expected_dist.arg_constraints:
        expected_arg = getattr(expected_dist, name)
        actual_arg = getattr(actual_dist, name)
        assert actual_arg is not None, f"arg {name} is None"
        if np.issubdtype(np.asarray(expected_arg).dtype, np.number):
            assert_allclose(actual_arg, expected_arg)
        else:
            assert (
                actual_arg.shape == expected_arg.shape
                and actual_arg.dtype == expected_arg.dtype
            )
    expected_sample = expected_dist.sample(random.PRNGKey(0))
    actual_sample = actual_dist.sample(random.PRNGKey(0))
    expected_log_prob = expected_dist.log_prob(expected_sample)
    actual_log_prob = actual_dist.log_prob(actual_sample)
    assert_allclose(actual_sample, expected_sample, rtol=1e-6)
    assert_allclose(actual_log_prob, expected_log_prob, rtol=2e-6)


@pytest.mark.parametrize(
    "method, arg", [("to_event", 1), ("mask", False), ("expand", [5])]
)
def test_special_dist_pytree(method, arg):
    def f(x):
        d = dist.Normal(np.zeros(1), np.ones(1))
        return getattr(d, method)(arg)

    jax.jit(f)(0)
    lax.map(f, np.ones(3))


def test_expand_no_unnecessary_batch_shape_expansion():
    # ExpandedDistribution can mutate the `batch_shape` of
    # its base distribution in order to make ExpandedDistribution
    # mappable, see #684. However, this mutation should not take
    # place if no mapping operation is performed.

    for arg in (jnp.array(1.0), jnp.ones((2,)), jnp.ones((2, 2))):
        # Low level test: ensure that (tree_flatten o tree_unflatten)(expanded_dist)
        # amounts to an identity operation.
        d = dist.Normal(arg, arg).expand([10, 3, *arg.shape])
        roundtripped_d = type(d).tree_unflatten(*d.tree_flatten()[::-1])
        assert d.batch_shape == roundtripped_d.batch_shape
        assert d.base_dist.batch_shape == roundtripped_d.base_dist.batch_shape
        assert d.base_dist.event_shape == roundtripped_d.base_dist.event_shape
        assert jnp.allclose(d.base_dist.loc, roundtripped_d.base_dist.loc)
        assert jnp.allclose(d.base_dist.scale, roundtripped_d.base_dist.scale)

        # High-level test: `jax.jit`ting a function returning an ExpandedDistribution
        # (which involves an instance of the low-level case as it will transform
        #  the original function by adding some flattening and unflattening steps)
        # should return same object as its non-jitted equivalent.
        def bs(arg):
            return dist.Normal(arg, arg).expand([10, 3, *arg.shape])

        d = bs(arg)
        dj = jax.jit(bs)(arg)

        assert isinstance(d, dist.ExpandedDistribution)
        assert isinstance(dj, dist.ExpandedDistribution)

        assert d.batch_shape == dj.batch_shape
        assert d.base_dist.batch_shape == dj.base_dist.batch_shape
        assert d.base_dist.event_shape == dj.base_dist.event_shape
        assert jnp.allclose(d.base_dist.loc, dj.base_dist.loc)
        assert jnp.allclose(d.base_dist.scale, dj.base_dist.scale)


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)], ids=str)
def test_kl_delta_normal_shape(batch_shape):
    v = np.random.normal(size=batch_shape)
    loc = np.random.normal(size=batch_shape)
    scale = np.exp(np.random.normal(size=batch_shape))
    p = dist.Delta(v)
    q = dist.Normal(loc, scale)
    assert kl_divergence(p, q).shape == batch_shape


def test_kl_delta_normal():
    v = np.random.normal()
    loc = np.random.normal()
    scale = np.exp(np.random.normal())
    p = dist.Delta(v, 10.0)
    q = dist.Normal(loc, scale)
    assert_allclose(kl_divergence(p, q), 10.0 - q.log_prob(v))


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)], ids=str)
@pytest.mark.parametrize("event_shape", [(), (4,), (2, 3)], ids=str)
def test_kl_independent_normal(batch_shape, event_shape):
    shape = batch_shape + event_shape
    p = dist.Normal(np.random.normal(size=shape), np.exp(np.random.normal(size=shape)))
    q = dist.Normal(np.random.normal(size=shape), np.exp(np.random.normal(size=shape)))
    actual = kl_divergence(
        dist.Independent(p, len(event_shape)), dist.Independent(q, len(event_shape))
    )
    expected = sum_rightmost(kl_divergence(p, q), len(event_shape))
    assert_allclose(actual, expected)


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)], ids=str)
@pytest.mark.parametrize("event_shape", [(), (4,), (2, 3)], ids=str)
def test_kl_expanded_normal(batch_shape, event_shape):
    shape = batch_shape + event_shape
    p = dist.Normal(np.random.normal(), np.exp(np.random.normal())).expand(shape)
    q = dist.Normal(np.random.normal(), np.exp(np.random.normal())).expand(shape)
    actual = kl_divergence(
        dist.Independent(p, len(event_shape)), dist.Independent(q, len(event_shape))
    )
    expected = sum_rightmost(kl_divergence(p, q), len(event_shape))
    assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "batch_shape_p, batch_shape_q",
    [
        ((1,), (1,)),
        ((2, 3), (2, 3)),
        ((5, 1, 3), (2, 3)),
        ((1, 3), (5, 2, 3)),
    ],
    ids=str,
)
@pytest.mark.parametrize("single_scale_p", [False, True], ids=str)
@pytest.mark.parametrize("single_loc_p", [False, True], ids=str)
@pytest.mark.parametrize("single_scale_q", [False, True], ids=str)
@pytest.mark.parametrize("single_loc_q", [False, True], ids=str)
def test_kl_multivariate_normal_consistency_with_independent_normals(
    batch_shape_p,
    batch_shape_q,
    single_scale_p,
    single_loc_p,
    single_scale_q,
    single_loc_q,
):
    event_shape = (5,)

    def make_dists(loc_batch_shape, scales_batch_shape):
        mus = np.random.normal(size=loc_batch_shape + event_shape)
        scales = np.exp(np.random.normal(size=scales_batch_shape + event_shape) * 0.1)

        def diagonalize(v, ignore_axes: int):
            if ignore_axes == 0:
                return jnp.diag(v)
            return vmap(diagonalize, in_axes=(0, None))(v, ignore_axes - 1)

        scale_tril = diagonalize(scales, len(scales_batch_shape))
        return (
            dist.Normal(mus, scales).to_event(len(event_shape)),
            dist.MultivariateNormal(mus, scale_tril=scale_tril),
        )

    p_uni, p_mvn = make_dists(
        () if single_loc_p else batch_shape_p, () if single_scale_p else batch_shape_p
    )
    q_uni, q_mvn = make_dists(
        () if single_loc_q else batch_shape_q, () if single_scale_q else batch_shape_q
    )

    actual = kl_divergence(p_mvn, q_mvn)
    expected = kl_divergence(p_uni, q_uni)
    assert_allclose(actual, expected, atol=1e-5)


def test_kl_multivariate_normal_nondiagonal_covariance():
    p_mvn = dist.MultivariateNormal(np.zeros(2), covariance_matrix=np.eye(2))
    q_mvn = dist.MultivariateNormal(
        np.ones(2), covariance_matrix=np.array([[2, 0.8], [0.8, 0.5]])
    )

    actual = kl_divergence(p_mvn, q_mvn)
    expected = 3.21138
    assert_allclose(actual, expected, atol=2e-5)


@pytest.mark.parametrize("shape", [(), (4,), (2, 3)], ids=str)
@pytest.mark.parametrize(
    "p_dist, q_dist",
    [
        (dist.Beta, dist.Beta),
        (dist.Gamma, dist.Gamma),
        (dist.Kumaraswamy, dist.Beta),
        (dist.Normal, dist.Normal),
        (dist.Weibull, dist.Gamma),
    ],
)
def test_kl_univariate(shape, p_dist, q_dist):
    def make_dist(dist_class):
        params = {}
        for k, c in dist_class.arg_constraints.items():
            if c is constraints.real:
                params[k] = np.random.normal(size=shape)
            elif c is constraints.positive:
                params[k] = np.exp(np.random.normal(size=shape))
            else:
                raise ValueError(f"Missing pattern for param {k}.")
        d = dist_class(**params)
        if dist_class is dist.Kumaraswamy:
            d.KL_KUMARASWAMY_BETA_TAYLOR_ORDER = 1000
        return d

    p = make_dist(p_dist)
    q = make_dist(q_dist)
    actual = kl_divergence(p, q)
    x = p.sample(random.PRNGKey(0), (10000,)).copy()
    expected = jnp.mean((p.log_prob(x) - q.log_prob(x)), 0)
    assert_allclose(actual, expected, rtol=0.05)


@pytest.mark.parametrize("shape", [(4,), (2, 3)], ids=str)
def test_kl_dirichlet_dirichlet(shape):
    p = dist.Dirichlet(np.exp(np.random.normal(size=shape)))
    q = dist.Dirichlet(np.exp(np.random.normal(size=shape)))
    actual = kl_divergence(p, q)
    x = p.sample(random.PRNGKey(0), (10_000,)).copy()
    expected = jnp.mean((p.log_prob(x) - q.log_prob(x)), 0)
    assert_allclose(actual, expected, rtol=0.05)


def test_vmapped_binomial_p0():
    # test that vmapped binomial with p = 0 does not have an infinite loop
    def sample_binomial_withp0(key):
        n = 2 * (random.uniform(key) > 0.5)
        _, key = random.split(key)
        return dist.Binomial(total_count=n, probs=0).sample(key)

    jax.vmap(sample_binomial_withp0)(random.split(random.PRNGKey(0), 1))


def _get_vmappable_dist_init_params(jax_dist):
    if jax_dist.__name__ == ("_TruncatedCauchy"):
        return [2, 3]
    elif jax_dist.__name__ == ("_TruncatedNormal"):
        return [2, 3]
    elif issubclass(jax_dist, dist.Distribution):
        init_parameters = list(inspect.signature(jax_dist.__init__).parameters.keys())[
            1:
        ]
        vmap_over_parameters = list(
            inspect.signature(vmap_over.dispatch(jax_dist)).parameters.keys()
        )[1:]
        return list(
            [
                i
                for i, name in enumerate(init_parameters)
                if name in vmap_over_parameters
            ]
        )
    else:
        raise ValueError


def _allclose_or_equal(a1, a2):
    if isinstance(a1, np.ndarray):
        return np.allclose(a2, a1)
    elif isinstance(a1, jnp.ndarray):
        return jnp.allclose(a2, a1)
    elif isinstance(a1, csr_matrix):
        return np.allclose(a2.todense(), a1.todense())
    else:
        return a2 == a1 or a2 is a1


def _tree_equal(t1, t2):
    t = jax.tree.map(_allclose_or_equal, t1, t2)
    return jnp.all(jax.flatten_util.ravel_pytree(t)[0])


@pytest.mark.parametrize(
    "jax_dist, sp_dist, params", CONTINUOUS + DISCRETE + DIRECTIONAL
)
def test_vmap_dist(jax_dist, sp_dist, params):
    param_names = list(inspect.signature(jax_dist).parameters.keys())
    vmappable_param_idxs = _get_vmappable_dist_init_params(jax_dist)
    vmappable_param_idxs = vmappable_param_idxs[: len(params)]

    if len(vmappable_param_idxs) == 0:
        return

    def make_jax_dist(*params):
        return jax_dist(*params)

    def sample(d: dist.Distribution):
        return d.sample(random.PRNGKey(0))

    d = make_jax_dist(*params)

    if isinstance(d, _SparseCAR) and d.is_sparse:
        # In this case, since csr arrays are not jittable,
        # _SparseCAR has a csr_matrix as part of its pytree
        # definition (not as a pytree leaf). This causes pytree
        # operations like jax.tree.map to fail, since these functions
        # compare the pytree def of each of the arguments using ==
        # which is ambiguous for array-like objects.
        return

    in_out_axes_cases = [
        # vmap over all args
        (
            tuple(0 if i in vmappable_param_idxs else None for i in range(len(params))),
            0,
        ),
        # vmap over a single arg, out over all attributes of a distribution
        *(
            ([0 if i == idx else None for i in range(len(params))], 0)
            for idx in vmappable_param_idxs
            if params[idx] is not None
        ),
        # vmap over a single arg, out over the associated attribute of the distribution
        *(
            (
                [0 if i == idx else None for i in range(len(params))],
                vmap_over(d, **{param_names[idx]: 0}),
            )
            for idx in vmappable_param_idxs
            if params[idx] is not None
        ),
        # vmap over a single arg, axis=1, (out single attribute, axis=1)
        *(
            (
                [1 if i == idx else None for i in range(len(params))],
                vmap_over(d, **{param_names[idx]: 1}),
            )
            for idx in vmappable_param_idxs
            if isinstance(params[idx], jnp.ndarray)
            and jnp.array(params[idx]).ndim > 0
            # skip this distribution because _GeneralMixture.__init__ turns
            # 1d inputs into 0d attributes, thus breaks the expectations of
            # the vmapping test case where in_axes=1, only done for rank>=1 tensors.
            and jax_dist is not _GeneralMixture
        ),
    ]

    for in_axes, out_axes in in_out_axes_cases:
        batched_params = [
            (
                jax.jax.tree.map(lambda x: jnp.expand_dims(x, ax), arg)
                if isinstance(ax, int)
                else arg
            )
            for arg, ax in zip(params, in_axes)
        ]
        # Recreate the jax_dist to avoid side effects coming from `d.sample`
        # triggering lazy_property computations, which, in a few cases, break
        # vmap_over's expectations regarding existing attributes to be vmapped.
        d = make_jax_dist(*params)
        batched_d = jax.vmap(make_jax_dist, in_axes=in_axes, out_axes=out_axes)(
            *batched_params
        )
        eq = vmap(lambda x, y: _tree_equal(x, y), in_axes=(out_axes, None))(
            batched_d, d
        )
        assert eq == jnp.array([True])

        samples_dist = sample(d)
        samples_batched_dist = jax.vmap(sample, in_axes=(out_axes,))(batched_d)
        assert samples_batched_dist.shape == (1, *samples_dist.shape)


def test_vmap_validate_args():
    # Test for #1684: vmapping distributions whould work when `validate_args=True`
    v_dist = jax.vmap(
        lambda loc, scale: dist.Normal(loc=loc, scale=scale, validate_args=True),
        in_axes=(0, 0),
    )(jnp.zeros((2,)), jnp.zeros((2,)))

    # non-regression test
    v_dist = jax.vmap(
        lambda loc, scale: dist.Normal(loc=loc, scale=scale, validate_args=False),
        in_axes=(0, 0),
    )(jnp.zeros((2,)), jnp.zeros((2,)))
    assert not v_dist._validate_args


def test_multinomial_abstract_total_count():
    probs = jnp.array([0.2, 0.5, 0.3])
    key = random.PRNGKey(0)

    def f(x):
        total_count = x.sum(-1)
        return dist.Multinomial(total_count, probs=probs, total_count_max=10).sample(
            key
        )

    x = dist.Multinomial(10, probs).sample(key)
    y = jax.jit(f)(x)
    assert_allclose(x, y, rtol=1e-6)


def test_normal_log_cdf():
    # test if log_cdf method agrees with jax.scipy.stats.norm.logcdf
    # and if exp(log_cdf) agrees with cdf
    loc = jnp.array([[0.0, -10.0, 20.0]])
    scale = jnp.array([[1, 5, 7]])
    values = jnp.linspace(-5, 5, 100).reshape(-1, 1)
    numpyro_log_cdf = dist.Normal(loc=loc, scale=scale).log_cdf(values)
    numpyro_cdf = dist.Normal(loc=loc, scale=scale).cdf(values)
    jax_log_cdf = jax_norm.logcdf(loc=loc, scale=scale, x=values)
    assert_allclose(numpyro_log_cdf, jax_log_cdf)
    assert_allclose(jnp.exp(numpyro_log_cdf), numpyro_cdf, rtol=1e-6)


@pytest.mark.parametrize(
    "value",
    [
        -15.0,
        jnp.array([[-15.0], [-10.0], [-5.0]]),
        jnp.array([[[-15.0], [-10.0], [-5.0]], [[-14.0], [-9.0], [-4.0]]]),
    ],
)
def test_truncated_normal_log_prob_in_tail(value):
    # define set of distributions truncated in tail of distribution
    loc = 1.35
    scale = jnp.geomspace(0.01, 1, 10)
    low, high = (-20, -1.0)
    a, b = (low - loc) / scale, (high - loc) / scale  # rescale for jax input

    numpyro_log_prob = dist.TruncatedNormal(loc, scale, low=low, high=high).log_prob(
        value
    )
    jax_log_prob = jax_truncnorm.logpdf(value, loc=loc, scale=scale, a=a, b=b)
    assert_allclose(numpyro_log_prob, jax_log_prob, rtol=1e-06)


def test_sample_truncated_normal_in_tail():
    # test, if samples from distributions truncated in
    # tail of distribution returns any inf's
    tail_dist = dist.TruncatedNormal(loc=0, scale=1, low=-16, high=-15)
    samples = tail_dist.sample(random.PRNGKey(0), sample_shape=(10_000,))
    assert ~jnp.isinf(samples).any()


@jax.enable_custom_prng()
def test_jax_custom_prng():
    samples = dist.Normal(0, 5).sample(random.PRNGKey(0), sample_shape=(1000,))
    assert ~jnp.isinf(samples).any()


def _assert_not_jax_issue_19885(
    capfd: pytest.CaptureFixture, func: Callable, *args, **kwargs
) -> None:
    # jit-ing identity plus matrix multiplication leads to performance degradation as
    # discussed in https://github.com/google/jax/issues/19885. This assertion verifies
    # that the issue does not affect perforance in numpyro.
    for jit in [True, False]:
        result = jax.jit(func)(*args, **kwargs)
        block_until_ready = getattr(result, "block_until_ready", None)
        if block_until_ready:
            result = block_until_ready()
        _, err = capfd.readouterr()
        assert (
            "MatMul reference implementation being executed" not in err
        ), f"jit: {jit}"
    return result


@pytest.mark.xfail
def test_jax_issue_19885(capfd: pytest.CaptureFixture) -> None:
    def func_with_warning(y) -> jnp.ndarray:
        return jnp.identity(y.shape[-1]) + jnp.matmul(y, y)

    _assert_not_jax_issue_19885(capfd, func_with_warning, jnp.ones((20, 100, 100)))


def test_lowrank_mvn_19885(capfd: pytest.CaptureFixture) -> None:
    # Create parameters.
    batch_size = 100
    event_size = 200
    sample_size = 40
    rank = 40
    loc, cov_diag = random.normal(random.key(0), (2, batch_size, event_size))
    cov_diag = jnp.exp(cov_diag)
    cov_factor = random.normal(random.key(1), (batch_size, event_size, rank))

    distribution = _assert_not_jax_issue_19885(
        capfd, dist.LowRankMultivariateNormal, loc, cov_factor, cov_diag
    )
    x = _assert_not_jax_issue_19885(
        capfd,
        lambda x: distribution.sample(random.key(0), x.shape),
        jnp.empty(sample_size),
    )
    assert x.shape == (sample_size, batch_size, event_size)
    log_prob = _assert_not_jax_issue_19885(capfd, distribution.log_prob, x)
    assert log_prob.shape == (sample_size, batch_size)


def test_gaussian_random_walk_linear_recursive_equivalence():
    dist1 = dist.GaussianRandomWalk(3.7, 15)
    dist2 = dist.TransformedDistribution(
        dist.Normal(0, 3.7).expand([15, 1]).to_event(2),
        dist.transforms.RecursiveLinearTransform(jnp.eye(1)),
    )
    x1 = dist1.sample(random.PRNGKey(7))
    x2 = dist2.sample(random.PRNGKey(7))
    assert jnp.allclose(x1, x2.squeeze())
    assert jnp.allclose(dist1.log_prob(x1), dist2.log_prob(x2))
