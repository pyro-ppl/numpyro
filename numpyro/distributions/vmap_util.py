# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import copy
from functools import singledispatch
from typing import Union

from numpyro.distributions import constraints
from numpyro.distributions.conjugate import (
    BetaBinomial,
    DirichletMultinomial,
    GammaPoisson,
    NegativeBinomial2,
    NegativeBinomialLogits,
    NegativeBinomialProbs,
)
from numpyro.distributions.constraints import Constraint
from numpyro.distributions.continuous import (
    CAR,
    LKJ,
    AsymmetricLaplace,
    AsymmetricLaplaceQuantile,
    Beta,
    BetaProportion,
    Cauchy,
    Chi2,
    Dirichlet,
    EulerMaruyama,
    Exponential,
    Gamma,
    GaussianRandomWalk,
    Gompertz,
    Gumbel,
    HalfCauchy,
    HalfNormal,
    InverseGamma,
    Kumaraswamy,
    Laplace,
    LKJCholesky,
    Logistic,
    LogNormal,
    LogUniform,
    LowRankMultivariateNormal,
    MatrixNormal,
    MultivariateNormal,
    MultivariateStudentT,
    Normal,
    Pareto,
    RelaxedBernoulliLogits,
    SoftLaplace,
    StudentT,
    Uniform,
    Weibull,
)
from numpyro.distributions.copula import GaussianCopula, GaussianCopulaBeta
from numpyro.distributions.directional import (
    ProjectedNormal,
    SineBivariateVonMises,
    SineSkewed,
    VonMises,
)
from numpyro.distributions.discrete import (
    BernoulliLogits,
    BernoulliProbs,
    BinomialLogits,
    BinomialProbs,
    CategoricalLogits,
    CategoricalProbs,
    DiscreteUniform,
    GeometricLogits,
    GeometricProbs,
    MultinomialLogits,
    MultinomialProbs,
    OrderedLogistic,
    Poisson,
    ZeroInflatedLogits,
    ZeroInflatedPoisson,
    ZeroInflatedProbs,
)
from numpyro.distributions.distribution import (
    Delta,
    Distribution,
    ExpandedDistribution,
    FoldedDistribution,
    ImproperUniform,
    TransformedDistribution,
)
from numpyro.distributions.mixtures import MixtureGeneral, MixtureSameFamily
from numpyro.distributions.transforms import (
    AffineTransform,
    CorrCholeskyTransform,
    PowerTransform,
    Transform,
)
from numpyro.distributions.truncated import (
    LeftTruncatedDistribution,
    RightTruncatedDistribution,
    TwoSidedTruncatedDistribution,
)


@singledispatch
def vmap_over(d: Union[Distribution, Transform, Constraint], **kwargs):
    raise NotImplementedError


@vmap_over.register
def _vmap_over_affine_transform(self: AffineTransform, loc, scale):
    dist_axes = copy.copy(self)
    dist_axes.loc = loc
    dist_axes.scale = scale
    dist_axes.domain = None
    return dist_axes


@vmap_over.register
def _vmap_over_greater_than(self: constraints._GreaterThan, lower_bound):
    axes = copy.copy(self)
    axes.lower_bound = lower_bound
    return axes


@vmap_over.register
def _vmap_over_less_than(self: constraints._LessThan, upper_bound):
    axes = copy.copy(self)
    axes.upper_bound = upper_bound
    return axes


@vmap_over.register
def _vmap_over_interval(self: constraints._Interval, lower_bound, upper_bound):
    import copy

    axes = copy.copy(self)
    axes.lower_bound = lower_bound
    axes.upper_bound = upper_bound
    return axes


@vmap_over.register
def _vmap_over_integer_interval(
    self: constraints._IntegerInterval, lower_bound=None, upper_bound=None
):
    dist_axes = copy.copy(self)
    dist_axes.lower_bound = lower_bound
    dist_axes.upper_bound = upper_bound
    return dist_axes


@vmap_over.register
def _vmap_over_corr_cholesky_transform(self: CorrCholeskyTransform):
    return None


@vmap_over.register
def _vmap_over_power_transform(self: PowerTransform, exponent):
    axes = copy.copy(self)
    axes.exponent = exponent
    return axes


@vmap_over.register
def _default_vmap_over(d: Distribution, **kwargs):
    pytree_fields = type(d).gather_pytree_data_fields()
    dist_axes = copy.copy(d)

    for f in pytree_fields:
        setattr(dist_axes, f, kwargs.get(f, None))

    return dist_axes


@vmap_over.register
def _(self: AsymmetricLaplaceQuantile, loc=None, scale=None, quantile=None):
    dist_axes = _default_vmap_over(self, loc=loc, scale=scale, quantile=quantile)
    dist_axes._ald = vmap_over(
        self._ald,
        loc=loc,
        scale=scale if scale is not None else quantile,
        asymmetry=quantile,
    )
    return dist_axes


@vmap_over.register
def _vmap_over_beta(self: Beta, concentration1=None, concentration0=None):
    dist_axes = _default_vmap_over(
        self, concentration1=concentration1, concentration0=concentration0
    )
    if concentration1 is not None or concentration0 is not None:
        dist_axes._dirichlet = 0
    else:
        dist_axes._dirichlet = None
    return dist_axes


@vmap_over.register
def _vmap_over_beta_proportion(self: BetaProportion, mean=None, concentration=None):
    dist_axes = vmap_over.dispatch(Beta)(
        self,
        concentration if concentration is not None else mean,
        concentration if concentration is not None else mean,
    )
    dist_axes.concentration = concentration
    return dist_axes


@vmap_over.register
def _vmap_over_chi2(self: Chi2, df=None):
    dist_axes = vmap_over.dispatch(Gamma)(self, rate=df, concentration=df)
    dist_axes.df = df
    return dist_axes


@vmap_over.register
def _vmap_over_gaussian_copula(
    self: GaussianCopula,
    marginal_dist=None,
    correlation_matrix=None,
    correlation_cholesky=None,
):
    dist_axes = copy.copy(self)
    dist_axes.marginal_dist = marginal_dist
    dist_axes.base_dist = vmap_over(
        dist_axes.base_dist,
        loc=correlation_matrix or correlation_cholesky,
        scale_tril=correlation_matrix or correlation_cholesky,
    )
    return dist_axes


@vmap_over.register
def _vmap_over_gausian_copula_beta(
    self: GaussianCopulaBeta,
    concentration1=None,
    concentration0=None,
    correlation_matrix=None,
    correlation_cholesky=None,
):
    d = vmap_over.dispatch(GaussianCopula)(
        self,
        vmap_over(
            self.marginal_dist,
            concentration1=concentration1,
            concentration0=concentration0,
        ),
        correlation_matrix=correlation_matrix,
        correlation_cholesky=correlation_cholesky,
    )
    d.concentration1 = concentration1
    d.concentration0 = concentration0
    return d


@vmap_over.register
def _vmap_over_half_cauchy(self: HalfCauchy, scale=None):
    dist_axes = _default_vmap_over(self, scale=scale)
    dist_axes._cauchy = vmap_over(self._cauchy, loc=scale, scale=scale)
    return dist_axes


@vmap_over.register
def _vmap_over_inverse_gamma(self: InverseGamma, concentration=None, rate=None):
    dist_axes = _default_vmap_over(self, concentration=concentration, rate=rate)
    dist_axes.base_dist = vmap_over(
        self.base_dist, concentration=concentration, rate=rate
    )
    dist_axes.transforms = None
    return dist_axes


@vmap_over.register
def _vmap_over_uniform(self: Uniform, low=None, high=None):
    dist_axes = _default_vmap_over(self, low=low, high=high)
    dist_axes._support = vmap_over(self._support, low, high)
    return dist_axes


@vmap_over.register
def _vmap_over_kumaraswamy(self: Kumaraswamy, concentration0=None, concentration1=None):
    dist_axes = _default_vmap_over(
        self, concentration0=concentration0, concentration1=concentration1
    )
    if isinstance(self.base_dist, Uniform):
        dist_axes.base_dist = vmap_over(self.base_dist, low=None, high=None)
    else:
        assert isinstance(self.base_dist, ExpandedDistribution)
        dist_axes.base_dist = vmap_over(self.base_dist, base_dist=None)

    dist_axes.transforms = [
        vmap_over(self.transforms[0], exponent=concentration0),
        vmap_over(self.transforms[1], loc=None, scale=None),
        vmap_over(self.transforms[2], exponent=concentration1),
    ]
    return dist_axes


@vmap_over.register
def _vmap_over_lkj(self: LKJ, concentration=None):
    dist_axes = _default_vmap_over(self, concentration=concentration)
    dist_axes.base_dist = vmap_over(self.base_dist, concentration)
    dist_axes.transforms = None
    return dist_axes


@vmap_over.register
def _vmap_over_lkj_cholesky(self: LKJCholesky, concentration):
    dist_axes = _default_vmap_over(self, concentration=concentration)
    if dist_axes.sample_method == "onion":
        dist_axes._beta = vmap_over(self._beta, None, concentration)
    elif dist_axes.sample_method == "cvine":
        dist_axes._beta = vmap_over(self._beta, concentration, concentration)
    return dist_axes


@vmap_over.register
def _vmap_over_lognormal(self: LogNormal, loc=None, scale=None):
    dist_axes = _default_vmap_over(self, loc=loc, scale=scale)
    dist_axes.transforms = None
    dist_axes.base_dist = vmap_over(self.base_dist, loc=loc, scale=scale)
    return dist_axes


@vmap_over.register
def _vmap_over_loguniform(self: LogUniform, low=None, high=None):
    dist_axes = _default_vmap_over(self, low=low, high=high)
    dist_axes.base_dist = vmap_over(self.base_dist, low, high)
    dist_axes._support = vmap_over(self._support, low, high)
    return dist_axes


@vmap_over.register
def _vmap_over_car(
    self: CAR, loc=None, correlation=None, conditional_precision=None, adj_matrix=None
):
    dist_axes = _default_vmap_over(
        self,
        loc=loc,
        correlation=correlation,
        conditional_precision=conditional_precision,
    )
    if not self.is_sparse:
        dist_axes.adj_matrix = adj_matrix
        dist_axes.precision_matrix = adj_matrix
    else:
        assert adj_matrix is None
    return dist_axes


@vmap_over.register
def _vmap_over_multivariate_student_t(
    self: MultivariateStudentT, df=None, loc=None, scale_tril=None
):
    dist_axes = _default_vmap_over(self, df=df, loc=loc, scale_tril=scale_tril)
    dist_axes._chi2 = vmap_over(self._chi2, df)
    return dist_axes


@vmap_over.register
def _vmap_over_low_rank_multivariate_normal(
    self: LowRankMultivariateNormal, loc=None, cov_factor=None, cov_diag=None
):
    dist_axes = _default_vmap_over(
        self, loc=loc, cov_factor=cov_factor, cov_diag=cov_diag
    )
    dist_axes._capacitance_tril = cov_diag if cov_diag is not None else cov_factor
    return dist_axes


@vmap_over.register
def _vmap_over_pareto(self: Pareto, scale=None, alpha=None):
    dist_axes = _default_vmap_over(self, scale=scale, alpha=alpha)
    dist_axes.base_dist = vmap_over(self.base_dist, rate=alpha)
    dist_axes.transforms = [None, vmap_over(self.transforms[1], None, scale)]
    return dist_axes


@vmap_over.register
def _vmap_over_relaxed_bernoulli_logits(
    self: RelaxedBernoulliLogits, temperature=None, logits=None
):
    dist_axes = _default_vmap_over(self, temperature=temperature, logits=logits)
    dist_axes.transforms = None
    dist_axes.base_dist = vmap_over(
        self.base_dist,
        loc=logits if logits is not None else temperature,
        scale=temperature,
    )
    return dist_axes


@vmap_over.register
def _vmap_over_student_t(self: StudentT, df=None, loc=None, scale=None):
    dist_axes = _default_vmap_over(self, df=df, loc=loc, scale=scale)
    dist_axes._chi2 = vmap_over(self._chi2, df)
    return dist_axes


@vmap_over.register
def _vmap_over_two_sided_truncated_distribution(
    self: TwoSidedTruncatedDistribution, low=None, high=None
):
    dist_axes = _default_vmap_over(self, low=low, high=high)
    dist_axes.base_dist = None
    dist_axes._support = vmap_over(self._support, low, high)
    return dist_axes


@vmap_over.register
def _vmap_over_left_truncated_distribution(self: LeftTruncatedDistribution, low=None):
    dist_axes = _default_vmap_over(self, low=low)
    dist_axes.base_dist = None
    dist_axes._support = vmap_over(self._support, low)
    return dist_axes


@vmap_over.register
def _vmap_over_right_truncated_distribution(
    self: RightTruncatedDistribution, high=None
):
    dist_axes = _default_vmap_over(self, high=high)
    dist_axes.base_dist = None
    dist_axes._support = vmap_over(self._support, high)
    return dist_axes


@vmap_over.register
def _vmap_over_beta_binomial(
    self: BetaBinomial, concentration1=None, concentration0=None, total_count=None
):
    dist_axes = _default_vmap_over(
        self,
        concentration1=concentration1,
        concentration0=concentration0,
        total_count=total_count,
    )
    dist_axes._beta = vmap_over(
        self._beta, concentration1=concentration1, concentration0=concentration0
    )
    return dist_axes


@vmap_over.register
def _vmap_over_dirichlet_multinomial(self: DirichletMultinomial, concentration=None):
    dist_axes = _default_vmap_over(self, concentration=concentration)
    dist_axes._dirichlet = vmap_over(self._dirichlet, concentration=concentration)
    return dist_axes


@vmap_over.register
def _vmap_over_gamma_poisson(self: GammaPoisson, concentration=None, rate=None):
    dist_axes = _default_vmap_over(self, concentration=concentration, rate=rate)
    dist_axes._gamma = vmap_over(self._gamma, concentration=concentration, rate=rate)
    return dist_axes


@vmap_over.register
def _vmap_over_negative_binomial_probs(
    self: NegativeBinomialProbs, total_count=None, probs=None
):
    dist_axes = vmap_over.dispatch(GammaPoisson)(
        self, concentration=total_count, rate=probs
    )
    dist_axes.total_count = total_count
    dist_axes.probs = probs
    return dist_axes


@vmap_over.register
def _vmap_over_negative_binomial_logits(
    self: NegativeBinomialLogits, total_count=None, logits=None
):
    dist_axes = vmap_over.dispatch(GammaPoisson)(
        self, concentration=total_count, rate=logits
    )
    dist_axes.total_count = total_count
    dist_axes.logits = logits
    return dist_axes


@vmap_over.register
def _vmap_over_negative_binomial_2(
    self: NegativeBinomial2, mean=None, concentration=None
):
    return vmap_over.dispatch(GammaPoisson)(
        self,
        concentration=concentration,
        rate=concentration if concentration is not None else mean,
    )


@vmap_over.register
def _vmap_over_ordered_logistic(self: OrderedLogistic, predictor=None, cutpoints=None):
    dist_axes = vmap_over.dispatch(CategoricalProbs)(
        self, probs=predictor if predictor is not None else cutpoints
    )
    dist_axes.predictor = predictor
    dist_axes.cutpoints = cutpoints
    return dist_axes


@vmap_over.register
def _vmap_over_discrete_uniform(self: DiscreteUniform, low=None, high=None):
    dist_axes = _default_vmap_over(self, low=low, high=high)
    dist_axes._support = vmap_over(self._support, low, high)
    return dist_axes


@vmap_over.register
def _vmap_over_zero_inflated_poisson(self: ZeroInflatedPoisson, gate=None, rate=None):
    dist_axes = vmap_over.dispatch(ZeroInflatedProbs)(
        self, base_dist=vmap_over(self.base_dist, rate=rate), gate=gate
    )
    dist_axes.rate = rate
    return dist_axes


@vmap_over.register
def _vmap_over_half_normal(self: HalfNormal, scale=None):
    dist_axes = _default_vmap_over(self, scale=scale)
    dist_axes._normal = vmap_over(self._normal, loc=scale, scale=scale)
    return dist_axes


vmap_over.register(AsymmetricLaplace, _default_vmap_over)
vmap_over.register(Gamma, _default_vmap_over)
vmap_over.register(Cauchy, _default_vmap_over)
vmap_over.register(Dirichlet, _default_vmap_over)
vmap_over.register(EulerMaruyama, _default_vmap_over)
vmap_over.register(Exponential, _default_vmap_over)
vmap_over.register(FoldedDistribution, _default_vmap_over)
vmap_over.register(GaussianRandomWalk, _default_vmap_over)
vmap_over.register(MultivariateNormal, _default_vmap_over)
vmap_over.register(Gompertz, _default_vmap_over)
vmap_over.register(Gumbel, _default_vmap_over)
vmap_over.register(TransformedDistribution, _default_vmap_over)
vmap_over.register(ImproperUniform, _default_vmap_over)
vmap_over.register(ExpandedDistribution, _default_vmap_over)
vmap_over.register(Laplace, _default_vmap_over)
vmap_over.register(Logistic, _default_vmap_over)
vmap_over.register(MatrixNormal, _default_vmap_over)
vmap_over.register(Normal, _default_vmap_over)
vmap_over.register(SoftLaplace, _default_vmap_over)
vmap_over.register(MixtureSameFamily, _default_vmap_over)
vmap_over.register(MixtureGeneral, _default_vmap_over)
vmap_over.register(VonMises, _default_vmap_over)
vmap_over.register(SineBivariateVonMises, _default_vmap_over)
vmap_over.register(ProjectedNormal, _default_vmap_over)
vmap_over.register(SineSkewed, _default_vmap_over)
vmap_over.register(BernoulliProbs, _default_vmap_over)
vmap_over.register(BernoulliLogits, _default_vmap_over)
vmap_over.register(BinomialProbs, _default_vmap_over)
vmap_over.register(BinomialLogits, _default_vmap_over)
vmap_over.register(CategoricalProbs, _default_vmap_over)
vmap_over.register(CategoricalLogits, _default_vmap_over)
vmap_over.register(Delta, _default_vmap_over)
vmap_over.register(GeometricProbs, _default_vmap_over)
vmap_over.register(GeometricLogits, _default_vmap_over)
vmap_over.register(MultinomialProbs, _default_vmap_over)
vmap_over.register(MultinomialLogits, _default_vmap_over)
vmap_over.register(Poisson, _default_vmap_over)
vmap_over.register(ZeroInflatedProbs, _default_vmap_over)
vmap_over.register(ZeroInflatedLogits, _default_vmap_over)
vmap_over.register(Weibull, _default_vmap_over)
