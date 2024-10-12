# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import copy
from functools import singledispatch
from typing import Union

import jax
import jax.numpy as jnp

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
    AsymmetricLaplaceQuantile,
    Beta,
    BetaProportion,
    Chi2,
    Gamma,
    HalfCauchy,
    HalfNormal,
    InverseGamma,
    Kumaraswamy,
    LKJCholesky,
    LogNormal,
    LogUniform,
    LowRankMultivariateNormal,
    MultivariateStudentT,
    Pareto,
    RelaxedBernoulliLogits,
    StudentT,
    Uniform,
)
from numpyro.distributions.copula import GaussianCopula, GaussianCopulaBeta
from numpyro.distributions.discrete import (
    CategoricalProbs,
    DiscreteUniform,
    OrderedLogistic,
    ZeroInflatedPoisson,
    ZeroInflatedProbs,
)
from numpyro.distributions.distribution import (
    Distribution,
    ExpandedDistribution,
    Independent,
    MaskedDistribution,
    Unit,
)
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
def _vmap_over_affine_transform(
    dist: AffineTransform, loc=None, scale=None, domain=None
):
    dist_axes = copy.copy(dist)
    dist_axes.loc = loc
    dist_axes.scale = scale
    dist_axes.domain = domain
    return dist_axes


@vmap_over.register
def _vmap_over_greater_than(dist: constraints._GreaterThan, lower_bound=None):
    axes = copy.copy(dist)
    axes.lower_bound = lower_bound
    return axes


@vmap_over.register
def _vmap_over_less_than(dist: constraints._LessThan, upper_bound=None):
    axes = copy.copy(dist)
    axes.upper_bound = upper_bound
    return axes


@vmap_over.register
def _vmap_over_interval(
    dist: constraints._Interval, lower_bound=None, upper_bound=None
):
    axes = copy.copy(dist)
    axes.lower_bound = lower_bound
    axes.upper_bound = upper_bound
    return axes


@vmap_over.register
def _vmap_over_integer_interval(
    dist: constraints._IntegerInterval, lower_bound=None, upper_bound=None
):
    dist_axes = copy.copy(dist)
    dist_axes.lower_bound = lower_bound
    dist_axes.upper_bound = upper_bound
    return dist_axes


@vmap_over.register
def _vmap_over_corr_cholesky_transform(dist: CorrCholeskyTransform):
    return None


@vmap_over.register
def _vmap_over_power_transform(dist: PowerTransform, exponent=None):
    axes = copy.copy(dist)
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
def _(dist: AsymmetricLaplaceQuantile, loc=None, scale=None, quantile=None):
    dist_axes = _default_vmap_over(
        dist,
        loc=loc,
        scale=scale,
        quantile=quantile,
        _ald=vmap_over(
            dist._ald,
            loc=loc,
            scale=scale if scale is not None else quantile,
            asymmetry=quantile,
        ),
    )
    return dist_axes


@vmap_over.register
def _vmap_over_beta(dist: Beta, concentration1=None, concentration0=None):
    dist_axes = _default_vmap_over(
        dist, concentration1=concentration1, concentration0=concentration0
    )
    if concentration1 is not None or concentration0 is not None:
        dist_axes._dirichlet = 0
    else:
        dist_axes._dirichlet = None
    return dist_axes


@vmap_over.register
def _vmap_over_beta_proportion(dist: BetaProportion, mean=None, concentration=None):
    dist_axes = vmap_over.dispatch(Beta)(
        dist,
        concentration1=concentration if concentration is not None else mean,
        concentration0=concentration if concentration is not None else mean,
    )
    dist_axes.concentration = concentration
    return dist_axes


@vmap_over.register
def _vmap_over_chi2(dist: Chi2, df=None):
    dist_axes = vmap_over.dispatch(Gamma)(dist, rate=df, concentration=df)
    dist_axes.df = df
    return dist_axes


@vmap_over.register
def _vmap_over_gaussian_copula(
    dist: GaussianCopula,
    marginal_dist=None,
    correlation_matrix=None,
    correlation_cholesky=None,
):
    dist_axes = _default_vmap_over(
        dist, marginal_dist=marginal_dist, correlation_matrix=correlation_matrix
    )
    dist_axes.base_dist = vmap_over(
        dist.base_dist,
        loc=correlation_matrix if correlation_matrix == 0 else correlation_cholesky,
        scale_tril=correlation_matrix
        if correlation_matrix == 0
        else correlation_cholesky,
        covariance_matrix=correlation_matrix,
    )
    return dist_axes


@vmap_over.register
def _vmap_over_gausian_copula_beta(
    dist: GaussianCopulaBeta,
    concentration1=None,
    concentration0=None,
    correlation_matrix=None,
    correlation_cholesky=None,
):
    d = vmap_over.dispatch(GaussianCopula)(
        dist,
        vmap_over(
            dist.marginal_dist,
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
def _vmap_over_half_cauchy(dist: HalfCauchy, scale=None):
    dist_axes = _default_vmap_over(dist, scale=scale)
    dist_axes._cauchy = vmap_over(dist._cauchy, loc=scale, scale=scale)
    return dist_axes


@vmap_over.register
def _vmap_over_inverse_gamma(dist: InverseGamma, concentration=None, rate=None):
    dist_axes = _default_vmap_over(dist, concentration=concentration, rate=rate)
    dist_axes.base_dist = vmap_over(
        dist.base_dist, concentration=concentration, rate=rate
    )
    dist_axes.transforms = None
    return dist_axes


@vmap_over.register
def _vmap_over_uniform(dist: Uniform, low=None, high=None):
    dist_axes = _default_vmap_over(dist, low=low, high=high)
    dist_axes._support = vmap_over(dist._support, lower_bound=low, upper_bound=high)
    return dist_axes


@vmap_over.register
def _vmap_over_kumaraswamy(dist: Kumaraswamy, concentration0=None, concentration1=None):
    dist_axes = _default_vmap_over(
        dist, concentration0=concentration0, concentration1=concentration1
    )
    dist_axes.concentration0 = concentration0
    dist_axes.concentration1 = concentration1
    return dist_axes


@vmap_over.register
def _vmap_over_lkj(dist: LKJ, concentration=None):
    dist_axes = _default_vmap_over(dist, concentration=concentration)
    dist_axes.base_dist = vmap_over(dist.base_dist, concentration=concentration)
    dist_axes.transforms = None
    return dist_axes


@vmap_over.register
def _vmap_over_lkj_cholesky(dist: LKJCholesky, concentration):
    dist_axes = _default_vmap_over(dist, concentration=concentration)
    if dist_axes.sample_method == "onion":
        dist_axes._beta = vmap_over(
            dist._beta, concentration1=None, concentration0=concentration
        )
    elif dist_axes.sample_method == "cvine":
        dist_axes._beta = vmap_over(
            dist._beta, concentration1=concentration, concentration0=concentration
        )
    return dist_axes


@vmap_over.register
def _vmap_over_lognormal(dist: LogNormal, loc=None, scale=None):
    dist_axes = _default_vmap_over(dist, loc=loc, scale=scale)
    dist_axes.transforms = None
    dist_axes.base_dist = vmap_over(dist.base_dist, loc=loc, scale=scale)
    return dist_axes


@vmap_over.register
def _vmap_over_loguniform(dist: LogUniform, low=None, high=None):
    dist_axes = _default_vmap_over(dist, low=low, high=high)
    dist_axes.base_dist = vmap_over(dist.base_dist, low=low, high=high)
    dist_axes._support = vmap_over(dist._support, lower_bound=low, upper_bound=high)
    return dist_axes


@vmap_over.register
def _vmap_over_car(
    dist: CAR, loc=None, correlation=None, conditional_precision=None, adj_matrix=None
):
    dist_axes = _default_vmap_over(
        dist,
        loc=loc,
        correlation=correlation,
        conditional_precision=conditional_precision,
    )
    if not dist.is_sparse:
        dist_axes.adj_matrix = adj_matrix
        dist_axes.precision_matrix = adj_matrix
    else:
        assert adj_matrix is None
    return dist_axes


@vmap_over.register
def _vmap_over_multivariate_student_t(
    dist: MultivariateStudentT, df=None, loc=None, scale_tril=None
):
    dist_axes = _default_vmap_over(dist, df=df, loc=loc, scale_tril=scale_tril)
    dist_axes._chi2 = vmap_over(dist._chi2, df=df)
    return dist_axes


@vmap_over.register
def _vmap_over_low_rank_multivariate_normal(
    dist: LowRankMultivariateNormal, loc=None, cov_factor=None, cov_diag=None
):
    dist_axes = _default_vmap_over(
        dist, loc=loc, cov_factor=cov_factor, cov_diag=cov_diag
    )
    dist_axes._capacitance_tril = cov_diag if cov_diag is not None else cov_factor
    return dist_axes


@vmap_over.register
def _vmap_over_pareto(dist: Pareto, scale=None, alpha=None):
    dist_axes = _default_vmap_over(dist, scale=scale, alpha=alpha)
    dist_axes.base_dist = vmap_over(dist.base_dist, rate=alpha)
    dist_axes.transforms = [None, vmap_over(dist.transforms[1], loc=None, scale=scale)]
    return dist_axes


@vmap_over.register
def _vmap_over_relaxed_bernoulli_logits(
    dist: RelaxedBernoulliLogits, temperature=None, logits=None
):
    dist_axes = _default_vmap_over(dist, temperature=temperature, logits=logits)
    dist_axes.transforms = None
    dist_axes.base_dist = vmap_over(
        dist.base_dist,
        loc=logits if logits is not None else temperature,
        scale=temperature,
    )
    return dist_axes


@vmap_over.register
def _vmap_over_student_t(dist: StudentT, df=None, loc=None, scale=None):
    dist_axes = _default_vmap_over(dist, df=df, loc=loc, scale=scale)
    dist_axes._chi2 = vmap_over(dist._chi2, df=df)
    return dist_axes


@vmap_over.register
def _vmap_over_two_sided_truncated_distribution(
    dist: TwoSidedTruncatedDistribution, low=None, high=None
):
    dist_axes = _default_vmap_over(dist, low=low, high=high)
    dist_axes.base_dist = None
    dist_axes._support = vmap_over(dist._support, lower_bound=low, upper_bound=high)
    return dist_axes


@vmap_over.register
def _vmap_over_left_truncated_distribution(dist: LeftTruncatedDistribution, low=None):
    dist_axes = _default_vmap_over(dist, low=low)
    dist_axes.base_dist = None
    dist_axes._support = vmap_over(dist._support, lower_bound=low)
    return dist_axes


@vmap_over.register
def _vmap_over_right_truncated_distribution(
    dist: RightTruncatedDistribution, high=None
):
    dist_axes = _default_vmap_over(dist, high=high)
    dist_axes.base_dist = None
    dist_axes._support = vmap_over(dist._support, upper_bound=high)
    return dist_axes


@vmap_over.register
def _vmap_over_beta_binomial(
    dist: BetaBinomial, concentration1=None, concentration0=None, total_count=None
):
    dist_axes = _default_vmap_over(
        dist,
        concentration1=concentration1,
        concentration0=concentration0,
        total_count=total_count,
    )
    dist_axes._beta = vmap_over(
        dist._beta, concentration1=concentration1, concentration0=concentration0
    )
    return dist_axes


@vmap_over.register
def _vmap_over_dirichlet_multinomial(dist: DirichletMultinomial, concentration=None):
    dist_axes = _default_vmap_over(dist, concentration=concentration)
    dist_axes._dirichlet = vmap_over(dist._dirichlet, concentration=concentration)
    return dist_axes


@vmap_over.register
def _vmap_over_gamma_poisson(dist: GammaPoisson, concentration=None, rate=None):
    dist_axes = _default_vmap_over(dist, concentration=concentration, rate=rate)
    dist_axes._gamma = vmap_over(dist._gamma, concentration=concentration, rate=rate)
    return dist_axes


@vmap_over.register
def _vmap_over_negative_binomial_probs(
    dist: NegativeBinomialProbs, total_count=None, probs=None
):
    dist_axes = vmap_over.dispatch(GammaPoisson)(
        dist, concentration=total_count, rate=probs
    )
    dist_axes.total_count = total_count
    dist_axes.probs = probs
    return dist_axes


@vmap_over.register
def _vmap_over_negative_binomial_logits(
    dist: NegativeBinomialLogits, total_count=None, logits=None
):
    dist_axes = vmap_over.dispatch(GammaPoisson)(
        dist, concentration=total_count, rate=logits
    )
    dist_axes.total_count = total_count
    dist_axes.logits = logits
    return dist_axes


@vmap_over.register
def _vmap_over_negative_binomial_2(
    dist: NegativeBinomial2, mean=None, concentration=None
):
    return vmap_over.dispatch(GammaPoisson)(
        dist,
        concentration=concentration,
        rate=concentration if concentration is not None else mean,
    )


@vmap_over.register
def _vmap_over_ordered_logistic(dist: OrderedLogistic, predictor=None, cutpoints=None):
    dist_axes = vmap_over.dispatch(CategoricalProbs)(
        dist, probs=predictor if predictor is not None else cutpoints
    )
    dist_axes.predictor = predictor
    dist_axes.cutpoints = cutpoints
    return dist_axes


@vmap_over.register
def _vmap_over_discrete_uniform(dist: DiscreteUniform, low=None, high=None):
    dist_axes = _default_vmap_over(dist, low=low, high=high)
    dist_axes._support = vmap_over(dist._support, lower_bound=low, upper_bound=high)
    return dist_axes


@vmap_over.register
def _vmap_over_zero_inflated_poisson(dist: ZeroInflatedPoisson, gate=None, rate=None):
    dist_axes = vmap_over.dispatch(ZeroInflatedProbs)(
        dist, base_dist=vmap_over(dist.base_dist, rate=rate), gate=gate
    )
    dist_axes.rate = rate
    return dist_axes


@vmap_over.register
def _vmap_over_half_normal(dist: HalfNormal, scale=None):
    dist_axes = _default_vmap_over(dist, scale=scale)
    dist_axes._normal = vmap_over(dist._normal, loc=scale, scale=scale)
    return dist_axes


@singledispatch
def promote_batch_shape(d: Distribution):
    raise NotImplementedError


@promote_batch_shape.register
def _default_promote_batch_shape(d: Distribution):
    attr_name = list(d.arg_constraints.keys())[0]
    attr_event_dim = d.arg_constraints[attr_name].event_dim
    attr = getattr(d, attr_name)
    resolved_batch_shape = attr.shape[
        : max(0, attr.ndim - d.event_dim - attr_event_dim)
    ]
    new_self = copy.deepcopy(d)
    new_self._batch_shape = resolved_batch_shape
    return new_self


@promote_batch_shape.register
def _promote_batch_shape_expanded(d: ExpandedDistribution):
    orig_delta_batch_shape = d.batch_shape[
        : len(d.batch_shape) - len(d.base_dist.batch_shape)
    ]

    new_self = copy.deepcopy(d)

    # new dimensions coming from a vmap or numpyro scan/enum operation
    promoted_base_dist = promote_batch_shape(new_self.base_dist)
    new_shapes_elems = promoted_base_dist.batch_shape[
        : len(promoted_base_dist.batch_shape) - len(d.base_dist.batch_shape)
    ]

    # The new dimensions are appended in front of the previous ExpandedDistribution
    # batch dimensions. However, these batch dimensions are now present in
    # the base distribution. Thus the dimensions present in the original
    # ExpandedDistribution batch_shape, but not in the original base distribution
    # batch_shape are now intermediate dimensions: to maintain broadcastability,
    # the attribute of the batch distribution are expanded with such intermediate
    # dimensions.
    new_self._batch_shape = (*new_shapes_elems, *d.batch_shape)

    new_self.base_dist._batch_shape = (
        *new_shapes_elems,
        *tuple(1 for _ in orig_delta_batch_shape),
        *d.base_dist.batch_shape,
    )
    new_axes_locs = range(
        len(new_shapes_elems),
        len(new_shapes_elems) + len(orig_delta_batch_shape),
    )
    new_base_dist = jax.tree.map(
        lambda x: jnp.expand_dims(x, axis=new_axes_locs), new_self.base_dist
    )

    new_self.base_dist = new_base_dist
    return new_self


@promote_batch_shape.register
def _promote_batch_shape_masked(d: MaskedDistribution):
    new_self = copy.copy(d)
    new_base_dist = promote_batch_shape(d.base_dist)
    new_self._batch_shape = new_base_dist.batch_shape
    new_self.base_dist = new_base_dist
    return new_self


@promote_batch_shape.register
def _promote_batch_shape_independent(d: Independent):
    new_self = copy.copy(d)
    new_base_dist = promote_batch_shape(d.base_dist)
    new_self._batch_shape = new_base_dist.batch_shape[: -d.event_dim]
    new_self.base_dist = new_base_dist
    return new_self


@promote_batch_shape.register
def _promote_batch_shape_unit(d: Unit):
    return d
