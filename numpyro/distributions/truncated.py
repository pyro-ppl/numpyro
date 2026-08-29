# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0


import math
from typing import Callable, Optional, Union, cast

import jax
from jax import Array, lax
import jax.numpy as jnp
import jax.random as random
from jax.scipy.special import gammainc, gammaincc, logsumexp
from jax.typing import ArrayLike

from numpyro._typing import NumLike
from numpyro.distributions import constraints
from numpyro.distributions.constraints import Constraint
from numpyro.distributions.continuous import (
    Cauchy,
    Gamma,
    Laplace,
    Logistic,
    Normal,
    SoftLaplace,
    StudentT,
)
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import (
    clamp_probs,
    lazy_property,
    promote_shapes,
    validate_sample,
)
from numpyro.util import is_prng_key


class LeftTruncatedDistribution(Distribution):
    arg_constraints = {"low": constraints.real}
    reparametrized_params = ["low"]
    supported_types = (Cauchy, Laplace, Logistic, Normal, SoftLaplace, StudentT)
    pytree_data_fields = ("base_dist", "low", "_support")
    _support: constraints.Constraint

    def __init__(
        self,
        base_dist: Union[Cauchy, Laplace, Logistic, Normal, SoftLaplace, StudentT],
        low: ArrayLike = 0.0,
        *,
        validate_args: Optional[bool] = None,
    ):
        assert isinstance(base_dist, self.supported_types), (
            f"{type(base_dist).__name__} is not supported by this class; for a Gamma "
            "base distribution use numpyro.distributions.TruncatedGamma."
        )
        assert base_dist.support is constraints.real, (
            "The base distribution should be univariate and have real support."
        )
        batch_shape = lax.broadcast_shapes(base_dist.batch_shape, jnp.shape(low))
        self.base_dist: Union[
            Cauchy, Laplace, Logistic, Normal, SoftLaplace, StudentT
        ] = jax.tree.map(lambda p: promote_shapes(p, shape=batch_shape)[0], base_dist)
        (self.low,) = promote_shapes(low, shape=batch_shape)
        self._support = constraints.greater_than_eq(cast(NumLike, low))
        super().__init__(batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self) -> Constraint:
        return self._support

    @lazy_property
    def _tail_prob_at_low(self):
        # if low < loc, returns cdf(low); otherwise returns 1 - cdf(low)
        loc = self.base_dist.loc
        sign = jnp.where(loc >= self.low, 1.0, -1.0)
        return self.base_dist.cdf(loc - sign * (loc - self.low))

    @lazy_property
    def _tail_prob_at_high(self):
        # if low < loc, returns cdf(high) = 1; otherwise returns 1 - cdf(high) = 0
        return jnp.where(self.low <= self.base_dist.loc, 1.0, 0.0)

    def sample(
        self, key: Optional[jax.Array], sample_shape: tuple[int, ...] = ()
    ) -> Array:
        assert is_prng_key(key)
        assert key is not None
        dtype = jnp.result_type(float)
        finfo = jnp.finfo(dtype)
        minval = finfo.tiny
        u = random.uniform(key, shape=sample_shape + self.batch_shape, minval=minval)
        return self.icdf(u)

    def icdf(self, q: ArrayLike) -> Array:
        loc = self.base_dist.loc
        sign = jnp.where(loc >= self.low, 1.0, -1.0)
        ppf = (1 - sign) * loc + sign * self.base_dist.icdf(
            (1 - q) * self._tail_prob_at_low + q * self._tail_prob_at_high
        )
        return jnp.where(jnp.less(q, 0), jnp.nan, ppf)

    def cdf(self, value: ArrayLike) -> Array:
        # For left truncated distribution: CDF(x) = (F(x) - F(low)) / (1 - F(low))
        # where F is the base distribution CDF
        base_cdf_value = self.base_dist.cdf(value)
        base_cdf_low = self.base_dist.cdf(self.low)

        # Handle the case where value < low (should be 0)
        # and value >= low (should be the truncated CDF)
        truncated_cdf = (base_cdf_value - base_cdf_low) / (1.0 - base_cdf_low)

        # Clamp to [0, 1] and handle values below the truncation point
        result = jnp.where(
            jnp.less(value, self.low), 0.0, jnp.clip(truncated_cdf, 0.0, 1.0)
        )
        return result

    @validate_sample
    def log_prob(self, value: ArrayLike) -> Array:
        sign = jnp.where(self.base_dist.loc >= self.low, 1.0, -1.0)
        return self.base_dist.log_prob(value) - jnp.log(
            sign * (self._tail_prob_at_high - self._tail_prob_at_low)
        )

    @property
    def mean(self) -> Array:
        if isinstance(self.base_dist, Normal):
            low_prob = jnp.exp(self.log_prob(self.low))
            return self.base_dist.loc + low_prob * self.base_dist.scale**2
        elif isinstance(self.base_dist, Cauchy):
            return jnp.full(self.batch_shape, jnp.nan)
        else:
            raise NotImplementedError("mean only available for Normal and Cauchy")

    @property
    def variance(self) -> Array:
        if isinstance(self.base_dist, Normal):
            low_prob = jnp.exp(self.log_prob(self.low))
            return (self.base_dist.scale**2) * (
                1
                + (self.low - self.base_dist.loc) * low_prob
                - (low_prob * self.base_dist.scale) ** 2
            )
        elif isinstance(self.base_dist, Cauchy):
            return jnp.full(self.batch_shape, jnp.nan)
        else:
            raise NotImplementedError("variance only available for Normal and Cauchy")


class RightTruncatedDistribution(Distribution):
    arg_constraints = {"high": constraints.real}
    reparametrized_params = ["high"]
    supported_types = (Cauchy, Laplace, Logistic, Normal, SoftLaplace, StudentT)
    pytree_data_fields = ("base_dist", "high", "_support")
    _support: constraints.Constraint

    def __init__(
        self,
        base_dist: Union[Cauchy, Laplace, Logistic, Normal, SoftLaplace, StudentT],
        high: ArrayLike = 0.0,
        *,
        validate_args: Optional[bool] = None,
    ):
        assert isinstance(base_dist, self.supported_types), (
            f"{type(base_dist).__name__} is not supported by this class; for a Gamma "
            "base distribution use numpyro.distributions.TruncatedGamma."
        )
        assert base_dist.support is constraints.real, (
            "The base distribution should be univariate and have real support."
        )
        batch_shape = lax.broadcast_shapes(base_dist.batch_shape, jnp.shape(high))
        self.base_dist: Union[
            Cauchy, Laplace, Logistic, Normal, SoftLaplace, StudentT
        ] = jax.tree.map(lambda p: promote_shapes(p, shape=batch_shape)[0], base_dist)
        (self.high,) = promote_shapes(high, shape=batch_shape)
        self._support = constraints.less_than_eq(cast(NumLike, high))
        super().__init__(batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self) -> Constraint:
        return self._support

    @lazy_property
    def _cdf_at_high(self) -> Array:
        return self.base_dist.cdf(self.high)

    def sample(
        self, key: Optional[jax.Array], sample_shape: tuple[int, ...] = ()
    ) -> Array:
        assert is_prng_key(key)
        assert key is not None
        dtype = jnp.result_type(float)
        finfo = jnp.finfo(dtype)
        minval = finfo.tiny
        u = random.uniform(key, shape=sample_shape + self.batch_shape, minval=minval)
        return self.icdf(u)

    def icdf(self, q: ArrayLike) -> Array:
        ppf = self.base_dist.icdf(q * self._cdf_at_high)
        return jnp.where(jnp.greater(q, 1), jnp.nan, ppf)

    def cdf(self, value: ArrayLike) -> Array:
        # For right truncated distribution: CDF(x) = F(x) / F(high)
        # where F is the base distribution CDF
        base_cdf_value = self.base_dist.cdf(value)
        base_cdf_high = self._cdf_at_high

        # Handle the case where value > high (should be 1)
        # and value <= high (should be the truncated CDF)
        truncated_cdf = base_cdf_value / base_cdf_high

        # Clamp to [0, 1] and handle values above the truncation point
        result = jnp.where(
            jnp.greater(value, self.high), 1.0, jnp.clip(truncated_cdf, 0.0, 1.0)
        )
        return result

    @validate_sample
    def log_prob(self, value: ArrayLike) -> Array:
        return self.base_dist.log_prob(value) - jnp.log(self._cdf_at_high)

    @property
    def mean(self) -> Array:
        if isinstance(self.base_dist, Normal):
            high_prob = jnp.exp(self.log_prob(self.high))
            return self.base_dist.loc - high_prob * self.base_dist.scale**2
        elif isinstance(self.base_dist, Cauchy):
            return jnp.full(self.batch_shape, jnp.nan)
        else:
            raise NotImplementedError("mean only available for Normal and Cauchy")

    @property
    def variance(self) -> Array:
        if isinstance(self.base_dist, Normal):
            high_prob = jnp.exp(self.log_prob(self.high))
            return (self.base_dist.scale**2) * (
                1
                - (self.high - self.base_dist.loc) * high_prob
                - (high_prob * self.base_dist.scale) ** 2
            )
        elif isinstance(self.base_dist, Cauchy):
            return jnp.full(self.batch_shape, jnp.nan)
        else:
            raise NotImplementedError("variance only available for Normal and Cauchy")


class TwoSidedTruncatedDistribution(Distribution):
    arg_constraints = {
        "low": constraints.dependent(is_discrete=False, event_dim=0),
        "high": constraints.dependent(is_discrete=False, event_dim=0),
    }
    reparametrized_params = ["low", "high"]
    supported_types = (Cauchy, Laplace, Logistic, Normal, SoftLaplace, StudentT)
    pytree_data_fields = ("base_dist", "low", "high", "_support")
    _support: constraints.Constraint

    def __init__(
        self,
        base_dist: Union[Cauchy, Laplace, Logistic, Normal, SoftLaplace, StudentT],
        low: ArrayLike = 0.0,
        high: ArrayLike = 1.0,
        *,
        validate_args: Optional[bool] = None,
    ):
        assert isinstance(base_dist, self.supported_types), (
            f"{type(base_dist).__name__} is not supported by this class; for a Gamma "
            "base distribution use numpyro.distributions.TruncatedGamma."
        )
        assert base_dist.support is constraints.real, (
            "The base distribution should be univariate and have real support."
        )
        batch_shape = lax.broadcast_shapes(
            base_dist.batch_shape, jnp.shape(low), jnp.shape(high)
        )
        self.base_dist: Union[
            Cauchy, Laplace, Logistic, Normal, SoftLaplace, StudentT
        ] = jax.tree.map(lambda p: promote_shapes(p, shape=batch_shape)[0], base_dist)
        (self.low,) = promote_shapes(low, shape=batch_shape)
        (self.high,) = promote_shapes(high, shape=batch_shape)
        self._support = constraints.interval(cast(NumLike, low), cast(NumLike, high))
        super().__init__(batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self) -> Constraint:
        return self._support

    @lazy_property
    def _tail_prob_at_low(self) -> Array:
        # if low < loc, returns cdf(low); otherwise returns 1 - cdf(low)
        loc = self.base_dist.loc
        sign = jnp.where(loc >= self.low, 1.0, -1.0)
        return self.base_dist.cdf(loc - sign * (loc - self.low))

    @lazy_property
    def _tail_prob_at_high(self) -> Array:
        # if low < loc, returns cdf(high); otherwise returns 1 - cdf(high)
        loc = self.base_dist.loc
        sign = jnp.where(loc >= self.low, 1.0, -1.0)
        return self.base_dist.cdf(loc - sign * (loc - self.high))

    @lazy_property
    def _log_diff_tail_probs(self) -> Array:
        # use log_cdf method, if available, to avoid inf's in log_prob
        # fall back to cdf, if log_cdf not available
        log_cdf = getattr(self.base_dist, "log_cdf", None)
        if callable(log_cdf):
            return logsumexp(
                a=jnp.stack([log_cdf(self.high), log_cdf(self.low)], axis=-1),
                axis=-1,
                b=jnp.array([1, -1]),  # subtract low from high
            )

        else:
            loc = self.base_dist.loc
            sign = jnp.where(loc >= self.low, 1.0, -1.0)
            return jnp.log(sign * (self._tail_prob_at_high - self._tail_prob_at_low))

    def sample(
        self, key: Optional[jax.Array], sample_shape: tuple[int, ...] = ()
    ) -> Array:
        assert is_prng_key(key)
        assert key is not None
        dtype = jnp.result_type(float)
        finfo = jnp.finfo(dtype)
        minval = finfo.tiny
        u = random.uniform(key, shape=sample_shape + self.batch_shape, minval=minval)
        return self.icdf(u)

    def icdf(self, q: ArrayLike) -> Array:
        # NB: we use a more numerically stable formula for a symmetric base distribution
        #   A = icdf(cdf(low) + (cdf(high) - cdf(low)) * q) = icdf[(1 - q) * cdf(low) + q * cdf(high)]
        # will suffer by precision issues when low is large;
        # If low < loc:
        #   A = icdf[(1 - q) * cdf(low) + q * cdf(high)]
        # Else
        #   A = 2 * loc - icdf[(1 - q) * cdf(2*loc-low)) + q * cdf(2*loc - high)]
        loc = self.base_dist.loc
        sign = jnp.where(loc >= self.low, 1.0, -1.0)
        ppf = (1 - sign) * loc + sign * self.base_dist.icdf(
            clamp_probs((1 - q) * self._tail_prob_at_low + q * self._tail_prob_at_high)
        )
        return jnp.where(
            jnp.logical_or(jnp.less(q, 0), jnp.greater(q, 1)), jnp.nan, ppf
        )

    def cdf(self, value: ArrayLike) -> Array:
        # For two-sided truncated distribution: CDF(x) = (F(x) - F(low)) / (F(high) - F(low))
        # where F is the base distribution CDF
        base_cdf_value = self.base_dist.cdf(value)
        base_cdf_low = self.base_dist.cdf(self.low)
        base_cdf_high = self.base_dist.cdf(self.high)

        # Calculate the normalization constant (F(high) - F(low))
        normalization = base_cdf_high - base_cdf_low

        # Calculate the truncated CDF
        truncated_cdf = (base_cdf_value - base_cdf_low) / normalization

        # Handle values outside the truncation interval
        result = jnp.where(
            jnp.less(value, self.low),
            0.0,
            jnp.where(
                jnp.greater(value, self.high), 1.0, jnp.clip(truncated_cdf, 0.0, 1.0)
            ),
        )
        return result

    @validate_sample
    def log_prob(self, value: ArrayLike) -> Array:
        # NB: we use a more numerically stable formula for a symmetric base distribution
        # if low < loc
        #   cdf(high) - cdf(low) = as-is
        # if low > loc
        #   cdf(high) - cdf(low) = cdf(2 * loc - low) - cdf(2 * loc - high)
        return self.base_dist.log_prob(value) - self._log_diff_tail_probs

    @property
    def mean(self) -> Array:
        if isinstance(self.base_dist, Normal):
            low_prob = jnp.exp(self.log_prob(self.low))
            high_prob = jnp.exp(self.log_prob(self.high))
            return self.base_dist.loc + (low_prob - high_prob) * self.base_dist.scale**2
        elif isinstance(self.base_dist, Cauchy):
            return jnp.full(self.batch_shape, jnp.nan)
        else:
            raise NotImplementedError("mean only available for Normal and Cauchy")

    @property
    def variance(self) -> Array:
        if isinstance(self.base_dist, Normal):
            low_prob = jnp.exp(self.log_prob(self.low))
            high_prob = jnp.exp(self.log_prob(self.high))
            return (self.base_dist.scale**2) * (
                1
                + (self.low - self.base_dist.loc) * low_prob
                - (self.high - self.base_dist.loc) * high_prob
                - ((low_prob - high_prob) * self.base_dist.scale) ** 2
            )
        elif isinstance(self.base_dist, Cauchy):
            return jnp.full(self.batch_shape, jnp.nan)
        else:
            raise NotImplementedError("variance only available for Normal and Cauchy")


def TruncatedDistribution(
    base_dist: Union[Cauchy, Laplace, Logistic, Normal, SoftLaplace, StudentT],
    low: Optional[ArrayLike] = None,
    high: Optional[ArrayLike] = None,
    *,
    validate_args: Optional[bool] = None,
):
    """
    A function to generate a truncated distribution.

    :param base_dist: The base distribution to be truncated. This should be a univariate
        distribution. Currently, only the following distributions are supported:
        Cauchy, Laplace, Logistic, Normal, and StudentT.
    :param low: the value which is used to truncate the base distribution from below.
        Setting this parameter to None to not truncate from below.
    :param high: the value which is used to truncate the base distribution from above.
        Setting this parameter to None to not truncate from above.
    """
    if high is None:
        if low is None:
            return base_dist
        else:
            return LeftTruncatedDistribution(
                base_dist, low=low, validate_args=validate_args
            )
    elif low is None:
        return RightTruncatedDistribution(
            base_dist, high=high, validate_args=validate_args
        )
    else:
        return TwoSidedTruncatedDistribution(
            base_dist, low=low, high=high, validate_args=validate_args
        )


def TruncatedCauchy(
    loc: ArrayLike = 0.0,
    scale: ArrayLike = 1.0,
    *,
    low: Optional[ArrayLike] = None,
    high: Optional[ArrayLike] = None,
    validate_args: Optional[bool] = None,
):
    return TruncatedDistribution(
        Cauchy(loc, scale), low=low, high=high, validate_args=validate_args
    )


def TruncatedNormal(
    loc: ArrayLike = 0.0,
    scale: ArrayLike = 1.0,
    *,
    low: Optional[ArrayLike] = None,
    high: Optional[ArrayLike] = None,
    validate_args: Optional[bool] = None,
):
    return TruncatedDistribution(
        Normal(loc, scale), low=low, high=high, validate_args=validate_args
    )


def _safe_log_normalizer(normalizer: ArrayLike) -> ArrayLike:
    """Log of a truncation normalizer, kept finite when the normalizer underflows.

    Callers pair this with a ``jnp.where`` selecting ``-inf`` wherever the normalizer
    is zero; taking the log of the clamped value keeps that branch's gradient finite
    rather than ``nan``.
    """
    return jnp.log(jnp.where(normalizer > 0.0, normalizer, 1.0))


def _truncated_log_prob(
    base_log_prob: ArrayLike, normalizer: ArrayLike, log_normalizer: ArrayLike
) -> ArrayLike:
    """Renormalized log density, guarding the underflowed case.

    Once the retained probability underflows to zero, ``jnp.log`` hands back ``-inf``
    and the subtraction would make ``log_prob`` ``+inf`` — attracting a gradient-based
    sampler rather than repelling it. Selecting ``-inf`` instead also makes the
    gradient zero there, which is the honest answer for a density that is
    identically zero across the region.

    ``cdf``, ``mean`` and ``variance`` return ``nan`` in the same regime rather than
    ``-inf``: a density that is identically zero still has a well defined log density,
    but conditioning on an event of zero probability leaves its distribution function
    and moments undefined.
    """
    return jnp.where(normalizer > 0.0, base_log_prob - log_normalizer, -jnp.inf)


def _nan_if_degenerate(value: ArrayLike, normalizer: ArrayLike) -> ArrayLike:
    """Mask a moment or probability that is undefined because no mass is retained."""
    return jnp.where(normalizer > 0.0, value, jnp.nan)


def _bisection_steps() -> int:
    """Bisection iterations needed to bracket the root tightly enough for Newton.

    Each step halves the bracket, so ``-log2(eps)`` steps reduce it by the working
    precision's worth of factors. The Newton step in :func:`_icdf_by_bisection`
    supplies the remaining digits, so no margin beyond that is useful — measured
    accuracy is unchanged from 20 steps upward and limited by the cdf itself.
    """
    return int(-math.log2(jnp.finfo(jnp.result_type(float)).eps))


def _bracket_above(
    cdf_fn: Callable, q: ArrayLike, low: ArrayLike, scale: ArrayLike
) -> ArrayLike:
    """Grow an upper bound from ``low`` until the cdf covers ``q``.

    Used for distributions truncated only from below, whose support has no upper end
    to bisect against. The offset doubles each step, so the bound reaches ``2**n``
    scale lengths in ``n`` steps.
    """
    shape = lax.broadcast_shapes(jnp.shape(q), jnp.shape(low), jnp.shape(scale))
    dtype = jnp.result_type(float)
    low = jnp.broadcast_to(low, shape).astype(dtype)
    high = jnp.broadcast_to(low + scale, shape).astype(dtype)

    def body(_, high):
        return jnp.where(cdf_fn(high) < q, low + 2.0 * (high - low), high)

    return lax.fori_loop(0, 64, body, high)


def _icdf_by_bisection(
    cdf_fn: Callable,
    pdf_fn: Callable,
    q: ArrayLike,
    low: ArrayLike,
    high: ArrayLike,
) -> ArrayLike:
    """Invert a monotone cdf on ``[low, high]`` by bisection.

    Used in place of an inverse incomplete gamma: ``tfp.math.igammainv`` and
    ``igammacinv`` return ``+inf`` once the tail probability they are asked to invert
    falls below roughly ``3e-8`` — an algorithm tolerance rather than a dtype limit,
    identical in single and double precision — which puts a draw outside the support
    whenever the retained mass is small. Bisecting the cdf has no such floor, costs a
    fixed number of incomplete gamma evaluations, and differentiates through
    ``cdf_fn``.
    """

    shape = lax.broadcast_shapes(jnp.shape(q), jnp.shape(low), jnp.shape(high))
    dtype = jnp.result_type(float)
    bounds = (
        jnp.broadcast_to(low, shape).astype(dtype),
        jnp.broadcast_to(high, shape).astype(dtype),
    )

    def body(_, bounds):
        lower, upper = bounds
        mid = 0.5 * (lower + upper)
        go_right = cdf_fn(mid) < q
        return (jnp.where(go_right, mid, lower), jnp.where(go_right, upper, mid))

    lower, upper = lax.fori_loop(0, _bisection_steps(), body, bounds)
    x = lax.stop_gradient(0.5 * (lower + upper))
    # Bisection branches on comparisons, so it is piecewise constant in the parameters
    # and carries no derivative. One Newton step taken from the detached root restores
    # the implicit-function derivative, dx/dtheta = -(dF/dtheta) / f(x), without moving
    # the value: the residual is already zero to the bracket's precision.
    density = pdf_fn(x)
    safe_density = jnp.where(density > 0.0, density, 1.0)
    correction = (cdf_fn(x) - q) / safe_density
    return x - jnp.where(density > 0.0, correction, 0.0)


class LeftTruncatedGamma(Distribution):
    r"""A :class:`~numpyro.distributions.continuous.Gamma` distribution truncated
    from below at ``low``.

    .. math::
        f(x \mid \alpha, \lambda, a) = \frac{
            \mathrm{Gamma}(x \mid \alpha, \lambda)
        }{
            Q(\alpha, \lambda a)
        }, \qquad x \geq a,

    where :math:`Q(\alpha, z)` is the regularized upper incomplete gamma function
    (:func:`~jax.scipy.special.gammaincc`). Taking the normalizer from the upper tail
    keeps it accurate however far into the tail ``low`` sits.

    :param base_dist: a :class:`~numpyro.distributions.continuous.Gamma` instance.
    :param low: the value at which the base distribution is truncated from below.

    .. note::
        ``icdf`` inverts the cdf by bisection rather than calling an inverse incomplete
        gamma. The latter saturates below a tail probability of roughly ``3e-8`` — an
        algorithm tolerance, identical in single and double precision — which would
        return draws outside the support whenever the retained mass is small. A Newton
        step from the bracketed root supplies the implicit derivative, so ``icdf`` and
        ``sample`` stay differentiable. The cost is a fixed number of incomplete gamma
        evaluations, roughly three to five times a single inverse; ``log_prob`` is
        unaffected.

        Where the normalizer underflows to zero, ``log_prob`` returns ``-inf`` rather
        than ``+inf`` and ``cdf``, ``mean`` and ``variance`` return ``nan``. Finally,
        ``variance`` is computed as ``E[X^2] - E[X]^2``, which cancels badly in single
        precision when the interval is narrow relative to its distance from the origin
        — ``Gamma(2, 1e-4)`` on ``[100, 101]`` is out by 70% — so enable
        ``jax_enable_x64`` for such configurations.
    """

    arg_constraints = {"low": constraints.greater_than_eq(0.0)}
    reparametrized_params = ["low"]
    supported_types = (Gamma,)
    pytree_data_fields = ("base_dist", "low", "_support")

    def __init__(
        self,
        base_dist: Gamma,
        low: ArrayLike = 0.0,
        *,
        validate_args: Optional[bool] = None,
    ):
        assert isinstance(base_dist, self.supported_types)
        batch_shape = lax.broadcast_shapes(base_dist.batch_shape, jnp.shape(low))
        self.base_dist: Gamma = jax.tree.map(
            lambda p: promote_shapes(p, shape=batch_shape)[0], base_dist
        )
        (self.low,) = promote_shapes(low, shape=batch_shape)
        self._support = constraints.greater_than_eq(low)
        super().__init__(batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self) -> Constraint:
        return self._support

    @lazy_property
    def _normalizer(self) -> ArrayLike:
        return gammaincc(self.base_dist.concentration, self.base_dist.rate * self.low)

    @lazy_property
    def _log_normalizer(self) -> ArrayLike:
        return _safe_log_normalizer(self._normalizer)

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> ArrayLike:
        assert is_prng_key(key)
        dtype = jnp.result_type(float)
        finfo = jnp.finfo(dtype)
        minval = finfo.tiny
        u = random.uniform(key, shape=sample_shape + self.batch_shape, minval=minval)
        return self.icdf(u)

    def icdf(self, q: ArrayLike) -> ArrayLike:
        # The support is unbounded above, so bracket by doubling before bisecting.
        scale = (self.base_dist.concentration + 1.0) / self.base_dist.rate
        high = _bracket_above(self.cdf, q, self.low, scale)
        x = _icdf_by_bisection(self.cdf, self._pdf, q, self.low, high)
        invalid = jnp.logical_or(jnp.logical_or(q < 0, q > 1), self._normalizer <= 0.0)
        x = jnp.where(invalid, jnp.nan, x)
        # q == 1 is legitimately infinite here: the support is unbounded above.
        return jnp.where(q >= 1.0, jnp.inf, x)

    def cdf(self, value: ArrayLike) -> ArrayLike:
        sf = gammaincc(self.base_dist.concentration, self.base_dist.rate * value)
        cdf = jnp.where(
            value < self.low, 0.0, jnp.clip(1 - sf / self._normalizer, 0.0, 1.0)
        )
        return jnp.where(self._normalizer > 0.0, cdf, jnp.nan)

    def _pdf(self, value: ArrayLike) -> ArrayLike:
        """Density without support validation, for the Newton step in ``icdf``."""
        return jnp.exp(
            _truncated_log_prob(
                self.base_dist.log_prob(value), self._normalizer, self._log_normalizer
            )
        )

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        return _truncated_log_prob(
            self.base_dist.log_prob(value), self._normalizer, self._log_normalizer
        )

    @property
    def mean(self) -> ArrayLike:
        concentration, rate = self.base_dist.concentration, self.base_dist.rate
        mean = (concentration / rate) * (
            gammaincc(concentration + 1, rate * self.low) / self._normalizer
        )
        return _nan_if_degenerate(mean, self._normalizer)

    @property
    def variance(self) -> ArrayLike:
        concentration, rate = self.base_dist.concentration, self.base_dist.rate
        second_moment = (concentration * (concentration + 1) / rate**2) * (
            gammaincc(concentration + 2, rate * self.low) / self._normalizer
        )
        return _nan_if_degenerate(second_moment - self.mean**2, self._normalizer)


class RightTruncatedGamma(Distribution):
    r"""A :class:`~numpyro.distributions.continuous.Gamma` distribution truncated
    from above at ``high``.

    .. math::
        f(x \mid \alpha, \lambda, b) = \frac{
            \mathrm{Gamma}(x \mid \alpha, \lambda)
        }{
            P(\alpha, \lambda b)
        }, \qquad 0 < x \leq b,

    where :math:`P(\alpha, z)` is the regularized lower incomplete gamma function
    (:func:`~jax.scipy.special.gammainc`).

    The support is closed at zero, so ``log_prob(0.)`` is ``+inf`` when
    ``concentration < 1``, where the density genuinely diverges. The base
    :class:`~numpyro.distributions.continuous.Gamma` has an open support and masks that
    point instead; matching it would need a half-open interval constraint, which does
    not exist yet (:func:`~numpyro.distributions.constraints.open_interval` would also
    wrongly exclude ``high``).

    :param base_dist: a :class:`~numpyro.distributions.continuous.Gamma` instance.
    :param high: the value at which the base distribution is truncated from above.

    .. note::
        ``icdf`` inverts the cdf by bisection rather than calling an inverse incomplete
        gamma. The latter saturates below a tail probability of roughly ``3e-8`` — an
        algorithm tolerance, identical in single and double precision — which would
        return draws outside the support whenever the retained mass is small. A Newton
        step from the bracketed root supplies the implicit derivative, so ``icdf`` and
        ``sample`` stay differentiable. The cost is a fixed number of incomplete gamma
        evaluations, roughly three to five times a single inverse; ``log_prob`` is
        unaffected.

        Where the normalizer underflows to zero, ``log_prob`` returns ``-inf`` rather
        than ``+inf`` and ``cdf``, ``mean`` and ``variance`` return ``nan``. Finally,
        ``variance`` is computed as ``E[X^2] - E[X]^2``, which cancels badly in single
        precision when the interval is narrow relative to its distance from the origin
        — ``Gamma(2, 1e-4)`` on ``[100, 101]`` is out by 70% — so enable
        ``jax_enable_x64`` for such configurations.
    """

    arg_constraints = {"high": constraints.positive}
    reparametrized_params = ["high"]
    supported_types = (Gamma,)
    pytree_data_fields = ("base_dist", "high", "_support")

    def __init__(
        self,
        base_dist: Gamma,
        high: ArrayLike = 1.0,
        *,
        validate_args: Optional[bool] = None,
    ):
        assert isinstance(base_dist, self.supported_types)
        batch_shape = lax.broadcast_shapes(base_dist.batch_shape, jnp.shape(high))
        self.base_dist: Gamma = jax.tree.map(
            lambda p: promote_shapes(p, shape=batch_shape)[0], base_dist
        )
        (self.high,) = promote_shapes(high, shape=batch_shape)
        self._support = constraints.interval(0.0, high)
        super().__init__(batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self) -> Constraint:
        return self._support

    @lazy_property
    def _normalizer(self) -> ArrayLike:
        return gammainc(self.base_dist.concentration, self.base_dist.rate * self.high)

    @lazy_property
    def _log_normalizer(self) -> ArrayLike:
        return _safe_log_normalizer(self._normalizer)

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> ArrayLike:
        assert is_prng_key(key)
        dtype = jnp.result_type(float)
        finfo = jnp.finfo(dtype)
        minval = finfo.tiny
        u = random.uniform(key, shape=sample_shape + self.batch_shape, minval=minval)
        return self.icdf(u)

    def icdf(self, q: ArrayLike) -> ArrayLike:
        x = _icdf_by_bisection(
            self.cdf, self._pdf, q, jnp.zeros_like(self.high), self.high
        )
        invalid = jnp.logical_or(jnp.logical_or(q < 0, q > 1), self._normalizer <= 0.0)
        return jnp.where(invalid, jnp.nan, x)

    def cdf(self, value: ArrayLike) -> ArrayLike:
        cdf = gammainc(self.base_dist.concentration, self.base_dist.rate * value)
        cdf = jnp.where(
            value > self.high, 1.0, jnp.clip(cdf / self._normalizer, 0.0, 1.0)
        )
        return jnp.where(self._normalizer > 0.0, cdf, jnp.nan)

    def _pdf(self, value: ArrayLike) -> ArrayLike:
        """Density without support validation, for the Newton step in ``icdf``."""
        return jnp.exp(
            _truncated_log_prob(
                self.base_dist.log_prob(value), self._normalizer, self._log_normalizer
            )
        )

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        return _truncated_log_prob(
            self.base_dist.log_prob(value), self._normalizer, self._log_normalizer
        )

    @property
    def mean(self) -> ArrayLike:
        concentration, rate = self.base_dist.concentration, self.base_dist.rate
        mean = (concentration / rate) * (
            gammainc(concentration + 1, rate * self.high) / self._normalizer
        )
        return _nan_if_degenerate(mean, self._normalizer)

    @property
    def variance(self) -> ArrayLike:
        concentration, rate = self.base_dist.concentration, self.base_dist.rate
        second_moment = (concentration * (concentration + 1) / rate**2) * (
            gammainc(concentration + 2, rate * self.high) / self._normalizer
        )
        return _nan_if_degenerate(second_moment - self.mean**2, self._normalizer)


class TwoSidedTruncatedGamma(Distribution):
    r"""A :class:`~numpyro.distributions.continuous.Gamma` distribution truncated to
    the interval ``[low, high]``.

    .. math::
        f(x \mid \alpha, \lambda, a, b) = \frac{
            \mathrm{Gamma}(x \mid \alpha, \lambda)
        }{
            Z(\alpha, \lambda, a, b)
        }, \qquad a \leq x \leq b,

    with normalizer :math:`Z = P(\alpha, \lambda b) - P(\alpha, \lambda a)`.

    Written that way the normalizer loses all of its significant digits when the
    interval sits far out in the right tail, where both terms are within rounding
    distance of one; for ``Gamma(2, 1)`` on ``[30, 40]`` in single precision the
    difference is exactly zero, which would make ``log_prob`` infinite. This class
    therefore uses the equivalent upper-tail form
    :math:`Z = Q(\alpha, \lambda a) - Q(\alpha, \lambda b)` whenever the interval lies
    above the median of the base distribution, and switches ``cdf``, ``mean`` and
    ``variance`` to match.

    When ``low`` is zero the support is closed there, so ``log_prob(0.)`` is ``+inf``
    for ``concentration < 1``; see :class:`RightTruncatedGamma` for why this differs
    from the base :class:`~numpyro.distributions.continuous.Gamma`.

    :param base_dist: a :class:`~numpyro.distributions.continuous.Gamma` instance.
    :param low: the value at which the base distribution is truncated from below.
    :param high: the value at which the base distribution is truncated from above.

    .. note::
        ``icdf`` inverts the cdf by bisection rather than calling an inverse incomplete
        gamma. The latter saturates below a tail probability of roughly ``3e-8`` — an
        algorithm tolerance, identical in single and double precision — which would
        return draws outside the support whenever the retained mass is small. A Newton
        step from the bracketed root supplies the implicit derivative, so ``icdf`` and
        ``sample`` stay differentiable. The cost is a fixed number of incomplete gamma
        evaluations, roughly three to five times a single inverse; ``log_prob`` is
        unaffected.

        Where the normalizer underflows to zero, ``log_prob`` returns ``-inf`` rather
        than ``+inf`` and ``cdf``, ``mean`` and ``variance`` return ``nan``. Finally,
        ``variance`` is computed as ``E[X^2] - E[X]^2``, which cancels badly in single
        precision when the interval is narrow relative to its distance from the origin
        — ``Gamma(2, 1e-4)`` on ``[100, 101]`` is out by 70% — so enable
        ``jax_enable_x64`` for such configurations.
    """

    arg_constraints = {
        "low": constraints.dependent(is_discrete=False, event_dim=0),
        "high": constraints.dependent(is_discrete=False, event_dim=0),
    }
    reparametrized_params = ["low", "high"]
    supported_types = (Gamma,)
    pytree_data_fields = ("base_dist", "low", "high", "_support")

    def __init__(
        self,
        base_dist: Gamma,
        low: ArrayLike = 0.0,
        high: ArrayLike = 1.0,
        *,
        validate_args: Optional[bool] = None,
    ):
        assert isinstance(base_dist, self.supported_types)
        batch_shape = lax.broadcast_shapes(
            base_dist.batch_shape, jnp.shape(low), jnp.shape(high)
        )
        self.base_dist: Gamma = jax.tree.map(
            lambda p: promote_shapes(p, shape=batch_shape)[0], base_dist
        )
        (self.low,) = promote_shapes(low, shape=batch_shape)
        (self.high,) = promote_shapes(high, shape=batch_shape)
        self._support = constraints.interval(low, high)
        super().__init__(batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self) -> Constraint:
        return self._support

    @lazy_property
    def _cdf_at_low(self) -> ArrayLike:
        return gammainc(self.base_dist.concentration, self.base_dist.rate * self.low)

    @lazy_property
    def _cdf_at_high(self) -> ArrayLike:
        return gammainc(self.base_dist.concentration, self.base_dist.rate * self.high)

    @lazy_property
    def _sf_at_low(self) -> ArrayLike:
        return gammaincc(self.base_dist.concentration, self.base_dist.rate * self.low)

    @lazy_property
    def _sf_at_high(self) -> ArrayLike:
        return gammaincc(self.base_dist.concentration, self.base_dist.rate * self.high)

    @lazy_property
    def _use_upper_tail(self) -> ArrayLike:
        # The interval lies above the median of the base distribution, so both lower
        # tail probabilities are close to one and their difference cancels.
        return self._cdf_at_low > 0.5

    @lazy_property
    def _normalizer(self) -> ArrayLike:
        return jnp.where(
            self._use_upper_tail,
            self._sf_at_low - self._sf_at_high,
            self._cdf_at_high - self._cdf_at_low,
        )

    @lazy_property
    def _log_normalizer(self) -> ArrayLike:
        return _safe_log_normalizer(self._normalizer)

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> ArrayLike:
        assert is_prng_key(key)
        dtype = jnp.result_type(float)
        finfo = jnp.finfo(dtype)
        minval = finfo.tiny
        u = random.uniform(key, shape=sample_shape + self.batch_shape, minval=minval)
        return self.icdf(u)

    def icdf(self, q: ArrayLike) -> ArrayLike:
        x = _icdf_by_bisection(self.cdf, self._pdf, q, self.low, self.high)
        invalid = jnp.logical_or(jnp.logical_or(q < 0, q > 1), self._normalizer <= 0.0)
        return jnp.where(invalid, jnp.nan, x)

    def cdf(self, value: ArrayLike) -> ArrayLike:
        concentration, rate = self.base_dist.concentration, self.base_dist.rate
        truncated_cdf = jnp.where(
            self._use_upper_tail,
            (self._sf_at_low - gammaincc(concentration, rate * value))
            / self._normalizer,
            (gammainc(concentration, rate * value) - self._cdf_at_low)
            / self._normalizer,
        )
        cdf = jnp.where(
            value < self.low,
            0.0,
            jnp.where(value > self.high, 1.0, jnp.clip(truncated_cdf, 0.0, 1.0)),
        )
        return jnp.where(self._normalizer > 0.0, cdf, jnp.nan)

    def _pdf(self, value: ArrayLike) -> ArrayLike:
        """Density without support validation, for the Newton step in ``icdf``."""
        return jnp.exp(
            _truncated_log_prob(
                self.base_dist.log_prob(value), self._normalizer, self._log_normalizer
            )
        )

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        return _truncated_log_prob(
            self.base_dist.log_prob(value), self._normalizer, self._log_normalizer
        )

    def _partial_moment(self, order: int) -> ArrayLike:
        """The integral of ``x**order`` over the interval, up to the falling factorial.

        This is the same incomplete gamma difference as the normalizer, but evaluated
        at ``concentration + order``. The tail switch must therefore be decided afresh:
        ``Gamma(alpha + k)`` has a larger median than ``Gamma(alpha)``, so an interval
        sitting above the median of the base distribution can fall deep into the
        *lower* tail at the shifted order, where the upper-tail form is the one that
        cancels. Reusing the order-zero branch there collapses the difference to zero
        and yields a negative variance.
        """
        rate = self.base_dist.rate
        shifted = self.base_dist.concentration + order
        use_upper_tail = gammainc(shifted, rate * self.low) > 0.5
        return jnp.where(
            use_upper_tail,
            gammaincc(shifted, rate * self.low) - gammaincc(shifted, rate * self.high),
            gammainc(shifted, rate * self.high) - gammainc(shifted, rate * self.low),
        )

    @property
    def mean(self) -> ArrayLike:
        concentration, rate = self.base_dist.concentration, self.base_dist.rate
        mean = (concentration / rate) * (self._partial_moment(1) / self._normalizer)
        return _nan_if_degenerate(mean, self._normalizer)

    @property
    def variance(self) -> ArrayLike:
        concentration, rate = self.base_dist.concentration, self.base_dist.rate
        second_moment = (concentration * (concentration + 1) / rate**2) * (
            self._partial_moment(2) / self._normalizer
        )
        return _nan_if_degenerate(second_moment - self.mean**2, self._normalizer)


def TruncatedGamma(
    concentration: ArrayLike = 1.0,
    rate: ArrayLike = 1.0,
    *,
    low: Optional[ArrayLike] = None,
    high: Optional[ArrayLike] = None,
    validate_args: Optional[bool] = None,
):
    """
    A function to generate a truncated gamma distribution.

    :param concentration: concentration parameter of the base
        :class:`~numpyro.distributions.continuous.Gamma` distribution.
    :param rate: rate parameter of the base distribution.
    :param low: the value which is used to truncate the base distribution from
        below. Setting this parameter to None to not truncate from below.
    :param high: the value which is used to truncate the base distribution from
        above. Setting this parameter to None to not truncate from above.

    **Example:**

    .. doctest::

            >>> from jax import numpy as jnp
            >>> from numpyro import distributions as dist
            >>> d = dist.TruncatedGamma(2.0, 1.0, low=0.5, high=3.0)
            >>> log_prob = d.log_prob(jnp.array([1.0, 2.0]))
            >>> lower_bounded = dist.TruncatedGamma(2.0, 1.0, low=0.5)
    """
    base_dist = Gamma(concentration, rate)
    if high is None:
        if low is None:
            return base_dist
        return LeftTruncatedGamma(base_dist, low=low, validate_args=validate_args)
    elif low is None:
        return RightTruncatedGamma(base_dist, high=high, validate_args=validate_args)
    else:
        return TwoSidedTruncatedGamma(
            base_dist, low=low, high=high, validate_args=validate_args
        )


class TruncatedPolyaGamma(Distribution):
    truncation_point = 2.5
    num_log_prob_terms = 7
    num_gamma_variates = 8
    assert num_log_prob_terms % 2 == 1

    arg_constraints = {}
    support = constraints.interval(0.0, truncation_point)

    def __init__(
        self, batch_shape: tuple[int, ...] = (), *, validate_args: Optional[bool] = None
    ):
        super(TruncatedPolyaGamma, self).__init__(
            batch_shape, validate_args=validate_args
        )

    def sample(
        self, key: Optional[jax.Array], sample_shape: tuple[int, ...] = ()
    ) -> Array:
        assert is_prng_key(key)
        assert key is not None
        denom = jnp.square(jnp.arange(0.5, self.num_gamma_variates))
        x = random.gamma(
            key, jnp.ones(self.batch_shape + sample_shape + (self.num_gamma_variates,))
        )
        x = jnp.sum(x / denom, axis=-1)
        return jnp.clip(x * (0.5 / jnp.pi**2), None, self.truncation_point)

    @validate_sample
    def log_prob(self, value: ArrayLike) -> Array:
        value = jnp.expand_dims(value, -1)
        all_indices = jnp.arange(0, self.num_log_prob_terms)
        two_n_plus_one = 2.0 * all_indices + 1.0
        log_terms = (
            jnp.log(two_n_plus_one)
            - 1.5 * jnp.log(value)
            - 0.125 * jnp.square(two_n_plus_one) / value
        )
        even_terms = jnp.take(log_terms, all_indices[::2], axis=-1)
        odd_terms = jnp.take(log_terms, all_indices[1::2], axis=-1)
        sum_even = jnp.exp(logsumexp(even_terms, axis=-1))
        sum_odd = jnp.exp(logsumexp(odd_terms, axis=-1))
        return jnp.log(sum_even - sum_odd) - 0.5 * jnp.log(2.0 * jnp.pi)


class DoublyTruncatedPowerLaw(Distribution):
    r"""Power law distribution with :math:`\alpha` index, and lower and upper bounds.
    We can define the power law distribution as,

    .. math::
        f(x; \alpha, a, b) = \frac{x^{\alpha}}{Z(\alpha, a, b)},

    where, :math:`a` and :math:`b` are the lower and upper bounds respectively,
    and :math:`Z(\alpha, a, b)` is the normalization constant. It is defined as,

    .. math::
        Z(\alpha, a, b) = \begin{cases}
            \log(b) - \log(a) & \text{if } \alpha = -1, \\
            \frac{b^{1 + \alpha} - a^{1 + \alpha}}{1 + \alpha} & \text{otherwise}.
        \end{cases}

    :param alpha: index of the power law distribution
    :param low: lower bound of the distribution
    :param high: upper bound of the distribution
    """

    arg_constraints = {
        "alpha": constraints.real,
        "low": constraints.greater_than_eq(0),
        "high": constraints.greater_than(0),
    }
    reparametrized_params = ["alpha", "low", "high"]
    pytree_aux_fields = ("_support",)
    pytree_data_fields = ("alpha", "low", "high")
    _support: constraints.Constraint

    def __init__(
        self,
        alpha: ArrayLike,
        low: ArrayLike,
        high: ArrayLike,
        *,
        validate_args: Optional[bool] = None,
    ):
        self.alpha, self.low, self.high = promote_shapes(alpha, low, high)
        self._support = constraints.interval(cast(NumLike, low), cast(NumLike, high))
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha), jnp.shape(low), jnp.shape(high)
        )
        super(DoublyTruncatedPowerLaw, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self) -> Constraint:
        return self._support

    @validate_sample
    def log_prob(self, value: ArrayLike) -> Array:
        r"""Logarithmic probability distribution:

        Z inequal minus one:

        .. math::

            \frac{(\alpha + 1)x^\alpha}{b^{\alpha + 1} - a^{\alpha + 1}}

        Z equal minus one:

        .. math::

            \frac{x^\alpha}{\log(b) - \log(a)}

        Derivations are calculated by Wolfram Alpha via the Jacobian matrix accordingly.
        """

        @jax.custom_jvp
        def f(x: ArrayLike, alpha: ArrayLike, low: ArrayLike, high: ArrayLike) -> Array:
            neq_neg1_mask = jnp.not_equal(alpha, -1.0)
            neq_neg1_alpha = jnp.where(neq_neg1_mask, alpha, 0.0)
            # eq_neg1_alpha = jnp.where(~neq_neg1_mask, alpha, -1.0)

            def neq_neg1_fn() -> Array:
                one_more_alpha = 1.0 + neq_neg1_alpha
                return jnp.log(
                    jnp.power(x, neq_neg1_alpha)
                    * (one_more_alpha)
                    / (jnp.power(high, one_more_alpha) - jnp.power(low, one_more_alpha))
                )

            def eq_neg1_fn() -> Array:
                return -jnp.log(x) - jnp.log(jnp.log(high) - jnp.log(low))

            return jnp.where(neq_neg1_mask, neq_neg1_fn(), eq_neg1_fn())

        @f.defjvp
        def f_jvp(
            primals: tuple[Array, Array, Array, Array],
            tangents: tuple[Array, Array, Array, Array],
        ) -> tuple[Array, Array]:
            x, alpha, low, high = primals
            x_t, alpha_t, low_t, high_t = tangents

            log_low = jnp.log(low)
            log_high = jnp.log(high)
            log_x = jnp.log(x)

            # Mask and alpha values
            delta_eq_neg1 = 10e-4
            neq_neg1_mask = jnp.not_equal(alpha, -1.0)
            neq_neg1_alpha = jnp.where(neq_neg1_mask, alpha, 0.0)
            eq_neg1_alpha = jnp.where(jnp.not_equal(alpha, 0.0), alpha, -1.0)

            primal_out = f(*primals)

            # Alpha tangent with approximation
            # Variable part for all values alpha unequal -1
            def alpha_tangent_variable(alpha: ArrayLike) -> Array:
                one_more_alpha = 1.0 + alpha
                low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
                high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
                return jnp.reciprocal(one_more_alpha) + (
                    low_pow_one_more_alpha * log_low
                    - high_pow_one_more_alpha * log_high
                ) / (high_pow_one_more_alpha - low_pow_one_more_alpha)

            # Alpha tangent
            alpha_tangent = jnp.where(
                neq_neg1_mask,
                log_x + alpha_tangent_variable(neq_neg1_alpha),
                # Approximate derivative with right and lefthand approximation
                log_x
                + (
                    alpha_tangent_variable(alpha - delta_eq_neg1)
                    + alpha_tangent_variable(alpha + delta_eq_neg1)
                )
                * 0.5,
            )

            # High and low tangents for alpha unequal -1
            one_more_alpha = 1.0 + neq_neg1_alpha
            low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
            high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
            change_sq = jnp.square(high_pow_one_more_alpha - low_pow_one_more_alpha)
            low_tangent_neq_neg1_common = (
                jnp.square(one_more_alpha) * jnp.power(x, neq_neg1_alpha) / change_sq
            )
            low_tangent_neq_neg1 = low_tangent_neq_neg1_common * jnp.power(
                low, neq_neg1_alpha
            )
            high_tangent_neq_neg1 = low_tangent_neq_neg1_common * jnp.power(
                high, neq_neg1_alpha
            )

            # High and low tangents for alpha equal -1
            low_tangent_eq_neg1_common = jnp.power(x, eq_neg1_alpha) / jnp.square(
                log_high - log_low
            )
            low_tangent_eq_neg1 = low_tangent_eq_neg1_common / low
            high_tangent_eq_neg1 = -low_tangent_eq_neg1_common / high

            # High and low tangents
            low_tangent = jnp.where(
                neq_neg1_mask, low_tangent_neq_neg1, low_tangent_eq_neg1
            )
            high_tangent = jnp.where(
                neq_neg1_mask, high_tangent_neq_neg1, high_tangent_eq_neg1
            )

            # Final tangents
            tangent_out = (
                alpha / x * x_t
                + alpha_tangent * alpha_t
                + low_tangent * low_t
                + high_tangent * high_t
            )
            return primal_out, tangent_out

        return f(value, self.alpha, self.low, self.high)

    def cdf(self, value: ArrayLike) -> Array:
        r"""Cumulated probability distribution:
        Z inequal minus one:

        .. math::

            \frac{x^{\alpha + 1} - a^{\alpha + 1}}{b^{\alpha + 1} - a^{\alpha + 1}}

        Z equal minus one:

        .. math::

            \frac{\log(x) - \log(a)}{\log(b) - \log(a)}

        Derivations are calculated by Wolfram Alpha via the Jacobian matrix accordingly.
        """

        @jax.custom_jvp
        def f(x: ArrayLike, alpha: ArrayLike, low: ArrayLike, high: ArrayLike) -> Array:
            neq_neg1_mask = jnp.not_equal(alpha, -1.0)
            neq_neg1_alpha = jnp.where(neq_neg1_mask, alpha, 0.0)

            def cdf_when_alpha_neq_neg1() -> Array:
                one_more_alpha = 1.0 + neq_neg1_alpha
                low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
                return (jnp.power(x, one_more_alpha) - low_pow_one_more_alpha) / (
                    jnp.power(high, one_more_alpha) - low_pow_one_more_alpha
                )

            def cdf_when_alpha_eq_neg1() -> Array:
                return jnp.log(x / low) / jnp.log(high / low)

            cdf_val = jnp.where(
                neq_neg1_mask,
                cdf_when_alpha_neq_neg1(),
                cdf_when_alpha_eq_neg1(),
            )
            return jnp.clip(cdf_val, 0.0, 1.0)

        @f.defjvp
        def f_jvp(
            primals: tuple[Array, Array, Array, Array],
            tangents: tuple[Array, Array, Array, Array],
        ) -> tuple[Array, Array]:
            x, alpha, low, high = primals
            x_t, alpha_t, low_t, high_t = tangents

            log_low = jnp.log(low)
            log_high = jnp.log(high)
            log_x = jnp.log(x)

            delta_eq_neg1 = 10e-4
            neq_neg1_mask = jnp.not_equal(alpha, -1.0)
            neq_neg1_alpha = jnp.where(neq_neg1_mask, alpha, 0.0)

            # Calculate primal
            primal_out = f(*primals)

            # Tangents for alpha not equals -1
            def x_neq_neg1(alpha: ArrayLike) -> Array:
                one_more_alpha = 1.0 + alpha
                return (one_more_alpha * jnp.power(x, alpha)) / (
                    jnp.power(high, one_more_alpha) - jnp.power(low, one_more_alpha)
                )

            def alpha_neq_neg1(alpha: ArrayLike) -> Array:
                one_more_alpha = 1.0 + alpha
                low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
                high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
                x_pow_one_more_alpha = jnp.power(x, one_more_alpha)
                term1 = (
                    x_pow_one_more_alpha * log_x - low_pow_one_more_alpha * log_low
                ) / (high_pow_one_more_alpha - low_pow_one_more_alpha)
                term2 = (
                    (x_pow_one_more_alpha - low_pow_one_more_alpha)
                    * (
                        high_pow_one_more_alpha * log_high
                        - low_pow_one_more_alpha * log_low
                    )
                ) / jnp.square(high_pow_one_more_alpha - low_pow_one_more_alpha)
                return term1 - term2

            def low_neq_neg1(alpha: ArrayLike) -> Array:
                one_more_alpha = 1.0 + alpha
                low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
                high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
                x_pow_one_more_alpha = jnp.power(x, one_more_alpha)
                change = high_pow_one_more_alpha - low_pow_one_more_alpha
                term2 = one_more_alpha * jnp.power(low, alpha) / change
                term1 = term2 * (x_pow_one_more_alpha - low_pow_one_more_alpha) / change
                return term1 - term2

            def high_neq_neg1(alpha: ArrayLike) -> Array:
                one_more_alpha = 1.0 + alpha
                low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
                high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
                x_pow_one_more_alpha = jnp.power(x, one_more_alpha)
                return -(
                    one_more_alpha
                    * jnp.power(high, alpha)
                    * (x_pow_one_more_alpha - low_pow_one_more_alpha)
                ) / jnp.square(high_pow_one_more_alpha - low_pow_one_more_alpha)

            # Tangents for alpha equals -1
            def x_eq_neg1() -> Array:
                return jnp.reciprocal(x * (log_high - log_low))

            def low_eq_neg1() -> Array:
                return (log_x - log_low) / (
                    jnp.square(log_high - log_low) * low
                ) - jnp.reciprocal((log_high - log_low) * low)

            def high_eq_neg1() -> Array:
                return (log_x - log_low) / (jnp.square(log_high - log_low) * high)

            # Including approximation for alpha = -1
            tangent_out = (
                jnp.where(neq_neg1_mask, x_neq_neg1(neq_neg1_alpha), x_eq_neg1()) * x_t
                + jnp.where(
                    neq_neg1_mask,
                    alpha_neq_neg1(neq_neg1_alpha),
                    (
                        alpha_neq_neg1(alpha - delta_eq_neg1)
                        + alpha_neq_neg1(alpha + delta_eq_neg1)
                    )
                    * 0.5,
                )
                * alpha_t
                + jnp.where(neq_neg1_mask, low_neq_neg1(neq_neg1_alpha), low_eq_neg1())
                * low_t
                + jnp.where(
                    neq_neg1_mask, high_neq_neg1(neq_neg1_alpha), high_eq_neg1()
                )
                * high_t
            )

            return primal_out, tangent_out

        return f(value, self.alpha, self.low, self.high)

    def icdf(self, q: ArrayLike) -> Array:
        r"""Inverse cumulated probability distribution:
        Z inequal minus one:

        .. math::
            a \left(\frac{b}{a}\right)^{q}

        Z equal minus one:

        .. math::
            \left(a^{1 + \alpha} + q (b^{1 + \alpha} - a^{1 + \alpha})\right)^{\frac{1}{1 + \alpha}}

        Derivations are calculated by Wolfram Alpha via the Jacobian matrix accordingly.
        """

        @jax.custom_jvp
        def f(q: ArrayLike, alpha: ArrayLike, low: ArrayLike, high: ArrayLike) -> Array:
            neq_neg1_mask = jnp.not_equal(alpha, -1.0)
            neq_neg1_alpha = jnp.where(neq_neg1_mask, alpha, 0.0)

            def icdf_alpha_neq_neg1() -> Array:
                one_more_alpha = 1.0 + neq_neg1_alpha
                low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
                high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
                return jnp.power(
                    low_pow_one_more_alpha
                    + q * (high_pow_one_more_alpha - low_pow_one_more_alpha),
                    jnp.reciprocal(one_more_alpha),
                )

            def icdf_alpha_eq_neg1() -> Array:
                return jnp.power(high / low, q) * low

            icdf_val = jnp.where(
                neq_neg1_mask,
                icdf_alpha_neq_neg1(),
                icdf_alpha_eq_neg1(),
            )
            return icdf_val

        @f.defjvp
        def f_jvp(
            primals: tuple[Array, Array, Array, Array],
            tangents: tuple[Array, Array, Array, Array],
        ) -> tuple[Array, Array]:
            x, alpha, low, high = primals
            x_t, alpha_t, low_t, high_t = tangents

            log_low = jnp.log(low)
            log_high = jnp.log(high)
            high_over_low = jnp.divide(high, low)

            delta_eq_neg1 = 10e-4
            neq_neg1_mask = jnp.not_equal(alpha, -1.0)
            neq_neg1_alpha = jnp.where(neq_neg1_mask, alpha, 0.0)

            primal_out = f(*primals)

            # Tangents for alpha not equal -1
            def x_neq_neg1(alpha: ArrayLike) -> Array:
                one_more_alpha = 1.0 + alpha
                low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
                high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
                change = high_pow_one_more_alpha - low_pow_one_more_alpha
                return (
                    change
                    * jnp.power(
                        low_pow_one_more_alpha + x * change,
                        jnp.reciprocal(one_more_alpha) - 1,
                    )
                ) / one_more_alpha

            def alpha_neq_neg1(alpha: ArrayLike) -> Array:
                one_more_alpha = 1.0 + alpha
                low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
                high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
                factor0 = low_pow_one_more_alpha + x * (
                    high_pow_one_more_alpha - low_pow_one_more_alpha
                )
                term1 = jnp.power(factor0, jnp.reciprocal(one_more_alpha))
                term2 = (
                    low_pow_one_more_alpha * log_low
                    + x
                    * (
                        high_pow_one_more_alpha * log_high
                        - low_pow_one_more_alpha * log_low
                    )
                ) / (one_more_alpha * factor0)
                term3 = jnp.log(factor0) / jnp.square(one_more_alpha)
                return term1 * (term2 - term3)

            def low_neq_neg1(alpha: ArrayLike) -> Array:
                one_more_alpha = 1.0 + alpha
                low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
                high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
                return (
                    (1.0 - x)
                    * jnp.power(low, alpha)
                    * jnp.power(
                        low_pow_one_more_alpha
                        + x * (high_pow_one_more_alpha - low_pow_one_more_alpha),
                        jnp.reciprocal(one_more_alpha) - 1,
                    )
                )

            def high_neq_neg1(alpha: ArrayLike) -> Array:
                one_more_alpha = 1.0 + alpha
                low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
                high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
                return (
                    x
                    * jnp.power(high, alpha)
                    * jnp.power(
                        low_pow_one_more_alpha
                        + x * (high_pow_one_more_alpha - low_pow_one_more_alpha),
                        jnp.reciprocal(one_more_alpha) - 1,
                    )
                )

            # Tangents for alpha equals -1
            def dx_eq_neg1() -> Array:
                return low * jnp.power(high_over_low, x) * (log_high - log_low)

            def low_eq_neg1() -> Array:
                return (
                    jnp.power(high_over_low, x)
                    - (high * x * jnp.power(high_over_low, x - 1)) / low
                )

            def high_eq_neg1() -> Array:
                return x * jnp.power(high_over_low, x - 1)

            # Including approximation for alpha = -1 \
            tangent_out = (
                jnp.where(neq_neg1_mask, x_neq_neg1(neq_neg1_alpha), dx_eq_neg1()) * x_t
                + jnp.where(
                    neq_neg1_mask,
                    alpha_neq_neg1(neq_neg1_alpha),
                    (
                        alpha_neq_neg1(alpha - delta_eq_neg1)
                        + alpha_neq_neg1(alpha + delta_eq_neg1)
                    )
                    * 0.5,
                )
                * alpha_t
                + jnp.where(neq_neg1_mask, low_neq_neg1(neq_neg1_alpha), low_eq_neg1())
                * low_t
                + jnp.where(
                    neq_neg1_mask, high_neq_neg1(neq_neg1_alpha), high_eq_neg1()
                )
                * high_t
            )

            return primal_out, tangent_out

        return f(q, self.alpha, self.low, self.high)

    def sample(
        self, key: Optional[jax.Array], sample_shape: tuple[int, ...] = ()
    ) -> Array:
        assert is_prng_key(key)
        assert key is not None
        u = random.uniform(key, sample_shape + self.batch_shape)
        samples = self.icdf(u)
        return samples


class LowerTruncatedPowerLaw(Distribution):
    r"""Lower truncated power law distribution with :math:`\alpha` index.
    We can define the power law distribution as,

    .. math::
        f(x; \alpha, a) = (-\alpha-1)a^{-\alpha - 1}x^{-\alpha},
        \qquad x \geq a, \qquad \alpha < -1,

    where, :math:`a` is the lower bound. The cdf of the distribution is given by,

    .. math::
        F(x; \alpha, a) = 1 - \left(\frac{x}{a}\right)^{1+\alpha}.

    The k-th moment of the distribution is given by,

    .. math::
        E[X^k] = \begin{cases}
            \frac{-\alpha-1}{-\alpha-1-k}a^k & \text{if } k < -\alpha-1, \\
            \infty & \text{otherwise}.
        \end{cases}

    :param alpha: index of the power law distribution
    :param low: lower bound of the distribution
    """

    arg_constraints = {
        "alpha": constraints.less_than(-1.0),
        "low": constraints.greater_than(0.0),
    }
    reparametrized_params = ["alpha", "low"]
    pytree_aux_fields = ("_support",)
    _support: constraints.Constraint

    def __init__(
        self, alpha: ArrayLike, low: ArrayLike, *, validate_args: Optional[bool] = None
    ):
        self.alpha, self.low = promote_shapes(alpha, low)
        batch_shape = lax.broadcast_shapes(jnp.shape(alpha), jnp.shape(low))
        self._support = constraints.greater_than(cast(NumLike, low))
        super(LowerTruncatedPowerLaw, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self) -> Constraint:
        return self._support

    @validate_sample
    def log_prob(self, value: ArrayLike) -> Array:
        one_more_alpha = 1.0 + self.alpha
        return (
            self.alpha * jnp.log(value)
            + jnp.log(-one_more_alpha)
            - one_more_alpha * jnp.log(self.low)
        )

    def cdf(self, value: ArrayLike) -> Array:
        cdf_val = jnp.where(
            jnp.less_equal(value, self.low),
            jnp.zeros_like(value),
            1.0 - jnp.power(value / self.low, 1.0 + self.alpha),
        )
        return cdf_val

    def icdf(self, q: ArrayLike) -> Array:
        nan_mask = jnp.logical_or(jnp.isnan(q), jnp.less(q, 0.0))
        nan_mask = jnp.logical_or(nan_mask, jnp.greater(q, 1.0))
        return jnp.where(
            nan_mask,
            jnp.nan,
            self.low * jnp.power(1.0 - q, jnp.reciprocal(1.0 + self.alpha)),
        )

    def sample(
        self, key: Optional[jax.Array], sample_shape: tuple[int, ...] = ()
    ) -> Array:
        assert is_prng_key(key)
        assert key is not None
        u = random.uniform(key, sample_shape + self.batch_shape)
        samples = self.icdf(u)
        return samples
