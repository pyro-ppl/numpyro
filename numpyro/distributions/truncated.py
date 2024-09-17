# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import jax
from jax import lax
import jax.numpy as jnp
import jax.random as random
from jax.scipy.special import logsumexp

from numpyro.distributions import constraints
from numpyro.distributions.continuous import (
    Cauchy,
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

    def __init__(self, base_dist, low=0.0, *, validate_args=None):
        assert isinstance(base_dist, self.supported_types)
        assert (
            base_dist.support is constraints.real
        ), "The base distribution should be univariate and have real support."
        batch_shape = lax.broadcast_shapes(base_dist.batch_shape, jnp.shape(low))
        self.base_dist = jax.tree.map(
            lambda p: promote_shapes(p, shape=batch_shape)[0], base_dist
        )
        (self.low,) = promote_shapes(low, shape=batch_shape)
        self._support = constraints.greater_than(low)
        super().__init__(batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
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

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        dtype = jnp.result_type(float)
        finfo = jnp.finfo(dtype)
        minval = finfo.tiny
        u = random.uniform(key, shape=sample_shape + self.batch_shape, minval=minval)
        loc = self.base_dist.loc
        sign = jnp.where(loc >= self.low, 1.0, -1.0)
        return (1 - sign) * loc + sign * self.base_dist.icdf(
            (1 - u) * self._tail_prob_at_low + u * self._tail_prob_at_high
        )

    @validate_sample
    def log_prob(self, value):
        sign = jnp.where(self.base_dist.loc >= self.low, 1.0, -1.0)
        return self.base_dist.log_prob(value) - jnp.log(
            sign * (self._tail_prob_at_high - self._tail_prob_at_low)
        )

    @property
    def mean(self):
        if isinstance(self.base_dist, Normal):
            low_prob = jnp.exp(self.log_prob(self.low))
            return self.base_dist.loc + low_prob * self.base_dist.scale**2
        elif isinstance(self.base_dist, Cauchy):
            return jnp.full(self.batch_shape, jnp.nan)
        else:
            raise NotImplementedError("mean only available for Normal and Cauchy")

    @property
    def var(self):
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
            raise NotImplementedError("var only available for Normal and Cauchy")


class RightTruncatedDistribution(Distribution):
    arg_constraints = {"high": constraints.real}
    reparametrized_params = ["high"]
    supported_types = (Cauchy, Laplace, Logistic, Normal, SoftLaplace, StudentT)
    pytree_data_fields = ("base_dist", "high", "_support")

    def __init__(self, base_dist, high=0.0, *, validate_args=None):
        assert isinstance(base_dist, self.supported_types)
        assert (
            base_dist.support is constraints.real
        ), "The base distribution should be univariate and have real support."
        batch_shape = lax.broadcast_shapes(base_dist.batch_shape, jnp.shape(high))
        self.base_dist = jax.tree.map(
            lambda p: promote_shapes(p, shape=batch_shape)[0], base_dist
        )
        (self.high,) = promote_shapes(high, shape=batch_shape)
        self._support = constraints.less_than(high)
        super().__init__(batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    @lazy_property
    def _cdf_at_high(self):
        return self.base_dist.cdf(self.high)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        dtype = jnp.result_type(float)
        finfo = jnp.finfo(dtype)
        minval = finfo.tiny
        u = random.uniform(key, shape=sample_shape + self.batch_shape, minval=minval)
        return self.base_dist.icdf(u * self._cdf_at_high)

    @validate_sample
    def log_prob(self, value):
        return self.base_dist.log_prob(value) - jnp.log(self._cdf_at_high)

    @property
    def mean(self):
        if isinstance(self.base_dist, Normal):
            high_prob = jnp.exp(self.log_prob(self.high))
            return self.base_dist.loc - high_prob * self.base_dist.scale**2
        elif isinstance(self.base_dist, Cauchy):
            return jnp.full(self.batch_shape, jnp.nan)
        else:
            raise NotImplementedError("mean only available for Normal and Cauchy")

    @property
    def var(self):
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
            raise NotImplementedError("var only available for Normal and Cauchy")


class TwoSidedTruncatedDistribution(Distribution):
    arg_constraints = {
        "low": constraints.dependent,
        "high": constraints.dependent,
    }
    reparametrized_params = ["low", "high"]
    supported_types = (Cauchy, Laplace, Logistic, Normal, SoftLaplace, StudentT)
    pytree_data_fields = ("base_dist", "low", "high", "_support")

    def __init__(self, base_dist, low=0.0, high=1.0, *, validate_args=None):
        assert isinstance(base_dist, self.supported_types)
        assert (
            base_dist.support is constraints.real
        ), "The base distribution should be univariate and have real support."
        batch_shape = lax.broadcast_shapes(
            base_dist.batch_shape, jnp.shape(low), jnp.shape(high)
        )
        self.base_dist = jax.tree.map(
            lambda p: promote_shapes(p, shape=batch_shape)[0], base_dist
        )
        (self.low,) = promote_shapes(low, shape=batch_shape)
        (self.high,) = promote_shapes(high, shape=batch_shape)
        self._support = constraints.interval(low, high)
        super().__init__(batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    @lazy_property
    def _tail_prob_at_low(self):
        # if low < loc, returns cdf(low); otherwise returns 1 - cdf(low)
        loc = self.base_dist.loc
        sign = jnp.where(loc >= self.low, 1.0, -1.0)
        return self.base_dist.cdf(loc - sign * (loc - self.low))

    @lazy_property
    def _tail_prob_at_high(self):
        # if low < loc, returns cdf(high); otherwise returns 1 - cdf(high)
        loc = self.base_dist.loc
        sign = jnp.where(loc >= self.low, 1.0, -1.0)
        return self.base_dist.cdf(loc - sign * (loc - self.high))

    @lazy_property
    def _log_diff_tail_probs(self):
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

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        dtype = jnp.result_type(float)
        finfo = jnp.finfo(dtype)
        minval = finfo.tiny
        u = random.uniform(key, shape=sample_shape + self.batch_shape, minval=minval)

        # NB: we use a more numerically stable formula for a symmetric base distribution
        #   A = icdf(cdf(low) + (cdf(high) - cdf(low)) * u) = icdf[(1 - u) * cdf(low) + u * cdf(high)]
        # will suffer by precision issues when low is large;
        # If low < loc:
        #   A = icdf[(1 - u) * cdf(low) + u * cdf(high)]
        # Else
        #   A = 2 * loc - icdf[(1 - u) * cdf(2*loc-low)) + u * cdf(2*loc - high)]
        loc = self.base_dist.loc
        sign = jnp.where(loc >= self.low, 1.0, -1.0)
        return (1 - sign) * loc + sign * self.base_dist.icdf(
            clamp_probs((1 - u) * self._tail_prob_at_low + u * self._tail_prob_at_high)
        )

    @validate_sample
    def log_prob(self, value):
        # NB: we use a more numerically stable formula for a symmetric base distribution
        # if low < loc
        #   cdf(high) - cdf(low) = as-is
        # if low > loc
        #   cdf(high) - cdf(low) = cdf(2 * loc - low) - cdf(2 * loc - high)
        return self.base_dist.log_prob(value) - self._log_diff_tail_probs

    @property
    def mean(self):
        if isinstance(self.base_dist, Normal):
            low_prob = jnp.exp(self.log_prob(self.low))
            high_prob = jnp.exp(self.log_prob(self.high))
            return self.base_dist.loc + (low_prob - high_prob) * self.base_dist.scale**2
        elif isinstance(self.base_dist, Cauchy):
            return jnp.full(self.batch_shape, jnp.nan)
        else:
            raise NotImplementedError("mean only available for Normal and Cauchy")

    @property
    def var(self):
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
            raise NotImplementedError("var only available for Normal and Cauchy")


def TruncatedDistribution(base_dist, low=None, high=None, *, validate_args=None):
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


def TruncatedCauchy(loc=0.0, scale=1.0, *, low=None, high=None, validate_args=None):
    return TruncatedDistribution(
        Cauchy(loc, scale), low=low, high=high, validate_args=validate_args
    )


def TruncatedNormal(loc=0.0, scale=1.0, *, low=None, high=None, validate_args=None):
    return TruncatedDistribution(
        Normal(loc, scale), low=low, high=high, validate_args=validate_args
    )


class TruncatedPolyaGamma(Distribution):
    truncation_point = 2.5
    num_log_prob_terms = 7
    num_gamma_variates = 8
    assert num_log_prob_terms % 2 == 1

    arg_constraints = {}
    support = constraints.interval(0.0, truncation_point)

    def __init__(self, batch_shape=(), *, validate_args=None):
        super(TruncatedPolyaGamma, self).__init__(
            batch_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        denom = jnp.square(jnp.arange(0.5, self.num_gamma_variates))
        x = random.gamma(
            key, jnp.ones(self.batch_shape + sample_shape + (self.num_gamma_variates,))
        )
        x = jnp.sum(x / denom, axis=-1)
        return jnp.clip(x * (0.5 / jnp.pi**2), None, self.truncation_point)

    @validate_sample
    def log_prob(self, value):
        value = value[..., None]
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

    def __init__(self, alpha, low, high, *, validate_args=None):
        self.alpha, self.low, self.high = promote_shapes(alpha, low, high)
        self._support = constraints.interval(low, high)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha), jnp.shape(low), jnp.shape(high)
        )
        super(DoublyTruncatedPowerLaw, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    @validate_sample
    def log_prob(self, value):
        r"""Logarithmic probability distribution:
        Z inequal minus one:
        .. math::
            (x^\alpha) (\alpha + 1)/(b^(\alpha + 1) - a^(\alpha + 1))

        Z equal minus one:
        .. math::
            (x^\alpha)/(log(b) - log(a))
        Derivations are calculated by Wolfram Alpha via the Jacobian matrix accordingly.
        """

        @jax.custom_jvp
        def f(x, alpha, low, high):
            neq_neg1_mask = jnp.not_equal(alpha, -1.0)
            neq_neg1_alpha = jnp.where(neq_neg1_mask, alpha, 0.0)
            # eq_neg1_alpha = jnp.where(~neq_neg1_mask, alpha, -1.0)

            def neq_neg1_fn():
                one_more_alpha = 1.0 + neq_neg1_alpha
                return jnp.log(
                    jnp.power(x, neq_neg1_alpha)
                    * (one_more_alpha)
                    / (jnp.power(high, one_more_alpha) - jnp.power(low, one_more_alpha))
                )

            def eq_neg1_fn():
                return -jnp.log(x) - jnp.log(jnp.log(high) - jnp.log(low))

            return jnp.where(neq_neg1_mask, neq_neg1_fn(), eq_neg1_fn())

        @f.defjvp
        def f_jvp(primals, tangents):
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
            def alpha_tangent_variable(alpha):
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
                # Approximate derivate with right an lefthand approximation
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

    def cdf(self, value):
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
        def f(x, alpha, low, high):
            neq_neg1_mask = jnp.not_equal(alpha, -1.0)
            neq_neg1_alpha = jnp.where(neq_neg1_mask, alpha, 0.0)

            def cdf_when_alpha_neq_neg1():
                one_more_alpha = 1.0 + neq_neg1_alpha
                low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
                return (jnp.power(x, one_more_alpha) - low_pow_one_more_alpha) / (
                    jnp.power(high, one_more_alpha) - low_pow_one_more_alpha
                )

            def cdf_when_alpha_eq_neg1():
                return jnp.log(x / low) / jnp.log(high / low)

            cdf_val = jnp.where(
                neq_neg1_mask,
                cdf_when_alpha_neq_neg1(),
                cdf_when_alpha_eq_neg1(),
            )
            return jnp.clip(cdf_val, 0.0, 1.0)

        @f.defjvp
        def f_jvp(primals, tangents):
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
            def x_neq_neg1(alpha):
                one_more_alpha = 1.0 + alpha
                return (one_more_alpha * jnp.power(x, alpha)) / (
                    jnp.power(high, one_more_alpha) - jnp.power(low, one_more_alpha)
                )

            def alpha_neq_neg1(alpha):
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

            def low_neq_neg1(alpha):
                one_more_alpha = 1.0 + alpha
                low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
                high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
                x_pow_one_more_alpha = jnp.power(x, one_more_alpha)
                change = high_pow_one_more_alpha - low_pow_one_more_alpha
                term2 = one_more_alpha * jnp.power(low, alpha) / change
                term1 = term2 * (x_pow_one_more_alpha - low_pow_one_more_alpha) / change
                return term1 - term2

            def high_neq_neg1(alpha):
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
            def x_eq_neg1():
                return jnp.reciprocal(x * (log_high - log_low))

            def low_eq_neg1():
                return (log_x - log_low) / (
                    jnp.square(log_high - log_low) * low
                ) - jnp.reciprocal((log_high - log_low) * low)

            def high_eq_neg1():
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

    def icdf(self, q):
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
        def f(q, alpha, low, high):
            neq_neg1_mask = jnp.not_equal(alpha, -1.0)
            neq_neg1_alpha = jnp.where(neq_neg1_mask, alpha, 0.0)

            def icdf_alpha_neq_neg1():
                one_more_alpha = 1.0 + neq_neg1_alpha
                low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
                high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
                return jnp.power(
                    low_pow_one_more_alpha
                    + q * (high_pow_one_more_alpha - low_pow_one_more_alpha),
                    jnp.reciprocal(one_more_alpha),
                )

            def icdf_alpha_eq_neg1():
                return jnp.power(high / low, q) * low

            icdf_val = jnp.where(
                neq_neg1_mask,
                icdf_alpha_neq_neg1(),
                icdf_alpha_eq_neg1(),
            )
            return icdf_val

        @f.defjvp
        def f_jvp(primals, tangents):
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
            def x_neq_neg1(alpha):
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

            def alpha_neq_neg1(alpha):
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

            def low_neq_neg1(alpha):
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

            def high_neq_neg1(alpha):
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
            def dx_eq_neg1():
                return low * jnp.power(high_over_low, x) * (log_high - log_low)

            def low_eq_neg1():
                return (
                    jnp.power(high_over_low, x)
                    - (high * x * jnp.power(high_over_low, x - 1)) / low
                )

            def high_eq_neg1():
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

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
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

    def __init__(self, alpha, low, *, validate_args=None):
        self.alpha, self.low = promote_shapes(alpha, low)
        batch_shape = lax.broadcast_shapes(jnp.shape(alpha), jnp.shape(low))
        self._support = constraints.greater_than(low)
        super(LowerTruncatedPowerLaw, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    @validate_sample
    def log_prob(self, value):
        one_more_alpha = 1.0 + self.alpha
        return (
            self.alpha * jnp.log(value)
            + jnp.log(-one_more_alpha)
            - one_more_alpha * jnp.log(self.low)
        )

    def cdf(self, value):
        cdf_val = jnp.where(
            jnp.less_equal(value, self.low),
            jnp.zeros_like(value),
            1.0 - jnp.power(value / self.low, 1.0 + self.alpha),
        )
        return cdf_val

    def icdf(self, q):
        nan_mask = jnp.logical_or(jnp.isnan(q), jnp.less(q, 0.0))
        nan_mask = jnp.logical_or(nan_mask, jnp.greater(q, 1.0))
        return jnp.where(
            nan_mask,
            jnp.nan,
            self.low * jnp.power(1.0 - q, jnp.reciprocal(1.0 + self.alpha)),
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        u = random.uniform(key, sample_shape + self.batch_shape)
        samples = self.icdf(u)
        return samples
