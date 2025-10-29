# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

import jax
from jax import lax
import jax.numpy as jnp
from jax.typing import ArrayLike

from numpyro._typing import ConstraintT, DistributionT
from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import (
    promote_shapes,
    validate_sample,
)


class LeftCensoredDistribution(Distribution):
    """
    Distribution wrapper for left-censored outcomes.

    This distribution augments a base distribution with left-censoring,
    so that the likelihood contribution depends on the censoring indicator.

    :param base_dist: Parametric distribution for the *uncensored* values
            (e.g., Exponential, Weibull, LogNormal, Normal, etc.).
            This distribution must implement a ``cdf`` method.
    :type base_dist: numpyro.distributions.Distribution
    :param censored: Censoring indicator per observation:
            0 → value is observed exactly
            1 → observation is left-censored at the reported value
            (true value occurred *on or before* the reported value)
    :type censored: array-like of {0,1}

    .. note::
            The ``log_prob(value)`` method expects ``value`` to be the observed upper bound
            for each observation. The contribution to the log-likelihood is:

                    log f(value)    if censored == 0
                    log F(value)    if censored == 1

            where f is the density and F the cumulative distribution function of ``base_dist``.

            This is commonly used in survival analysis, where event times are positive,
            but the approach is more general and can be applied to any distribution
            with a cumulative distribution function, regardless of support.

            In R's ``survival`` package notation, this corresponds to
            ``Surv(time, event, type = 'left')``.

            Example:

                    Surv(time = c(2, 4, 6), event = c(0, 1, 0), type='left')

            means:

                    subject 1 had an event exactly at t=2
                    subject 2 had an event before or at t=4 (left-censored)
                    subject 3 had an event exactly at t=6

    **Example:**

    .. doctest::

            >>> from jax import numpy as jnp
            >>> from numpyro import distributions as dist
            >>> base = dist.LogNormal(0., 1.)
            >>> surv_dist = dist.LeftCensoredDistribution(base, censored=jnp.array([0, 1, 1]))
            >>> loglik = surv_dist.log_prob(jnp.array([2., 4., 6.]))
    """

    arg_constraints = {"censored": constraints.boolean}
    pytree_data_fields = ("base_dist", "censored", "_support")

    def __init__(
        self,
        base_dist: DistributionT,
        censored: ArrayLike = False,
        *,
        validate_args: Optional[bool] = None,
    ):
        # test if base_dist has an implemented cdf method
        if not hasattr(base_dist, "cdf"):
            raise TypeError(
                f"{type(base_dist).__name__} does not have a 'cdf' method. "
                "Censored distributions require a base distribution with an "
                "implemented cumulative distribution function."
            )

        # Optionally test that cdf actually works (in validate_args mode)
        if validate_args:
            try:
                test_val = base_dist.support.feasible_like(jnp.array(0.0))
                _ = base_dist.cdf(test_val)
            except (NotImplementedError, AttributeError) as e:
                raise TypeError(
                    f"{type(base_dist).__name__}.cdf() is not properly implemented."
                ) from e
        batch_shape = lax.broadcast_shapes(base_dist.batch_shape, jnp.shape(censored))
        self.base_dist = jax.tree.map(
            lambda p: promote_shapes(p, shape=batch_shape)[0], base_dist
        )
        self.censored = jnp.array(
            promote_shapes(censored, shape=batch_shape)[0], dtype=jnp.bool
        )
        self._support = base_dist.support
        super().__init__(batch_shape, validate_args=validate_args)

    def sample(
        self, key: Optional[jax.dtypes.prng_key], sample_shape: tuple[int, ...] = ()
    ) -> ArrayLike:
        return self.base_dist.expand(self.batch_shape).sample(key, sample_shape)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self) -> ConstraintT:
        return self._support

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        dtype = jnp.result_type(value, float)
        minval = 100.0 * jnp.finfo(dtype).tiny

        def log_cdf_censored(x):
            # log(F(x)) with stability
            return jnp.log(jnp.clip(self.base_dist.cdf(x), minval, 1.0))

        return jnp.where(
            self.censored,
            log_cdf_censored(value),  # left-censored observations: log F(t)
            self.base_dist.log_prob(value),  # observed values: log f(t)
        )


class RightCensoredDistribution(Distribution):
    """
    Distribution wrapper for right-censored outcomes.

    This distribution augments a base distribution with right-censoring,
    so that the likelihood contribution depends on the censoring indicator.

    :param base_dist: Parametric distribution for the *uncensored* values
            (e.g., Exponential, Weibull, LogNormal, Normal, etc.).
            This distribution must implement a ``cdf`` method.
    :type base_dist: numpyro.distributions.Distribution
    :param censored: Censoring indicator per observation:
            0 → value is observed exactly
            1 → observation is right-censored at the reported value
            (true value occurred *on or after* the reported value)
    :type censored: array-like of {0,1}

    .. note::
            The ``log_prob(value)`` method expects ``value`` to be the observed lower bound
            for each observation. The contribution to the log-likelihood is:

                    log f(value)    if censored == 0
                    log (1 - F(value))    if censored == 1

            where f is the density and F the cumulative distribution function of ``base_dist``.

            This is commonly used in survival analysis, where event times are positive,
            but the approach is more general and can be applied to any distribution
            with a cumulative distribution function, regardless of support.

            In R's ``survival`` package notation, this corresponds to
            ``Surv(time, event, type = 'right')``.

            Example:

                    Surv(time = c(5, 8, 10), event = c(1, 0, 1))

            means:

                    subject 1 had an event at t=5
                    subject 2 was censored at t=8
                    subject 3 had an event at t=10

    **Example:**

    .. doctest::

            >>> from jax import numpy as jnp
            >>> from numpyro import distributions as dist
            >>> base = dist.Exponential(rate=0.1)
            >>> surv_dist = dist.RightCensoredDistribution(base, censored=jnp.array([0, 1, 0]))
            >>> loglik = surv_dist.log_prob(jnp.array([5., 8., 10.]))
    """

    arg_constraints = {"censored": constraints.boolean}
    pytree_data_fields = ("base_dist", "censored", "_support")

    def __init__(
        self,
        base_dist: DistributionT,
        censored: ArrayLike = False,
        *,
        validate_args: Optional[bool] = None,
    ):
        # test if base_dist has an implemented cdf method
        if not hasattr(base_dist, "cdf"):
            raise TypeError(
                f"{type(base_dist).__name__} does not have a 'cdf' method. "
                "Censored distributions require a base distribution with an "
                "implemented cumulative distribution function."
            )

        # Optionally test that cdf actually works (in validate_args mode)
        if validate_args:
            try:
                test_val = base_dist.support.feasible_like(jnp.array(0.0))
                _ = base_dist.cdf(test_val)
            except (NotImplementedError, AttributeError) as e:
                raise TypeError(
                    f"{type(base_dist).__name__}.cdf() is not properly implemented."
                ) from e
        batch_shape = lax.broadcast_shapes(base_dist.batch_shape, jnp.shape(censored))
        self.base_dist = jax.tree.map(
            lambda p: promote_shapes(p, shape=batch_shape)[0], base_dist
        )
        self.censored = jnp.array(
            promote_shapes(censored, shape=batch_shape)[0], dtype=jnp.bool
        )
        self._support = base_dist.support
        super().__init__(batch_shape, validate_args=validate_args)

    def sample(
        self, key: Optional[jax.dtypes.prng_key], sample_shape: tuple[int, ...] = ()
    ) -> ArrayLike:
        return self.base_dist.expand(self.batch_shape).sample(key, sample_shape)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self) -> ConstraintT:
        return self._support

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        dtype = jnp.result_type(value, float)
        eps = jnp.finfo(dtype).eps

        def log_survival_censored(x):
            # log(1 - F(x)) with stability
            Fx = jnp.clip(self.base_dist.cdf(x), 0.0, 1 - eps)
            return jnp.log1p(-Fx)

        return jnp.where(
            self.censored,
            log_survival_censored(value),  # censored observations: log S(t)
            self.base_dist.log_prob(value),  # observed values: log f(t)
        )


class IntervalCensoredDistribution(Distribution):
    r"""
    Distribution wrapper for interval-censored outcomes.

    This distribution augments a base distribution with interval censoring,
    so that the likelihood contribution depends on whether the observation is
    left-censored, right-censored, interval-censored, or doubly-censored
    (meaning value is known to lie outside of the interval).

    Parameters
    ----------
    base_dist : numpyro.distributions.Distribution
        Parametric distribution for the *uncensored* values
        (e.g., Exponential, Weibull, LogNormal, Normal, etc.).
        This distribution must implement a `cdf` method.
    left_censored : array-like of {0,1}
        Indicator per observation:
        - 1 → observation is left-censored at the reported upper bound
        - 0 → not left-censored
    right_censored : array-like of {0,1}
        Indicator per observation:
        - 1 → observation is right-censored at the reported lower bound
        - 0 → not right-censored

    Notes
    -----
    - The `log_prob(value)` method expects `value` to be a two-dimensional array
      of shape `(batch_size, 2)`, where each row is `(lower, upper)`:

        * If `left_censored[i] == 1`:
            Interval = (-inf, upper] → left-censored at `upper`
            Contribution = log F(upper)

        * If `right_censored[i] == 1`:
            Interval = (lower, inf) → right-censored at `lower`
            Contribution = log(1 - F(lower))

        * If both `left_censored[i] == 0` and `right_censored[i] == 0`:
            Interval = (lower, upper] → event occurred within the interval
            Contribution = log(F(upper) - F(lower))

        * If both `left_censored[i] == 1` and `right_censored[i] == 1`:
            Interval = (-inf, lower] | [upper, inf) → event occurred outside of the interval
            Contribution = log(1 - (F(upper) - F(lower)))

        * If `jnp.isclose(lower, upper)`:
            Contribution = log F(upper) (i.e., the CDF at that point)

    where f is the density and F the cumulative distribution function of `base_dist`.

    - This approach is commonly used in survival analysis, where event times are positive,
      but it is more general and can be applied to any distribution with a cumulative
      distribution function, regardless of support.

    - This matches the semantics of R’s **survival** package with
    `Surv(l, r, type = "interval2")`.

        Example:
        `Surv(l = c(2, 4, 6), r = c(5, Inf, 9), type="interval2")`
        means:
        * subject 1: event occurred in (2, 5]
        * subject 2: event interval (2,5]
        * subject 3: event occurred in (6, 9]

    Examples
    --------
    >>> base = dist.Weibull(concentration=2.0, scale=3.0)
    >>> left_censored = jnp.array([0, 0, 0])
    >>> right_censored = jnp.array([0, 1, 0])
    >>> surv_dist = IntervalCensoredDistribution(base, left_censored=left_censored, right_censored=right_censored)
    >>> # Three observations: interval-censored, right-censored, interval-censored
    >>> values = jnp.array([
    ...     [2.0, 5.0],   # left-censored at 4
    ...     [4.0, jnp.inf], # right-censored at 4
    ...     [6.0, 9.0],   # interval (6,9]
    ... ])
    >>> loglik = surv_dist.log_prob(values)
    # loglik[0] = log (F(5) - F(2))
    # loglik[1] = log (1 - F(4))
    # loglik[2] = log (F(9) - F(6))
    """

    arg_constraints = {
        "left_censored": constraints.boolean,
        "right_censored": constraints.boolean,
    }
    pytree_data_fields = ("base_dist", "left_censored", "right_censored", "_support")

    def __init__(
        self,
        base_dist: DistributionT,
        left_censored: ArrayLike,
        right_censored: ArrayLike,
        *,
        validate_args: Optional[bool] = None,
    ):
        # Optionally test that cdf actually works (in validate_args mode)
        if validate_args:
            try:
                test_val = base_dist.support.feasible_like(jnp.array(0.0))
                _ = base_dist.cdf(test_val)
            except (NotImplementedError, AttributeError) as e:
                raise TypeError(
                    f"{type(base_dist).__name__}.cdf() is not properly implemented."
                ) from e
        batch_shape = lax.broadcast_shapes(
            base_dist.batch_shape, jnp.shape(left_censored), jnp.shape(right_censored)
        )
        self.base_dist = jax.tree.map(
            lambda p: promote_shapes(p, shape=batch_shape)[0], base_dist
        )
        self.left_censored = jnp.array(
            promote_shapes(left_censored, shape=batch_shape)[0], dtype=jnp.bool
        )
        self.right_censored = jnp.array(
            promote_shapes(right_censored, shape=batch_shape)[0], dtype=jnp.bool
        )
        self._support = base_dist.support
        super().__init__(batch_shape, event_shape=(2,), validate_args=validate_args)

    def sample(
        self, key: Optional[jax.dtypes.prng_key], sample_shape: tuple[int, ...] = ()
    ) -> ArrayLike:
        return self.base_dist.expand(self.batch_shape).sample(key, sample_shape)

    @constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self) -> ConstraintT:
        return self._support

    @validate_sample
    def log_prob(self, value):
        dtype = jnp.result_type(value, float)
        minval = 100.0 * jnp.finfo(dtype).tiny  # for values close to 0
        eps = jnp.finfo(dtype).eps  # otherwise

        x1 = jnp.take(value, 0, axis=-1)  # left bound
        x2 = jnp.take(value, 1, axis=-1)  # right bound

        # make masks based on censoring indicators
        m_left = self.left_censored & (~self.right_censored)  # left-censored only
        m_right = self.right_censored & (~self.left_censored)  # right-censored only
        m_int = (~self.left_censored) & (~self.right_censored)  # interval censored
        m_double = self.left_censored & self.right_censored  # doubly censored
        m_point = jnp.isclose(x1, x2)  # point interval

        # Replace potential out-of-support values with finite placeholder BEFORE cdf
        # (value doesn't matter; it will be overwritten)
        feasible_value = self.support.feasible_like(x1)
        x1_finite = jnp.where(m_left, feasible_value, x1)
        x2_finite = jnp.where(m_right, feasible_value, x2)

        # Calculate CDF on safe values
        F1_tmp = self.base_dist.cdf(x1_finite)
        F2_tmp = self.base_dist.cdf(x2_finite)

        # Overwrite with correct limit values on censored rows
        # Left-censored: F1 := 0
        F1 = jnp.where(m_left, 0.0, F1_tmp)
        # Right-censored: F2 := 1
        F2 = jnp.where(m_right, 1.0, F2_tmp)

        # Stabilize against log(0) and tiny intervals
        F1 = jnp.clip(F1, minval, 1.0 - eps)
        F2 = jnp.clip(F2, minval, 1.0 - eps)

        # Use a stable log-diff for intervals (also covers left/right cases)
        # log(F2 - F1) = logF2 + log1p(-exp(logF1 - logF2))
        logF1 = jnp.log(F1)
        logF2 = jnp.log(F2)

        lp_interval = logF2 + jnp.log1p(-jnp.exp(jnp.clip(logF1 - logF2, max=-minval)))
        # handle point intervals (x1 == x2) by returning log density instead of log prob
        lp_interval = jnp.where(m_point, self.base_dist.log_prob(x1), lp_interval)

        # for doubly censored data, the value is not in the interval, so computation is 1 - exp(lp_interval)
        lp_double = jnp.log1p(-jnp.exp(lp_interval))

        # Select the right expression per row
        # left: log F(x2)
        lp_left = logF2
        # right: log (1 - F(x1)) = log1p(-F1)
        lp_right = jnp.log1p(-F1)

        logp = jnp.zeros_like(logF1)
        logp = jnp.where(m_left, lp_left, logp)
        logp = jnp.where(m_right, lp_right, logp)
        logp = jnp.where(m_int, lp_interval, logp)
        logp = jnp.where(m_double, lp_double, logp)
        return logp
