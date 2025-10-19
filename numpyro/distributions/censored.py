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
