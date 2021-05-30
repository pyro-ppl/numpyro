# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from jax import lax
import jax.numpy as jnp
import jax.random as random
from jax.scipy.special import logsumexp
from jax.tree_util import tree_map

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
    is_prng_key,
    lazy_property,
    promote_shapes,
    validate_sample,
)


class LeftTruncatedDistribution(Distribution):
    arg_constraints = {"low": constraints.real}
    reparametrized_params = ["low"]
    supported_types = (Cauchy, Laplace, Logistic, Normal, SoftLaplace, StudentT)

    def __init__(self, base_dist, low=0.0, validate_args=None):
        assert isinstance(base_dist, self.supported_types)
        assert (
            base_dist.support is constraints.real
        ), "The base distribution should be univariate and have real support."
        batch_shape = lax.broadcast_shapes(base_dist.batch_shape, jnp.shape(low))
        self.base_dist = tree_map(
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
        u = random.uniform(key, sample_shape + self.batch_shape)
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

    def tree_flatten(self):
        base_flatten, base_aux = self.base_dist.tree_flatten()
        if isinstance(self._support.lower_bound, (int, float)):
            return base_flatten, (
                type(self.base_dist),
                base_aux,
                self._support.lower_bound,
            )
        else:
            return (base_flatten, self.low), (type(self.base_dist), base_aux)

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        if len(aux_data) == 2:
            base_flatten, low = params
            base_cls, base_aux = aux_data
        else:
            base_flatten = params
            base_cls, base_aux, low = aux_data
        base_dist = base_cls.tree_unflatten(base_aux, base_flatten)
        return cls(base_dist, low=low)


class RightTruncatedDistribution(Distribution):
    arg_constraints = {"high": constraints.real}
    reparametrized_params = ["high"]
    supported_types = (Cauchy, Laplace, Logistic, Normal, SoftLaplace, StudentT)

    def __init__(self, base_dist, high=0.0, validate_args=None):
        assert isinstance(base_dist, self.supported_types)
        assert (
            base_dist.support is constraints.real
        ), "The base distribution should be univariate and have real support."
        batch_shape = lax.broadcast_shapes(base_dist.batch_shape, jnp.shape(high))
        self.base_dist = tree_map(
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
        u = random.uniform(key, sample_shape + self.batch_shape)
        return self.base_dist.icdf(u * self._cdf_at_high)

    @validate_sample
    def log_prob(self, value):
        return self.base_dist.log_prob(value) - jnp.log(self._cdf_at_high)

    def tree_flatten(self):
        base_flatten, base_aux = self.base_dist.tree_flatten()
        if isinstance(self._support.upper_bound, (int, float)):
            return base_flatten, (
                type(self.base_dist),
                base_aux,
                self._support.upper_bound,
            )
        else:
            return (base_flatten, self.high), (type(self.base_dist), base_aux)

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        if len(aux_data) == 2:
            base_flatten, high = params
            base_cls, base_aux = aux_data
        else:
            base_flatten = params
            base_cls, base_aux, high = aux_data
        base_dist = base_cls.tree_unflatten(base_aux, base_flatten)
        return cls(base_dist, high=high)


class TwoSidedTruncatedDistribution(Distribution):
    arg_constraints = {"low": constraints.dependent, "high": constraints.dependent}
    reparametrized_params = ["low", "high"]
    supported_types = (Cauchy, Laplace, Logistic, Normal, SoftLaplace, StudentT)

    def __init__(self, base_dist, low=0.0, high=1.0, validate_args=None):
        assert isinstance(base_dist, self.supported_types)
        assert (
            base_dist.support is constraints.real
        ), "The base distribution should be univariate and have real support."
        batch_shape = lax.broadcast_shapes(
            base_dist.batch_shape, jnp.shape(low), jnp.shape(high)
        )
        self.base_dist = tree_map(
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

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        u = random.uniform(key, sample_shape + self.batch_shape)

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
            (1 - u) * self._tail_prob_at_low + u * self._tail_prob_at_high
        )

    @validate_sample
    def log_prob(self, value):
        # NB: we use a more numerically stable formula for a symmetric base distribution
        # if low < loc
        #   cdf(high) - cdf(low) = as-is
        # if low > loc
        #   cdf(high) - cdf(low) = cdf(2 * loc - low) - cdf(2 * loc - high)
        sign = jnp.where(self.base_dist.loc >= self.low, 1.0, -1.0)
        return self.base_dist.log_prob(value) - jnp.log(
            sign * (self._tail_prob_at_high - self._tail_prob_at_low)
        )

    def tree_flatten(self):
        base_flatten, base_aux = self.base_dist.tree_flatten()
        if isinstance(self._support.lower_bound, (int, float)) and isinstance(
            self._support.upper_bound, (int, float)
        ):
            return base_flatten, (
                type(self.base_dist),
                base_aux,
                self._support.lower_bound,
                self._support.upper_bound,
            )
        else:
            return (base_flatten, self.low, self.high), (type(self.base_dist), base_aux)

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        if len(aux_data) == 2:
            base_flatten, low, high = params
            base_cls, base_aux = aux_data
        else:
            base_flatten = params
            base_cls, base_aux, low, high = aux_data
        base_dist = base_cls.tree_unflatten(base_aux, base_flatten)
        return cls(base_dist, low=low, high=high)


def TruncatedDistribution(base_dist, low=None, high=None, validate_args=None):
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


class TruncatedCauchy(LeftTruncatedDistribution):
    arg_constraints = {
        "low": constraints.real,
        "loc": constraints.real,
        "scale": constraints.positive,
    }
    reparametrized_params = ["low", "loc", "scale"]

    def __init__(self, low=0.0, loc=0.0, scale=1.0, validate_args=None):
        self.low, self.loc, self.scale = promote_shapes(low, loc, scale)
        super().__init__(
            Cauchy(self.loc, self.scale), low=self.low, validate_args=validate_args
        )

    @property
    def mean(self):
        return jnp.full(self.batch_shape, jnp.nan)

    @property
    def variance(self):
        return jnp.full(self.batch_shape, jnp.nan)

    def tree_flatten(self):
        if isinstance(self._support.lower_bound, (int, float)):
            aux_data = self._support.lower_bound
        else:
            aux_data = None
        return (self.low, self.loc, self.scale), aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        d = cls(*params)
        if aux_data is not None:
            d._support = constraints.greater_than(aux_data)
        return d


class TruncatedNormal(LeftTruncatedDistribution):
    arg_constraints = {
        "low": constraints.real,
        "loc": constraints.real,
        "scale": constraints.positive,
    }
    reparametrized_params = ["low", "loc", "scale"]

    def __init__(self, low=0.0, loc=0.0, scale=1.0, validate_args=None):
        self.low, self.loc, self.scale = promote_shapes(low, loc, scale)
        super().__init__(
            Normal(self.loc, self.scale), low=self.low, validate_args=validate_args
        )

    @property
    def mean(self):
        low_prob = jnp.exp(self.log_prob(self.low))
        return self.loc + low_prob * self.scale ** 2

    @property
    def variance(self):
        low_prob = jnp.exp(self.log_prob(self.low))
        return (self.scale ** 2) * (
            1 + (self.low - self.loc) * low_prob - (low_prob * self.scale) ** 2
        )

    def tree_flatten(self):
        if isinstance(self._support.lower_bound, (int, float)):
            aux_data = self._support.lower_bound
        else:
            aux_data = None
        return (self.low, self.loc, self.scale), aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        d = cls(*params)
        if aux_data is not None:
            d._support = constraints.greater_than(aux_data)
        return d


class TruncatedPolyaGamma(Distribution):
    truncation_point = 2.5
    num_log_prob_terms = 7
    num_gamma_variates = 8
    assert num_log_prob_terms % 2 == 1

    arg_constraints = {}
    support = constraints.interval(0.0, truncation_point)

    def __init__(self, batch_shape=(), validate_args=None):
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
        return jnp.clip(x * (0.5 / jnp.pi ** 2), a_max=self.truncation_point)

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

    def tree_flatten(self):
        return (), self.batch_shape

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        return cls(batch_shape=aux_data)
