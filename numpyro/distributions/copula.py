# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import copy

from jax import lax, numpy as jnp

import numpyro.distributions.constraints as constraints
from numpyro.distributions.continuous import Beta, MultivariateNormal, Normal
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import (
    clamp_probs,
    is_prng_key,
    lazy_property,
    validate_sample,
)


class GaussianCopula(Distribution):
    """
    A distribution that links the `batch_shape[:-1]` of marginal distribution `marginal_dist`
    with a multivariate Gaussian copula modelling the correlation between the axes.

    :param Distribution marginal_dist: Distribution whose last batch axis is to be coupled.
    :param array_like correlation_matrix: Correlation matrix of coupling multivariate normal distribution.
    :param array_like correlation_cholesky: Correlation Cholesky factor of coupling multivariate normal distribution.
    """

    arg_constraints = {
        "correlation_matrix": constraints.corr_matrix,
        "correlation_cholesky": constraints.corr_cholesky,
    }
    reparametrized_params = [
        "correlation_matrix",
        "correlation_cholesky",
    ]

    def __init__(
        self,
        marginal_dist,
        correlation_matrix=None,
        correlation_cholesky=None,
        *,
        validate_args=None,
    ):
        if len(marginal_dist.event_shape) > 0:
            raise ValueError("`marginal_dist` needs to be a univariate distribution.")

        self.marginal_dist = marginal_dist
        self.base_dist = MultivariateNormal(
            covariance_matrix=correlation_matrix,
            scale_tril=correlation_cholesky,
        )

        event_shape = self.base_dist.event_shape
        batch_shape = lax.broadcast_shapes(
            self.marginal_dist.batch_shape[:-1],
            self.base_dist.batch_shape,
        )

        super(GaussianCopula, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)

        shape = sample_shape + self.batch_shape
        normal_samples = self.base_dist.expand(shape).sample(key)
        cdf = Normal().cdf(normal_samples)
        return self.marginal_dist.icdf(cdf)

    @validate_sample
    def log_prob(self, value):
        # Ref: https://en.wikipedia.org/wiki/Copula_(probability_theory)#Gaussian_copula
        # see also https://github.com/pyro-ppl/numpyro/pull/1506#discussion_r1037525015
        marginal_lps = self.marginal_dist.log_prob(value)
        probs = self.marginal_dist.cdf(value)
        quantiles = Normal().icdf(clamp_probs(probs))

        copula_lp = (
            self.base_dist.log_prob(quantiles)
            + 0.5 * (quantiles**2).sum(-1)
            + 0.5 * jnp.log(2 * jnp.pi) * quantiles.shape[-1]
        )
        return copula_lp + marginal_lps.sum(axis=-1)

    @property
    def mean(self):
        return jnp.broadcast_to(self.marginal_dist.mean, self.shape())

    @property
    def variance(self):
        return jnp.broadcast_to(self.marginal_dist.variance, self.shape())

    @constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self):
        return constraints.independent(self.marginal_dist.support, 1)

    @lazy_property
    def correlation_matrix(self):
        return self.base_dist.covariance_matrix

    @lazy_property
    def correlation_cholesky(self):
        return self.base_dist.scale_tril

    def tree_flatten(self):
        arg_constraints = {"marginal_dist": None, "base_dist": None}
        return (
            tuple(getattr(self, param) for param in arg_constraints.keys()),
            (self.batch_shape, self.event_shape),
        )

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        arg_constraints = {"marginal_dist": None, "base_dist": None}
        batch_shape, event_shape = aux_data
        assert isinstance(batch_shape, tuple)
        assert isinstance(event_shape, tuple)
        d = cls.__new__(cls)
        for k, v in zip(arg_constraints.keys(), params):
            setattr(d, k, v)
        Distribution.__init__(d, batch_shape, event_shape)
        return d

    def vmap_over(
        self,
        marginal_dist=None,
        correlation_matrix=None,
        correlation_cholesky=None,
    ):
        dist_axes = copy.copy(self)
        dist_axes.marginal_dist = marginal_dist
        dist_axes.base_dist = dist_axes.base_dist.vmap_over(
            loc=correlation_matrix or correlation_cholesky,
            scale_tril=correlation_matrix or correlation_cholesky,
        )
        return dist_axes


class GaussianCopulaBeta(GaussianCopula):
    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
        "correlation_matrix": constraints.corr_matrix,
        "correlation_cholesky": constraints.corr_cholesky,
    }
    support = constraints.independent(constraints.unit_interval, 1)

    def __init__(
        self,
        concentration1,
        concentration0,
        correlation_matrix=None,
        correlation_cholesky=None,
        *,
        validate_args=False,
    ):
        # set initially to allow argument validation
        self.concentration1, self.concentration0 = concentration1, concentration0

        super().__init__(
            Beta(concentration1, concentration0),
            correlation_matrix,
            correlation_cholesky,
            validate_args=validate_args,
        )

    def vmap_over(
        self,
        concentration1=None,
        concentration0=None,
        correlation_matrix=None,
        correlation_cholesky=None,
    ):
        return super().vmap_over(
            self.marginal_dist.vmap_over(
                concentration1=concentration1, concentration0=concentration0
            ),
            correlation_matrix=correlation_matrix,
            correlation_cholesky=correlation_cholesky,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        d = super().tree_unflatten(aux_data, params)
        d.concentration0 = d.marginal_dist.concentration0
        d.concentration1 = d.marginal_dist.concentration1
        return d
