# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from jax import lax, numpy as jnp

import numpyro.distributions.constraints as constraints
from numpyro.distributions.continuous import MultivariateNormal, Normal, Beta
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import is_prng_key, promote_shapes, validate_sample


class GaussianCopula(Distribution):
    arg_constraints = {
        "correlation_matrix": constraints.corr_matrix,
        "correlation_cholesky": constraints.lower_cholesky,
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
        validate_args=None,
    ):
        self.marginal_dist = marginal_dist
        self.base_dist = MultivariateNormal(
            covariance_matrix=correlation_matrix,
            scale_tril=correlation_cholesky,
        )
        self.normal = Normal()

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
        cdf = self.normal.cdf(normal_samples)
        return self.marginal_dist.icdf(cdf)

    @validate_sample
    def log_prob(self, value):
        # Ref: https://en.wikipedia.org/wiki/Copula_(probability_theory)#Gaussian_copula
        # see also https://github.com/pyro-ppl/numpyro/pull/1506#discussion_r1037525015
        marginal_lps = self.marginal_dist.log_prob(value)
        quantiles = self.normal.icdf(value)
        #
        copula_lp = (
            self.base_dist.log_prob(quantiles)
            + 0.5 * (quantiles**2).sum(-1)
            + 0.5 * jnp.log(2 * jnp.pi) * quantiles.shape[-1]
        )
        return copula_lp + marginal_lps.sum(axis=-1)

    @property
    def mean(self):
        return jnp.broadcast_to(self.marginal_dist.mean, self.shape())

    @constraints.dependent_property(event_dim=1)
    def support(self):
        return constraints.independent(self.marginal_dist.support, 1)


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
        super().__init__(
            Beta(concentration1, concentration0),
            correlation_matrix,
            correlation_cholesky,
            validate_args=validate_args,
        )
        self.concentration1, self.concentration0 = promote_shapes(
            concentration1, concentration0, shape=self.batch_shape + self.event_shape
        )
