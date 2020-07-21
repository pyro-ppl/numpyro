# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0
import math

from jax import lax
import jax.numpy as jnp

from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import promote_shapes, validate_sample, von_mises_centered


class VonMises(Distribution):
    arg_constraints = {'loc': constraints.real, 'concentration': constraints.positive}

    support = constraints.interval(-math.pi, math.pi)

    def __init__(self, loc, concentration, validate_args=None):
        """  von Mises distribution for sampling directions.

        :param loc: center of distribution
        :param concentration: concentration of distribution
        """
        self.loc, self.concentration = promote_shapes(loc, concentration)

        batch_shape = lax.broadcast_shapes(jnp.shape(concentration), jnp.shape(loc))

        super(VonMises, self).__init__(batch_shape=batch_shape,
                                       validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        """ Generate sample from von Mises distribution

            :param sample_shape: shape of samples
            :param key: random number generator key
            :return: samples from von Mises
        """
        samples = von_mises_centered(key, self.concentration, sample_shape + self.shape())
        samples = samples + self.loc  # VM(0, concentration) -> VM(loc,concentration)
        samples = (samples + jnp.pi) % (2. * jnp.pi) - jnp.pi

        return samples

    @validate_sample
    def log_prob(self, value):
        return -(jnp.log(2 * jnp.pi) + lax.bessel_i0e(self.concentration)) + (
                self.concentration * jnp.cos(value - self.loc))

    @property
    def mean(self):
        """ Computes circular mean of distribution. NOTE: same as location when mapped to support [-pi, pi] """
        return jnp.broadcast_to((self.loc + jnp.pi) % (2. * jnp.pi) - jnp.pi, self.batch_shape)

    @property
    def variance(self):
        """ Computes circular variance of distribution """
        return jnp.broadcast_to(1. - lax.bessel_i1e(self.concentration) / lax.bessel_i0e(self.concentration),
                                self.batch_shape)
