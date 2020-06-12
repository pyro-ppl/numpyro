# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0
import math

import jax.numpy as np
from jax import lax, dtypes

from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import validate_sample
from numpyro.util import copy_docs_from
from numpyro.distributions.util import promote_shapes
from numpyro.distributions.util import _von_mises_centered


@copy_docs_from(Distribution)
class VonMises(Distribution):
    arg_constraints = {'location': constraints.real, 'concentration': constraints.positive}

    @property
    def support(self):
        return constraints.interval(-math.pi, math.pi)

    def __init__(self, loc, concentration, validate_args=None):
        self.loc, self.concentration = promote_shapes(loc, concentration)
        self.concentration = concentration
        self.loc = (loc + np.pi) % 2. * np.pi - np.pi

        batch_shape = lax.broadcast_shapes(np.shape(concentration), np.shape(loc))

        super(VonMises, self).__init__(batch_shape=batch_shape,
                                       validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        """ Generate sample from von Mises distribution

            :param sample_shape: shape of samples
            :param rng_key: random number generator key
            :return: samples from von Mises
        """
        samples = _von_mises_centered(key=key, concentration=self.concentration, shape=sample_shape,
                                      dtype=dtypes.canonicalize_dtype(np.float64))
        samples = samples + self.loc  # VM(0, concentration) -> VM(loc,concentration)
        samples = samples % (2. * math.pi) - math.pi

        return samples

    @validate_sample
    def log_prob(self, value):
        return -(np.log(2 * np.pi) + lax.bessel_i0e(self._concentration)) + (
                self._concentration * np.cos(value - self._loc))

    @property
    def mean(self):
        return np.broadcast_to((self.loc + np.pi) % 2. * np.pi - np.pi, self.batch_shape)

    @property
    def location(self):
        return self.loc

    @property
    def concentration(self):
        return self.concentration

    @property
    def variance(self):
        """ Computes circular variance of distribution """
        return np.broadcast_to(1 - lax.bessel_i1e(self._concentration) / lax.bessel_i0e(self._concentration),
                               self.batch_shape)
