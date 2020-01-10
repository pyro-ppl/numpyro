# Copyright (c) 2017-2020 Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from jax import lax, random
import jax.numpy as np
from jax.scipy.special import gammaln

from numpyro.distributions import constraints
from numpyro.distributions.continuous import Beta, Gamma
from numpyro.distributions.discrete import Binomial, Poisson
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import promote_shapes, validate_sample


def _log_beta(x, y):
    return gammaln(x) + gammaln(y) - gammaln(x + y)


class BetaBinomial(Distribution):
    r"""
    Compound distribution comprising of a beta-binomial pair. The probability of
    success (``probs`` for the :class:`~numpyro.distributions.Binomial` distribution)
    is unknown and randomly drawn from a :class:`~numpyro.distributions.Beta` distribution
    prior to a certain number of Bernoulli trials given by ``total_count``.

    :param numpy.ndarray concentration1: 1st concentration parameter (alpha) for the
        Beta distribution.
    :param numpy.ndarray concentration0: 2nd concentration parameter (beta) for the
        Beta distribution.
    :param numpy.ndarray total_count: number of Bernoulli trials.
    """
    arg_constraints = {'concentration1': constraints.positive, 'concentration0': constraints.positive,
                       'total_count': constraints.nonnegative_integer}

    def __init__(self, concentration1, concentration0, total_count=1, validate_args=None):
        batch_shape = lax.broadcast_shapes(np.shape(concentration1), np.shape(concentration0),
                                           np.shape(total_count))
        self.concentration1 = np.broadcast_to(concentration1, batch_shape)
        self.concentration0 = np.broadcast_to(concentration0, batch_shape)
        self.total_count, = promote_shapes(total_count, shape=batch_shape)
        self._beta = Beta(self.concentration1, self.concentration0)
        super(BetaBinomial, self).__init__(batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        key_beta, key_binom = random.split(key)
        probs = self._beta.sample(key_beta, sample_shape)
        return Binomial(self.total_count, probs).sample(key_binom)

    @validate_sample
    def log_prob(self, value):
        log_factorial_n = gammaln(self.total_count + 1)
        log_factorial_k = gammaln(value + 1)
        log_factorial_nmk = gammaln(self.total_count - value + 1)
        return (log_factorial_n - log_factorial_k - log_factorial_nmk +
                _log_beta(value + self.concentration1, self.total_count - value + self.concentration0) -
                _log_beta(self.concentration0, self.concentration1))

    @property
    def mean(self):
        return self._beta.mean * self.total_count

    @property
    def variance(self):
        return self._beta.variance * self.total_count * (self.concentration0 + self.concentration1 + self.total_count)

    @property
    def support(self):
        return constraints.integer_interval(0, self.total_count)


class GammaPoisson(Distribution):
    r"""
    Compound distribution comprising of a gamma-poisson pair, also referred to as
    a gamma-poisson mixture. The ``rate`` parameter for the
    :class:`~numpyro.distributions.Poisson` distribution is unknown and randomly
    drawn from a :class:`~numpyro.distributions.Gamma` distribution.

    :param numpy.ndarray concentration: shape parameter (alpha) of the Gamma distribution.
    :param numpy.ndarray rate: rate parameter (beta) for the Gamma distribution.
    """
    arg_constraints = {'concentration': constraints.positive, 'rate': constraints.positive}
    support = constraints.nonnegative_integer

    def __init__(self, concentration, rate=1., validate_args=None):
        self._gamma = Gamma(concentration, rate)
        self.concentration = self._gamma.concentration
        self.rate = self._gamma.rate
        super(GammaPoisson, self).__init__(self._gamma.batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        key_gamma, key_poisson = random.split(key)
        rate = self._gamma.sample(key_gamma, sample_shape)
        return Poisson(rate).sample(key_poisson)

    @validate_sample
    def log_prob(self, value):
        post_value = self.concentration + value
        return -_log_beta(self.concentration, value + 1) - np.log(post_value) + \
            self.concentration * np.log(self.rate) - post_value * np.log1p(self.rate)

    @property
    def mean(self):
        return self.concentration / self.rate

    @property
    def variance(self):
        return self.concentration / np.square(self.rate) * (1 + self.rate)
