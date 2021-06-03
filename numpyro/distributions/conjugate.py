# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from jax import lax, nn, random
import jax.numpy as jnp
from jax.scipy.special import betainc, betaln, gammaln

from numpyro.distributions import constraints
from numpyro.distributions.continuous import Beta, Dirichlet, Gamma
from numpyro.distributions.discrete import (
    BinomialProbs,
    MultinomialProbs,
    Poisson,
    ZeroInflatedDistribution,
)
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import is_prng_key, promote_shapes, validate_sample


def _log_beta_1(alpha, value):
    # XXX: support sparse `value`
    return gammaln(1 + value) + gammaln(alpha) - gammaln(value + alpha)


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
    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
        "total_count": constraints.nonnegative_integer,
    }
    has_enumerate_support = True
    enumerate_support = BinomialProbs.enumerate_support

    def __init__(
        self, concentration1, concentration0, total_count=1, validate_args=None
    ):
        self.concentration1, self.concentration0, self.total_count = promote_shapes(
            concentration1, concentration0, total_count
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(concentration1), jnp.shape(concentration0), jnp.shape(total_count)
        )
        concentration1 = jnp.broadcast_to(concentration1, batch_shape)
        concentration0 = jnp.broadcast_to(concentration0, batch_shape)
        self._beta = Beta(concentration1, concentration0)
        super(BetaBinomial, self).__init__(batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        key_beta, key_binom = random.split(key)
        probs = self._beta.sample(key_beta, sample_shape)
        return BinomialProbs(total_count=self.total_count, probs=probs).sample(
            key_binom
        )

    @validate_sample
    def log_prob(self, value):
        return (
            -_log_beta_1(self.total_count - value + 1, value)
            + betaln(
                value + self.concentration1,
                self.total_count - value + self.concentration0,
            )
            - betaln(self.concentration0, self.concentration1)
        )

    @property
    def mean(self):
        return self._beta.mean * self.total_count

    @property
    def variance(self):
        return (
            self._beta.variance
            * self.total_count
            * (self.concentration0 + self.concentration1 + self.total_count)
        )

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self):
        return constraints.integer_interval(0, self.total_count)


class DirichletMultinomial(Distribution):
    r"""
    Compound distribution comprising of a dirichlet-multinomial pair. The probability of
    classes (``probs`` for the :class:`~numpyro.distributions.Multinomial` distribution)
    is unknown and randomly drawn from a :class:`~numpyro.distributions.Dirichlet`
    distribution prior to a certain number of Categorical trials given by
    ``total_count``.

    :param numpy.ndarray concentration: concentration parameter (alpha) for the
        Dirichlet distribution.
    :param numpy.ndarray total_count: number of Categorical trials.
    """
    arg_constraints = {
        "concentration": constraints.independent(constraints.positive, 1),
        "total_count": constraints.nonnegative_integer,
    }

    def __init__(self, concentration, total_count=1, validate_args=None):
        if jnp.ndim(concentration) < 1:
            raise ValueError(
                "`concentration` parameter must be at least one-dimensional."
            )

        batch_shape = lax.broadcast_shapes(
            jnp.shape(concentration)[:-1], jnp.shape(total_count)
        )
        concentration_shape = batch_shape + jnp.shape(concentration)[-1:]
        (self.concentration,) = promote_shapes(concentration, shape=concentration_shape)
        (self.total_count,) = promote_shapes(total_count, shape=batch_shape)
        concentration = jnp.broadcast_to(self.concentration, concentration_shape)
        self._dirichlet = Dirichlet(concentration)
        super().__init__(
            self._dirichlet.batch_shape,
            self._dirichlet.event_shape,
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        key_dirichlet, key_multinom = random.split(key)
        probs = self._dirichlet.sample(key_dirichlet, sample_shape)
        return MultinomialProbs(total_count=self.total_count, probs=probs).sample(
            key_multinom
        )

    @validate_sample
    def log_prob(self, value):
        alpha = self.concentration
        return _log_beta_1(alpha.sum(-1), value.sum(-1)) - _log_beta_1(
            alpha, value
        ).sum(-1)

    @property
    def mean(self):
        return self._dirichlet.mean * jnp.expand_dims(self.total_count, -1)

    @property
    def variance(self):
        n = jnp.expand_dims(self.total_count, -1)
        alpha = self.concentration
        alpha_sum = self.concentration.sum(-1, keepdims=True)
        alpha_ratio = alpha / alpha_sum
        return n * alpha_ratio * (1 - alpha_ratio) * (n + alpha_sum) / (1 + alpha_sum)

    @constraints.dependent_property(is_discrete=True, event_dim=1)
    def support(self):
        return constraints.multinomial(self.total_count)

    @staticmethod
    def infer_shapes(concentration, total_count=()):
        batch_shape = lax.broadcast_shapes(concentration[:-1], total_count)
        event_shape = concentration[-1:]
        return batch_shape, event_shape


class GammaPoisson(Distribution):
    r"""
    Compound distribution comprising of a gamma-poisson pair, also referred to as
    a gamma-poisson mixture. The ``rate`` parameter for the
    :class:`~numpyro.distributions.Poisson` distribution is unknown and randomly
    drawn from a :class:`~numpyro.distributions.Gamma` distribution.

    :param numpy.ndarray concentration: shape parameter (alpha) of the Gamma distribution.
    :param numpy.ndarray rate: rate parameter (beta) for the Gamma distribution.
    """
    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
    }
    support = constraints.nonnegative_integer

    def __init__(self, concentration, rate=1.0, validate_args=None):
        self.concentration, self.rate = promote_shapes(concentration, rate)
        self._gamma = Gamma(concentration, rate)
        super(GammaPoisson, self).__init__(
            self._gamma.batch_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        key_gamma, key_poisson = random.split(key)
        rate = self._gamma.sample(key_gamma, sample_shape)
        return Poisson(rate).sample(key_poisson)

    @validate_sample
    def log_prob(self, value):
        post_value = self.concentration + value
        return (
            -betaln(self.concentration, value + 1)
            - jnp.log(post_value)
            + self.concentration * jnp.log(self.rate)
            - post_value * jnp.log1p(self.rate)
        )

    @property
    def mean(self):
        return self.concentration / self.rate

    @property
    def variance(self):
        return self.concentration / jnp.square(self.rate) * (1 + self.rate)

    def cdf(self, value):
        bt = betainc(self.concentration, value + 1.0, self.rate / (self.rate + 1.0))
        return bt


def NegativeBinomial(total_count, probs=None, logits=None, validate_args=None):
    if probs is not None:
        return NegativeBinomialProbs(total_count, probs, validate_args=validate_args)
    elif logits is not None:
        return NegativeBinomialLogits(total_count, logits, validate_args=validate_args)
    else:
        raise ValueError("One of `probs` or `logits` must be specified.")


class NegativeBinomialProbs(GammaPoisson):
    arg_constraints = {
        "total_count": constraints.positive,
        "probs": constraints.unit_interval,
    }
    support = constraints.nonnegative_integer

    def __init__(self, total_count, probs, validate_args=None):
        self.total_count, self.probs = promote_shapes(total_count, probs)
        concentration = total_count
        rate = 1.0 / probs - 1.0
        super().__init__(concentration, rate, validate_args=validate_args)


class NegativeBinomialLogits(GammaPoisson):
    arg_constraints = {
        "total_count": constraints.positive,
        "logits": constraints.real,
    }
    support = constraints.nonnegative_integer

    def __init__(self, total_count, logits, validate_args=None):
        self.total_count, self.logits = promote_shapes(total_count, logits)
        concentration = total_count
        rate = jnp.exp(-logits)
        super().__init__(concentration, rate, validate_args=validate_args)

    @validate_sample
    def log_prob(self, value):
        return -(
            self.total_count * nn.softplus(self.logits)
            + value * nn.softplus(-self.logits)
            + _log_beta_1(self.total_count, value)
        )


class NegativeBinomial2(GammaPoisson):
    """
    Another parameterization of GammaPoisson with `rate` is replaced by `mean`.
    """

    arg_constraints = {
        "mean": constraints.positive,
        "concentration": constraints.positive,
    }
    support = constraints.nonnegative_integer

    def __init__(self, mean, concentration, validate_args=None):
        rate = concentration / mean
        super().__init__(concentration, rate, validate_args=validate_args)


def ZeroInflatedNegativeBinomial2(
    mean, concentration, *, gate=None, gate_logits=None, validate_args=None
):
    return ZeroInflatedDistribution(
        NegativeBinomial2(mean, concentration, validate_args=validate_args),
        gate=gate,
        gate_logits=gate_logits,
        validate_args=validate_args,
    )
