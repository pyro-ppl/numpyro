# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

# The implementation largely follows the design in PyTorch's `torch.distributions`
#
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


from typing import Optional, Union

import numpy as np

import jax
from jax import Array, lax
from jax.nn import softmax, softplus
import jax.numpy as jnp
import jax.random as random
from jax.scipy.special import expit, gammaincc, gammaln, logsumexp, xlog1py, xlogy
from jax.typing import ArrayLike

from numpyro.distributions import constraints, transforms
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import (
    assert_one_of,
    binary_cross_entropy_with_logits,
    binomial,
    categorical,
    clamp_probs,
    lazy_property,
    multinomial,
    promote_shapes,
    validate_sample,
)
from numpyro.util import is_prng_key, not_jax_tracer


def _to_probs_bernoulli(logits: ArrayLike) -> ArrayLike:
    return expit(logits)


def _to_logits_bernoulli(probs: ArrayLike) -> ArrayLike:
    ps_clamped = clamp_probs(probs)
    return jnp.log(ps_clamped) - jnp.log1p(-ps_clamped)


def _to_probs_multinom(logits: ArrayLike) -> ArrayLike:
    return softmax(logits, axis=-1)


def _to_logits_multinom(probs: ArrayLike) -> ArrayLike:
    minval = jnp.finfo(jnp.result_type(probs)).min
    return jnp.clip(jnp.log(probs), minval)


class BernoulliProbs(Distribution):
    r"""A Bernoulli discrete random variable parameterizing the probability of a binary
    outcome.

    The Probability Mass Function (PMF) of the Bernoulli distribution is defined as:

    .. math::
        P(X = k | p) = p^k (1 - p)^{1-k}, \quad k \in \{0, 1\}

    Where, :math:`p` represents the success probability parameter (:attr:`probs`),
    :math:`k` represents the observed binary outcome (:attr:`value`).
    The support domain is :math:`k \in \{0, 1\}`.
    """

    arg_constraints = {"probs": constraints.unit_interval}
    support = constraints.boolean
    r"""The support of the Bernoulli distribution is the set of binary outcomes :math:`\{0, 1\}`."""

    has_enumerate_support = True

    def __init__(self, probs: ArrayLike, *, validate_args: Optional[bool] = None):
        r"""
        :param probs: Success probability in the interval :math:`[0, 1]`.
        :param validate_args: If True, enforce domain constraints during initialization.
        """
        self.probs = probs
        super(BernoulliProbs, self).__init__(
            batch_shape=jnp.shape(self.probs), validate_args=validate_args
        )

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> ArrayLike:
        r"""Draw samples from the Bernoulli distribution.

        This method invokes :func:`~jax.random.bernoulli` directly, which generates
        binary samples from the Bernoulli parametrization. Samples are mapped across
        the specified batch dimensions and sample dimensions via shape broadcasting.

        :param key: A JAX random number generator key (PRNG state).
        :param sample_shape: Desired sample dimensions to prepend to the batch shape.
        :return: Binary-valued samples (0 or 1) drawn from the Bernoulli distribution.
        """
        assert is_prng_key(key)
        samples = random.bernoulli(
            key, self.probs, shape=sample_shape + self.batch_shape
        )
        return samples.astype(jnp.result_type(samples, int))

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        r"""Evaluate the log probability mass function at specified binary
        configurations.

        .. math::
            \ln P(X=k | p) = k\ln(p) + (1-k)\ln(1-p)

        The log probability mass function is evaluated using numerically-stable log-space operations.
        Rather than computing :math:`\ln(p)` and :math:`\ln(1-p)` directly from clamped
        probabilities, this implementation employs the primitives :func:`~jax.scipy.special.xlogy`
        and :func:`~jax.scipy.special.xlog1py`, which handle edge cases gracefully:

        - When :math:`p = 0` or :math:`p = 1`, the log-probability computation is protected
          from logarithmic singularities via masking.
        - The clamped probability values prevent numerical underflow in extreme configurations.

        :param value: Binary observation(s) to score (:math:`k \in \{0, 1\}`).
        :return: Log probability scores evaluated under the Bernoulli PMF.
        """
        ps_clamped = clamp_probs(self.probs)
        value = jnp.array(value, jnp.result_type(float))
        return xlogy(value, ps_clamped) + xlog1py(1 - value, -ps_clamped)

    @lazy_property
    def logits(self) -> ArrayLike:
        r"""The log-odds (logits) parameter of the Bernoulli distribution is given by
        the logit transformation of the success probability:

        .. math::
            \alpha = \text{logit}(p) = \ln\left(\frac{p}{1-p}\right)
        """
        return _to_logits_bernoulli(self.probs)

    @property
    def mean(self) -> ArrayLike:
        r"""The mean of the Bernoulli distribution is given by the success probability
        parameter:

        .. math::
            E[X] = p

        :return: The mean of the Bernoulli distribution, which is equal to the success
            probability :attr:`probs`.
        """
        return self.probs

    @property
    def variance(self) -> ArrayLike:
        r"""The variance of the Bernoulli distribution is given by:

        .. math::
            \mathrm{Var}[X] = p (1 - p)

        :return: The variance of the Bernoulli distribution, which is the product of
            the success probability and its complement.
        """
        return self.probs * (1 - self.probs)

    def enumerate_support(self, expand: bool = True) -> ArrayLike:
        values = jnp.arange(2).reshape((-1,) + (1,) * len(self.batch_shape))
        if expand:
            values = jnp.broadcast_to(values, values.shape[:1] + self.batch_shape)
        return values

    def entropy(self) -> ArrayLike:
        r"""The entropy of the Bernoulli distribution is given by:

        .. math::
            H[X] = -p \ln p - (1-p) \ln (1-p)
        """
        return -xlogy(self.probs, self.probs) - xlog1py(1 - self.probs, -self.probs)


class BernoulliLogits(Distribution):
    r"""A Bernoulli discrete random variable parameterized by log-odds (logits).

    The Probability Mass Function (PMF) of the Bernoulli distribution is:

    .. math::
        P(X = k | \alpha) = \sigma(\alpha)^k (1 - \sigma(\alpha))^{1-k},
        \quad k \in \{0, 1\}

    Where :math:`\alpha = \text{logits}` is the log-odds parameter and
    :math:`\sigma(\alpha) = 1/(1 + \exp{(-\alpha)})` is the sigmoid function.
    """

    arg_constraints = {"logits": constraints.real}

    support = constraints.boolean
    r"""The support of the Bernoulli distribution is the set of binary outcomes :math:`\{0, 1\}`."""

    has_enumerate_support = True

    def __init__(self, logits: ArrayLike, *, validate_args: Optional[bool] = None):
        r"""
        :param logits: Log-odds parameter spanning the full real line :math:`\alpha \in \mathbb{R}`.
        :param validate_args: If True, enforce domain constraints during initialization.
        """
        self.logits = logits
        super(BernoulliLogits, self).__init__(
            batch_shape=jnp.shape(self.logits), validate_args=validate_args
        )

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> ArrayLike:
        r"""Draw samples from the Bernoulli distribution.

        The method first converts :attr:`logits` to probabilities via the sigmoid
        function (accessed via the lazy property :attr:`probs`), then invokes
        :func:`~jax.random.bernoulli` for sampling.

        :param key: A JAX random number generator key (PRNG state).
        :param sample_shape: Desired sample dimensions to prepend to the batch shape.
        :return: Binary-valued samples (0 or 1) drawn from the Bernoulli distribution.
        """
        assert is_prng_key(key)
        samples = random.bernoulli(
            key, self.probs, shape=sample_shape + self.batch_shape
        )
        return samples.astype(jnp.result_type(samples, int))

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        r"""Evaluate the log probability mass function at specified binary configurations.

        The log probability mass function leverages the numerically-stable
        :func:`~jax.nn.binary_cross_entropy_with_logits` primitive, which computes the
        Bernoulli negative log-likelihood directly in log-odds space:

        .. math::
            \ln P(X = k | \alpha) = -\mathrm{BCEWithLogits}(\alpha, k)
            = k \ln(\sigma(\alpha)) + (1-k) \ln(1 - \sigma(\alpha))

        This formulation avoids explicit exponential evaluation for large
        :math:`|\alpha|`, protecting against overflow (:math:`e^\alpha \to \infty` for
        :math:`\alpha \gg 0`) and underflow (:math:`e^{-\alpha} \to 0` for
        :math:`\alpha \ll -0`).

        :param value: Binary observation(s) to score (:math:`k \in \{0, 1\}`).
        :return: Log probability scores evaluated under the Bernoulli PMF.
        """
        return -binary_cross_entropy_with_logits(self.logits, value)

    @lazy_property
    def probs(self) -> ArrayLike:
        r"""The success probability parameter of the Bernoulli distribution is given by
        the sigmoid of the log-odds parameter:

        .. math::
            p = \sigma(\alpha) = \frac{1}{1 + e^{-\alpha}}
        """
        return _to_probs_bernoulli(self.logits)

    @property
    def mean(self) -> ArrayLike:
        r"""The mean of the Bernoulli distribution is given by the sigmoid of the
        log-odds parameter:

        .. math::
            E[X] = \sigma(\alpha) = \frac{1}{1 + e^{-\alpha}}
        """
        return self.probs

    @property
    def variance(self) -> ArrayLike:
        r"""The variance of the Bernoulli distribution is given by:

        .. math::
            \mathrm{Var}[X] = \sigma(\alpha) (1 - \sigma(\alpha))
        """
        return self.probs * (1 - self.probs)

    def enumerate_support(self, expand: bool = True) -> ArrayLike:
        values = jnp.arange(2).reshape((-1,) + (1,) * len(self.batch_shape))
        if expand:
            values = jnp.broadcast_to(values, values.shape[:1] + self.batch_shape)
        return values

    def entropy(self) -> ArrayLike:
        r"""The entropy of the Bernoulli distribution is given by:

        .. math::
            H[X] = -p \ln p - (1-p) \ln (1-p)

        where :math:`p = \sigma(\alpha)` is the mean of the distribution.

        The implementation is of following form to maintain numerical stability across
        the full range of log-odds values:

        .. math::
            H[X] = \frac{(1 + e^{-\alpha}) \ln(1 + e^{-\alpha})
                + e^{-\alpha} \alpha}{1 + e^{-\alpha}}
        """
        nexp = jnp.exp(-self.logits)
        return ((1 + nexp) * jnp.log1p(nexp) + nexp * self.logits) / (1 + nexp)


def Bernoulli(
    probs: Optional[ArrayLike] = None,
    logits: Optional[ArrayLike] = None,
    *,
    validate_args: Optional[bool] = None,
) -> Union[BernoulliProbs, BernoulliLogits]:
    r"""Factory function to create a Bernoulli distribution instance from either
    probability or log-odds parameterization.

    :param probs: The success probability parameter in the unit interval :math:`[0, 1]`,
        defaults to None
    :param logits: The log-odds parameter, defaults to None
    :param validate_args: Optional toggle to enforce domain constraints during
        graph construction. Default is None.
    :return: The created Bernoulli distribution instance.
    """
    assert_one_of(probs=probs, logits=logits)
    if probs is not None:
        return BernoulliProbs(probs, validate_args=validate_args)
    elif logits is not None:
        return BernoulliLogits(logits, validate_args=validate_args)


class BinomialProbs(Distribution):
    r"""A Binomial discrete random variable parameterizing the count of successes in
    repeated trials.

    The Probability Mass Function (PMF) of the Binomial distribution is defined as:

    .. math::
        P(X = k | n, p) = \binom{n}{k} p^k (1 - p)^{n-k}, \quad k \in \{0, 1, \dots, n\}

    Where, :math:`n` is the number of trials (:attr:`total_count`),
    :math:`p` is the success probability per trial (:attr:`probs`),
    :math:`k` is the observed count of successes (:attr:`value`).
    """

    arg_constraints = {
        "probs": constraints.unit_interval,
        "total_count": constraints.nonnegative_integer,
    }
    has_enumerate_support = True

    def __init__(
        self,
        probs: ArrayLike,
        total_count: ArrayLike = 1,
        *,
        validate_args: Optional[bool] = None,
    ):
        r"""
        :param probs: Success probability per trial in :math:`[0, 1]`.
        :param total_count: Number of trials (non-negative integer).
        :param validate_args: If True, enforce domain constraints during initialization.
        """
        self.probs, self.total_count = promote_shapes(probs, total_count)
        batch_shape = lax.broadcast_shapes(jnp.shape(probs), jnp.shape(total_count))
        super(BinomialProbs, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> ArrayLike:
        r"""Draw samples from the Binomial distribution.

        This method uses the internal :func:`~numpyro.distributions.util.binomial`
        utility function to generate count samples.

        :param key: A JAX random number generator key (PRNG state).
        :param sample_shape: Desired sample dimensions to prepend to the batch shape.
        :return: Non-negative integer samples representing success counts.
        """
        assert is_prng_key(key)
        return binomial(
            key, self.probs, n=self.total_count, shape=sample_shape + self.batch_shape
        )

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        r"""Evaluate the log probability mass function at specified count configurations.

        The log probability mass function is fully evaluated in log-space to prevent
        factorial overflow and underflow:

        .. math::
            \ln P(X = k | n, p) = \ln \binom{n}{k} + k \ln p + (n-k) \log(1-p)

        The binomial coefficient in log-space is computed using the log-gamma function:

        .. math::
            \ln \binom{n}{k} = \ln\Gamma(n + 1) - \ln\Gamma(k + 1)
            - \ln\Gamma(n - k + 1)

        This approach using :func:`~jax.scipy.special.gammaln` avoids
        computing factorials explicitly. The probability terms are evaluated using
        :func:`~jax.scipy.special.xlogy` and :func:`~jax.scipy.special.xlog1py`
        to handle boundary cases gracefully (:math:`p = 0`, :math:`p = 1`, etc.).

        :param value: Count observation(s) in the range :math:`[0, n]`.
        :return: Log probability scores evaluated under the Binomial PMF.
        """
        value = jnp.array(value, jnp.result_type(float))
        log_factorial_n = gammaln(self.total_count + 1)
        log_factorial_k = gammaln(value + 1)
        log_factorial_nmk = gammaln(self.total_count - value + 1)
        probs = clamp_probs(self.probs)
        return (
            log_factorial_n
            - log_factorial_k
            - log_factorial_nmk
            + xlogy(value, probs)
            + xlog1py(self.total_count - value, -probs)
        )

    @lazy_property
    def logits(self) -> ArrayLike:
        r"""The log-odds (logits) parameter of the Binomial distribution is given by
        the logit transformation of the success probability:

        .. math::
            \alpha = \text{logit}(p) = \ln\left(\frac{p}{1-p}\right)
        """
        return _to_logits_bernoulli(self.probs)

    @property
    def mean(self) -> ArrayLike:
        r"""The mean of the Binomial distribution is given by:

        .. math::
            E[X] = n p
        """
        return jnp.broadcast_to(self.total_count * self.probs, self.batch_shape)

    @property
    def variance(self) -> ArrayLike:
        r"""The variance of the Binomial distribution is given by:

        .. math::
            \mathrm{Var}[X] = n p (1 - p)
        """
        return jnp.broadcast_to(
            self.total_count * self.probs * (1 - self.probs), self.batch_shape
        )

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self) -> constraints.Constraint:
        r"""The support of the Binomial distribution is the set of integer counts
        from 0 to the total count.
        """
        return constraints.integer_interval(0, self.total_count)

    def enumerate_support(self, expand: bool = True) -> ArrayLike:
        if not_jax_tracer(self.total_count):
            total_count = np.amax(self.total_count)
            # NB: the error can't be raised if inhomogeneous issue happens when tracing
            if np.amin(self.total_count) != total_count:
                raise NotImplementedError(
                    "Inhomogeneous total count not supported by `enumerate_support`."
                )
        else:
            total_count = jnp.amax(self.total_count)
        values = jnp.arange(total_count + 1).reshape(
            (-1,) + (1,) * len(self.batch_shape)
        )
        if expand:
            values = jnp.broadcast_to(values, values.shape[:1] + self.batch_shape)
        return values


class BinomialLogits(Distribution):
    r"""
    A Binomial discrete random variable parameterized by log-odds (logits).

    The Probability Mass Function (PMF) of the Binomial distribution is:

    .. math::
        P(X = k | n, \alpha) = \binom{n}{k} \sigma(\alpha)^k (1 - \sigma(\alpha))^{n-k}

    Where :math:`\alpha = \text{logits}` and
    :math:`\sigma(\alpha) = 1/(1 + \exp(-\alpha))`.
    """

    arg_constraints = {
        "logits": constraints.real,
        "total_count": constraints.nonnegative_integer,
    }
    has_enumerate_support = True
    enumerate_support = BinomialProbs.enumerate_support

    def __init__(
        self,
        logits: ArrayLike,
        total_count: ArrayLike = 1,
        *,
        validate_args: Optional[bool] = None,
    ):
        r"""
        :param logits: Log-odds parameter spanning :math:`\mathbb{R}`.
        :param total_count: Number of trials (non-negative integer).
        :param validate_args: If True, enforce domain constraints during initialization.
        """
        self.logits, self.total_count = promote_shapes(logits, total_count)
        batch_shape = lax.broadcast_shapes(jnp.shape(logits), jnp.shape(total_count))
        super(BinomialLogits, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> ArrayLike:
        r"""Draw samples from the Binomial distribution.

        The method first converts :attr:`logits` to probabilities via the sigmoid function
        (via the lazy property :attr:`probs`), then uses the internal :func:`binomial`
        utility for sampling. This maintains numerical stability across extreme log-odds values.

        :param key: A JAX random number generator key (PRNG state).
        :param sample_shape: Desired sample dimensions to prepend to the batch shape.
        :return: Non-negative integer samples representing success counts.
        """
        assert is_prng_key(key)
        return binomial(
            key, self.probs, n=self.total_count, shape=sample_shape + self.batch_shape
        )

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        r"""Evaluate the log probability mass function at specified count
        configurations.

        The log probability mass function is computed entirely in log-space using a
        numerically-stable formulation that avoids sigmoid underflow/overflow:

        .. math::
            \ln P(X = k | n, \alpha) = \ln \binom{n}{k} + (k - n) \alpha
            - n \ln(1 + \sigma(-|\alpha|))

        The binomial coefficient in log-space is computed using the log-gamma function:

        .. math::
            \ln \binom{n}{k} = \ln\Gamma(n + 1) - \ln\Gamma(k + 1)
            - \ln\Gamma(n - k + 1)

        This approach using :func:`~jax.scipy.special.gammaln` avoids
        computing factorials explicitly.

        :param value: Count observation(s) in the range :math:`[0, n]`.
        :return: Log probability scores evaluated under the Binomial PMF.
        """
        total_count = jnp.array(self.total_count, dtype=jnp.result_type(float))
        log_factorial_n = gammaln(total_count + 1)
        log_factorial_k = gammaln(value + 1)
        log_factorial_nmk = gammaln(total_count - value + 1)
        normalize_term = (
            self.total_count * jnp.clip(self.logits, 0)
            + xlog1py(total_count, jnp.exp(-jnp.abs(self.logits)))
            - log_factorial_n
        )
        return (
            value * self.logits - log_factorial_k - log_factorial_nmk - normalize_term
        )

    @lazy_property
    def probs(self) -> ArrayLike:
        r"""The success probability per trial of the Binomial distribution is given by
        the sigmoid of the log-odds parameter:

        .. math::
            p = \sigma(\alpha) = \frac{1}{1 + e^{-\alpha}}
        """
        return _to_probs_bernoulli(self.logits)

    @property
    def mean(self) -> ArrayLike:
        r"""The mean of the Binomial distribution is given by:

        .. math::
            E[X] = n \sigma(\alpha)
        """
        return jnp.broadcast_to(self.total_count * self.probs, self.batch_shape)

    @property
    def variance(self) -> ArrayLike:
        r"""The variance of the Binomial distribution is given by:

        .. math::
            \mathrm{Var}[X] = n \sigma(\alpha) (1 - \sigma(\alpha))
        """
        return jnp.broadcast_to(
            self.total_count * self.probs * (1 - self.probs), self.batch_shape
        )

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self) -> constraints.Constraint:
        r"""The support of the Binomial distribution is the set of integer counts from
        0 to the total count."""
        return constraints.integer_interval(0, self.total_count)


def Binomial(
    total_count: ArrayLike = 1,
    probs: Optional[ArrayLike] = None,
    logits: Optional[ArrayLike] = None,
    *,
    validate_args: Optional[bool] = None,
) -> Union[BinomialProbs, BinomialLogits]:
    r"""Factory function to create a Binomial distribution instance from either
    probability or log-odds parameterization.

    :param total_count: Number of trials (non-negative integer), defaults to 1
    :param probs: The success probability parameter in the unit interval :math:`[0, 1]`,
        defaults to None
    :param logits: The log-odds parameter, defaults to None
    :param validate_args: Optional toggle to enforce simplex constraint during
        graph construction. Default is None
    :return: A Binomial distribution instance corresponding to the specified
        parameterization.
    """
    assert_one_of(probs=probs, logits=logits)
    if probs is not None:
        return BinomialProbs(probs, total_count, validate_args=validate_args)
    elif logits is not None:
        return BinomialLogits(logits, total_count, validate_args=validate_args)


class CategoricalProbs(Distribution):
    r"""A Categorical discrete random variable over :math:`K` mutually exclusive
    outcomes, parameterized by a probability vector on the simplex.

    The Probability Mass Function (PMF) of the Categorical distribution is defined as:

    .. math::
        P(X = k \mid \mathbf{p}) = p_k, \quad k \in \{0, 1, \dots, K-1\}

    where the probability vector :math:`\mathbf{p} = (p_0, p_1, \dots, p_{K-1})`
    satisfies :math:`p_k \ge 0` and :math:`\sum_{k=0}^{K-1} p_k = 1`.

    Where, :math:`\mathbf{p}` represents the category probability vector
    (:attr:`probs`), :math:`K` is the number of categories (the size of the trailing
    dimension of :attr:`probs`), and :math:`k` is the observed category index
    (:attr:`value`). The support domain is :math:`k \in \{0, 1, \dots, K-1\}`.
    """

    arg_constraints = {"probs": constraints.simplex}
    has_enumerate_support = True

    def __init__(self, probs: Array, *, validate_args: Optional[bool] = None):
        r"""
        :param probs: Category probability vector on the simplex; the trailing
            dimension indexes the :math:`K` categories and must sum to one.
        :param validate_args: If True, enforce domain constraints during initialization.
        """
        if jnp.ndim(probs) < 1:
            raise ValueError("`probs` parameter must be at least one-dimensional.")
        self.probs = probs
        super(CategoricalProbs, self).__init__(
            batch_shape=jnp.shape(self.probs)[:-1], validate_args=validate_args
        )

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> ArrayLike:
        r"""Draw samples from the Categorical distribution.

        This method delegates to :func:`~numpyro.distributions.util.categorical`, which
        internally relies on :func:`~jax.random.categorical` over the log-probabilities
        of :attr:`probs`.

        :param key: A JAX random number generator key (PRNG state).
        :param sample_shape: Desired sample dimensions to prepend to the batch shape.
        :return: Integer-valued samples in :math:`\{0, 1, \dots, K-1\}` drawn from the
            Categorical distribution.
        """
        assert is_prng_key(key)
        return categorical(key, self.probs, shape=sample_shape + self.batch_shape)

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        r"""Evaluate the log probability mass function at specified category indices.

        .. math::
            \ln P(X = k \mid \mathbf{p}) = \ln p_k

        The implementation gathers the log-probabilities from :attr:`logits` (which are
        already normalized log-probabilities computed from :attr:`probs`) at the
        positions indicated by ``value`` using :func:`~jax.numpy.take_along_axis`.

        :param value: Category index/indices to score (:math:`k \in \{0, 1, \dots, K-1\}`).
        :return: Log probability scores evaluated under the Categorical PMF.
        """
        batch_shape = lax.broadcast_shapes(jnp.shape(value), self.batch_shape)
        value = jnp.expand_dims(value, axis=-1)
        value = jnp.broadcast_to(value, batch_shape + (1,))
        logits = self.logits
        log_pmf = jnp.broadcast_to(logits, batch_shape + jnp.shape(logits)[-1:])
        return jnp.take_along_axis(log_pmf, value, axis=-1)[..., 0]

    @lazy_property
    def logits(self) -> ArrayLike:
        r"""The log-probability (logits) parameter of the Categorical distribution is
        the (already-normalized) log of the category probabilities:

        .. math::
            \alpha_k = \ln p_k, \quad k \in \{0, 1, \dots, K-1\}
        """
        return _to_logits_multinom(self.probs)

    @property
    def mean(self) -> ArrayLike:
        r"""The mean of a Categorical distribution over arbitrary unordered categories
        is not well-defined. This property therefore returns ``NaN``.

        :return: An array of NaNs with shape equal to :attr:`batch_shape`.
        """
        return jnp.full(self.batch_shape, jnp.nan, dtype=jnp.result_type(self.probs))

    @property
    def variance(self) -> ArrayLike:
        r"""The variance of a Categorical distribution over arbitrary unordered
        categories is not well-defined. This property therefore returns ``NaN``.

        :return: An array of NaNs with shape equal to :attr:`batch_shape`.
        """
        return jnp.full(self.batch_shape, jnp.nan, dtype=jnp.result_type(self.probs))

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self) -> constraints.Constraint:
        r"""The support of the Categorical distribution is the set of integers
        :math:`\{0, 1, \dots, K-1\}`, where :math:`K` is the number of categories
        inferred from the trailing dimension of :attr:`probs`.
        """
        return constraints.integer_interval(0, jnp.shape(self.probs)[-1] - 1)

    def enumerate_support(self, expand: bool = True) -> ArrayLike:
        r"""Enumerate all values in the support of the Categorical distribution.

        :param expand: Whether to broadcast the enumerated values across the batch
            shape. Default is True.
        :return: An array of integer category indices :math:`\{0, 1, \dots, K-1\}`,
            optionally broadcast across the batch dimensions.
        """
        values = jnp.arange(self.probs.shape[-1]).reshape(
            (-1,) + (1,) * len(self.batch_shape)
        )
        if expand:
            values = jnp.broadcast_to(values, values.shape[:1] + self.batch_shape)
        return values

    def entropy(self) -> ArrayLike:
        r"""The entropy of the Categorical distribution is given by:

        .. math::
            H[X] = -\sum_{k=0}^{K-1} p_k \ln p_k

        :return: The entropy of the Categorical distribution.
        """
        return -(self.probs * jnp.log(self.probs)).sum(axis=-1)


class CategoricalLogits(Distribution):
    r"""A Categorical discrete random variable over :math:`K` mutually exclusive
    outcomes, parameterized by unnormalized log-probabilities (logits).

    The Probability Mass Function (PMF) of the Categorical distribution is defined,
    via the softmax transformation of the logits, as:

    .. math::
        P(X = k \mid \boldsymbol{\alpha}) = \frac{\exp(\alpha_k)}{\sum_{j=0}^{K-1}
        \exp(\alpha_j)}, \quad k \in \{0, 1, \dots, K-1\}

    Where, :math:`\boldsymbol{\alpha} = (\alpha_0, \alpha_1, \dots, \alpha_{K-1})`
    is the real-valued logits vector (:attr:`logits`), :math:`K` is the number of
    categories (the size of the trailing dimension of :attr:`logits`), and :math:`k`
    is the observed category index (:attr:`value`). The support domain is
    :math:`k \in \{0, 1, \dots, K-1\}`.
    """

    arg_constraints = {"logits": constraints.real_vector}
    has_enumerate_support = True

    def __init__(self, logits: Array, *, validate_args: Optional[bool] = None):
        r"""
        :param logits: Real-valued logits vector; the trailing dimension indexes the
            :math:`K` categories. Logits are unnormalized and converted to
            probabilities via the softmax function.
        :param validate_args: If True, enforce domain constraints during initialization.
        """
        if jnp.ndim(logits) < 1:
            raise ValueError("`logits` parameter must be at least one-dimensional.")
        self.logits = logits
        super(CategoricalLogits, self).__init__(
            batch_shape=jnp.shape(logits)[:-1], validate_args=validate_args
        )

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> ArrayLike:
        r"""Draw samples from the Categorical distribution.

        This method invokes :func:`~jax.random.categorical` directly, which samples in
        logit-space using the Gumbel-max trick and therefore avoids materializing the
        softmax-normalized probabilities.

        :param key: A JAX random number generator key (PRNG state).
        :param sample_shape: Desired sample dimensions to prepend to the batch shape.
        :return: Integer-valued samples in :math:`\{0, 1, \dots, K-1\}` drawn from the
            Categorical distribution.
        """
        assert is_prng_key(key)
        return random.categorical(
            key, self.logits, shape=sample_shape + self.batch_shape
        )

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        r"""Evaluate the log probability mass function at specified category indices.

        .. math::
            \ln P(X = k \mid \boldsymbol{\alpha}) = \alpha_k - \ln\!\sum_{j=0}^{K-1}
            \exp(\alpha_j)

        The normalizing log-partition is computed via :func:`~jax.scipy.special.logsumexp`,
        which uses the standard max-subtraction trick to guarantee numerical stability
        in the presence of large or widely-spread logit magnitudes. After
        normalization, the relevant log-probability is gathered with
        :func:`~jax.numpy.take_along_axis`.

        :param value: Category index/indices to score (:math:`k \in \{0, 1, \dots, K-1\}`).
        :return: Log probability scores evaluated under the Categorical PMF.
        """
        batch_shape = lax.broadcast_shapes(jnp.shape(value), self.batch_shape)
        value = jnp.expand_dims(value, -1)
        value = jnp.broadcast_to(value, batch_shape + (1,))
        log_pmf = self.logits - logsumexp(self.logits, axis=-1, keepdims=True)
        log_pmf = jnp.broadcast_to(log_pmf, batch_shape + jnp.shape(log_pmf)[-1:])
        return jnp.take_along_axis(log_pmf, value, -1)[..., 0]

    @lazy_property
    def probs(self) -> ArrayLike:
        r"""The probability vector of the Categorical distribution is given by the
        softmax of the logits:

        .. math::
            p_k = \frac{\exp(\alpha_k)}{\sum_{j=0}^{K-1} \exp(\alpha_j)},
            \quad k \in \{0, 1, \dots, K-1\}
        """
        return _to_probs_multinom(self.logits)

    @property
    def mean(self) -> ArrayLike:
        r"""The mean of a Categorical distribution over arbitrary unordered categories
        is not well-defined. This property therefore returns ``NaN``.

        :return: An array of NaNs with shape equal to :attr:`batch_shape`.
        """
        return jnp.full(self.batch_shape, jnp.nan, dtype=jnp.result_type(self.logits))

    @property
    def variance(self) -> ArrayLike:
        r"""The variance of a Categorical distribution over arbitrary unordered
        categories is not well-defined. This property therefore returns ``NaN``.

        :return: An array of NaNs with shape equal to :attr:`batch_shape`.
        """
        return jnp.full(self.batch_shape, jnp.nan, dtype=jnp.result_type(self.logits))

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self) -> constraints.Constraint:
        r"""The support of the Categorical distribution is the set of integers
        :math:`\{0, 1, \dots, K-1\}`, where :math:`K` is the number of categories
        inferred from the trailing dimension of :attr:`logits`.
        """
        return constraints.integer_interval(0, jnp.shape(self.logits)[-1] - 1)

    def enumerate_support(self, expand: bool = True) -> ArrayLike:
        r"""Enumerate all values in the support of the Categorical distribution.

        :param expand: Whether to broadcast the enumerated values across the batch
            shape. Default is True.
        :return: An array of integer category indices :math:`\{0, 1, \dots, K-1\}`,
            optionally broadcast across the batch dimensions.
        """
        values = jnp.arange(self.logits.shape[-1]).reshape(
            (-1,) + (1,) * len(self.batch_shape)
        )
        if expand:
            values = jnp.broadcast_to(values, values.shape[:1] + self.batch_shape)
        return values

    def entropy(self) -> ArrayLike:
        r"""The entropy of the Categorical distribution is given by:

        .. math::
            H[X] = -\sum_{k=0}^{K-1} p_k \ln p_k
            = \ln\!\sum_{j=0}^{K-1} \exp(\alpha_j) - \sum_{k=0}^{K-1} p_k\, \alpha_k

        where :math:`p_k = \mathrm{softmax}(\boldsymbol{\alpha})_k`. The implementation
        uses :func:`~jax.scipy.special.logsumexp` for the log-partition term, ensuring
        numerical stability for large or widely-spread logits.

        :return: The entropy of the Categorical distribution.
        """
        probs = softmax(self.logits, axis=-1)
        return -(probs * self.logits).sum(axis=-1) + logsumexp(self.logits, axis=-1)


def Categorical(probs=None, logits=None, *, validate_args: Optional[bool] = None):
    assert_one_of(probs=probs, logits=logits)
    if probs is not None:
        return CategoricalProbs(probs, validate_args=validate_args)
    elif logits is not None:
        return CategoricalLogits(logits, validate_args=validate_args)


class DiscreteUniform(Distribution):
    r"""A discrete uniform random variable over the inclusive integer interval
    :math:`\{a, a+1, \dots, b\}`, where :math:`a` (:attr:`low`) is the inclusive
    lower bound and :math:`b` (:attr:`high`) is the inclusive upper bound of the
    support.

    The Probability Mass Function (PMF) of the discrete uniform distribution is
    defined as:

    .. math::
        P(X = k \mid a, b) = \frac{1}{b - a + 1},
        \quad k \in \{a, a+1, \dots, b\}

    Where :math:`k` is the observed integer value (:attr:`value`).
    """

    arg_constraints = {
        "low": constraints.dependent(is_discrete=True, event_dim=0),
        "high": constraints.dependent(is_discrete=True, event_dim=0),
    }
    has_enumerate_support = True
    pytree_data_fields = ("low", "high", "_support")

    def __init__(
        self,
        low: ArrayLike = 0,
        high: ArrayLike = 1,
        *,
        validate_args: Optional[bool] = None,
    ):
        r"""
        :param low: Inclusive lower bound of the integer support. Default is 0.
        :param high: Inclusive upper bound of the integer support. Must satisfy
            ``high >= low``. Default is 1.
        :param validate_args: If True, enforce domain constraints during initialization.
        """
        self.low, self.high = promote_shapes(low, high)
        batch_shape = lax.broadcast_shapes(jnp.shape(low), jnp.shape(high))
        self._support = constraints.integer_interval(low, high)
        super().__init__(batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self) -> constraints.Constraint:
        r"""The support of the discrete uniform distribution is the set of integers
        :math:`\{\text{low}, \text{low}+1, \dots, \text{high}\}`.
        """
        return self._support

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> ArrayLike:
        r"""Draw samples from the discrete uniform distribution.

        This method invokes :func:`~jax.random.randint` directly, which generates
        uniformly distributed integers in the half-open interval ``[low, high + 1)``,
        equivalent to the inclusive interval :math:`\{a, \dots, b\}`.

        :param key: A JAX random number generator key (PRNG state).
        :param sample_shape: Desired sample dimensions to prepend to the batch shape.
        :return: Integer-valued samples drawn uniformly from
            :math:`\{a, \dots, b\}`.
        """
        shape = sample_shape + self.batch_shape
        return random.randint(key, shape=shape, minval=self.low, maxval=self.high + 1)

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        r"""Evaluate the log probability mass function at specified integer values.

        .. math::
            \ln P(X = k \mid a, b) = -\ln(b - a + 1)

        The log-PMF is constant over the support, so the implementation simply
        broadcasts the negative log of the support cardinality to the requested shape.

        :param value: Integer observation(s) to score
            (:math:`k \in \{a, \dots, b\}`).
        :return: Log probability scores evaluated under the discrete uniform PMF.
        """
        shape = lax.broadcast_shapes(jnp.shape(value), self.batch_shape)
        return -jnp.broadcast_to(jnp.log(self.high + 1 - self.low), shape)

    def cdf(self, value: ArrayLike) -> ArrayLike:
        r"""Evaluate the cumulative distribution function (CDF) of the discrete
        uniform distribution.

        .. math::
            F(x) = \frac{\lfloor x \rfloor + 1 - a}{b - a + 1},
            \quad \text{clipped to } [0, 1]

        :param value: Point(s) at which to evaluate the CDF.
        :return: The CDF evaluated at ``value``, clipped to the unit interval.
        """
        cdf = (jnp.floor(value) + 1 - self.low) / (self.high - self.low + 1)
        return jnp.clip(cdf, 0.0, 1.0)

    def icdf(self, value: ArrayLike) -> ArrayLike:
        r"""Evaluate the inverse cumulative distribution function (quantile function)
        of the discrete uniform distribution.

        .. math::
            F^{-1}(u) = a + u\,(b - a + 1) - 1, \quad u \in [0, 1]

        :param value: Quantile level(s) :math:`u \in [0, 1]`.
        :return: The inverse CDF evaluated at ``value``.
        """
        return self.low + value * (self.high - self.low + 1) - 1

    @property
    def mean(self) -> ArrayLike:
        r"""The mean of the discrete uniform distribution is the midpoint of the
        support:

        .. math::
            E[X] = \frac{a + b}{2}

        :return: The mean of the discrete uniform distribution.
        """
        return self.low + (self.high - self.low) / 2.0

    @property
    def variance(self) -> ArrayLike:
        r"""The variance of the discrete uniform distribution is given by:

        .. math::
            \mathrm{Var}[X] = \frac{(b - a + 1)^2 - 1}{12}

        :return: The variance of the discrete uniform distribution.
        """
        return ((self.high - self.low + 1) ** 2 - 1) / 12.0

    def enumerate_support(self, expand: bool = True) -> ArrayLike:
        r"""Enumerate all values in the support of the discrete uniform distribution.

        Both :attr:`low` and :attr:`high` must be concrete (non-JAX-tracer) values and
        homogeneous across the batch shape; otherwise a :class:`NotImplementedError`
        is raised.

        :param expand: Whether to broadcast the enumerated values across the batch
            shape. Default is True.
        :return: An array of integer values
            :math:`\{a, a+1, \dots, b\}`, optionally
            broadcast across the batch dimensions.
        """
        if not not_jax_tracer(self.high) or not not_jax_tracer(self.low):
            raise NotImplementedError("Both `low` and `high` must not be a JAX Tracer.")
        if np.any(np.amax(self.low) != self.low):
            # NB: the error can't be raised if inhomogeneous issue happens when tracing
            raise NotImplementedError(
                "Inhomogeneous `low` not supported by `enumerate_support`."
            )
        if np.any(np.amax(self.high) != self.high):
            # NB: the error can't be raised if inhomogeneous issue happens when tracing
            raise NotImplementedError(
                "Inhomogeneous `high` not supported by `enumerate_support`."
            )
        low = np.reshape(self.low, -1)[0]
        high = np.reshape(self.high, -1)[0]
        values = jnp.arange(low, high + 1).reshape((-1,) + (1,) * len(self.batch_shape))
        if expand:
            values = jnp.broadcast_to(values, values.shape[:1] + self.batch_shape)
        return values

    def entropy(self) -> ArrayLike:
        r"""The entropy of the discrete uniform distribution is given by:

        .. math::
            H[X] = \ln(\text{high} - \text{low} + 1)

        :return: The entropy of the discrete uniform distribution.
        """
        return jnp.log(self.high - self.low + 1)


class OrderedLogistic(CategoricalProbs):
    """
    A categorical distribution with ordered outcomes.

    **References:**

    1. *Stan Functions Reference, v2.20 section 12.6*,
       Stan Development Team

    :param numpy.ndarray predictor: prediction in real domain; typically this is output
        of a linear model.
    :param numpy.ndarray cutpoints: positions in real domain to separate categories.
    """

    arg_constraints = {
        "predictor": constraints.real,
        "cutpoints": constraints.ordered_vector,
    }

    def __init__(
        self,
        predictor: ArrayLike,
        cutpoints: ArrayLike,
        *,
        validate_args: Optional[bool] = None,
    ):
        if jnp.ndim(predictor) == 0:
            (predictor,) = promote_shapes(predictor, shape=(1,))
        else:
            predictor = predictor[..., None]
        predictor, self.cutpoints = promote_shapes(predictor, cutpoints)
        self.predictor = predictor[..., 0]
        probs = transforms.SimplexToOrderedTransform(self.predictor).inv(self.cutpoints)
        super(OrderedLogistic, self).__init__(probs, validate_args=validate_args)

    @staticmethod
    def infer_shapes(predictor, cutpoints):
        batch_shape = lax.broadcast_shapes(predictor, cutpoints[:-1])
        event_shape = ()
        return batch_shape, event_shape

    def entropy(self) -> ArrayLike:
        raise NotImplementedError


class MultinomialProbs(Distribution):
    arg_constraints = {
        "probs": constraints.simplex,
        "total_count": constraints.nonnegative_integer,
    }
    pytree_data_fields = ("probs",)
    pytree_aux_fields = ("total_count", "total_count_max")

    def __init__(
        self,
        probs: Array,
        total_count: ArrayLike = 1,
        *,
        total_count_max: Optional[int] = None,
        validate_args: Optional[bool] = None,
    ):
        if jnp.ndim(probs) < 1:
            raise ValueError("`probs` parameter must be at least one-dimensional.")
        batch_shape, event_shape = self.infer_shapes(
            jnp.shape(probs), jnp.shape(total_count)
        )
        self.probs = promote_shapes(probs, shape=batch_shape + jnp.shape(probs)[-1:])[0]
        self.total_count = promote_shapes(total_count, shape=batch_shape)[0]
        self.total_count_max = total_count_max
        super(MultinomialProbs, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> ArrayLike:
        assert is_prng_key(key)
        return multinomial(
            key,
            self.probs,
            self.total_count,
            shape=sample_shape + self.batch_shape,
            total_count_max=self.total_count_max,
        )

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        value = jnp.array(value, jnp.result_type(float))
        return gammaln(self.total_count + 1) + jnp.sum(
            xlogy(value, self.probs) - gammaln(value + 1), axis=-1
        )

    @lazy_property
    def logits(self) -> ArrayLike:
        return _to_logits_multinom(self.probs)

    @property
    def mean(self) -> ArrayLike:
        return self.probs * jnp.expand_dims(self.total_count, -1)

    @property
    def variance(self) -> ArrayLike:
        return jnp.expand_dims(self.total_count, -1) * self.probs * (1 - self.probs)

    @constraints.dependent_property(is_discrete=True, event_dim=1)
    def support(self) -> constraints.Constraint:
        return constraints.multinomial(self.total_count)

    @staticmethod
    def infer_shapes(
        probs: Array, total_count: ArrayLike
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        batch_shape = lax.broadcast_shapes(probs[:-1], total_count)
        event_shape = probs[-1:]
        return batch_shape, event_shape


class MultinomialLogits(Distribution):
    arg_constraints = {
        "logits": constraints.real_vector,
        "total_count": constraints.nonnegative_integer,
    }
    pytree_data_fields = ("logits",)
    pytree_aux_fields = ("total_count", "total_count_max")

    def __init__(
        self,
        logits: Array,
        total_count: ArrayLike = 1,
        *,
        total_count_max: Optional[int] = None,
        validate_args: Optional[bool] = None,
    ):
        if jnp.ndim(logits) < 1:
            raise ValueError("`logits` parameter must be at least one-dimensional.")
        batch_shape, event_shape = self.infer_shapes(
            jnp.shape(logits), jnp.shape(total_count)
        )
        self.logits = promote_shapes(
            logits, shape=batch_shape + jnp.shape(logits)[-1:]
        )[0]
        self.total_count = promote_shapes(total_count, shape=batch_shape)[0]
        self.total_count_max = total_count_max
        super(MultinomialLogits, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> ArrayLike:
        assert is_prng_key(key)
        return multinomial(
            key,
            self.probs,
            self.total_count,
            shape=sample_shape + self.batch_shape,
            total_count_max=self.total_count_max,
        )

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        if self._validate_args:
            self._validate_sample(value)
        normalize_term = self.total_count * logsumexp(self.logits, axis=-1) - gammaln(
            self.total_count + 1
        )
        return (
            jnp.sum(value * self.logits - gammaln(value + 1), axis=-1) - normalize_term
        )

    @lazy_property
    def probs(self) -> ArrayLike:
        return _to_probs_multinom(self.logits)

    @property
    def mean(self) -> ArrayLike:
        return jnp.expand_dims(self.total_count, -1) * self.probs

    @property
    def variance(self) -> ArrayLike:
        return jnp.expand_dims(self.total_count, -1) * self.probs * (1 - self.probs)

    @constraints.dependent_property(is_discrete=True, event_dim=1)
    def support(self) -> constraints.Constraint:
        return constraints.multinomial(self.total_count)

    @staticmethod
    def infer_shapes(
        logits: Array, total_count: ArrayLike
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        batch_shape = lax.broadcast_shapes(logits[:-1], total_count)
        event_shape = logits[-1:]
        return batch_shape, event_shape


def Multinomial(
    total_count=1,
    probs: Array = None,
    logits: Array = None,
    *,
    total_count_max: Optional[int] = None,
    validate_args: Optional[bool] = None,
) -> Union[MultinomialProbs, MultinomialLogits]:
    """Multinomial distribution.

    :param total_count: number of trials. If this is a JAX array,
        it is required to specify `total_count_max`.
    :param probs: event probabilities
    :param logits: event log probabilities
    :param int total_count_max: the maximum number of trials,
        i.e. `max(total_count)`
    """
    assert_one_of(probs=probs, logits=logits)
    if probs is not None:
        return MultinomialProbs(
            probs,
            total_count,
            total_count_max=total_count_max,
            validate_args=validate_args,
        )
    elif logits is not None:
        return MultinomialLogits(
            logits,
            total_count,
            total_count_max=total_count_max,
            validate_args=validate_args,
        )


class Poisson(Distribution):
    r"""
    Creates a Poisson distribution parameterized by rate, the rate parameter.

    Samples are nonnegative integers, with a pmf given by

    .. math::
      \mathrm{rate}^k \frac{e^{-\mathrm{rate}}}{k!}

    :param numpy.ndarray rate: The rate parameter
    :param bool is_sparse: Whether to assume value is mostly zero when computing
        :meth:`log_prob`, which can speed up computation when data is sparse.
    """

    arg_constraints = {"rate": constraints.greater_than_eq(0.0)}
    support = constraints.nonnegative_integer
    pytree_aux_fields = ("is_sparse",)

    def __init__(
        self,
        rate: ArrayLike,
        *,
        is_sparse: bool = False,
        validate_args: Optional[bool] = None,
    ):
        self.rate = rate
        self.is_sparse = is_sparse
        super(Poisson, self).__init__(jnp.shape(rate), validate_args=validate_args)

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> ArrayLike:
        assert is_prng_key(key)
        return random.poisson(key, self.rate, shape=sample_shape + self.batch_shape)

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        # Using an integer vs. floating-point `rate` leads to differing results.
        # To ensure consistent behavior, `rate` is explicitly cast to a floating-point type.
        # See: https://github.com/pyro-ppl/numpyro/issues/2181
        ftype = jnp.result_type(float)
        rate = jnp.astype(self.rate, ftype)

        if (
            self.is_sparse
            and not isinstance(value, jax.core.Tracer)
            and jnp.size(value) > 1
        ):
            shape = lax.broadcast_shapes(self.batch_shape, jnp.shape(value))
            rate = jnp.broadcast_to(rate, shape).reshape(-1)
            nonzero = np.broadcast_to(jax.device_get(value) > 0, shape).reshape(-1)
            value = jnp.broadcast_to(value, shape).reshape(-1)
            sparse_value = value[nonzero]
            sparse_rate = rate[nonzero]
            return (
                jnp.asarray(-rate, dtype=ftype)
                .at[nonzero]
                .add(
                    jnp.log(sparse_rate) * sparse_value - gammaln(sparse_value + 1),
                )
                .reshape(shape)
            )
        _value = jnp.astype(value, ftype)
        return xlogy(_value, rate) - gammaln(_value + 1.0) - rate

    @property
    def mean(self) -> ArrayLike:
        return self.rate

    @property
    def variance(self) -> ArrayLike:
        return self.rate

    def cdf(self, value: ArrayLike) -> ArrayLike:
        k = jnp.floor(value) + 1
        return gammaincc(k, self.rate)


class ZeroInflatedProbs(Distribution):
    arg_constraints = {"gate": constraints.unit_interval}
    pytree_data_fields = ("base_dist", "gate")

    def __init__(
        self,
        base_dist: Distribution,
        gate: ArrayLike,
        *,
        validate_args: Optional[bool] = None,
    ):
        batch_shape = lax.broadcast_shapes(jnp.shape(gate), base_dist.batch_shape)
        (self.gate,) = promote_shapes(gate, shape=batch_shape)
        assert base_dist.support.is_discrete
        if base_dist.event_shape:
            raise ValueError(
                "ZeroInflatedProbs expected empty base_dist.event_shape but got {}".format(
                    base_dist.event_shape
                )
            )
        # XXX: we might need to promote parameters of base_dist but let's keep
        # this simplified for now
        self.base_dist = base_dist.expand(batch_shape)
        super(ZeroInflatedProbs, self).__init__(
            batch_shape, validate_args=validate_args
        )

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> ArrayLike:
        assert is_prng_key(key)
        key_bern, key_base = random.split(key)
        shape = sample_shape + self.batch_shape
        mask = random.bernoulli(key_bern, self.gate, shape)
        samples = self.base_dist(rng_key=key_base, sample_shape=sample_shape)
        return jnp.where(mask, 0, samples)

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        log_prob = jnp.log1p(-self.gate) + self.base_dist.log_prob(value)
        return jnp.where(value == 0, jnp.log(self.gate + jnp.exp(log_prob)), log_prob)

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self) -> constraints.Constraint:
        return self.base_dist.support

    @lazy_property
    def mean(self) -> ArrayLike:
        return (1 - self.gate) * self.base_dist.mean

    @lazy_property
    def variance(self) -> ArrayLike:
        return (1 - self.gate) * (
            self.base_dist.mean**2 + self.base_dist.variance
        ) - self.mean**2

    @property
    def has_enumerate_support(self):
        return self.base_dist.has_enumerate_support

    def enumerate_support(self, expand: bool = True) -> ArrayLike:
        return self.base_dist.enumerate_support(expand=expand)


class ZeroInflatedLogits(ZeroInflatedProbs):
    arg_constraints = {"gate_logits": constraints.real}

    def __init__(
        self,
        base_dist: Distribution,
        gate_logits: ArrayLike,
        *,
        validate_args: Optional[bool] = None,
    ):
        gate = _to_probs_bernoulli(gate_logits)
        batch_shape = lax.broadcast_shapes(jnp.shape(gate), base_dist.batch_shape)
        (self.gate_logits,) = promote_shapes(gate_logits, shape=batch_shape)
        super().__init__(base_dist, gate, validate_args=validate_args)

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        log_prob_minus_log_gate = -self.gate_logits + self.base_dist.log_prob(value)
        log_gate = -softplus(-self.gate_logits)
        log_prob = log_prob_minus_log_gate + log_gate
        zero_log_prob = softplus(log_prob_minus_log_gate) + log_gate
        return jnp.where(value == 0, zero_log_prob, log_prob)


def ZeroInflatedDistribution(
    base_dist: Distribution,
    *,
    gate: Optional[ArrayLike] = None,
    gate_logits: Optional[ArrayLike] = None,
    validate_args: Optional[bool] = None,
) -> Union[ZeroInflatedProbs, ZeroInflatedLogits]:
    """
    Generic Zero Inflated distribution.

    :param Distribution base_dist: the base distribution.
    :param numpy.ndarray gate: probability of extra zeros given via a Bernoulli distribution.
    :param numpy.ndarray gate_logits: logits of extra zeros given via a Bernoulli distribution.
    """
    assert_one_of(gate=gate, gate_logits=gate_logits)
    if gate is not None:
        return ZeroInflatedProbs(base_dist, gate, validate_args=validate_args)
    else:
        return ZeroInflatedLogits(base_dist, gate_logits, validate_args=validate_args)


class ZeroInflatedPoisson(ZeroInflatedProbs):
    """
    A Zero Inflated Poisson distribution.

    :param numpy.ndarray gate: probability of extra zeros.
    :param numpy.ndarray rate: rate of Poisson distribution.
    """

    arg_constraints = {"gate": constraints.unit_interval, "rate": constraints.positive}
    support = constraints.nonnegative_integer
    pytree_data_fields = ("rate",)

    # TODO: resolve inconsistent parameter order w.r.t. Pyro
    # and support `gate_logits` argument
    def __init__(
        self,
        gate: ArrayLike,
        rate: ArrayLike = 1.0,
        *,
        validate_args: Optional[bool] = None,
    ) -> None:
        _, self.rate = promote_shapes(gate, rate)
        super().__init__(Poisson(self.rate), gate, validate_args=validate_args)


class HurdleProbs(Distribution):
    r"""Generic hurdle distribution parameterized by a probability :math:`g` (``gate``)
    of the structural zero and an arbitrary base distribution.

    **Hurdle mechanism.** A hurdle model is a two-part model. A Bernoulli "hurdle"
    decides whether the outcome is zero (with probability :math:`g`, the *gate*) or
    strictly positive (with probability :math:`1 - g`). Conditional on the outcome
    being positive, the magnitude is drawn from the base distribution -
    *zero-truncated* in the discrete case so the base distribution cannot itself
    produce a zero. With :math:`B` denoting the base PMF/PDF:

    .. math::

        P(X = 0) = g, \qquad
        P(X = k) = (1 - g) \, \frac{B(k)}{1 - B(0)} \;\text{for } k \geq 1
        \;\text{(discrete base)}

    For a continuous base on :math:`\mathbb{R}_{>0}` the truncation factor
    :math:`1 - B(0)` equals 1 and the formula simplifies to a point mass at 0 with
    weight :math:`g` mixed with :math:`(1 - g) \, b(x)` on :math:`x > 0`.

    **Assumptions.**

    1. *All zeros are structural* - they originate exclusively from the hurdle
       process. This contrasts with zero-inflated models, which mix structural
       zeros with sampling zeros from the base distribution.
    2. The hurdle decision (zero vs. positive) and the magnitude (given positive)
       are *conditionally independent* given the parameters.
    3. For a discrete base, :math:`P(\text{base} = 0) < 1` so the truncation
       factor :math:`1 - B(0)` is well-defined. For a continuous base supported
       on :math:`\mathbb{R}_{>0}`, :math:`P(\text{base} = 0) = 0` and no
       truncation is needed.

    .. note::
        ``gate`` is the probability of a structural zero. This matches the convention
        used by :class:`ZeroInflatedDistribution`, and corresponds to ``1 - psi`` in
        PyMC's hurdle distributions.

    :param Distribution base_dist: the base distribution.
    :param ArrayLike gate: probability of a structural zero, in :math:`[0, 1]`.

    **References:**

    1. Cragg, J. G. (1971). Some Statistical Models for Limited Dependent
       Variables with Application to the Demand for Durable Goods.
       *Econometrica*, 39(5), 829-844.
    2. Mullahy, J. (1986). Specification and testing of some modified count
       data models. *Journal of Econometrics*, 33(3), 341-365.
    """

    arg_constraints = {"gate": constraints.unit_interval}
    pytree_data_fields = ("base_dist", "gate")
    pytree_aux_fields = ("_is_discrete",)

    def __init__(
        self,
        base_dist: Distribution,
        gate: ArrayLike,
        *,
        validate_args: Optional[bool] = None,
    ) -> None:
        batch_shape = lax.broadcast_shapes(jnp.shape(gate), base_dist.batch_shape)
        (self.gate,) = promote_shapes(gate, shape=batch_shape)
        if base_dist.event_shape:
            raise ValueError(
                "HurdleProbs expected empty base_dist.event_shape but got {}".format(
                    base_dist.event_shape
                )
            )
        self.base_dist = base_dist.expand(batch_shape)
        self._is_discrete = base_dist.support.is_discrete
        super(HurdleProbs, self).__init__(batch_shape, validate_args=validate_args)

    @constraints.dependent_property
    def support(self) -> constraints.Constraint:
        return self.base_dist.support

    def _log_one_minus_p_zero(self) -> ArrayLike:
        # log(1 - B(0)) for the discrete base, used to renormalize the truncated PMF.
        log_p0 = self.base_dist.log_prob(jnp.zeros((), dtype=jnp.result_type(int)))
        return jax.nn.log1mexp(-log_p0)

    def _log_gate(self) -> ArrayLike:
        return jnp.log(self.gate)

    def _log_one_minus_gate(self) -> ArrayLike:
        return jnp.log1p(-self.gate)

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> ArrayLike:
        assert is_prng_key(key)
        key_bern, key_base = random.split(key)
        shape = sample_shape + self.batch_shape
        zero_mask = random.bernoulli(key_bern, self.gate, shape)
        if self._is_discrete:
            samples = self._sample_truncated(key_base, sample_shape)
        else:
            samples = self.base_dist(rng_key=key_base, sample_shape=sample_shape)
        return jnp.where(zero_mask, jnp.zeros_like(samples), samples)

    def _sample_truncated(
        self, key: jax.Array, sample_shape: tuple[int, ...]
    ) -> ArrayLike:
        # Rejection sampling from the zero-truncated base distribution: redraw any
        # element that came back as 0 until all elements are strictly positive.
        first = self.base_dist(rng_key=key, sample_shape=sample_shape)

        def cond_fun(state: tuple) -> ArrayLike:
            _, current = state
            return jnp.any(current == 0)

        def body_fun(state: tuple) -> tuple:
            k, current = state
            k, sub = random.split(k)
            candidate = self.base_dist(rng_key=sub, sample_shape=sample_shape)
            current = jnp.where(current == 0, candidate, current)
            return (k, current)

        _, samples = lax.while_loop(cond_fun, body_fun, (key, first))
        return samples

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        log_gate = self._log_gate()
        log_one_minus_gate = self._log_one_minus_gate()
        # Replace zeros with ones before evaluating the base log_prob to avoid
        # -inf / NaN gradients when the base PDF is undefined at 0 (e.g. Gamma).
        safe_value = jnp.where(value == 0, jnp.ones_like(value), value)
        log_prob_base = self.base_dist.log_prob(safe_value)
        if self._is_discrete:
            log_nonzero = (
                log_one_minus_gate + log_prob_base - self._log_one_minus_p_zero()
            )
        else:
            log_nonzero = log_one_minus_gate + log_prob_base
        return jnp.where(value == 0, log_gate, log_nonzero)

    @lazy_property
    def mean(self) -> ArrayLike:
        if self._is_discrete:
            trunc = -jnp.expm1(
                self.base_dist.log_prob(jnp.zeros((), dtype=jnp.result_type(int)))
            )
            return (1 - self.gate) * self.base_dist.mean / trunc
        return (1 - self.gate) * self.base_dist.mean

    @lazy_property
    def variance(self) -> ArrayLike:
        if self._is_discrete:
            trunc = -jnp.expm1(
                self.base_dist.log_prob(jnp.zeros((), dtype=jnp.result_type(int)))
            )
            second_moment_trunc = (
                self.base_dist.mean**2 + self.base_dist.variance
            ) / trunc
            return (1 - self.gate) * second_moment_trunc - self.mean**2
        return (1 - self.gate) * (
            self.base_dist.mean**2 + self.base_dist.variance
        ) - self.mean**2


class HurdleLogits(HurdleProbs):
    r"""Hurdle distribution parameterized by ``gate_logits`` (the log-odds of the
    structural zero) instead of a probability.

    Like :class:`HurdleProbs`, this is a two-part model where a Bernoulli
    "hurdle" - here parameterized in logit space - selects between an exact
    zero and a positive draw from the (zero-truncated, for discrete bases) base
    distribution. See :class:`HurdleProbs` for the full mechanism, assumptions,
    and underlying PMF/PDF.

    :param Distribution base_dist: the base distribution.
    :param ArrayLike gate_logits: log-odds of a structural zero,
        :math:`\mathrm{logit}(g) = \log\frac{g}{1 - g}`.

    **References:**

    1. Cragg, J. G. (1971). Some Statistical Models for Limited Dependent
       Variables with Application to the Demand for Durable Goods.
       *Econometrica*, 39(5), 829-844.
    2. Mullahy, J. (1986). Specification and testing of some modified count
       data models. *Journal of Econometrics*, 33(3), 341-365.
    """

    arg_constraints = {"gate_logits": constraints.real}
    pytree_data_fields = ("base_dist", "gate_logits")

    def __init__(
        self,
        base_dist: Distribution,
        gate_logits: ArrayLike,
        *,
        validate_args: Optional[bool] = None,
    ) -> None:
        gate = _to_probs_bernoulli(gate_logits)
        batch_shape = lax.broadcast_shapes(jnp.shape(gate), base_dist.batch_shape)
        (self.gate_logits,) = promote_shapes(gate_logits, shape=batch_shape)
        super().__init__(base_dist, gate, validate_args=validate_args)

    def _log_gate(self) -> ArrayLike:
        return -softplus(-self.gate_logits)

    def _log_one_minus_gate(self) -> ArrayLike:
        return -softplus(self.gate_logits)


def HurdleDistribution(
    base_dist: Distribution,
    *,
    gate: Optional[ArrayLike] = None,
    gate_logits: Optional[ArrayLike] = None,
    validate_args: Optional[bool] = None,
) -> Union[HurdleProbs, HurdleLogits]:
    r"""Generic hurdle distribution.

    A hurdle model is a two-part model: a Bernoulli "hurdle" selects between an
    exact zero (with probability ``gate``) and a positive draw from the
    (zero-truncated, for discrete bases) base distribution. Returns a
    :class:`HurdleProbs` if ``gate`` is supplied, or a :class:`HurdleLogits` if
    ``gate_logits`` is supplied. Exactly one of the two must be provided. See
    :class:`HurdleProbs` for the full mechanism, assumptions, and PMF/PDF.

    :param Distribution base_dist: the base distribution.
    :param ArrayLike gate: probability of a structural zero.
    :param ArrayLike gate_logits: log-odds of a structural zero.

    **References:**

    1. Cragg, J. G. (1971). Some Statistical Models for Limited Dependent
       Variables with Application to the Demand for Durable Goods.
       *Econometrica*, 39(5), 829-844.
    2. Mullahy, J. (1986). Specification and testing of some modified count
       data models. *Journal of Econometrics*, 33(3), 341-365.
    """
    assert_one_of(gate=gate, gate_logits=gate_logits)
    if gate is not None:
        return HurdleProbs(base_dist, gate, validate_args=validate_args)
    return HurdleLogits(base_dist, gate_logits, validate_args=validate_args)


class HurdlePoisson(HurdleProbs):
    r"""A hurdle Poisson distribution: a two-part model in which structural zeros
    are produced by a Bernoulli "hurdle" with probability :math:`g` and positive
    counts follow a zero-truncated :math:`\mathrm{Poisson}(\lambda)`. The hurdle
    and the magnitude (given a positive count) are conditionally independent;
    see :class:`HurdleProbs` for the full mechanism and assumptions.

    The probability mass function is

    .. math::

        P(X = 0) = g, \qquad
        P(X = k) = (1 - g) \, \frac{\lambda^{k} e^{-\lambda} / k!}{1 - e^{-\lambda}}
        \;\text{for } k \geq 1.

    :param ArrayLike gate: probability of a structural zero, :math:`g \in [0, 1]`.
    :param ArrayLike rate: rate :math:`\lambda > 0` of the underlying Poisson.

    **References:**

    1. Mullahy, J. (1986). Specification and testing of some modified count
       data models. *Journal of Econometrics*, 33(3), 341-365.
    2. Cragg, J. G. (1971). Some Statistical Models for Limited Dependent
       Variables with Application to the Demand for Durable Goods.
       *Econometrica*, 39(5), 829-844.
    """

    arg_constraints = {"gate": constraints.unit_interval, "rate": constraints.positive}
    support = constraints.nonnegative_integer
    pytree_data_fields = ("rate",)

    def __init__(
        self,
        gate: ArrayLike,
        rate: ArrayLike = 1.0,
        *,
        validate_args: Optional[bool] = None,
    ) -> None:
        _, self.rate = promote_shapes(gate, rate)
        super().__init__(Poisson(self.rate), gate, validate_args=validate_args)


class GeometricProbs(Distribution):
    r"""A Geometric discrete random variable representing the number of failures
    before the first success, parameterized by the success probability
    (:attr:`probs`).

    The probability mass function (PMF) is defined as:

    .. math::

        P(X = k; p) = p(1-p)^k

    where :math:`p \in (0,1]` is the probability of success on each independent trial.
    :math:`k \in \{0, 1, 2, \ldots\}` is the number of failures before first success.
    Equivalently, the first success occurs on trial :math:`k+1`.

    :param probs: Probability of success on each trial (:math:`p`).
    :type probs: ArrayLike
    :param validate_args: Whether to validate input constraints, defaults to
        ``None``.
    :type validate_args: bool, optional
    """

    arg_constraints = {"probs": constraints.unit_interval}
    support = constraints.nonnegative_integer

    def __init__(self, probs: ArrayLike, *, validate_args: Optional[bool] = None):
        r"""
        :param probs: Probability of success on each trial (:math:`p`).
        :param validate_args: If True, enforce domain constraints during initialization.
        """
        self.probs = probs
        super(GeometricProbs, self).__init__(
            batch_shape=jnp.shape(self.probs), validate_args=validate_args
        )

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> ArrayLike:
        r"""Generates samples using inverse CDF method.

        For a uniform random variable :math:`U \sim \mathrm{Uniform}[0, 1)`,
        a Geometric sample is obtained as:

        .. math::
            X = \left\lfloor \frac{\log(1-U)}{\log(1-p)} \right\rfloor.

        :param key: JAX PRNGKey for reproducibility.
        :type key: jax.Array
        :param sample_shape: The shape of the samples to be generated.
        :type sample_shape: tuple[int, ...]
        :return: Samples from Geometric distribution of shape ``sample_shape + batch_shape``.
        :rtype: ArrayLike
        """
        assert is_prng_key(key)
        probs = self.probs
        dtype = jnp.result_type(probs)
        shape = sample_shape + self.batch_shape
        u = random.uniform(key, shape, dtype)
        return jnp.floor(jnp.log1p(-u) / jnp.log1p(-probs))

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        r"""Calculates the log of the probability mass function.

        .. math::
            \log P(X = k; p) = k\log(1-p) + \log p.

        :param value: Values at which to evaluate the log density. Values must be nonnegative integers.
        :type value: ArrayLike
        :return: Log probability mass.
        :rtype: ArrayLike
        """
        probs = jnp.where((self.probs == 1) & (value == 0), 0, self.probs)
        return value * jnp.log1p(-probs) + jnp.log(probs)

    @lazy_property
    def logits(self) -> ArrayLike:
        r"""Calculates the logits corresponding to the success probability.

        .. math::
            \ell = \log\left(\frac{p}{1-p}\right).
        """
        return _to_logits_bernoulli(self.probs)

    @property
    def mean(self) -> ArrayLike:
        r"""Calculates the mean of the Geometric distribution.

        .. math::
            \mathbb{E}[X] = \frac{1-p}{p}.
        """
        return 1.0 / self.probs - 1.0

    @property
    def variance(self) -> ArrayLike:
        r"""Calculates the variance of the Geometric distribution.

        .. math::
            \operatorname{Var}(X) = \frac{1-p}{p^2}.
        """
        return (1.0 / self.probs - 1.0) / self.probs

    def entropy(self) -> ArrayLike:
        r"""Entropy of the Geometric distribution.

        .. math::
            H(X) = -\log p - \frac{1-p}{p}\log(1-p).

        :return: Entropy of the Geometric distribution.
        :rtype: ArrayLike
        """
        return -(1 - self.probs) * jnp.log1p(-self.probs) / self.probs - jnp.log(
            self.probs
        )


class GeometricLogits(Distribution):
    r"""Geometric distribution parameterized by logits (:attr:`logits`).

    .. math::
        P(X = k \mid \ell) = \sigma(\ell)
        \left(1-\sigma(\ell)\right)^k,
        \qquad k \in \{0, 1, 2, \ldots\}.

    where :math:`\ell` denote the logits parameter,
    :math:`p = \sigma(\ell) = \frac{1}{1+\exp(-\ell)}` is the probability of success.

    :param logits: Logits of success on each trial (:math:`logits`).
    :type logits: ArrayLike
    :param validate_args: Whether to validate input constraints, defaults to
        ``None``.
    :type validate_args: bool, optional
    """

    arg_constraints = {"logits": constraints.real}
    support = constraints.nonnegative_integer

    def __init__(self, logits: ArrayLike, *, validate_args: Optional[bool] = None):
        r"""
        :param logits: Logits of success on each trial (:math:`logits`).
        :param validate_args: If True, enforce domain constraints during initialization.
        """
        self.logits = logits
        super(GeometricLogits, self).__init__(
            batch_shape=jnp.shape(self.logits), validate_args=validate_args
        )

    @lazy_property
    def probs(self) -> ArrayLike:
        r"""The success probability obtained by applying the sigmoid function
        to the logits.

        .. math::
            p = \sigma(\ell) = \frac{1}{1+\exp(-\ell)}.
        """
        return _to_probs_bernoulli(self.logits)

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> ArrayLike:
        r"""Generates samples using inverse CDF technique in logit space.

        :param key: JAX pseudo-random number generator key.
        :type key: jax.Array
        :param sample_shape: Sample dimensions to prepend to the batch shape.
        :type sample_shape: tuple[int, ...]
        :return: Samples from the Geometric distribution of shape
            ``sample_shape + batch_shape``.
        :rtype: ArrayLike
        """
        assert is_prng_key(key)
        logits = self.logits
        dtype = jnp.result_type(logits)
        shape = sample_shape + self.batch_shape
        u = random.uniform(key, shape, dtype)
        return jnp.floor(jnp.log1p(-u) / -softplus(logits))

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        r"""Calculates the log probability mass function.

        .. math::
            \log P(X = k; \ell) = \log\sigma(\ell)
            + k\log\left(1-\sigma(\ell)\right).

        :param value: Number of failures before the first success. Values must
            be nonnegative integers.
        :type value: ArrayLike
        :return: Log probability mass.
        :rtype: ArrayLike
        """
        return (-value - 1) * softplus(self.logits) + self.logits

    @property
    def mean(self) -> ArrayLike:
        r"""Calculates the mean of the Geometric distribution.

        .. math::
            E[X] = \frac{1-p}{p},

        where :math:`p=\sigma(\ell)`.
        """
        return 1.0 / self.probs - 1.0

    @property
    def variance(self) -> ArrayLike:
        r"""Calculates the variance of the Geometric distribution.

        .. math::
            \operatorname{Var}(X) = \frac{1-p}{p^2},

        where :math:`p=\sigma(\ell)`.
        """
        return (1.0 / self.probs - 1.0) / self.probs

    def entropy(self) -> ArrayLike:
        r"""Calculates the entropy of the Geometric distribution.

        .. math::
            H(X) = -\log p - \frac{1-p}{p}\log(1-p),

        where :math:`p=\sigma(\ell)`.

        :return: Entropy of the Geometric distribution.
        :rtype: ArrayLike
        """
        logq = -jax.nn.softplus(self.logits)
        logp = -jax.nn.softplus(-self.logits)
        p = jax.scipy.special.expit(self.logits)
        p_clip = jnp.clip(p, jnp.finfo(p.dtype).tiny)
        return -(1 - p) * logq / p_clip - logp


def Geometric(
    probs: Optional[ArrayLike] = None,
    logits: Optional[ArrayLike] = None,
    *,
    validate_args: Optional[bool] = None,
) -> Union[GeometricProbs, GeometricLogits]:
    r"""Geometric distribution parameterized by either probabilities
    or logits.

    Exactly one of :attr:`probs` or :attr:`logits` must be specified.

    :param probs: Probability of success on each independent trial (:math:`p`).
    :type probs: ArrayLike, optional
    :param logits: Logits of success on each independent trial.
    :type logits: ArrayLike, optional
    :param validate_args: Whether to validate input constraints, defaults to
        ``None``.
    :type validate_args: bool, optional
    :return: A probability- or logit-parameterized Geometric distribution.
    :rtype: Union[GeometricProbs, GeometricLogits]
    :raises ValueError: If both or neither of ``probs`` and ``logits`` are specified.
    """
    assert_one_of(probs=probs, logits=logits)
    if probs is not None:
        return GeometricProbs(probs, validate_args=validate_args)
    elif logits is not None:
        return GeometricLogits(logits, validate_args=validate_args)
