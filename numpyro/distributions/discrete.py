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

import warnings

import numpy as np

import jax
from jax import lax
from jax.nn import softmax, softplus
import jax.numpy as jnp
from jax.ops import index_add
import jax.random as random
from jax.scipy.special import expit, gammaincc, gammaln, logsumexp, xlog1py, xlogy

from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import (
    binary_cross_entropy_with_logits,
    binomial,
    categorical,
    clamp_probs,
    is_prng_key,
    lazy_property,
    multinomial,
    promote_shapes,
    validate_sample,
)
from numpyro.util import not_jax_tracer


def _to_probs_bernoulli(logits):
    return expit(logits)


def _to_logits_bernoulli(probs):
    ps_clamped = clamp_probs(probs)
    return jnp.log(ps_clamped) - jnp.log1p(-ps_clamped)


def _to_probs_multinom(logits):
    return softmax(logits, axis=-1)


def _to_logits_multinom(probs):
    minval = jnp.finfo(jnp.result_type(probs)).min
    return jnp.clip(jnp.log(probs), a_min=minval)


class BernoulliProbs(Distribution):
    arg_constraints = {"probs": constraints.unit_interval}
    support = constraints.boolean
    has_enumerate_support = True

    def __init__(self, probs, validate_args=None):
        self.probs = probs
        super(BernoulliProbs, self).__init__(
            batch_shape=jnp.shape(self.probs), validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        samples = random.bernoulli(
            key, self.probs, shape=sample_shape + self.batch_shape
        )
        return samples.astype(jnp.result_type(samples, int))

    @validate_sample
    def log_prob(self, value):
        return xlogy(value, self.probs) + xlog1py(1 - value, -self.probs)

    @lazy_property
    def logits(self):
        return _to_logits_bernoulli(self.probs)

    @property
    def mean(self):
        return self.probs

    @property
    def variance(self):
        return self.probs * (1 - self.probs)

    def enumerate_support(self, expand=True):
        values = jnp.arange(2).reshape((-1,) + (1,) * len(self.batch_shape))
        if expand:
            values = jnp.broadcast_to(values, values.shape[:1] + self.batch_shape)
        return values


class BernoulliLogits(Distribution):
    arg_constraints = {"logits": constraints.real}
    support = constraints.boolean
    has_enumerate_support = True

    def __init__(self, logits=None, validate_args=None):
        self.logits = logits
        super(BernoulliLogits, self).__init__(
            batch_shape=jnp.shape(self.logits), validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        samples = random.bernoulli(
            key, self.probs, shape=sample_shape + self.batch_shape
        )
        return samples.astype(jnp.result_type(samples, int))

    @validate_sample
    def log_prob(self, value):
        return -binary_cross_entropy_with_logits(self.logits, value)

    @lazy_property
    def probs(self):
        return _to_probs_bernoulli(self.logits)

    @property
    def mean(self):
        return self.probs

    @property
    def variance(self):
        return self.probs * (1 - self.probs)

    def enumerate_support(self, expand=True):
        values = jnp.arange(2).reshape((-1,) + (1,) * len(self.batch_shape))
        if expand:
            values = jnp.broadcast_to(values, values.shape[:1] + self.batch_shape)
        return values


def Bernoulli(probs=None, logits=None, validate_args=None):
    if probs is not None:
        return BernoulliProbs(probs, validate_args=validate_args)
    elif logits is not None:
        return BernoulliLogits(logits, validate_args=validate_args)
    else:
        raise ValueError("One of `probs` or `logits` must be specified.")


class BinomialProbs(Distribution):
    arg_constraints = {
        "probs": constraints.unit_interval,
        "total_count": constraints.nonnegative_integer,
    }
    has_enumerate_support = True

    def __init__(self, probs, total_count=1, validate_args=None):
        self.probs, self.total_count = promote_shapes(probs, total_count)
        batch_shape = lax.broadcast_shapes(jnp.shape(probs), jnp.shape(total_count))
        super(BinomialProbs, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return binomial(
            key, self.probs, n=self.total_count, shape=sample_shape + self.batch_shape
        )

    @validate_sample
    def log_prob(self, value):
        log_factorial_n = gammaln(self.total_count + 1)
        log_factorial_k = gammaln(value + 1)
        log_factorial_nmk = gammaln(self.total_count - value + 1)
        return (
            log_factorial_n
            - log_factorial_k
            - log_factorial_nmk
            + xlogy(value, self.probs)
            + xlog1py(self.total_count - value, -self.probs)
        )

    @lazy_property
    def logits(self):
        return _to_logits_bernoulli(self.probs)

    @property
    def mean(self):
        return jnp.broadcast_to(self.total_count * self.probs, self.batch_shape)

    @property
    def variance(self):
        return jnp.broadcast_to(
            self.total_count * self.probs * (1 - self.probs), self.batch_shape
        )

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self):
        return constraints.integer_interval(0, self.total_count)

    def enumerate_support(self, expand=True):
        if not_jax_tracer(self.total_count):
            total_count = np.amax(self.total_count)
            # NB: the error can't be raised if inhomogeneous issue happens when tracing
            if np.amin(self.total_count) != total_count:
                raise NotImplementedError(
                    "Inhomogeneous total count not supported" " by `enumerate_support`."
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
    arg_constraints = {
        "logits": constraints.real,
        "total_count": constraints.nonnegative_integer,
    }
    has_enumerate_support = True
    enumerate_support = BinomialProbs.enumerate_support

    def __init__(self, logits, total_count=1, validate_args=None):
        self.logits, self.total_count = promote_shapes(logits, total_count)
        batch_shape = lax.broadcast_shapes(jnp.shape(logits), jnp.shape(total_count))
        super(BinomialLogits, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return binomial(
            key, self.probs, n=self.total_count, shape=sample_shape + self.batch_shape
        )

    @validate_sample
    def log_prob(self, value):
        log_factorial_n = gammaln(self.total_count + 1)
        log_factorial_k = gammaln(value + 1)
        log_factorial_nmk = gammaln(self.total_count - value + 1)
        normalize_term = (
            self.total_count * jnp.clip(self.logits, 0)
            + xlog1py(self.total_count, jnp.exp(-jnp.abs(self.logits)))
            - log_factorial_n
        )
        return (
            value * self.logits - log_factorial_k - log_factorial_nmk - normalize_term
        )

    @lazy_property
    def probs(self):
        return _to_probs_bernoulli(self.logits)

    @property
    def mean(self):
        return jnp.broadcast_to(self.total_count * self.probs, self.batch_shape)

    @property
    def variance(self):
        return jnp.broadcast_to(
            self.total_count * self.probs * (1 - self.probs), self.batch_shape
        )

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self):
        return constraints.integer_interval(0, self.total_count)


def Binomial(total_count=1, probs=None, logits=None, validate_args=None):
    if probs is not None:
        return BinomialProbs(probs, total_count, validate_args=validate_args)
    elif logits is not None:
        return BinomialLogits(logits, total_count, validate_args=validate_args)
    else:
        raise ValueError("One of `probs` or `logits` must be specified.")


class CategoricalProbs(Distribution):
    arg_constraints = {"probs": constraints.simplex}
    has_enumerate_support = True

    def __init__(self, probs, validate_args=None):
        if jnp.ndim(probs) < 1:
            raise ValueError("`probs` parameter must be at least one-dimensional.")
        self.probs = probs
        super(CategoricalProbs, self).__init__(
            batch_shape=jnp.shape(self.probs)[:-1], validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return categorical(key, self.probs, shape=sample_shape + self.batch_shape)

    @validate_sample
    def log_prob(self, value):
        batch_shape = lax.broadcast_shapes(jnp.shape(value), self.batch_shape)
        value = jnp.expand_dims(value, axis=-1)
        value = jnp.broadcast_to(value, batch_shape + (1,))
        logits = self.logits
        log_pmf = jnp.broadcast_to(logits, batch_shape + jnp.shape(logits)[-1:])
        return jnp.take_along_axis(log_pmf, value, axis=-1)[..., 0]

    @lazy_property
    def logits(self):
        return _to_logits_multinom(self.probs)

    @property
    def mean(self):
        return jnp.full(self.batch_shape, jnp.nan, dtype=jnp.result_type(self.probs))

    @property
    def variance(self):
        return jnp.full(self.batch_shape, jnp.nan, dtype=jnp.result_type(self.probs))

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self):
        return constraints.integer_interval(0, jnp.shape(self.probs)[-1] - 1)

    def enumerate_support(self, expand=True):
        values = jnp.arange(self.probs.shape[-1]).reshape(
            (-1,) + (1,) * len(self.batch_shape)
        )
        if expand:
            values = jnp.broadcast_to(values, values.shape[:1] + self.batch_shape)
        return values


class CategoricalLogits(Distribution):
    arg_constraints = {"logits": constraints.real_vector}
    has_enumerate_support = True

    def __init__(self, logits, validate_args=None):
        if jnp.ndim(logits) < 1:
            raise ValueError("`logits` parameter must be at least one-dimensional.")
        self.logits = logits
        super(CategoricalLogits, self).__init__(
            batch_shape=jnp.shape(logits)[:-1], validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return random.categorical(
            key, self.logits, shape=sample_shape + self.batch_shape
        )

    @validate_sample
    def log_prob(self, value):
        batch_shape = lax.broadcast_shapes(jnp.shape(value), self.batch_shape)
        value = jnp.expand_dims(value, -1)
        value = jnp.broadcast_to(value, batch_shape + (1,))
        log_pmf = self.logits - logsumexp(self.logits, axis=-1, keepdims=True)
        log_pmf = jnp.broadcast_to(log_pmf, batch_shape + jnp.shape(log_pmf)[-1:])
        return jnp.take_along_axis(log_pmf, value, -1)[..., 0]

    @lazy_property
    def probs(self):
        return _to_probs_multinom(self.logits)

    @property
    def mean(self):
        return jnp.full(self.batch_shape, jnp.nan, dtype=jnp.result_type(self.logits))

    @property
    def variance(self):
        return jnp.full(self.batch_shape, jnp.nan, dtype=jnp.result_type(self.logits))

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self):
        return constraints.integer_interval(0, jnp.shape(self.logits)[-1] - 1)

    def enumerate_support(self, expand=True):
        values = jnp.arange(self.logits.shape[-1]).reshape(
            (-1,) + (1,) * len(self.batch_shape)
        )
        if expand:
            values = jnp.broadcast_to(values, values.shape[:1] + self.batch_shape)
        return values


def Categorical(probs=None, logits=None, validate_args=None):
    if probs is not None:
        return CategoricalProbs(probs, validate_args=validate_args)
    elif logits is not None:
        return CategoricalLogits(logits, validate_args=validate_args)
    else:
        raise ValueError("One of `probs` or `logits` must be specified.")


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

    def __init__(self, predictor, cutpoints, validate_args=None):
        if jnp.ndim(predictor) == 0:
            (predictor,) = promote_shapes(predictor, shape=(1,))
        else:
            predictor = predictor[..., None]
        predictor, self.cutpoints = promote_shapes(predictor, cutpoints)
        self.predictor = predictor[..., 0]
        cumulative_probs = expit(cutpoints - predictor)
        # add two boundary points 0 and 1
        pad_width = [(0, 0)] * (jnp.ndim(cumulative_probs) - 1) + [(1, 1)]
        cumulative_probs = jnp.pad(cumulative_probs, pad_width, constant_values=(0, 1))
        probs = cumulative_probs[..., 1:] - cumulative_probs[..., :-1]
        super(OrderedLogistic, self).__init__(probs, validate_args=validate_args)

    @staticmethod
    def infer_shapes(predictor, cutpoints):
        batch_shape = lax.broadcast_shapes(predictor, cutpoints[:-1])
        event_shape = ()
        return batch_shape, event_shape


class PRNGIdentity(Distribution):
    """
    Distribution over :func:`~jax.random.PRNGKey`. This can be used to
    draw a batch of :func:`~jax.random.PRNGKey` using the :class:`~numpyro.handlers.seed`
    handler. Only `sample` method is supported.
    """

    def __init__(self):
        warnings.warn(
            "PRNGIdentity distribution is deprecated. To get a random "
            "PRNG key, you can use `numpyro.prng_key()` instead.",
            FutureWarning,
        )
        super(PRNGIdentity, self).__init__(event_shape=(2,))

    def sample(self, key, sample_shape=()):
        return jnp.reshape(
            random.split(key, np.prod(sample_shape).astype(np.int32)),
            sample_shape + self.event_shape,
        )


class MultinomialProbs(Distribution):
    arg_constraints = {
        "probs": constraints.simplex,
        "total_count": constraints.nonnegative_integer,
    }

    def __init__(self, probs, total_count=1, validate_args=None):
        if jnp.ndim(probs) < 1:
            raise ValueError("`probs` parameter must be at least one-dimensional.")
        batch_shape, event_shape = self.infer_shapes(
            jnp.shape(probs), jnp.shape(total_count)
        )
        self.probs = promote_shapes(probs, shape=batch_shape + jnp.shape(probs)[-1:])[0]
        self.total_count = promote_shapes(total_count, shape=batch_shape)[0]
        super(MultinomialProbs, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return multinomial(
            key, self.probs, self.total_count, shape=sample_shape + self.batch_shape
        )

    @validate_sample
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return gammaln(self.total_count + 1) + jnp.sum(
            xlogy(value, self.probs) - gammaln(value + 1), axis=-1
        )

    @lazy_property
    def logits(self):
        return _to_logits_multinom(self.probs)

    @property
    def mean(self):
        return self.probs * jnp.expand_dims(self.total_count, -1)

    @property
    def variance(self):
        return jnp.expand_dims(self.total_count, -1) * self.probs * (1 - self.probs)

    @constraints.dependent_property(is_discrete=True, event_dim=1)
    def support(self):
        return constraints.multinomial(self.total_count)

    @staticmethod
    def infer_shapes(probs, total_count):
        batch_shape = lax.broadcast_shapes(probs[:-1], total_count)
        event_shape = probs[-1:]
        return batch_shape, event_shape


class MultinomialLogits(Distribution):
    arg_constraints = {
        "logits": constraints.real_vector,
        "total_count": constraints.nonnegative_integer,
    }

    def __init__(self, logits, total_count=1, validate_args=None):
        if jnp.ndim(logits) < 1:
            raise ValueError("`logits` parameter must be at least one-dimensional.")
        batch_shape, event_shape = self.infer_shapes(
            jnp.shape(logits), jnp.shape(total_count)
        )
        self.logits = promote_shapes(
            logits, shape=batch_shape + jnp.shape(logits)[-1:]
        )[0]
        self.total_count = promote_shapes(total_count, shape=batch_shape)[0]
        super(MultinomialLogits, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return multinomial(
            key, self.probs, self.total_count, shape=sample_shape + self.batch_shape
        )

    @validate_sample
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        normalize_term = self.total_count * logsumexp(self.logits, axis=-1) - gammaln(
            self.total_count + 1
        )
        return (
            jnp.sum(value * self.logits - gammaln(value + 1), axis=-1) - normalize_term
        )

    @lazy_property
    def probs(self):
        return _to_probs_multinom(self.logits)

    @property
    def mean(self):
        return jnp.expand_dims(self.total_count, -1) * self.probs

    @property
    def variance(self):
        return jnp.expand_dims(self.total_count, -1) * self.probs * (1 - self.probs)

    @constraints.dependent_property(is_discrete=True, event_dim=1)
    def support(self):
        return constraints.multinomial(self.total_count)

    @staticmethod
    def infer_shapes(logits, total_count):
        batch_shape = lax.broadcast_shapes(logits[:-1], total_count)
        event_shape = logits[-1:]
        return batch_shape, event_shape


def Multinomial(total_count=1, probs=None, logits=None, validate_args=None):
    if probs is not None:
        return MultinomialProbs(probs, total_count, validate_args=validate_args)
    elif logits is not None:
        return MultinomialLogits(logits, total_count, validate_args=validate_args)
    else:
        raise ValueError("One of `probs` or `logits` must be specified.")


class Poisson(Distribution):
    arg_constraints = {"rate": constraints.positive}
    support = constraints.nonnegative_integer

    def __init__(self, rate, *, is_sparse=False, validate_args=None):
        self.rate = rate
        self.is_sparse = is_sparse
        super(Poisson, self).__init__(jnp.shape(rate), validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return random.poisson(key, self.rate, shape=sample_shape + self.batch_shape)

    @validate_sample
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value = jax.device_get(value)
        if (
            self.is_sparse
            and not isinstance(value, jax.core.Tracer)
            and jnp.size(value) > 1
        ):
            shape = lax.broadcast_shapes(self.batch_shape, jnp.shape(value))
            rate = jnp.broadcast_to(self.rate, shape).reshape(-1)
            value = jnp.broadcast_to(value, shape).reshape(-1)
            nonzero = value > 0
            sparse_value = value[nonzero]
            sparse_rate = rate[nonzero]
            return index_add(
                -rate,
                nonzero,
                jnp.log(sparse_rate) * sparse_value - gammaln(sparse_value + 1),
            ).reshape(shape)
        return (jnp.log(self.rate) * value) - gammaln(value + 1) - self.rate

    @property
    def mean(self):
        return self.rate

    @property
    def variance(self):
        return self.rate

    def cdf(self, value):
        k = jnp.floor(value) + 1
        return gammaincc(k, self.rate)


class ZeroInflatedProbs(Distribution):
    arg_constraints = {"gate": constraints.unit_interval}

    def __init__(self, base_dist, gate, *, validate_args=None):
        batch_shape = lax.broadcast_shapes(jnp.shape(gate), base_dist.batch_shape)
        (self.gate,) = promote_shapes(gate, shape=batch_shape)
        assert base_dist.is_discrete
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

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        key_bern, key_base = random.split(key)
        shape = sample_shape + self.batch_shape
        mask = random.bernoulli(key_bern, self.gate, shape)
        samples = self.base_dist(rng_key=key_base, sample_shape=sample_shape)
        return jnp.where(mask, 0, samples)

    @validate_sample
    def log_prob(self, value):
        log_prob = jnp.log1p(-self.gate) + self.base_dist.log_prob(value)
        return jnp.where(value == 0, jnp.log(self.gate + jnp.exp(log_prob)), log_prob)

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self):
        return self.base_dist.support

    @lazy_property
    def mean(self):
        return (1 - self.gate) * self.base_dist.mean

    @lazy_property
    def variance(self):
        return (1 - self.gate) * (
            self.base_dist.mean ** 2 + self.base_dist.variance
        ) - self.mean ** 2


class ZeroInflatedLogits(ZeroInflatedProbs):
    arg_constraints = {"gate_logits": constraints.real}

    def __init__(self, base_dist, gate_logits, *, validate_args=None):
        gate = _to_probs_bernoulli(gate_logits)
        batch_shape = lax.broadcast_shapes(jnp.shape(gate), base_dist.batch_shape)
        (self.gate_logits,) = promote_shapes(gate_logits, shape=batch_shape)
        super().__init__(base_dist, gate, validate_args=validate_args)

    @validate_sample
    def log_prob(self, value):
        log_prob_minus_log_gate = -self.gate_logits + self.base_dist.log_prob(value)
        log_gate = -softplus(-self.gate_logits)
        log_prob = log_prob_minus_log_gate + log_gate
        zero_log_prob = softplus(log_prob_minus_log_gate) + log_gate
        return jnp.where(value == 0, zero_log_prob, log_prob)


def ZeroInflatedDistribution(
    base_dist, *, gate=None, gate_logits=None, validate_args=None
):
    """
    Generic Zero Inflated distribution.

    :param Distribution base_dist: the base distribution.
    :param numpy.ndarray gate: probability of extra zeros given via a Bernoulli distribution.
    :param numpy.ndarray gate_logits: logits of extra zeros given via a Bernoulli distribution.
    """
    if (gate is None) == (gate_logits is None):
        raise ValueError(
            "Either `gate` or `gate_logits` must be specified, but not both."
        )
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

    # TODO: resolve inconsistent parameter order w.r.t. Pyro
    # and support `gate_logits` argument
    def __init__(self, gate, rate=1.0, validate_args=None):
        _, self.rate = promote_shapes(gate, rate)
        super().__init__(Poisson(self.rate), gate, validate_args=validate_args)


class GeometricProbs(Distribution):
    arg_constraints = {"probs": constraints.unit_interval}
    support = constraints.nonnegative_integer

    def __init__(self, probs, validate_args=None):
        self.probs = probs
        super(GeometricProbs, self).__init__(
            batch_shape=jnp.shape(self.probs), validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        probs = self.probs
        dtype = jnp.result_type(probs)
        shape = sample_shape + self.batch_shape
        u = random.uniform(key, shape, dtype)
        return jnp.floor(jnp.log1p(-u) / jnp.log1p(-probs))

    @validate_sample
    def log_prob(self, value):
        probs = jnp.where((self.probs == 1) & (value == 0), 0, self.probs)
        return value * jnp.log1p(-probs) + jnp.log(probs)

    @lazy_property
    def logits(self):
        return _to_logits_bernoulli(self.probs)

    @property
    def mean(self):
        return 1.0 / self.probs - 1.0

    @property
    def variance(self):
        return (1.0 / self.probs - 1.0) / self.probs


class GeometricLogits(Distribution):
    arg_constraints = {"logits": constraints.real}
    support = constraints.nonnegative_integer

    def __init__(self, logits, validate_args=None):
        self.logits = logits
        super(GeometricLogits, self).__init__(
            batch_shape=jnp.shape(self.logits), validate_args=validate_args
        )

    @lazy_property
    def probs(self):
        return _to_probs_bernoulli(self.logits)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        logits = self.logits
        dtype = jnp.result_type(logits)
        shape = sample_shape + self.batch_shape
        u = random.uniform(key, shape, dtype)
        return jnp.floor(jnp.log1p(-u) / -softplus(logits))

    @validate_sample
    def log_prob(self, value):
        return (-value - 1) * softplus(self.logits) + self.logits

    @property
    def mean(self):
        return 1.0 / self.probs - 1.0

    @property
    def variance(self):
        return (1.0 / self.probs - 1.0) / self.probs


def Geometric(probs=None, logits=None, validate_args=None):
    if probs is not None:
        return GeometricProbs(probs, validate_args=validate_args)
    elif logits is not None:
        return GeometricLogits(logits, validate_args=validate_args)
    else:
        raise ValueError("One of `probs` or `logits` must be specified.")
