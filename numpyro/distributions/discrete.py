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

from jax import device_put, lax
from jax.dtypes import canonicalize_dtype
from jax.nn import softmax, softplus
import jax.numpy as jnp
import jax.random as random
from jax.scipy.special import expit, gammaln, logsumexp, xlog1py, xlogy

from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import (
    binary_cross_entropy_with_logits,
    binomial,
    categorical,
    clamp_probs,
    get_dtype,
    lazy_property,
    multinomial,
    promote_shapes,
    sum_rightmost,
    validate_sample
)
from numpyro.util import not_jax_tracer


def _to_probs_bernoulli(logits):
    return 1 / (1 + jnp.exp(-logits))


def _to_logits_bernoulli(probs):
    ps_clamped = clamp_probs(probs)
    return jnp.log(ps_clamped) - jnp.log1p(-ps_clamped)


def _to_probs_multinom(logits):
    return softmax(logits, axis=-1)


def _to_logits_multinom(probs):
    minval = jnp.finfo(get_dtype(probs)).min
    return jnp.clip(jnp.log(probs), a_min=minval)


class BernoulliProbs(Distribution):
    arg_constraints = {'probs': constraints.unit_interval}
    support = constraints.boolean
    has_enumerate_support = True
    is_discrete = True

    def __init__(self, probs, validate_args=None):
        self.probs = probs
        super(BernoulliProbs, self).__init__(batch_shape=jnp.shape(self.probs), validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        return random.bernoulli(key, self.probs, shape=sample_shape + self.batch_shape)

    @validate_sample
    def log_prob(self, value):
        return xlogy(value, self.probs) + xlog1py(1 - value, -self.probs)

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
    arg_constraints = {'logits': constraints.real}
    support = constraints.boolean
    has_enumerate_support = True
    is_discrete = True

    def __init__(self, logits=None, validate_args=None):
        self.logits = logits
        super(BernoulliLogits, self).__init__(batch_shape=jnp.shape(self.logits), validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        return random.bernoulli(key, self.probs, shape=sample_shape + self.batch_shape)

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
        raise ValueError('One of `probs` or `logits` must be specified.')


class BinomialProbs(Distribution):
    arg_constraints = {'probs': constraints.unit_interval,
                       'total_count': constraints.nonnegative_integer}
    has_enumerate_support = True
    is_discrete = True

    def __init__(self, probs, total_count=1, validate_args=None):
        self.probs, self.total_count = promote_shapes(probs, total_count)
        batch_shape = lax.broadcast_shapes(jnp.shape(probs), jnp.shape(total_count))
        super(BinomialProbs, self).__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        return binomial(key, self.probs, n=self.total_count, shape=sample_shape + self.batch_shape)

    @validate_sample
    def log_prob(self, value):
        log_factorial_n = gammaln(self.total_count + 1)
        log_factorial_k = gammaln(value + 1)
        log_factorial_nmk = gammaln(self.total_count - value + 1)
        return (log_factorial_n - log_factorial_k - log_factorial_nmk +
                xlogy(value, self.probs) + xlog1py(self.total_count - value, -self.probs))

    @property
    def mean(self):
        return jnp.broadcast_to(self.total_count * self.probs, self.batch_shape)

    @property
    def variance(self):
        return jnp.broadcast_to(self.total_count * self.probs * (1 - self.probs), self.batch_shape)

    @property
    def support(self):
        return constraints.integer_interval(0, self.total_count)

    def enumerate_support(self, expand=True):
        total_count = jnp.amax(self.total_count)
        if not_jax_tracer(total_count):
            # NB: the error can't be raised if inhomogeneous issue happens when tracing
            if jnp.amin(self.total_count) != total_count:
                raise NotImplementedError("Inhomogeneous total count not supported"
                                          " by `enumerate_support`.")
        values = jnp.arange(total_count + 1).reshape((-1,) + (1,) * len(self.batch_shape))
        if expand:
            values = jnp.broadcast_to(values, values.shape[:1] + self.batch_shape)
        return values


class BinomialLogits(Distribution):
    arg_constraints = {'logits': constraints.real,
                       'total_count': constraints.nonnegative_integer}
    has_enumerate_support = True
    is_discrete = True

    def __init__(self, logits, total_count=1, validate_args=None):
        self.logits, self.total_count = promote_shapes(logits, total_count)
        batch_shape = lax.broadcast_shapes(jnp.shape(logits), jnp.shape(total_count))
        super(BinomialLogits, self).__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        return binomial(key, self.probs, n=self.total_count, shape=sample_shape + self.batch_shape)

    @validate_sample
    def log_prob(self, value):
        log_factorial_n = gammaln(self.total_count + 1)
        log_factorial_k = gammaln(value + 1)
        log_factorial_nmk = gammaln(self.total_count - value + 1)
        normalize_term = (self.total_count * jnp.clip(self.logits, 0) +
                          xlog1py(self.total_count, jnp.exp(-jnp.abs(self.logits))) -
                          log_factorial_n)
        return value * self.logits - log_factorial_k - log_factorial_nmk - normalize_term

    @lazy_property
    def probs(self):
        return _to_probs_bernoulli(self.logits)

    @property
    def mean(self):
        return jnp.broadcast_to(self.total_count * self.probs, self.batch_shape)

    @property
    def variance(self):
        return jnp.broadcast_to(self.total_count * self.probs * (1 - self.probs), self.batch_shape)

    @property
    def support(self):
        return constraints.integer_interval(0, self.total_count)

    def enumerate_support(self, expand=True):
        total_count = jnp.amax(self.total_count)
        if not_jax_tracer(total_count):
            # NB: the error can't be raised if inhomogeneous issue happens when tracing
            if jnp.amin(self.total_count) != total_count:
                raise NotImplementedError("Inhomogeneous total count not supported"
                                          " by `enumerate_support`.")
        values = jnp.arange(total_count + 1).reshape((-1,) + (1,) * len(self.batch_shape))
        if expand:
            values = jnp.broadcast_to(values, values.shape[:1] + self.batch_shape)
        return values


def Binomial(total_count=1, probs=None, logits=None, validate_args=None):
    if probs is not None:
        return BinomialProbs(probs, total_count, validate_args=validate_args)
    elif logits is not None:
        return BinomialLogits(logits, total_count, validate_args=validate_args)
    else:
        raise ValueError('One of `probs` or `logits` must be specified.')


class CategoricalProbs(Distribution):
    arg_constraints = {'probs': constraints.simplex}
    has_enumerate_support = True
    is_discrete = True

    def __init__(self, probs, validate_args=None):
        if jnp.ndim(probs) < 1:
            raise ValueError("`probs` parameter must be at least one-dimensional.")
        self.probs = probs
        super(CategoricalProbs, self).__init__(batch_shape=jnp.shape(self.probs)[:-1],
                                               validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        return categorical(key, self.probs, shape=sample_shape + self.batch_shape)

    @validate_sample
    def log_prob(self, value):
        batch_shape = lax.broadcast_shapes(jnp.shape(value), self.batch_shape)
        value = jnp.expand_dims(value, axis=-1)
        value = jnp.broadcast_to(value, batch_shape + (1,))
        logits = _to_logits_multinom(self.probs)
        log_pmf = jnp.broadcast_to(logits, batch_shape + jnp.shape(logits)[-1:])
        return jnp.take_along_axis(log_pmf, value, axis=-1)[..., 0]

    @property
    def mean(self):
        return jnp.full(self.batch_shape, jnp.nan, dtype=get_dtype(self.probs))

    @property
    def variance(self):
        return jnp.full(self.batch_shape, jnp.nan, dtype=get_dtype(self.probs))

    @property
    def support(self):
        return constraints.integer_interval(0, jnp.shape(self.probs)[-1] - 1)

    def enumerate_support(self, expand=True):
        values = jnp.arange(self.probs.shape[-1]).reshape((-1,) + (1,) * len(self.batch_shape))
        if expand:
            values = jnp.broadcast_to(values, values.shape[:1] + self.batch_shape)
        return values


class CategoricalLogits(Distribution):
    arg_constraints = {'logits': constraints.real_vector}
    has_enumerate_support = True
    is_discrete = True

    def __init__(self, logits, validate_args=None):
        if jnp.ndim(logits) < 1:
            raise ValueError("`logits` parameter must be at least one-dimensional.")
        self.logits = logits
        super(CategoricalLogits, self).__init__(batch_shape=jnp.shape(logits)[:-1],
                                                validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        return random.categorical(key, self.logits, shape=sample_shape + self.batch_shape)

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
        return jnp.full(self.batch_shape, jnp.nan, dtype=get_dtype(self.logits))

    @property
    def variance(self):
        return jnp.full(self.batch_shape, jnp.nan, dtype=get_dtype(self.logits))

    @property
    def support(self):
        return constraints.integer_interval(0, jnp.shape(self.logits)[-1] - 1)

    def enumerate_support(self, expand=True):
        values = jnp.arange(self.logits.shape[-1]).reshape((-1,) + (1,) * len(self.batch_shape))
        if expand:
            values = jnp.broadcast_to(values, values.shape[:1] + self.batch_shape)
        return values


def Categorical(probs=None, logits=None, validate_args=None):
    if probs is not None:
        return CategoricalProbs(probs, validate_args=validate_args)
    elif logits is not None:
        return CategoricalLogits(logits, validate_args=validate_args)
    else:
        raise ValueError('One of `probs` or `logits` must be specified.')


class Delta(Distribution):
    arg_constraints = {'v': constraints.real, 'log_density': constraints.real}
    support = constraints.real
    is_discrete = True

    def __init__(self, v=0., log_density=0., event_dim=0, validate_args=None, value=None):
        if value is not None:
            v = value
            warnings.warn("`value` argument has been deprecated in favor of `v` argument.",
                          FutureWarning)

        if event_dim > jnp.ndim(v):
            raise ValueError('Expected event_dim <= v.dim(), actual {} vs {}'
                             .format(event_dim, jnp.ndim(v)))
        batch_dim = jnp.ndim(v) - event_dim
        batch_shape = jnp.shape(v)[:batch_dim]
        event_shape = jnp.shape(v)[batch_dim:]
        self.v = lax.convert_element_type(v, canonicalize_dtype(jnp.float64))
        # NB: following Pyro implementation, log_density should be broadcasted to batch_shape
        self.log_density = promote_shapes(log_density, shape=batch_shape)[0]
        super(Delta, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return jnp.broadcast_to(device_put(self.v), shape)

    @validate_sample
    def log_prob(self, value):
        log_prob = jnp.log(value == self.v)
        log_prob = sum_rightmost(log_prob, len(self.event_shape))
        return log_prob + self.log_density

    @property
    def mean(self):
        return self.v

    @property
    def variance(self):
        return jnp.zeros(self.batch_shape + self.event_shape)

    def tree_flatten(self):
        return (self.v, self.log_density), self.event_dim

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        return cls(*params, event_dim=aux_data)


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
    arg_constraints = {'predictor': constraints.real,
                       'cutpoints': constraints.ordered_vector}

    def __init__(self, predictor, cutpoints, validate_args=None):
        predictor, self.cutpoints = promote_shapes(jnp.expand_dims(predictor, -1), cutpoints)
        self.predictor = predictor[..., 0]
        cumulative_probs = expit(cutpoints - predictor)
        # add two boundary points 0 and 1
        pad_width = [(0, 0)] * (jnp.ndim(cumulative_probs) - 1) + [(1, 1)]
        cumulative_probs = jnp.pad(cumulative_probs, pad_width, constant_values=(0, 1))
        probs = cumulative_probs[..., 1:] - cumulative_probs[..., :-1]
        super(OrderedLogistic, self).__init__(probs, validate_args=validate_args)


class PRNGIdentity(Distribution):
    """
    Distribution over :func:`~jax.random.PRNGKey`. This can be used to
    draw a batch of :func:`~jax.random.PRNGKey` using the :class:`~numpyro.handlers.seed`
    handler. Only `sample` method is supported.
    """
    is_discrete = True

    def __init__(self):
        super(PRNGIdentity, self).__init__(event_shape=(2,))

    def sample(self, key, sample_shape=()):
        return jnp.reshape(random.split(key, np.prod(sample_shape).astype(np.int32)),
                           sample_shape + self.event_shape)


class MultinomialProbs(Distribution):
    arg_constraints = {'probs': constraints.simplex,
                       'total_count': constraints.nonnegative_integer}
    is_discrete = True

    def __init__(self, probs, total_count=1, validate_args=None):
        if jnp.ndim(probs) < 1:
            raise ValueError("`probs` parameter must be at least one-dimensional.")
        batch_shape = lax.broadcast_shapes(jnp.shape(probs)[:-1], jnp.shape(total_count))
        self.probs = promote_shapes(probs, shape=batch_shape + jnp.shape(probs)[-1:])[0]
        self.total_count = promote_shapes(total_count, shape=batch_shape)[0]
        super(MultinomialProbs, self).__init__(batch_shape=batch_shape,
                                               event_shape=jnp.shape(self.probs)[-1:],
                                               validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        return multinomial(key, self.probs, self.total_count, shape=sample_shape + self.batch_shape)

    @validate_sample
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return gammaln(self.total_count + 1) \
            + jnp.sum(xlogy(value, self.probs) - gammaln(value + 1), axis=-1)

    @property
    def mean(self):
        return self.probs * jnp.expand_dims(self.total_count, -1)

    @property
    def variance(self):
        return jnp.expand_dims(self.total_count, -1) * self.probs * (1 - self.probs)

    @property
    def support(self):
        return constraints.multinomial(self.total_count)


class MultinomialLogits(Distribution):
    arg_constraints = {'logits': constraints.real_vector,
                       'total_count': constraints.nonnegative_integer}
    is_discrete = True

    def __init__(self, logits, total_count=1, validate_args=None):
        if jnp.ndim(logits) < 1:
            raise ValueError("`logits` parameter must be at least one-dimensional.")
        batch_shape = lax.broadcast_shapes(jnp.shape(logits)[:-1], jnp.shape(total_count))
        self.logits = promote_shapes(logits, shape=batch_shape + jnp.shape(logits)[-1:])[0]
        self.total_count = promote_shapes(total_count, shape=batch_shape)[0]
        super(MultinomialLogits, self).__init__(batch_shape=batch_shape,
                                                event_shape=jnp.shape(self.logits)[-1:],
                                                validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        return multinomial(key, self.probs, self.total_count, shape=sample_shape + self.batch_shape)

    @validate_sample
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        normalize_term = self.total_count * logsumexp(self.logits, axis=-1) \
            - gammaln(self.total_count + 1)
        return jnp.sum(value * self.logits - gammaln(value + 1), axis=-1) - normalize_term

    @lazy_property
    def probs(self):
        return _to_probs_multinom(self.logits)

    @property
    def mean(self):
        return jnp.expand_dims(self.total_count, -1) * self.probs

    @property
    def variance(self):
        return jnp.expand_dims(self.total_count, -1) * self.probs * (1 - self.probs)

    @property
    def support(self):
        return constraints.multinomial(self.total_count)


def Multinomial(total_count=1, probs=None, logits=None, validate_args=None):
    if probs is not None:
        return MultinomialProbs(probs, total_count, validate_args=validate_args)
    elif logits is not None:
        return MultinomialLogits(logits, total_count, validate_args=validate_args)
    else:
        raise ValueError('One of `probs` or `logits` must be specified.')


class Poisson(Distribution):
    arg_constraints = {'rate': constraints.positive}
    support = constraints.nonnegative_integer
    is_discrete = True

    def __init__(self, rate, validate_args=None):
        self.rate = rate
        super(Poisson, self).__init__(jnp.shape(rate), validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        return random.poisson(key, self.rate, shape=sample_shape + self.batch_shape)

    @validate_sample
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return (jnp.log(self.rate) * value) - gammaln(value + 1) - self.rate

    @property
    def mean(self):
        return self.rate

    @property
    def variance(self):
        return self.rate


class ZeroInflatedPoisson(Distribution):
    """
    A Zero Inflated Poisson distribution.

    :param numpy.ndarray gate: probability of extra zeros.
    :param numpy.ndarray rate: rate of Poisson distribution.
    """
    arg_constraints = {'gate': constraints.unit_interval, 'rate': constraints.positive}
    support = constraints.nonnegative_integer
    is_discrete = True

    def __init__(self, gate, rate=1., validate_args=None):
        batch_shape = lax.broadcast_shapes(jnp.shape(gate), jnp.shape(rate))
        self.gate, self.rate = promote_shapes(gate, rate)
        super(ZeroInflatedPoisson, self).__init__(batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        key_bern, key_poisson = random.split(key)
        shape = sample_shape + self.batch_shape
        mask = random.bernoulli(key_bern, self.gate, shape)
        samples = random.poisson(key_poisson, device_put(self.rate), shape)
        return jnp.where(mask, 0, samples)

    @validate_sample
    def log_prob(self, value):
        log_prob = jnp.log(self.rate) * value - gammaln(value + 1) + (jnp.log1p(-self.gate) - self.rate)
        return jnp.where(value == 0, jnp.logaddexp(jnp.log(self.gate), log_prob), log_prob)

    @lazy_property
    def mean(self):
        return (1 - self.gate) * self.rate

    @lazy_property
    def variance(self):
        return (1 - self.gate) * self.rate * (1 + self.rate * self.gate)


class GeometricProbs(Distribution):
    arg_constraints = {'probs': constraints.unit_interval}
    support = constraints.nonnegative_integer
    is_discrete = True

    def __init__(self, probs, validate_args=None):
        self.probs = probs
        super(GeometricProbs, self).__init__(batch_shape=jnp.shape(self.probs),
                                             validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        probs = self.probs
        dtype = get_dtype(probs)
        shape = sample_shape + self.batch_shape
        u = random.uniform(key, shape, dtype)
        return jnp.floor(jnp.log1p(-u) / jnp.log1p(-probs))

    @validate_sample
    def log_prob(self, value):
        probs = jnp.where((self.probs == 1) & (value == 0), 0, self.probs)
        return value * jnp.log1p(-probs) + jnp.log(probs)

    @property
    def mean(self):
        return 1. / self.probs - 1.

    @property
    def variance(self):
        return (1. / self.probs - 1.) / self.probs


class GeometricLogits(Distribution):
    arg_constraints = {'logits': constraints.real}
    support = constraints.nonnegative_integer
    is_discrete = True

    def __init__(self, logits, validate_args=None):
        self.logits = logits
        super(GeometricLogits, self).__init__(batch_shape=jnp.shape(self.logits),
                                              validate_args=validate_args)

    @lazy_property
    def probs(self):
        return _to_probs_bernoulli(self.logits)

    def sample(self, key, sample_shape=()):
        logits = self.logits
        dtype = get_dtype(logits)
        shape = sample_shape + self.batch_shape
        u = random.uniform(key, shape, dtype)
        return jnp.floor(jnp.log1p(-u) / -softplus(logits))

    @validate_sample
    def log_prob(self, value):
        return (-value - 1) * softplus(self.logits) + self.logits

    @property
    def mean(self):
        return 1. / self.probs - 1.

    @property
    def variance(self):
        return (1. / self.probs - 1.) / self.probs


def Geometric(probs=None, logits=None, validate_args=None):
    if probs is not None:
        return GeometricProbs(probs, validate_args=validate_args)
    elif logits is not None:
        return GeometricLogits(logits, validate_args=validate_args)
    else:
        raise ValueError('One of `probs` or `logits` must be specified.')
