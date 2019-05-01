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


import jax.numpy as np
import jax.random as random
from jax import lax
from jax.scipy.special import gammaln, logsumexp

from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import (
    binary_cross_entropy_with_logits,
    binomial,
    categorical_rvs,
    clamp_probs,
    get_dtypes,
    lazy_property,
    multinomial_rvs,
    poisson,
    promote_shapes,
    xlog1py,
    xlogy
)


def _to_probs_bernoulli(logits):
    return 1 / (1 + np.exp(-logits))


def _to_logits_bernoulli(probs):
    ps_clamped = clamp_probs(probs)
    return np.log(ps_clamped) - np.log1p(-ps_clamped)


def _to_probs_multinom(logits):
    x = np.exp(logits - np.max(logits, -1, keepdims=True))
    return x / x.sum(-1, keepdims=True)


def _to_logits_multinom(probs):
    minval = np.finfo(get_dtypes(probs)[0]).min
    return np.clip(np.log(probs), a_min=minval)


class Bernoulli(Distribution):
    arg_constraints = {'probs': constraints.unit_interval}
    support = constraints.boolean

    def __init__(self, probs, validate_args=None):
        self.probs = probs
        super(Bernoulli, self).__init__(batch_shape=np.shape(self.probs), validate_args=validate_args)

    def sample(self, key, size=()):
        return random.bernoulli(key, self.probs, shape=size + self.batch_shape)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return xlogy(value, self.probs) + xlog1py(1 - value, -self.probs)

    @property
    def mean(self):
        return np.broadcast_to(self.probs, self.batch_shape)

    @property
    def variance(self):
        return np.broadcast_to(self.probs * (1 - self.probs), self.batch_shape)


class BernoulliWithLogits(Distribution):
    arg_constraints = {'logits': constraints.real}
    support = constraints.boolean

    def __init__(self, logits=None, validate_args=None):
        self.logits = logits
        super(BernoulliWithLogits, self).__init__(batch_shape=np.shape(self.logits), validate_args=validate_args)

    def sample(self, key, size=()):
        return random.bernoulli(key, self.probs, shape=size + self.batch_shape)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        dtype = get_dtypes(self.logits)[0]
        value = lax.convert_element_type(value, dtype)
        return -binary_cross_entropy_with_logits(self.logits, value)

    @lazy_property
    def probs(self):
        return _to_probs_bernoulli(self.logits)

    @property
    def mean(self):
        return np.broadcast_to(self.probs, self.batch_shape)

    @property
    def variance(self):
        return np.broadcast_to(self.probs * (1 - self.probs), self.batch_shape)


class Binomial(Distribution):
    arg_constraints = {'total_count': constraints.nonnegative_integer,
                       'probs': constraints.unit_interval}

    def __init__(self, probs, total_count=1, validate_args=None):
        self.probs, self.total_count = promote_shapes(probs, total_count)
        batch_shape = lax.broadcast_shapes(np.shape(probs), np.shape(total_count))
        super(Binomial, self).__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, size=()):
        return binomial(key, self.probs, n=self.total_count, shape=size + self.batch_shape)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        dtype = get_dtypes(self.probs)[0]
        value = lax.convert_element_type(value, dtype)
        total_count = lax.convert_element_type(self.total_count, dtype)
        log_factorial_n = gammaln(total_count + 1)
        log_factorial_k = gammaln(value + 1)
        log_factorial_nmk = gammaln(total_count - value + 1)
        return (log_factorial_n - log_factorial_k - log_factorial_nmk +
                xlogy(value, self.probs) + xlog1py(total_count - value, -self.probs))

    @property
    def mean(self):
        return np.broadcast_to(self.total_count * self.probs, self.batch_shape)

    @property
    def variance(self):
        return np.broadcast_to(self.total_count * self.probs * (1 - self.probs), self.batch_shape)

    @property
    def support(self):
        return constraints.integer_interval(0, self.total_count)


class BinomialWithLogits(Distribution):
    arg_constraints = {'total_count': constraints.nonnegative_integer,
                       'logits': constraints.real}

    def __init__(self, logits, total_count=1, validate_args=None):
        self.logits, self.total_count = promote_shapes(logits, total_count)
        batch_shape = lax.broadcast_shapes(np.shape(logits), np.shape(total_count))
        super(BinomialWithLogits, self).__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, size=()):
        return binomial(key, self.probs, n=self.total_count, shape=size + self.batch_shape)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        dtype = get_dtypes(self.logits)[0]
        value = lax.convert_element_type(value, dtype)
        total_count = lax.convert_element_type(self.total_count, dtype)
        log_factorial_n = gammaln(total_count + 1)
        log_factorial_k = gammaln(value + 1)
        log_factorial_nmk = gammaln(total_count - value + 1)
        normalize_term = (total_count * np.clip(self.logits, 0) +
                          xlog1py(self.total_count, np.exp(-np.abs(self.logits))) -
                          log_factorial_n)
        return value * self.logits - log_factorial_k - log_factorial_nmk - normalize_term

    @lazy_property
    def probs(self):
        return _to_probs_bernoulli(self.logits)

    @property
    def mean(self):
        return np.broadcast_to(self.total_count * self.probs, self.batch_shape)

    @property
    def variance(self):
        return np.broadcast_to(self.total_count * self.probs * (1 - self.probs), self.batch_shape)

    @property
    def support(self):
        return constraints.integer_interval(0, self.total_count)


class Multinomial(Distribution):
    arg_constraints = {'total_count': constraints.nonnegative_integer,
                       'probs': constraints.simplex}

    def __init__(self, probs, total_count=1, validate_args=None):
        if np.ndim(probs) < 1:
            raise ValueError("`probs` parameter must be at least one-dimensional.")
        batch_shape = lax.broadcast_shapes(np.shape(probs)[:-1], np.shape(total_count))
        self.probs = promote_shapes(probs, shape=batch_shape + np.shape(probs)[-1:])[0]
        self.total_count = promote_shapes(total_count, shape=batch_shape)[0]
        super(Multinomial, self).__init__(batch_shape=batch_shape,
                                          event_shape=np.shape(self.probs)[-1:],
                                          validate_args=validate_args)

    def sample(self, key, size=()):
        return multinomial_rvs(key, self.total_count, self.probs, shape=size + self.batch_shape)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        dtype = get_dtypes(self.probs)[0]
        value = lax.convert_element_type(value, dtype)
        total_count = lax.convert_element_type(self.total_count, dtype)
        return gammaln(total_count + 1) + np.sum(xlogy(value, self.probs) - gammaln(value + 1), axis=-1)

    @property
    def mean(self):
        return np.broadcast_to(self.probs * np.expand_dims(self.total_count, -1),
                               self.batch_shape + self.event_shape)

    @property
    def variance(self):
        return np.broadcast_to(np.expand_dims(self.total_count, -1) * self.probs * (1 - self.probs),
                               self.batch_shape + self.event_shape)

    @property
    def support(self):
        return constraints.multinomial(self.total_count)


class MultinomialWithLogits(Distribution):
    arg_constraints = {'total_count': constraints.nonnegative_integer,
                       'logits': constraints.real}

    def __init__(self, logits, total_count=1, validate_args=None):
        if np.ndim(logits) < 1:
            raise ValueError("`logits` parameter must be at least one-dimensional.")
        batch_shape = lax.broadcast_shapes(np.shape(logits)[:-1], np.shape(total_count))
        logits = logits - logsumexp(logits)
        self.logits = promote_shapes(logits, shape=batch_shape + np.shape(logits)[-1:])[0]
        self.total_count = promote_shapes(total_count, shape=batch_shape)[0]
        super(MultinomialWithLogits, self).__init__(batch_shape=batch_shape,
                                                    event_shape=np.shape(self.logits)[-1:],
                                                    validate_args=validate_args)

    def sample(self, key, size=()):
        return multinomial_rvs(key, self.total_count, self.probs, shape=size + self.batch_shape)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        dtype = get_dtypes(self.logits)[0]
        value = lax.convert_element_type(value, dtype)
        total_count = lax.convert_element_type(self.total_count, dtype)
        return gammaln(total_count + 1) + np.sum(value * self.logits - gammaln(value + 1), axis=-1)

    @lazy_property
    def probs(self):
        return _to_probs_multinom(self.logits)

    @property
    def mean(self):
        return np.broadcast_to(np.expand_dims(self.total_count, -1) * self.probs,
                               self.batch_shape + self.event_shape)

    @property
    def variance(self):
        return np.broadcast_to(np.expand_dims(self.total_count, -1) * self.probs * (1 - self.probs),
                               self.batch_shape + self.event_shape)

    @property
    def support(self):
        return constraints.multinomial(self.total_count)


class Categorical(Distribution):
    arg_constraints = {'probs': constraints.simplex}

    def __init__(self, probs, validate_args=None):
        if np.ndim(probs) < 1:
            raise ValueError("`probs` parameter must be at least one-dimensional.")
        self.probs = probs
        super(Categorical, self).__init__(batch_shape=np.shape(self.probs)[:-1],
                                          validate_args=validate_args)

    def sample(self, key, size=()):
        return categorical_rvs(key, self.probs, shape=size + self.batch_shape)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        batch_shape = lax.broadcast_shapes(np.shape(value), self.batch_shape)
        value = np.expand_dims(value, axis=-1)
        value = np.broadcast_to(value, batch_shape + (1,))
        logits = _to_logits_multinom(self.probs)
        log_pmf = np.broadcast_to(logits, batch_shape + np.shape(logits)[-1:])
        return np.take_along_axis(log_pmf, value, axis=-1)[..., 0]

    @property
    def mean(self):
        return lax.full(self.batch_shape, np.nan, dtype=self.probs.dtype)

    @property
    def variance(self):
        return lax.full(self.batch_shape, np.nan, dtype=self.probs.dtype)

    @property
    def support(self):
        return constraints.integer_interval(0, np.shape(self.probs)[-1])


class CategoricalWithLogits(Distribution):
    arg_constraints = {'logits': constraints.real}

    def __init__(self, logits, validate_args=None):
        if np.ndim(logits) < 1:
            raise ValueError("`logits` parameter must be at least one-dimensional.")
        logits = logits - logsumexp(logits)
        self.logits = logits
        super(CategoricalWithLogits, self).__init__(batch_shape=np.shape(logits)[:-1],
                                                    validate_args=validate_args)

    def sample(self, key, size=()):
        return categorical_rvs(key, self.probs, shape=size + self.batch_shape)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value = np.expand_dims(value, -1)
        value, log_pmf = promote_shapes(value, self.logits)
        value = value[..., :1]
        return np.take_along_axis(log_pmf, value, -1)[..., 0]

    @lazy_property
    def probs(self):
        return _to_probs_multinom(self.logits)

    @property
    def mean(self):
        return lax.full(self.batch_shape, np.nan, dtype=self.logits.dtype)

    @property
    def variance(self):
        return lax.full(self.batch_shape, np.nan, dtype=self.logits.dtype)

    @property
    def support(self):
        return constraints.integer_interval(0, np.shape(self.logits)[-1])


class Poisson(Distribution):
    arg_constraints = {'rate': constraints.positive}
    support = constraints.nonnegative_integer

    def __init__(self, rate, validate_args=None):
        self.rate = rate
        super(Poisson, self).__init__(np.shape(rate), validate_args=validate_args)

    def sample(self, key, size=()):
        return poisson(key, self.rate, shape=size + self.batch_shape)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value = lax.convert_element_type(value, get_dtypes(self.rate)[0])
        return (np.log(self.rate) * value) - gammaln(value + 1) - self.rate

    @property
    def mean(self):
        return self.rate

    @property
    def variance(self):
        return self.rate
