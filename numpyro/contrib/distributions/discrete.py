import jax.numpy as np
import jax.random as random
from jax import lax
from jax.scipy.special import gammaln

from numpyro.contrib.distributions.distribution import Distribution
from numpyro.distributions import constraints
from numpyro.distributions.util import (
    binary_cross_entropy_with_logits,
    binomial,
    get_dtypes,
    lazy_property,
    promote_shapes,
    xlog1py,
    xlogy
)


def _to_probs_bernoulli(logits):
    return 1 / (1 + np.exp(-logits))


def clamp_probs(probs):
    eps = np.finfo(get_dtypes(probs)[0]).eps
    return np.clip(probs, a_min=eps, a_max=1 - eps)


def _to_logits_bernoulli(probs):
    ps_clamped = clamp_probs(probs)
    return np.log(ps_clamped) - np.log1p(-ps_clamped)


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
        return xlogy(value, self.probs) + xlog1py(1-value, -self.probs)

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
        super(Binomial, self).__init__(batch_shape=np.shape(self.probs), validate_args=validate_args)

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


class BinomialWithLogits(Distribution):
    arg_constraints = {'total_count': constraints.nonnegative_integer,
                       'logits': constraints.real}

    def __init__(self, logits, total_count=1, validate_args=None):
        self.logits, self.total_count = promote_shapes(logits, total_count)
        super(BinomialWithLogits, self).__init__(batch_shape=np.shape(self.probs), validate_args=validate_args)

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
        return log_factorial_n - log_factorial_k - log_factorial_nmk + value * self.logits \
            - total_count * np.clip(self.logits, 0) - xlog1py(self.total_count, np.exp(-np.abs(self.logits)))

    @lazy_property
    def probs(self):
        return _to_probs_bernoulli(self.logits)

    @property
    def mean(self):
        return np.broadcast_to(self.total_count * self.probs, self.batch_shape)

    @property
    def variance(self):
        return np.broadcast_to(self.total_count * self.probs * (1 - self.probs), self.batch_shape)
