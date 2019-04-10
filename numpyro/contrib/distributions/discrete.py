import jax.numpy as np
import jax.random as random

from numpyro.contrib.distributions.distribution import Distribution
from numpyro.distributions.util import binary_cross_entropy_with_logits, get_dtypes


def _to_probs_bernoulli(logits):
    return 1 / (1 + np.exp(logits))


def clamp_probs(probs):
    eps = np.finfo(get_dtypes(probs)[0]).eps
    return np.clip(probs, a_min=eps, a_max=1 - eps)


def _to_logits_bernoulli(probs):
    ps_clamped = clamp_probs(probs)
    return np.log(ps_clamped) - np.log1p(-ps_clamped)


class Bernoulli(Distribution):
    def __init__(self, probs=None, logits=None, validate_args=None):
        assert (probs is None) != (logits is None), \
            'Only one of `probs` or `logits` must be specified.'
        self.probs = probs if probs is not None else _to_probs_bernoulli(logits)
        self.logits = logits if logits is not None else _to_logits_bernoulli(probs)
        super(Bernoulli, self).__init__(batch_shape=np.shape(self.probs), validate_args=validate_args)

    def sample(self, key, size=()):
        return random.bernoulli(key, self.probs, shape=size)

    def log_prob(self, value):
        return -binary_cross_entropy_with_logits(self.logits, value)

    @property
    def mean(self):
        return self.probs

    @property
    def variance(self):
        return self.probs * (1 - self.probs)
