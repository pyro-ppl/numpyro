import jax.numpy as np
import jax.random as random
from jax import lax

from numpyro.contrib.distributions.distribution import Distribution
from numpyro.distributions.util import get_dtypes, promote_shapes


class Normal(Distribution):
    is_reparametrized = True

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = promote_shapes(loc, scale)
        batch_shape = lax.broadcast_shapes(np.shape(loc), np.shape(scale))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, size=()):
        eps = random.normal(key, shape=size)
        return self.loc + eps * self.scale

    def log_prob(self, value):
        return -((value - self.loc) ** 2) / (2.0 * self.scale ** 2) \
               - np.log(self.scale) - np.log(np.sqrt(2 * np.pi))

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self.scale


class Uniform(Distribution):
    is_reparametrized = False

    def __init__(self, low, high, validate_args=None):
        self.low, self.high = promote_shapes(low, high)
        batch_shape = lax.broadcast_shapes(np.shape(low), np.shape(high))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, size=()):
        size = size or self.batch_shape
        return self.low + random.uniform(key, shape=size) * (self.high - self.low)

    def log_prob(self, value):
        within_bounds = ((value >= self.low) & (value < self.high))
        return np.log(lax.convert_element_type(within_bounds, get_dtypes(self.low)[0])) - \
            np.log(self.high - self.low)

    @property
    def mean(self):
        return (self.high - self.low) / 2.

    @property
    def variance(self):
        return (self.high - self.low) ** 2 / 12.
