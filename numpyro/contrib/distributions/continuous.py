from torch.distributions import AbsTransform, TransformedDistribution

import jax.numpy as np
import jax.random as random
from jax import lax

from numpyro.contrib.distributions.distribution import Distribution
from numpyro.distributions import constraints
from numpyro.distributions.util import get_dtypes, promote_shapes


class Cauchy(Distribution):
    is_reparametrized = True
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = promote_shapes(loc, scale)
        batch_shape = lax.broadcast_shapes(np.shape(loc), np.shape(scale))
        super(Cauchy, self).__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, size=()):
        u = random.uniform(key, shape=size)
        eps = np.tan(np.pi * (u - 0.5))
        return self.loc + eps * self.scale

    def log_prob(self, value):
        return - np.log(np.pi) - np.log(self.scale) - (1.0 + ((value - self.loc) / self.scale) ** 2)

    @property
    def mean(self):
        return np.broadcast_to(self.loc, self.batch_shape)

    @property
    def variance(self):
        return np.broadcast_to(self.scale, self.batch_shape)


class Exponential(Distribution):
    is_reparametrized = True
    arg_constraints = {'rate': constraints.positive}
    support = constraints.positive

    def __init__(self, rate, validate_args=None):
        self.rate = rate
        super(Exponential, self).__init__(batch_shape=np.shape(rate), validate_args=validate_args)

    def sample(self, key, size=()):
        u = random.uniform(key, shape=size)
        return np.log1p(-(-u)) / self.rate

    def log_prob(self, value):
        return self.rate.log() - self.rate * value

    @property
    def mean(self):
        return np.reciprocal(self.rate)

    @property
    def variance(self):
        return np.reciprocal(self.rate)


class HalfCauchy(TransformedDistribution):
    is_reparametrized = True
    arg_constraints = {'scale': constraints.positive}
    support = constraints.positive

    def __init__(self, scale, validate_args=None):
        base_dist = Cauchy(0, scale)
        super(HalfCauchy, self).__init__(base_dist, AbsTransform(),
                                         validate_args=validate_args)

    def log_prob(self, value):
        log_prob = self.base_dist.log_prob(value) + np.log(2)
        log_prob[value.expand(log_prob.shape) < 0] = -np.inf
        return log_prob

    @property
    def mean(self):
        return self.base_dist.mean

    @property
    def variance(self):
        return self.base_dist.variance


class Normal(Distribution):
    is_reparametrized = True
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = promote_shapes(loc, scale)
        batch_shape = lax.broadcast_shapes(np.shape(loc), np.shape(scale))
        super(Normal, self).__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, size=()):
        eps = random.normal(key, shape=size)
        return self.loc + eps * self.scale

    def log_prob(self, value):
        return -((value - self.loc) ** 2) / (2.0 * self.scale ** 2) \
               - np.log(self.scale) - np.log(np.sqrt(2 * np.pi))

    @property
    def mean(self):
        return np.broadcast_to(self.loc, self.batch_shape)

    @property
    def variance(self):
        return np.broadcast_to(self.scale, self.batch_shape)


class Uniform(Distribution):
    is_reparametrized = True
    arg_constraints = {'low': constraints.dependent, 'high': constraints.dependent}

    def __init__(self, low, high, validate_args=None):
        self.low, self.high = promote_shapes(low, high)
        batch_shape = lax.broadcast_shapes(np.shape(low), np.shape(high))
        super(Uniform, self).__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, size=()):
        size = size or self.batch_shape
        return self.low + random.uniform(key, shape=size) * (self.high - self.low)

    def log_prob(self, value):
        within_bounds = ((value >= self.low) & (value < self.high))
        return np.log(lax.convert_element_type(within_bounds, get_dtypes(self.low)[0])) - \
            np.log(self.high - self.low)

    @property
    def mean(self):
        return np.broadcast_to((self.high - self.low) / 2., self.batch_shape)

    @property
    def variance(self):
        return np.broadcast_to((self.high - self.low) ** 2 / 12., self.batch_shape)

    @property
    def support(self):
        return constraints.interval(self.low, self.high)
