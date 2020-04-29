import random as rand
from jax import lax, ops
import jax.numpy as np
import jax.scipy as scipy
import jax.random as random
from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.continuous import Dirichlet, Normal
from numpyro.distributions.discrete import Categorical
from numpyro.util import copy_docs_from

@copy_docs_from(Distribution)
class NormalMixture(Distribution):
    arg_constraints = {'weights': constraints.simplex, 'locs': constraints.real, 'scales': constraints.positive}
    support = constraints.real
    def __init__(self, weights, locs, scales, validate_args=None):
        batch_shape = lax.broadcast_shapes(np.shape(weights)[:-1], np.shape(locs)[:-1], np.shape(scales)[:-1])
        self.mixture_shape = lax.broadcast_shapes(np.shape(weights)[-1:], np.shape(locs)[-1:], np.shape(scales)[-1:])
        self.weights = np.broadcast_to(weights, batch_shape + self.mixture_shape)
        self.locs = np.broadcast_to(locs, batch_shape + self.mixture_shape)
        self.scales = np.broadcast_to(scales, batch_shape + self.mixture_shape)
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        ps = Dirichlet(self.weights).sample(key, sample_shape=sample_shape)
        zs = np.expand_dims(Categorical(ps).sample(key), axis=-1)
        locs = np.broadcast_to(self.locs, sample_shape + self.batch_shape + self.event_shape + self.mixture_shape)
        scales = np.broadcast_to(self.scales, sample_shape + self.batch_shape + self.event_shape + self.mixture_shape)
        mlocs = np.squeeze(np.take_along_axis(locs, zs, axis=-1), axis=-1)
        mscales = np.squeeze(np.take_along_axis(scales, zs, axis=-1), axis=-1)
        return Normal(mlocs, mscales).sample(key)

    def log_prob(self, value):
        value = np.expand_dims(value, axis=-1)
        mlog_prob = Normal(self.locs, self.scales).log_prob(value)
        if self._validate_args:
            self._validate_sample(value)
        wlog_prob = np.log(self.weights) + mlog_prob
        return scipy.special.logsumexp(wlog_prob, axis=-1)
