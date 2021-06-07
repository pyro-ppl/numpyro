import jax.numpy as jnp
import jax.scipy as jscipy
from jax import lax

from numpyro.distributions import constraints
from numpyro.distributions.continuous import Dirichlet, Normal
from numpyro.distributions.discrete import Categorical
from numpyro.distributions.distribution import Distribution


class NormalMixture(Distribution):
    arg_constraints = {
        "weights": constraints.simplex,
        "locs": constraints.real,
        "scales": constraints.positive,
    }
    support = constraints.real

    def __init__(self, weights, locs, scales, validate_args=None):
        batch_shape = lax.broadcast_shapes(
            jnp.shape(weights)[:-1], jnp.shape(locs)[:-1], jnp.shape(scales)[:-1]
        )
        self.mixture_shape = lax.broadcast_shapes(
            jnp.shape(weights)[-1:], jnp.shape(locs)[-1:], jnp.shape(scales)[-1:]
        )
        self.weights = jnp.broadcast_to(weights, batch_shape + self.mixture_shape)
        self.locs = jnp.broadcast_to(locs, batch_shape + self.mixture_shape)
        self.scales = jnp.broadcast_to(scales, batch_shape + self.mixture_shape)
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        ps = Dirichlet(self.weights).sample(key, sample_shape=sample_shape)
        zs = jnp.expand_dims(Categorical(ps).sample(key), axis=-1)
        locs = jnp.broadcast_to(
            self.locs,
            sample_shape + self.batch_shape + self.event_shape + self.mixture_shape,
        )
        scales = jnp.broadcast_to(
            self.scales,
            sample_shape + self.batch_shape + self.event_shape + self.mixture_shape,
        )
        mlocs = jnp.squeeze(jnp.take_along_axis(locs, zs, axis=-1), axis=-1)
        mscales = jnp.squeeze(jnp.take_along_axis(scales, zs, axis=-1), axis=-1)
        return Normal(mlocs, mscales).sample(key)

    def log_prob(self, value):
        value = jnp.expand_dims(value, axis=-1)
        mlog_prob = Normal(self.locs, self.scales).log_prob(value)
        if self._validate_args:
            self._validate_sample(value)
        wlog_prob = jnp.log(self.weights) + mlog_prob
        return jscipy.special.logsumexp(wlog_prob, axis=-1)
