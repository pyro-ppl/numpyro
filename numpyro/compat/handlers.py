from jax import random

from numpyro.handlers import *  # noqa: F401, F403
from numpyro.handlers import seed as _seed


# This is so that users do not have to import PRNGKey from jax.random.
# XXX: Should we make this the default?
class seed(_seed):
    def __init__(self, fn, rng):
        super(seed, self).__init__(fn, random.PRNGKey(rng))
