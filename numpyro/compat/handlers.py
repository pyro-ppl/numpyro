from numpyro.handlers import *  # noqa: F401, F403
from numpyro.handlers import seed as numpyro_seed


# Compatibility wrapper for matching arg names
def seed(fn=None, rng_seed=None):
    return numpyro_seed(fn=fn, rng_key=rng_seed)
