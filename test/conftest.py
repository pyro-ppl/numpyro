from numpyro.util import set_rng_seed
from numpyro.distributions.distribution import jax_frozen


def pytest_runtest_setup(item):
    # TODO use context manager instead
    jax_frozen._validate_args = False
    set_rng_seed(0)
