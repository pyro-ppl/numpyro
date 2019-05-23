from jax.config import config

from numpyro.util import set_rng_seed


def pytest_runtest_setup(item):
    config.update('jax_platform_name', 'cpu')
    set_rng_seed(0)
