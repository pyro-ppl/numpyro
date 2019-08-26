import os

from jax.config import config; config.update('jax_platform_name', 'cpu')  # noqa: E702
from numpyro.util import set_rng_seed


def pytest_runtest_setup(item):
    if 'JAX_ENABLE_x64' in os.environ:
        config.update('jax_enable_x64', True)
    set_rng_seed(0)
