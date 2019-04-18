import os

from numpyro.util import set_rng_seed


def pytest_runtest_setup(item):
    set_rng_seed(0)
    os.environ['XLA_FLAGS'] = '--xla_cpu_enable_fast_math=false'
