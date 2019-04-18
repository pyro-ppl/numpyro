from numpyro.util import set_rng_seed


def pytest_runtest_setup(item):
    set_rng_seed(0)
