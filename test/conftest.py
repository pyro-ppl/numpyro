# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import os

from jax.config import config

from numpyro.util import set_rng_seed

config.update("jax_platform_name", "cpu")  # noqa: E702


def pytest_runtest_setup(item):
    if "JAX_ENABLE_X64" in os.environ:
        config.update("jax_enable_x64", True)
    set_rng_seed(0)
