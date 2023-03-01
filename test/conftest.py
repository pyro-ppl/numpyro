# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import os

os.environ["JAX_JIT_PJIT_API_MERGE"] = "0"

from jax.config import config  # noqa: E402

from numpyro.util import set_rng_seed  # noqa: E402

config.update("jax_platform_name", "cpu")  # noqa: E702


def pytest_runtest_setup(item):
    if "JAX_ENABLE_X64" in os.environ:
        config.update("jax_enable_x64", True)
    set_rng_seed(0)
