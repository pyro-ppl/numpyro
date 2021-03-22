# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from numpyro import compat, diagnostics, distributions, handlers, infer, optim
from numpyro.distributions.distribution import enable_validation, validation_enabled
import numpyro.patch  # noqa: F401
from numpyro.primitives import (
    deterministic,
    factor,
    get_mask,
    module,
    param,
    plate,
    plate_stack,
    prng_key,
    sample,
    subsample
)
from numpyro.util import enable_x64, set_host_device_count, set_platform
from numpyro.contrib.render import render_model
from numpyro.version import __version__

set_platform("cpu")


__all__ = [
    "__version__",
    "compat",
    "deterministic",
    "diagnostics",
    "distributions",
    "enable_x64",
    "enable_validation",
    "factor",
    "get_mask",
    "handlers",
    "infer",
    "module",
    "optim",
    "param",
    "plate",
    "plate_stack",
    "prng_key",
    "render_model",
    "sample",
    "subsample",
    "set_host_device_count",
    "set_platform",
    "validation_enabled",
]
