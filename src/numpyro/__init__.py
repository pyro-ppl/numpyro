# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging
import warnings

warnings.filterwarnings(
    "ignore", message=".*Attempting to hash a tracer.*", category=FutureWarning
)

# ruff: noqa: E402

from numpyro import compat, diagnostics, distributions, handlers, infer, ops, optim
from numpyro.distributions.distribution import enable_validation, validation_enabled
from numpyro.infer.inspect import render_model
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
    subsample,
)
from numpyro.util import enable_x64, set_host_device_count, set_platform
from numpyro.version import __version__


# filter out this annoying warning, which raises even when we install CPU-only jaxlib
def _filter_absl_cpu_warning(record):
    return not record.getMessage().startswith("No GPU/TPU found, falling back to CPU.")


logging.getLogger("absl").addFilter(_filter_absl_cpu_warning)


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
    "ops",
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
