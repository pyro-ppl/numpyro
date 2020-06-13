# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

try:
    import funsor
except ImportError:
    raise ImportError("`funsor` package is missing. You can install it with `pip install funsor`.")

from numpyro.contrib.funsor.enum_messenger import (enum, infer_config, markov, plate,
                                                   to_data, to_funsor, trace)
from numpyro.contrib.funsor.infer_util import config_enumerate, enum_log_density

funsor.set_backend("jax")


__all__ = [
    "config_enumerate",
    "enum",
    "enum_log_density",
    "infer_config",
    "markov",
    "plate",
    "to_data",
    "to_funsor",
    "trace",
]
