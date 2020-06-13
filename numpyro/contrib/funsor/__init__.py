# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

# TODO: add a try catch message to ensure users install funsor with jax backend properly
import funsor

from numpyro.contrib.funsor.enum_messenger import (enum, infer_config, markov, plate,
                                                   to_data, to_funsor, trace)
from numpyro.contrib.funsor.infer_util import enum_log_density

funsor.set_backend("jax")


__all__ = [
    "enum",
    "enum_log_density",
    "infer_config",
    "markov",
    "plate",
    "to_data",
    "to_funsor",
    "trace",
]
