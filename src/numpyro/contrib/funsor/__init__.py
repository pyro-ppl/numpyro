# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

try:
    import funsor
except ImportError as e:
    raise ImportError(
        "Looking like you want to do inference for models with "
        "discrete latent variables. This is an experimental feature. "
        "You need to install `funsor` to be able to use this feature. "
        "It can be installed with `pip install funsor`."
    ) from e

from numpyro.contrib.funsor.discrete import infer_discrete
from numpyro.contrib.funsor.enum_messenger import (
    enum,
    infer_config,
    markov,
    plate,
    to_data,
    to_funsor,
    trace,
)
from numpyro.contrib.funsor.infer_util import (
    config_enumerate,
    config_kl,
    log_density,
    plate_to_enum_plate,
)

funsor.set_backend("jax")


__all__ = [
    "config_enumerate",
    "config_kl",
    "enum",
    "infer_config",
    "infer_discrete",
    "log_density",
    "markov",
    "plate",
    "plate_to_enum_plate",
    "to_data",
    "to_funsor",
    "trace",
]
