# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import warnings

from numpyro.compat.util import UnsupportedAPIWarning
from numpyro.primitives import module, param as _param, plate, sample  # noqa: F401

_PARAM_STORE = {}


def get_param_store():
    warnings.warn(
        "A limited parameter store is provided for compatibility with Pyro. "
        "Value of SVI parameters should be obtained via SVI.get_params() method.",
        category=UnsupportedAPIWarning,
    )
    # Return an empty dict for compatibility
    return _PARAM_STORE


def clear_param_store():
    return _PARAM_STORE.clear()


def param(name, *args, **kwargs):
    # Get value assuming statement is wrapped in a substitute handler.
    val = _param(name, *args, **kwargs)
    # If no value is found, check param store.
    if val is None:
        # If other arguments are provided, e.g. for initialization, raise error.
        if args or kwargs:
            raise NotImplementedError
        param_store = get_param_store()
        if name in param_store:
            val = param_store[name]
    return val
