# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from numpyro import compat, diagnostics, distributions, handlers, infer, optim
from numpyro.distributions.distribution import enable_validation, validation_enabled
import numpyro.patch  # noqa: F401
from numpyro.primitives import factor, module, param, plate, sample
from numpyro.util import enable_x64, set_host_device_count, set_platform
from numpyro.version import __version__

set_platform('cpu')


__all__ = [
    '__version__',
    'compat',
    'diagnostics',
    'distributions',
    'enable_x64',
    'enable_validation',
    'factor',
    'handlers',
    'infer',
    'module',
    'optim',
    'param',
    'plate',
    'sample',
    'set_host_device_count',
    'set_platform',
    'validation_enabled',
]
