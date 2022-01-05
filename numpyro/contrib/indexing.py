# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import warnings

from numpyro.ops import Vindex, vindex  # noqa: F401
from numpyro.util import find_stack_level

warnings.warn(
    "`indexing` module has been moved from `numpyro.contrib` to `numpyro.ops`."
    " Please import Vindex or vindex functions from `numpyro.ops.indexing`.",
    FutureWarning,
    stacklevel=find_stack_level(),
)
