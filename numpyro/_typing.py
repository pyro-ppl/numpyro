# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0


from collections import OrderedDict
from collections.abc import Callable
from typing import (
    Any,
    ParamSpec,
    TypeAlias,
    TypeVar,
    Union,
)

import numpy as np

import jax

P = ParamSpec("P")
ModelT: TypeAlias = Callable[P, Any]

Message: TypeAlias = dict[str, Any]
TraceT: TypeAlias = OrderedDict[str, Message]


NonScalarArray = Union[np.ndarray, jax.Array]
"""An alias for array-like types excluding scalars."""


NumLike = Union[NonScalarArray, np.number, int, float, complex]
"""An alias for array-like types excluding `np.bool_` and `bool`."""


PyTree: TypeAlias = Any
"""A generic type for a pytree, i.e. a nested structure of lists, tuples, dicts, and arrays."""


PositionDict: TypeAlias = dict[str, jax.Array]
"""An unconstrained position dict keyed by sample-site name.

Used as the canonical input/output type for log-density and postprocess
callables exposed to external samplers (see
:class:`~numpyro.infer.LogDensityInfo` and
:class:`~numpyro.infer.ExternalKernel`)."""


NumLikeT = TypeVar("NumLikeT", bound=NumLike)
