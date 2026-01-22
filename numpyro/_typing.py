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


NumLikeT = TypeVar("NumLikeT", bound=NumLike)
