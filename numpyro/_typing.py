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


ModelArgs: TypeAlias = tuple[Any, ...]
"""Positional arguments of a model, as passed to ``MCMC.run(rng_key, *args)``."""

ModelKwargs: TypeAlias = dict[str, Any]
"""Keyword arguments of a model; may carry reserved keys such as ``GIBBS_SITES_KWARG``."""

SiteValues: TypeAlias = dict[str, jax.Array]
"""Values keyed by site name (a sample, a set of init params, a conditioning set)."""

PotentialFn: TypeAlias = Callable[[SiteValues], jax.Array]
"""Negative log joint as a function of (unconstrained) site values."""

ConstrainFn: TypeAlias = Callable[[SiteValues], SiteValues]
"""Maps site values to site values (constrain / postprocess)."""

StateT = TypeVar("StateT")
"""A kernel state pytree; used where a method returns the same state type it received."""
