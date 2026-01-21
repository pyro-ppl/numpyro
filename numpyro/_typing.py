# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0


from collections import OrderedDict
from collections.abc import Callable
from typing import (
    Any,
    Optional,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
    Union,
    runtime_checkable,
)
import weakref

import numpy as np

import jax
from jax.typing import ArrayLike

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


@runtime_checkable
class ConstraintT(Protocol):
    """A protocol for typing constraints."""

    @property
    def is_discrete(self) -> bool: ...
    @property
    def event_dim(self) -> int: ...

    def __call__(self, x: NumLike) -> ArrayLike: ...
    def __repr__(self) -> str: ...
    def check(self, value: NumLike) -> ArrayLike: ...
    def feasible_like(self, prototype: NumLike) -> NumLike: ...


@runtime_checkable
class TransformT(Protocol):
    _inv: Optional[Union["TransformT", weakref.ref]] = ...

    @property
    def domain(self) -> ConstraintT: ...
    @property
    def codomain(self) -> ConstraintT: ...
    @property
    def inv(self) -> "TransformT": ...
    @property
    def sign(self) -> NumLike: ...

    def __call__(self, x: NumLike) -> NumLike: ...
    def _inverse(self, y: NumLike) -> NumLike: ...
    def log_abs_det_jacobian(
        self, x: NumLike, y: NumLike, intermediates: Optional[PyTree] = None
    ) -> NumLike: ...
    def call_with_intermediates(
        self, x: NumLike
    ) -> tuple[NumLike, Optional[PyTree]]: ...
    def forward_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]: ...
    def inverse_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]: ...
