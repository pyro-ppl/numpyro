# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0


from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Optional, Protocol, TypeVar, Union, runtime_checkable
import weakref

try:
    from typing import ParamSpec, TypeAlias
except ImportError:
    from typing_extensions import ParamSpec, TypeAlias


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


# ConstraintLike represents any constraint object (both Constraint[NumLike] and
# Constraint[NonScalarArray]). We use Any because:
# 1. Constraint[T] is a generic class, and mypy struggles with Protocol subtyping
#    of generic classes due to variance issues
# 2. Some constraints only accept NonScalarArray while others accept full NumLike,
#    making a single Protocol definition either too restrictive or too permissive
# 3. Creating a structural Protocol causes mypy errors when assigning concrete
#    constraint instances to protocol-typed variables
# At runtime, all constraints share the interface (is_discrete, event_dim, __call__, etc.)
# and work correctly. This type alias provides better documentation than raw `Any`
# while acknowledging the limitations of Python's type system for this use case.
ConstraintLike: TypeAlias = Any


@runtime_checkable
class ConstraintT(Protocol):
    """A protocol for typing constraints that accept NumLike inputs.

    This is a more specific protocol for constraints that can handle both
    arrays and scalars (NumLike type).
    """

    @property
    def is_discrete(self) -> bool: ...
    @property
    def event_dim(self) -> int: ...
    @is_discrete.setter
    def is_discrete(self, value: bool): ...
    @event_dim.setter
    def event_dim(self, value: int): ...

    def __call__(self, x: NumLike) -> ArrayLike: ...
    def __repr__(self) -> str: ...
    def check(self, value: NumLike) -> ArrayLike: ...
    def feasible_like(self, prototype: NumLike) -> NumLike: ...


@runtime_checkable
class DistributionT(Protocol):
    """A protocol for typing distributions.

    Used to type object of type numpyro.distributions.Distribution, funsor.Funsor
    or tensorflow_probability.distributions.Distribution.
    """

    arg_constraints: dict[str, ConstraintT] = ...
    support: ConstraintT = ...
    has_enumerate_support: bool = ...
    reparametrized_params: list[str] = ...
    _validate_args: bool = ...
    pytree_data_fields: tuple = ...
    pytree_aux_fields: tuple = ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

    def rsample(
        self, key: jax.dtypes.prng_key, sample_shape: tuple[int, ...] = ()
    ) -> ArrayLike: ...
    def sample(
        self, key: jax.dtypes.prng_key, sample_shape: tuple[int, ...] = ()
    ) -> ArrayLike: ...
    def log_prob(self, value: ArrayLike) -> ArrayLike: ...
    def cdf(self, value: ArrayLike) -> ArrayLike: ...
    def icdf(self, q: ArrayLike) -> ArrayLike: ...
    def entropy(self) -> ArrayLike: ...
    def enumerate_support(self, expand: bool = True) -> ArrayLike: ...
    def shape(self, sample_shape: tuple[int, ...] = ()) -> tuple[int, ...]: ...

    @property
    def batch_shape(self) -> tuple[int, ...]: ...
    @property
    def event_shape(self) -> tuple[int, ...]: ...
    @property
    def event_dim(self) -> int: ...
    @property
    def has_rsample(self) -> bool: ...

    @property
    def mean(self) -> ArrayLike: ...
    @property
    def variance(self) -> ArrayLike: ...

    @property
    def is_discrete(self) -> bool: ...


# To avoid breaking changes for user code that uses `DistributionLike`
DistributionLike = DistributionT


@runtime_checkable
class TransformT(Protocol):
    _inv: Optional[Union["TransformT", weakref.ref]] = ...

    @property
    def domain(self) -> ConstraintLike: ...
    @property
    def codomain(self) -> ConstraintLike: ...
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
