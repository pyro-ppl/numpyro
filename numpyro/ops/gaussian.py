# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Information-form Gaussian factors for linear-Gaussian state-space models.

A :class:`Gaussian` is an unnormalized log-quadratic function
``g(x) = log_normalizer + x . info_vec - 0.5 x^T precision x`` whose precision
may be rank deficient. Factors over pairs of consecutive states compose
associatively, so a chain of ``T`` factors reduces in ``O(log T)`` parallel
depth (:func:`sequential_gaussian_tensordot`) and posterior state paths can be
sampled with the same depth (:func:`sequential_gaussian_filter_sample`).

Matrices act on the left: :func:`matrix_and_mvn_to_gaussian` encodes
``y = matrix @ x + noise`` with ``matrix`` of shape ``(..., y_dim, x_dim)``.
A factor over ``(x, y)`` stores ``x`` first.

Every factor keeps all of its fields at one common batch shape and never
carries sample dimensions; callers handle extra leading dimensions with
:func:`jax.vmap`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar, Sequence, Union

import jax
from jax import Array, lax
import jax.numpy as jnp

__all__ = ["Gaussian"]

_LOG_2PI = math.log(2 * math.pi)


def _mv(matrix: Array, vector: Array) -> Array:
    return jnp.einsum("...ij,...j->...i", matrix, vector)


def _mt(matrix: Array) -> Array:
    return jnp.swapaxes(matrix, -1, -2)


def _pad_event(x: Array, event_ndim: int, left: int, right: int) -> Array:
    return jnp.pad(x, [(0, 0)] * (x.ndim - event_ndim) + [(left, right)] * event_ndim)


def _with_batch(x: Array, event_ndim: int, batch_shape: tuple[int, ...]) -> Array:
    return jnp.broadcast_to(x, batch_shape + x.shape[x.ndim - event_ndim :])


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Gaussian:
    """
    Unnormalized log-quadratic factor
    ``log_normalizer + x . info_vec - 0.5 x^T precision x``.

    Parameters
    ----------
    log_normalizer : Array
        Shape ``batch_shape``.
    info_vec : Array
        Shape ``batch_shape + (dim,)``.
    precision : Array
        Shape ``batch_shape + (dim, dim)``; may be rank deficient.
    """

    log_normalizer: Array
    info_vec: Array
    precision: Array

    event_ndims: ClassVar[tuple[int, int, int]] = (0, 1, 2)

    @property
    def dim(self) -> int:
        return self.info_vec.shape[-1]

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return lax.broadcast_shapes(
            self.log_normalizer.shape,
            self.info_vec.shape[:-1],
            self.precision.shape[:-2],
        )

    def _map(self, fn) -> Gaussian:
        fields = (self.log_normalizer, self.info_vec, self.precision)
        return Gaussian(*(fn(x, k) for x, k in zip(fields, self.event_ndims)))

    def expand(self, batch_shape: Sequence[int]) -> Gaussian:
        """Broadcast every field to ``batch_shape``."""
        return self._map(lambda x, k: _with_batch(x, k, tuple(batch_shape)))

    def reshape(self, batch_shape: Sequence[int]) -> Gaussian:
        """Reshape the batch dimensions of every field to ``batch_shape``."""
        return self._map(
            lambda x, k: x.reshape(tuple(batch_shape) + x.shape[x.ndim - k :])
        )

    def __getitem__(self, index: Union[int, slice, tuple]) -> Gaussian:
        index = index if isinstance(index, tuple) else (index,)
        return self._map(lambda x, k: x[index + (slice(None),) * k])

    @staticmethod
    def cat(parts: Sequence[Gaussian], axis: int = 0) -> Gaussian:
        """Concatenate factors along a batch axis."""
        axis = axis % len(parts[0].batch_shape)
        return Gaussian(
            jnp.concatenate([p.log_normalizer for p in parts], axis),
            jnp.concatenate([p.info_vec for p in parts], axis),
            jnp.concatenate([p.precision for p in parts], axis),
        )

    def event_pad(self, left: int = 0, right: int = 0) -> Gaussian:
        """Embed the factor into a larger event space with zero coupling."""
        return Gaussian(
            self.log_normalizer,
            _pad_event(self.info_vec, 1, left, right),
            _pad_event(self.precision, 2, left, right),
        )

    def event_permute(self, perm: Array) -> Gaussian:
        """Permute event coordinates."""
        return Gaussian(
            self.log_normalizer,
            self.info_vec[..., perm],
            self.precision[..., perm, :][..., :, perm],
        )

    def _broadcast(self) -> Gaussian:
        return self.expand(self.batch_shape)

    def __add__(self, other: Union[Gaussian, Array, float]) -> Gaussian:
        if type(other).__name__ == "AffineNormal":
            other = other.to_gaussian()
        if isinstance(other, Gaussian):
            return Gaussian(
                self.log_normalizer + other.log_normalizer,
                self.info_vec + other.info_vec,
                self.precision + other.precision,
            )._broadcast()
        return Gaussian(
            self.log_normalizer + other, self.info_vec, self.precision
        )._broadcast()

    def __sub__(self, other: Union[Array, float]) -> Gaussian:
        return Gaussian(
            self.log_normalizer - other, self.info_vec, self.precision
        )._broadcast()

    def log_density(self, value: Array) -> Array:
        """Evaluate the factor at ``value`` of shape ``(..., dim)``."""
        if self.dim == 0:
            return jnp.broadcast_to(
                self.log_normalizer,
                lax.broadcast_shapes(value.shape[:-1], self.batch_shape),
            )
        quadratic = (value * (-0.5 * _mv(self.precision, value) + self.info_vec)).sum(
            -1
        )
        return quadratic + self.log_normalizer
