# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Information-form Gaussian factors for linear-Gaussian state-space models.

A :class:`Gaussian` is an unnormalized log-quadratic function
``g(x) = log_normalizer + x . info_vec - 0.5 x^T precision x`` whose precision
may be rank deficient. Factors over pairs of consecutive states compose
associatively, so a chain of ``T`` factors reduces in ``O(log T)`` parallel
depth (:func:`sequential_gaussian_tensordot`) and posterior state paths can be
sampled with the same depth (:func:`sequential_gaussian_filter_sample`). Both
algorithms follow the parallel-scan formulation of Sarkka and
Garcia-Fernandez, "Temporal parallelization of Bayesian smoothers", IEEE
Transactions on Automatic Control 66(1), 2021 (arXiv:1905.13002), and the
module is a port of Pyro's ``pyro.ops.gaussian``.

Matrices act on the left: :func:`matrix_and_mvn_to_gaussian` encodes
``y = matrix @ x + noise`` with ``matrix`` of shape ``(..., y_dim, x_dim)``.
A factor over ``(x, y)`` stores ``x`` first. Pyro's factories use
``y = x @ matrix`` with ``matrix`` of shape ``(..., x_dim, y_dim)``, so a
matrix taken from a Pyro model must be transposed.

The fields of a factor must broadcast to one batch shape. Factor fields never
carry sample dimensions; :meth:`Gaussian.condition`,
:meth:`Gaussian.left_condition` and their :class:`AffineNormal` counterparts
accept a ``value`` with leading dimensions beyond ``batch_shape`` and return a
factor whose batch shape is the broadcast of the two (the conditioned blocks
are materialized once per leading element). Every other operation expects
inputs already aligned to ``batch_shape``; use :func:`jax.vmap` for additional
leading dimensions. Shape operations (``__getitem__``, ``reshape``, ``cat``)
broadcast the fields to that shape first, and every factory and operation in
this module already returns broadcast fields via ``_broadcast()``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, ClassVar, Optional, Self, Sequence, TypeVar, Union

import numpy as np

import jax
from jax import Array, lax, random
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, solve_triangular

from numpyro.distributions.continuous import MultivariateNormal, Normal
from numpyro.distributions.distribution import (
    Distribution,
    ExpandedDistribution,
    Independent,
)
from numpyro.distributions.util import (
    cholesky_of_inverse,
    jitter_if_singular,
    safe_cholesky,
)

__all__ = [
    "AffineNormal",
    "Gaussian",
    "gaussian_tensordot",
    "loc_and_scale_tril",
    "matrix_and_gaussian_to_gaussian",
    "matrix_and_mvn_to_gaussian",
    "mvn_moments",
    "mvn_to_gaussian",
    "sequential_gaussian_filter_sample",
    "sequential_gaussian_tensordot",
]

_LOG_2PI = math.log(2 * math.pi)


def _mv(matrix: Array, vector: Array) -> Array:
    return jnp.einsum("...ij,...j->...i", matrix, vector)


def _mt(matrix: Array) -> Array:
    return jnp.swapaxes(matrix, -1, -2)


def _log_diag_sum(chol: Array) -> Array:
    """``sum(log(diag(chol)))`` via a masked reduce rather than a gather."""
    return jnp.log(jnp.einsum("...ii->...i", chol)).sum(-1)


def _static_runs(perm: np.ndarray) -> list[slice]:
    """
    Split a permutation into maximal runs of consecutive indices.

    :param numpy.ndarray perm: permutation of ``range(n)``.
    :return: one slice per run, in the order of ``perm``; ``[]`` for an empty
        permutation.
    :rtype: list[slice]
    """
    if perm.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(perm) != 1) + 1
    bounds = np.concatenate([[0], cuts, [perm.size]])
    return [
        slice(int(perm[i]), int(perm[i]) + int(j - i))
        for i, j in zip(bounds[:-1], bounds[1:])
    ]


def _pad_event(x: Array, event_ndim: int, left: int, right: int) -> Array:
    return jnp.pad(x, [(0, 0)] * (x.ndim - event_ndim) + [(left, right)] * event_ndim)


def _with_batch(x: Array, event_ndim: int, batch_shape: tuple[int, ...]) -> Array:
    return jnp.broadcast_to(x, batch_shape + x.shape[x.ndim - event_ndim :])


def _noise(
    noise: Optional[Array],
    key: Optional[Array],
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> Array:
    """
    Return ``noise`` after checking its shape, or draw standard normals.

    :param Array noise: draws of exactly ``shape``, or ``None`` to draw them.
    :param Array key: PRNG key; required when ``noise`` is ``None``.
    :param tuple shape: required shape of the draws.
    :param dtype: dtype of the fresh draws, canonicalized to the enabled
        precision so NumPy float64 fields do not trigger x64 warnings.
    :raises ValueError: if neither ``key`` nor ``noise`` is given, or if
        ``noise.shape != shape``.
    """
    if noise is None:
        if key is None:
            raise ValueError("either key or noise is required")
        return random.normal(key, shape, jax.dtypes.canonicalize_dtype(dtype))
    if noise.shape != shape:
        raise ValueError(f"noise must have shape {shape}, got {noise.shape}")
    return noise


class _FactorShapeOps:
    """
    Batch-shape operations shared by the factor dataclasses.

    A subclass is a frozen dataclass whose fields are arrays with, for field
    ``i``, ``event_ndims[i]`` trailing event dimensions; the leading
    dimensions broadcast to ``batch_shape``. The mixin adds no dataclass
    fields and rebuilds instances positionally from ``_fields()``, which must
    return the fields in declaration order.
    """

    event_ndims: ClassVar[tuple[int, ...]]

    def _fields(self) -> tuple[Array, ...]:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        """Event dimension of the factor."""
        raise NotImplementedError

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return lax.broadcast_shapes(
            *(x.shape[: x.ndim - k] for x, k in zip(self._fields(), self.event_ndims))
        )

    def _map(self, fn: Callable[[Array, int], Array]) -> Self:
        return type(self)(*(fn(x, k) for x, k in zip(self._fields(), self.event_ndims)))

    def _broadcast(self) -> Self:
        return self.expand(self.batch_shape)

    def expand(self, batch_shape: Sequence[int]) -> Self:
        """Broadcast every field to ``batch_shape``."""
        return self._map(lambda x, k: _with_batch(x, k, tuple(batch_shape)))

    def reshape(self, batch_shape: Sequence[int]) -> Self:
        """Reshape the batch dimensions of every field to ``batch_shape``."""
        return self._broadcast()._map(
            lambda x, k: x.reshape(tuple(batch_shape) + x.shape[x.ndim - k :])
        )

    def __getitem__(self, index: Union[int, slice, tuple]) -> Self:
        index = index if isinstance(index, tuple) else (index,)
        return self._broadcast()._map(lambda x, k: x[index + (slice(None),) * k])

    @classmethod
    def cat(cls, parts: Sequence[Self], axis: int = 0) -> Self:
        """Concatenate factors along a batch axis."""
        if not parts:
            raise ValueError("cat requires at least one factor")
        parts = [p._broadcast() for p in parts]
        if not parts[0].batch_shape:
            raise ValueError("cannot concatenate factors without batch dimensions")
        axis = axis % len(parts[0].batch_shape)
        return cls(
            *(
                jnp.concatenate(fields, axis)
                for fields in zip(*(p._fields() for p in parts))
            )
        )


F = TypeVar("F", bound=_FactorShapeOps)


def _schur_marginalize(
    precision: Array, info_vec: Array, left: int, right: int
) -> tuple[Array, Array, Array, Array]:
    """
    Schur complement of the ``left`` leading and ``right`` trailing
    coordinates, shared by the ``marginalize`` methods.

    One-sided calls index with static slices; only the two-sided case gathers
    with index arrays.

    :return: the reduced precision, the reduced information vector,
        ``b_tmp = L^-1 info_vec[drop]`` and ``sum(log(diag(L)))`` with ``L``
        the Cholesky factor of the dropped block, which must be positive
        definite.
    :rtype: tuple[Array, Array, Array, Array]
    """
    n = precision.shape[-1]
    if left and right:
        keep = np.arange(left, n - right)
        drop = np.concatenate([np.arange(left), np.arange(n - right, n)])
        P_aa = precision[..., keep[:, None], keep]
        P_ba = precision[..., drop[:, None], keep]
        P_bb = precision[..., drop[:, None], drop]
    else:
        keep = slice(left, n - right)
        drop = slice(0, left) if left else slice(n - right, n)
        P_aa = precision[..., keep, keep]
        P_ba = precision[..., drop, keep]
        P_bb = precision[..., drop, drop]
    chol = safe_cholesky(P_bb)
    P_a = solve_triangular(chol, P_ba, lower=True)
    b_tmp = solve_triangular(chol, info_vec[..., drop][..., None], lower=True)[..., 0]
    return (
        P_aa - _mt(P_a) @ P_a,
        info_vec[..., keep] - _mv(_mt(P_a), b_tmp),
        b_tmp,
        _log_diag_sum(chol),
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class Gaussian(_FactorShapeOps):
    """
    Unnormalized log-quadratic factor
    ``log_normalizer + x . info_vec - 0.5 x^T precision x``.

    :param Array log_normalizer: shape ``batch_shape``.
    :param Array info_vec: shape ``batch_shape + (dim,)``.
    :param Array precision: shape ``batch_shape + (dim, dim)``; may be rank
        deficient.
    :raises ValueError: if the trailing shapes of ``info_vec`` and
        ``precision`` disagree.
    """

    log_normalizer: Array
    info_vec: Array
    precision: Array

    event_ndims: ClassVar[tuple[int, int, int]] = (0, 1, 2)

    def __post_init__(self) -> None:
        if not (hasattr(self.info_vec, "shape") and hasattr(self.precision, "shape")):
            return
        if len(self.info_vec.shape) < 1 or len(self.precision.shape) < 2:
            raise ValueError("info_vec must have rank >= 1 and precision rank >= 2")
        dim = self.info_vec.shape[-1]
        if self.precision.shape[-2:] != (dim, dim):
            raise ValueError(
                f"precision must have trailing shape {(dim, dim)}, "
                f"got {self.precision.shape[-2:]}"
            )

    @property
    def dim(self) -> int:
        return self.info_vec.shape[-1]

    def _fields(self) -> tuple[Array, ...]:
        return (self.log_normalizer, self.info_vec, self.precision)

    def event_pad(self, left: int = 0, right: int = 0) -> Gaussian:
        """Embed the factor into a larger event space with zero coupling."""
        return Gaussian(
            self.log_normalizer,
            _pad_event(self.info_vec, 1, left, right),
            _pad_event(self.precision, 2, left, right),
        )

    def event_permute(self, perm: Union[Array, np.ndarray]) -> Gaussian:
        """
        Permute event coordinates.

        A static ``numpy`` permutation lowers to slices and concatenations;
        a traced permutation gathers.
        """
        if self.dim == 0:
            return self
        if isinstance(perm, np.ndarray):
            runs = _static_runs(perm)
            info_vec = jnp.concatenate([self.info_vec[..., s] for s in runs], -1)
            precision = jnp.concatenate([self.precision[..., s, :] for s in runs], -2)
            precision = jnp.concatenate([precision[..., :, s] for s in runs], -1)
        else:
            info_vec = self.info_vec[..., perm]
            precision = self.precision[..., perm, :][..., :, perm]
        return Gaussian(self.log_normalizer, info_vec, precision)

    def __add__(self, other: Union[Gaussian, AffineNormal, Array, float]) -> Gaussian:
        if isinstance(other, AffineNormal):
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

    def __sub__(self, other: Union[Gaussian, AffineNormal, Array, float]) -> Gaussian:
        """
        Subtract a scalar or a factor; the difference of two factors is a factor
        that is not necessarily normalizable.
        """
        if isinstance(other, AffineNormal):
            other = other.to_gaussian()
        if isinstance(other, Gaussian):
            return Gaussian(
                self.log_normalizer - other.log_normalizer,
                self.info_vec - other.info_vec,
                self.precision - other.precision,
            )._broadcast()
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

    def condition(self, value: Array) -> Gaussian:
        """
        Condition on the trailing block of coordinates.

        :param Array value: shape ``(..., right)`` with ``right <= dim``;
            leading dimensions broadcast against ``batch_shape``.
        :return: factor over the leading ``dim - right`` coordinates with the
            conditioned density folded into ``log_normalizer``, so
            ``g.log_density(concat([a, b])) == g.condition(b).log_density(a)``.
        :rtype: Gaussian
        """
        if value.shape[-1] > self.dim:
            raise ValueError(
                f"value may condition at most {self.dim} coordinates, "
                f"got {value.shape[-1]}"
            )
        n = self.dim - value.shape[-1]
        info_a, info_b = self.info_vec[..., :n], self.info_vec[..., n:]
        P_aa = self.precision[..., :n, :n]
        P_ab = self.precision[..., :n, n:]
        P_bb = self.precision[..., n:, n:]
        log_normalizer = (
            self.log_normalizer
            - 0.5 * (value * _mv(P_bb, value)).sum(-1)
            + (value * info_b).sum(-1)
        )
        return Gaussian(log_normalizer, info_a - _mv(P_ab, value), P_aa)._broadcast()

    def left_condition(self, value: Array) -> Gaussian:
        """
        Condition on the leading block of coordinates.

        :param Array value: shape ``(..., left)`` with ``left <= dim``;
            leading dimensions broadcast against ``batch_shape``.
        :return: factor over the trailing ``dim - left`` coordinates (see
            :meth:`condition`).
        :rtype: Gaussian
        """
        n = value.shape[-1]
        perm = np.concatenate([np.arange(n, self.dim), np.arange(n)])
        return self.event_permute(perm).condition(value)

    def marginalize(self, left: int = 0, right: int = 0) -> Gaussian:
        """
        Integrate out ``left`` leading and ``right`` trailing coordinates.

        One-sided calls index with static slices; only the two-sided case
        gathers with index arrays.

        :param int left: number of leading coordinates to integrate out.
        :param int right: number of trailing coordinates to integrate out.
        :return: factor over the remaining coordinates with ``event_logsumexp``
            preserved. The integrated block must have positive-definite
            precision.
        :rtype: Gaussian
        """
        if left == 0 and right == 0:
            return self
        precision, info_vec, b_tmp, log_diag = _schur_marginalize(
            self.precision, self.info_vec, left, right
        )
        log_normalizer = (
            self.log_normalizer
            + 0.5 * (left + right) * _LOG_2PI
            - log_diag
            + 0.5 * (b_tmp * b_tmp).sum(-1)
        )
        return Gaussian(log_normalizer, info_vec, precision)._broadcast()

    def event_logsumexp(self) -> Array:
        """Integrate the factor over all coordinates; requires positive-definite precision."""
        chol = safe_cholesky(self.precision)
        u = solve_triangular(chol, self.info_vec[..., None], lower=True)[..., 0]
        return (
            self.log_normalizer
            + 0.5 * self.dim * _LOG_2PI
            + 0.5 * (u * u).sum(-1)
            - _log_diag_sum(chol)
        )

    def sample(
        self,
        key: Optional[Array] = None,
        sample_shape: tuple[int, ...] = (),
        noise: Optional[Array] = None,
    ) -> Array:
        """
        Draw from the normalized Gaussian ``N(precision^-1 info_vec, precision^-1)``.

        :param Optional[Array] key: PRNG key; required when ``noise`` is
            ``None``.
        :param tuple sample_shape: leading sample dimensions.
        :param Optional[Array] noise: standard normal draws of shape
            ``sample_shape + batch_shape + (dim,)``; ``zeros`` yields the mean.
        :return: draws of shape ``sample_shape + batch_shape + (dim,)``.
        :rtype: Array
        :raises ValueError: if neither ``key`` nor ``noise`` is given, or if
            ``noise`` has the wrong shape.
        """
        shape = tuple(sample_shape) + self.batch_shape + (self.dim,)
        noise = _noise(noise, key, shape, self.precision.dtype)
        chol = safe_cholesky(self.precision)
        loc = cho_solve((chol, True), self.info_vec[..., None])[..., 0]

        def draw(eps: Array) -> Array:
            return (
                loc
                + solve_triangular(chol, eps[..., None], lower=True, trans=1)[..., 0]
            )

        for _ in sample_shape:
            draw = jax.vmap(draw)
        return draw(noise)


def _diag_normal_params(d: Distribution) -> Optional[tuple[Array, Array]]:
    """Return ``(loc, scale)`` of an ``Independent(Normal, 1)``, else ``None``."""
    shape = d.batch_shape + d.event_shape
    if isinstance(d, ExpandedDistribution):
        d = d.base_dist
    if not isinstance(d, Independent) or d.reinterpreted_batch_ndims != 1:
        return None
    base = d.base_dist
    base = base.base_dist if isinstance(base, ExpandedDistribution) else base
    if not isinstance(base, Normal):
        return None
    return jnp.broadcast_to(base.loc, shape), jnp.broadcast_to(base.scale, shape)


def _type_name(d: Distribution) -> str:
    """Name ``d`` with its wrapped bases, e.g. ``Independent(StudentT)``."""
    base = getattr(d, "base_dist", None)
    name = type(d).__name__
    return name if base is None else f"{name}({_type_name(base)})"


def _mvn_params(d: Distribution) -> tuple[Array, Array]:
    base = d.base_dist if isinstance(d, ExpandedDistribution) else d
    if not isinstance(base, MultivariateNormal):
        raise TypeError(
            "expected MultivariateNormal or Independent(Normal, 1), "
            f"got {_type_name(d)}"
        )
    shape = d.batch_shape + d.event_shape
    return (
        jnp.broadcast_to(base.loc, shape),
        jnp.broadcast_to(base.scale_tril, shape + shape[-1:]),
    )


def mvn_moments(d: Distribution) -> tuple[Array, Array]:
    """
    Mean and covariance of a Gaussian distribution, broadcast to its full shape.

    :param Distribution d: ``MultivariateNormal`` or ``Independent(Normal, 1)``,
        possibly wrapped in ``ExpandedDistribution``.
    :return: ``loc`` of shape ``d.batch_shape + d.event_shape`` and
        ``covariance`` of shape ``d.batch_shape + d.event_shape * 2``.
    :rtype: tuple[Array, Array]
    :raises TypeError: if ``d`` is not a supported Gaussian distribution.
    """
    diag = _diag_normal_params(d)
    if diag is not None:
        loc, scale = diag
        return loc, jnp.eye(loc.shape[-1], dtype=loc.dtype) * (scale**2)[..., None]
    loc, scale_tril = _mvn_params(d)
    return loc, scale_tril @ _mt(scale_tril)


def _sqrt_form(scale_tril: Array, loc: Array, matrix: Array) -> Gaussian:
    """
    Factor with ``precision = R^T R``, ``R = L^-1 matrix``, from noise
    ``N(loc, L L^T)`` with ``L = scale_tril``.
    """
    R = solve_triangular(scale_tril, matrix, lower=True)
    v = solve_triangular(scale_tril, loc[..., None], lower=True)[..., 0]
    log_normalizer = (
        -0.5 * loc.shape[-1] * _LOG_2PI
        - 0.5 * (v * v).sum(-1)
        - _log_diag_sum(scale_tril)
    )
    return Gaussian(log_normalizer, _mv(_mt(R), v), _mt(R) @ R)


def mvn_to_gaussian(d: Distribution) -> Gaussian:
    """
    Convert a Gaussian distribution to a normalized :class:`Gaussian` factor.

    ``Independent(Normal, 1)`` inputs take an elementwise path with no
    triangular solves.

    :param Distribution d: ``MultivariateNormal`` or ``Independent(Normal, 1)``,
        possibly wrapped in ``ExpandedDistribution``.
    :return: factor with ``batch_shape == d.batch_shape`` whose ``log_density``
        equals ``d.log_prob``.
    :rtype: Gaussian
    """
    diag = _diag_normal_params(d)
    if diag is not None:
        loc, scale = diag
        inv_var = scale**-2
        v = loc / scale
        log_normalizer = (
            -0.5 * loc.shape[-1] * _LOG_2PI
            - 0.5 * (v * v).sum(-1)
            - jnp.log(scale).sum(-1)
        )
        eye = jnp.eye(loc.shape[-1], dtype=loc.dtype)
        return Gaussian(log_normalizer, loc * inv_var, eye * inv_var[..., None])
    loc, scale_tril = _mvn_params(d)
    eye = jnp.broadcast_to(jnp.eye(loc.shape[-1], dtype=loc.dtype), scale_tril.shape)
    return _sqrt_form(scale_tril, loc, eye)


def matrix_and_gaussian_to_gaussian(matrix: Array, y_gaussian: Gaussian) -> Gaussian:
    """
    Joint factor over ``(x, y)`` for ``y - matrix @ x ~ y_gaussian``.

    :param Array matrix: shape ``(..., y_dim, x_dim)``.
    :param Gaussian y_gaussian: factor over ``y`` with ``dim == y_dim``.
    :return: factor over ``(x, y)`` with the broadcast batch shape of
        ``matrix`` and ``y_gaussian``.
    :rtype: Gaussian
    """
    batch_shape = lax.broadcast_shapes(matrix.shape[:-2], y_gaussian.batch_shape)
    matrix = _with_batch(matrix, 2, batch_shape)
    y_gaussian = y_gaussian.expand(batch_shape)
    P_yy = y_gaussian.precision
    P_xy = -_mt(matrix) @ P_yy
    P_xx = -P_xy @ matrix
    precision = jnp.concatenate(
        [jnp.concatenate([P_xx, P_xy], -1), jnp.concatenate([_mt(P_xy), P_yy], -1)],
        -2,
    )
    info_vec = jnp.concatenate(
        [-_mv(_mt(matrix), y_gaussian.info_vec), y_gaussian.info_vec], -1
    )
    return Gaussian(y_gaussian.log_normalizer, info_vec, precision)


def matrix_and_mvn_to_gaussian(
    matrix: Array, d: Distribution
) -> Union[Gaussian, AffineNormal]:
    """
    Factor over ``(x, y)`` for ``y = matrix @ x + noise`` with ``noise ~ d``.

    :param Array matrix: shape ``(..., y_dim, x_dim)``.
    :param Distribution d: noise distribution with ``event_shape == (y_dim,)``;
        ``MultivariateNormal`` or ``Independent(Normal, 1)``, possibly wrapped
        in ``ExpandedDistribution``.
    :return: :class:`AffineNormal` for diagonal-normal noise, otherwise a
        :class:`Gaussian` in square-root form.
    :rtype: Union[Gaussian, AffineNormal]
    :raises ValueError: if ``d.event_shape`` does not match the rows of
        ``matrix``.
    :raises TypeError: if ``d`` is not a supported Gaussian distribution.
    """
    y_dim, x_dim = matrix.shape[-2:]
    if d.event_shape != (y_dim,):
        raise ValueError(
            f"noise event_shape {d.event_shape} does not match matrix rows {y_dim}"
        )
    batch_shape = lax.broadcast_shapes(matrix.shape[:-2], d.batch_shape)
    matrix = jnp.broadcast_to(matrix, batch_shape + (y_dim, x_dim))
    diag = _diag_normal_params(d)
    if diag is not None:
        loc, scale = diag
        return AffineNormal(
            matrix,
            jnp.broadcast_to(loc, batch_shape + (y_dim,)),
            jnp.broadcast_to(scale, batch_shape + (y_dim,)),
        )
    loc, scale_tril = _mvn_params(d)
    eye = jnp.broadcast_to(
        jnp.eye(y_dim, dtype=matrix.dtype), batch_shape + (y_dim, y_dim)
    )
    return _sqrt_form(
        jnp.broadcast_to(scale_tril, batch_shape + (y_dim, y_dim)),
        jnp.broadcast_to(loc, batch_shape + (y_dim,)),
        jnp.concatenate([-matrix, eye], -1),
    )


def gaussian_tensordot(x: Gaussian, y: Gaussian, dims: int = 0) -> Gaussian:
    """
    Contract two factors over ``dims`` shared coordinates:
    ``(x @ y)(a, c) = log int exp(x(a, b) + y(b, c)) db``.

    :param Gaussian x: factor over ``(a, b)`` with ``b`` the trailing ``dims``
        coordinates.
    :param Gaussian y: factor over ``(b, c)`` with ``b`` the leading ``dims``
        coordinates.
    :param int dims: number of shared coordinates; the shared block must have
        positive-definite precision in ``x + y``.
    :return: factor over ``(a, c)`` with the broadcast batch shape of ``x``
        and ``y``.
    :rtype: Gaussian
    :raises ValueError: if ``dims`` is negative or exceeds the event dimension
        of a factor.
    """
    if dims < 0:
        raise ValueError(f"dims must be non-negative, got {dims}")
    na, nb, nc = x.dim - dims, dims, y.dim - dims
    if na < 0 or nc < 0:
        raise ValueError("dims exceeds the event dimension of a factor")
    Paa = x.precision[..., :na, :na]
    Pba = x.precision[..., na:, :na]
    Pbb = x.precision[..., na:, na:]
    Qbb = y.precision[..., :nb, :nb]
    Qbc = y.precision[..., :nb, nb:]
    Qcc = y.precision[..., nb:, nb:]
    xa, xb = x.info_vec[..., :na], x.info_vec[..., na:]
    yb, yc = y.info_vec[..., :nb], y.info_vec[..., nb:]
    precision = _pad_event(Paa, 2, 0, nc) + _pad_event(Qcc, 2, na, 0)
    info_vec = _pad_event(xa, 1, 0, nc) + _pad_event(yc, 1, na, 0)
    log_normalizer = x.log_normalizer + y.log_normalizer
    if nb > 0:
        B = _pad_event(Pba, 1, 0, nc) + _pad_event(Qbc, 1, na, 0)
        b = xb + yb
        chol = safe_cholesky(Pbb + Qbb)
        LinvB = solve_triangular(chol, B, lower=True)
        Linvb = solve_triangular(chol, b[..., None], lower=True)[..., 0]
        precision = precision - _mt(LinvB) @ LinvB
        info_vec = info_vec - _mv(_mt(LinvB), Linvb)
        log_normalizer = (
            log_normalizer
            + 0.5 * nb * _LOG_2PI
            + 0.5 * (Linvb * Linvb).sum(-1)
            - _log_diag_sum(chol)
        )
    return Gaussian(log_normalizer, info_vec, precision)._broadcast()


def _sequential_tensordot(factor: F, tensordot: Callable[[F, F, int], F]) -> F:
    """
    Reduce pairwise factors over the last batch axis with ``log2(T)`` batched
    contractions of ``tensordot`` over ``dim // 2`` shared coordinates.

    :raises ValueError: if the time axis is empty or the event dimension is odd.
    """
    if factor.batch_shape[-1] == 0:
        raise ValueError("cannot reduce over an empty time axis")
    if factor.dim % 2:
        raise ValueError(
            f"pairwise factors need an even event dimension, got {factor.dim}"
        )
    state_dim = factor.dim // 2
    while factor.batch_shape[-1] > 1:
        num_steps = factor.batch_shape[-1]
        even = num_steps // 2 * 2
        contracted = tensordot(factor[..., 0:even:2], factor[..., 1:even:2], state_dim)
        if num_steps > even:
            contracted = type(factor).cat([contracted, factor[..., -1:]], axis=-1)
        factor = contracted
    return factor[..., 0]


def sequential_gaussian_tensordot(gaussian: Gaussian) -> Gaussian:
    """
    Reduce a time series of pairwise factors to one factor over ``(z_0, z_T)``.

    :param Gaussian gaussian: factors over ``(z_{t-1}, z_t)`` with
        ``dim == 2 * state_dim`` and time on the last batch axis.
    :return: factor over ``(z_0, z_T)`` with batch shape
        ``gaussian.batch_shape[:-1]``, computed with ``log2(T)`` batched
        contractions.
    :rtype: Gaussian
    :raises ValueError: if the time axis is empty or the event dimension is odd.
    """
    return _sequential_tensordot(gaussian, gaussian_tensordot)


def sequential_gaussian_filter_sample(
    key: Optional[Array],
    init: Gaussian,
    trans: Gaussian,
    sample_shape: tuple[int, ...] = (),
    noise: Optional[Array] = None,
) -> Array:
    """
    Sample state paths from a chain of pairwise factors with ``O(log T)`` parallel depth.

    :param Optional[Array] key: PRNG key; required when ``noise`` is ``None``.
    :param Gaussian init: factor over ``z_0``.
    :param Gaussian trans: factors over ``(z_{t-1}, z_t)`` with time on the
        last batch axis (``T`` steps).
    :param tuple sample_shape: leading sample dimensions.
    :param Optional[Array] noise: standard normal draws of shape
        ``sample_shape + batch_shape + (T + 1, state_dim)``. ``zeros`` yields
        the posterior mean and ``[n, 0, -n]`` an antithetic triple;
        ``sample(key)`` equals ``sample(noise=random.normal(key, ...))``.
        Fresh draws use the promoted dtype of ``init`` and ``trans``.
    :return: state paths of shape
        ``sample_shape + batch_shape + (T + 1, state_dim)`` including ``z_0``.
    :rtype: Array
    :raises ValueError: if ``trans.dim != 2 * init.dim``, if neither ``key``
        nor ``noise`` is given, or if ``noise`` has the wrong shape.
    """
    state_dim = init.dim
    if trans.dim != 2 * state_dim:
        raise ValueError(
            f"trans.dim must equal 2 * init.dim = {2 * state_dim}, got {trans.dim}"
        )
    num_steps = trans.batch_shape[-1]
    batch_shape = lax.broadcast_shapes(trans.batch_shape[:-1], init.batch_shape)
    trans = trans.expand(batch_shape + (num_steps,))
    perm = np.concatenate(
        [
            np.arange(state_dim, 2 * state_dim),
            np.arange(state_dim),
            np.arange(2 * state_dim, 3 * state_dim),
        ]
    )

    tape = []
    gaussian = trans
    while gaussian.batch_shape[-1] > 1:
        time = gaussian.batch_shape[-1]
        even = time // 2 * 2
        x = gaussian[..., 0:even:2].event_pad(right=state_dim)
        y = gaussian[..., 1:even:2].event_pad(left=state_dim)
        joint = (x + y).event_permute(perm)
        tape.append(joint)
        contracted = joint.marginalize(left=state_dim)
        if time > even:
            contracted = Gaussian.cat([contracted, gaussian[..., -1:]], axis=-1)
        gaussian = contracted
    final = gaussian[..., 0] + init.expand(batch_shape).event_pad(right=state_dim)

    shape = tuple(sample_shape) + batch_shape + (num_steps + 1, state_dim)
    noise = _noise(noise, key, shape, jnp.result_type(init.precision, trans.precision))

    def backward(eps: Array) -> Array:
        result = final.sample(
            noise=eps[..., :2, :].reshape(batch_shape + (2 * state_dim,))
        )
        result = result.reshape(batch_shape + (2, state_dim))
        position = 2
        for joint in reversed(tape):
            pairs = joint.batch_shape[-1]
            cond = jnp.concatenate(
                [result[..., :pairs, :], result[..., 1 : pairs + 1, :]], -1
            )
            sample = joint.condition(cond).sample(
                noise=eps[..., position : position + pairs, :]
            )
            position += pairs
            head = jnp.stack([result[..., :pairs, :], sample], -2).reshape(
                batch_shape + (2 * pairs, state_dim)
            )
            result = jnp.concatenate([head, result[..., pairs:, :]], -2)
        return result

    for _ in sample_shape:
        backward = jax.vmap(backward)
    return backward(noise)


def loc_and_scale_tril(info_vec: Array, precision: Array) -> tuple[Array, Array]:
    """
    Moments of the normalized Gaussian with the given information parameters.

    :func:`~numpyro.distributions.util.cholesky_of_inverse` factorizes the
    reversed precision, so that is the ordering handed to
    :func:`~numpyro.distributions.util.jitter_if_singular`: a positive-definite
    precision is factorized exactly and one that is singular at rounding level
    receives a diagonal jitter before the single factorization. Probing the
    natural ordering instead would leave the factorized ordering unjittered
    whenever only the latter fails. The jitter is relative to the diagonal,
    which the reversal only permutes, so jittering before or after the reversal
    gives the same matrix.

    :param Array info_vec: shape ``(..., dim)``.
    :param Array precision: shape ``(..., dim, dim)``, positive definite.
    :return: ``loc = precision^-1 info_vec`` and the lower Cholesky factor of
        ``precision^-1``.
    :rtype: tuple[Array, Array]
    """
    reversed_precision = jitter_if_singular(precision[..., ::-1, ::-1])
    scale_tril = cholesky_of_inverse(reversed_precision[..., ::-1, ::-1])
    return _mv(scale_tril, _mv(_mt(scale_tril), info_vec)), scale_tril


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class AffineNormal(_FactorShapeOps):
    """
    Conditional ``y | x ~ Normal(matrix @ x + loc, scale)`` standing in for a
    joint factor over ``(x, y)``.

    :param Array matrix: shape ``batch_shape + (y_dim, x_dim)``.
    :param Array loc: shape ``batch_shape + (y_dim,)``.
    :param Array scale: shape ``batch_shape + (y_dim,)``.
    :raises ValueError: if the trailing shapes of the fields disagree.
    """

    matrix: Array
    loc: Array
    scale: Array

    event_ndims: ClassVar[tuple[int, int, int]] = (2, 1, 1)

    def __post_init__(self) -> None:
        fields = (self.matrix, self.loc, self.scale)
        if not all(hasattr(x, "shape") for x in fields):
            return
        if (
            len(self.matrix.shape) < 2
            or min(len(self.loc.shape), len(self.scale.shape)) < 1
        ):
            raise ValueError("matrix must have rank >= 2 and loc and scale rank >= 1")
        y_dim = self.matrix.shape[-2]
        if self.loc.shape[-1] != y_dim or self.scale.shape[-1] != y_dim:
            raise ValueError(
                f"loc and scale must have trailing dimension {y_dim}, "
                f"got {self.loc.shape[-1]} and {self.scale.shape[-1]}"
            )

    @property
    def dim(self) -> int:
        return self.matrix.shape[-1] + self.matrix.shape[-2]

    def _fields(self) -> tuple[Array, ...]:
        return (self.matrix, self.loc, self.scale)

    def to_gaussian(self) -> Gaussian:
        """Promote to a full :class:`Gaussian` over ``(x, y)``."""
        noise = Independent(Normal(self.loc, self.scale), 1)
        return matrix_and_gaussian_to_gaussian(self.matrix, mvn_to_gaussian(noise))

    def condition(self, value: Array) -> Gaussian:
        """
        Condition on the trailing block of coordinates.

        Conditioning on all of ``y`` needs no Cholesky factorization; other
        block sizes go through :meth:`to_gaussian`.

        :param Array value: shape ``(..., right)`` with ``right <= dim``;
            leading dimensions broadcast against ``batch_shape``.
        :return: factor over the leading ``dim - right`` coordinates (see
            :meth:`Gaussian.condition`).
        :rtype: Gaussian
        """
        if value.shape[-1] != self.matrix.shape[-2]:
            return self.to_gaussian().condition(value)
        W = self.matrix / self.scale[..., :, None]
        delta = (value - self.loc) / self.scale
        log_normalizer = (
            -0.5 * value.shape[-1] * _LOG_2PI
            - 0.5 * (delta * delta).sum(-1)
            - jnp.log(self.scale).sum(-1)
        )
        return Gaussian(log_normalizer, _mv(_mt(W), delta), _mt(W) @ W)._broadcast()

    def left_condition(self, value: Array) -> Union[AffineNormal, Gaussian]:
        """
        Condition on the leading block of coordinates.

        :param Array value: shape ``(..., left)`` with ``left <= dim``;
            leading dimensions broadcast against ``batch_shape``.
        :return: for ``left == x_dim``, an :class:`AffineNormal` with no
            remaining inputs; otherwise a :class:`Gaussian` via
            :meth:`to_gaussian`.
        :rtype: Union[AffineNormal, Gaussian]
        """
        if value.shape[-1] != self.matrix.shape[-1]:
            return self.to_gaussian().left_condition(value)
        loc = _mv(self.matrix, value) + self.loc
        batch_shape = loc.shape[:-1]
        empty = jnp.zeros(batch_shape + (loc.shape[-1], 0), self.matrix.dtype)
        return AffineNormal(empty, loc, jnp.broadcast_to(self.scale, loc.shape))

    def sample(
        self,
        key: Optional[Array] = None,
        sample_shape: tuple[int, ...] = (),
        noise: Optional[Array] = None,
    ) -> Array:
        """
        Draw ``y`` once all inputs are conditioned away (``x_dim == 0``).

        :param Optional[Array] key: PRNG key; required when ``noise`` is
            ``None``.
        :param tuple sample_shape: leading sample dimensions.
        :param Optional[Array] noise: standard normal draws of shape
            ``sample_shape + batch_shape + (y_dim,)``.
        :return: draws of shape ``sample_shape + batch_shape + (y_dim,)``.
        :rtype: Array
        :raises NotImplementedError: if ``x_dim != 0``.
        :raises ValueError: if neither ``key`` nor ``noise`` is given, or if
            ``noise`` has the wrong shape.
        """
        if self.matrix.shape[-1] != 0:
            raise NotImplementedError(
                "AffineNormal.sample requires all inputs to be conditioned"
            )
        shape = tuple(sample_shape) + self.loc.shape
        return self.loc + _noise(noise, key, shape, self.loc.dtype) * self.scale

    def marginalize(self, left: int = 0, right: int = 0) -> Gaussian:
        """
        Integrate out coordinates; integrating out all of ``y`` yields an exact
        zero factor over ``x``.
        """
        x_dim = self.matrix.shape[-1]
        if left == 0 and right == self.matrix.shape[-2]:
            batch_shape = self.batch_shape
            return Gaussian(
                jnp.zeros(batch_shape, self.loc.dtype),
                jnp.zeros(batch_shape + (x_dim,), self.loc.dtype),
                jnp.zeros(batch_shape + (x_dim, x_dim), self.loc.dtype),
            )
        return self.to_gaussian().marginalize(left, right)

    def event_pad(self, left: int = 0, right: int = 0) -> Gaussian:
        """Embed the factor into a larger event space with zero coupling."""
        return self.to_gaussian().event_pad(left, right)

    def event_permute(self, perm: Union[Array, np.ndarray]) -> Gaussian:
        """Permute event coordinates (see :meth:`Gaussian.event_permute`)."""
        return self.to_gaussian().event_permute(perm)

    def log_density(self, value: Array) -> Array:
        """Evaluate the factor at ``value`` of shape ``(..., x_dim + y_dim)``."""
        return self.to_gaussian().log_density(value)

    def __add__(self, other: Union[Gaussian, AffineNormal, Array, float]) -> Gaussian:
        return self.to_gaussian() + other
