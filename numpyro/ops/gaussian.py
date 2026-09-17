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
from typing import ClassVar, Optional, Sequence, Union

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
    relative_jitter,
    safe_cholesky,
)

__all__ = [
    "AffineNormal",
    "Gaussian",
    "gaussian_tensordot",
    "loc_and_scale_tril",
    "matrix_and_gaussian_to_gaussian",
    "matrix_and_mvn_to_gaussian",
    "mvn_to_gaussian",
    "sequential_gaussian_filter_sample",
    "sequential_gaussian_tensordot",
]

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

    def condition(self, value: Array) -> Gaussian:
        """
        Condition on the trailing block of coordinates.

        Parameters
        ----------
        value : Array
            Shape ``batch_shape + (right,)`` with ``right <= dim``.

        Returns
        -------
        Gaussian
            Factor over the leading ``dim - right`` coordinates with the
            conditioned density folded into ``log_normalizer``, so
            ``g.log_density(concat([a, b])) == g.condition(b).log_density(a)``.
        """
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
        """Condition on the leading block of coordinates (see :meth:`condition`)."""
        n = value.shape[-1]
        perm = jnp.concatenate([jnp.arange(n, self.dim), jnp.arange(n)])
        return self.event_permute(perm).condition(value)

    def marginalize(self, left: int = 0, right: int = 0) -> Gaussian:
        """
        Integrate out ``left`` leading and ``right`` trailing coordinates.

        Returns
        -------
        Gaussian
            Factor over the remaining coordinates with ``event_logsumexp``
            preserved. The integrated block must have positive-definite
            precision.
        """
        if left == 0 and right == 0:
            return self
        n = self.dim
        keep = jnp.arange(left, n - right)
        drop = jnp.concatenate([jnp.arange(left), jnp.arange(n - right, n)])
        P_aa = self.precision[..., keep[:, None], keep]
        P_ba = self.precision[..., drop[:, None], keep]
        P_bb = self.precision[..., drop[:, None], drop]
        chol = safe_cholesky(P_bb)
        P_a = solve_triangular(chol, P_ba, lower=True)
        b_tmp = solve_triangular(chol, self.info_vec[..., drop, None], lower=True)[
            ..., 0
        ]
        log_normalizer = (
            self.log_normalizer
            + 0.5 * (left + right) * _LOG_2PI
            - jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)).sum(-1)
            + 0.5 * (b_tmp * b_tmp).sum(-1)
        )
        return Gaussian(
            log_normalizer,
            self.info_vec[..., keep] - _mv(_mt(P_a), b_tmp),
            P_aa - _mt(P_a) @ P_a,
        )._broadcast()

    def event_logsumexp(self) -> Array:
        """Integrate the factor over all coordinates; requires positive-definite precision."""
        chol = safe_cholesky(self.precision)
        u = solve_triangular(chol, self.info_vec[..., None], lower=True)[..., 0]
        return (
            self.log_normalizer
            + 0.5 * self.dim * _LOG_2PI
            + 0.5 * (u * u).sum(-1)
            - jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)).sum(-1)
        )

    def sample(
        self,
        key: Optional[Array] = None,
        sample_shape: tuple[int, ...] = (),
        noise: Optional[Array] = None,
    ) -> Array:
        """
        Draw from the normalized Gaussian ``N(precision^-1 info_vec, precision^-1)``.

        Parameters
        ----------
        key : Array, optional
            PRNG key; required when ``noise`` is ``None``.
        sample_shape : tuple[int, ...]
            Leading sample dimensions.
        noise : Array, optional
            Standard normal draws of shape ``sample_shape + batch_shape + (dim,)``;
            ``zeros`` yields the mean.

        Returns
        -------
        Array
            Shape ``sample_shape + batch_shape + (dim,)``.
        """
        shape = tuple(sample_shape) + self.batch_shape + (self.dim,)
        if noise is None:
            noise = random.normal(key, shape, self.precision.dtype)
        noise = noise.reshape(shape)
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


def _is_diag_normal(d: Distribution) -> bool:
    if not isinstance(d, Independent) or d.reinterpreted_batch_ndims != 1:
        return False
    base = d.base_dist
    base = base.base_dist if isinstance(base, ExpandedDistribution) else base
    return isinstance(base, Normal)


def _diag_normal_params(d: Distribution) -> tuple[Array, Array]:
    if not _is_diag_normal(d):
        raise TypeError(f"expected Independent(Normal, 1), got {type(d).__name__}")
    base = d.base_dist
    base = base.base_dist if isinstance(base, ExpandedDistribution) else base
    shape = d.batch_shape + d.event_shape
    return jnp.broadcast_to(base.loc, shape), jnp.broadcast_to(base.scale, shape)


def _mvn_params(d: Distribution) -> tuple[Array, Array]:
    base = d.base_dist if isinstance(d, ExpandedDistribution) else d
    if not isinstance(base, MultivariateNormal):
        raise TypeError(
            "expected MultivariateNormal or Independent(Normal, 1), "
            f"got {type(d).__name__}"
        )
    shape = d.batch_shape + d.event_shape
    return (
        jnp.broadcast_to(base.loc, shape),
        jnp.broadcast_to(base.scale_tril, shape + shape[-1:]),
    )


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
        - jnp.log(jnp.diagonal(scale_tril, axis1=-2, axis2=-1)).sum(-1)
    )
    return Gaussian(log_normalizer, _mv(_mt(R), v), _mt(R) @ R)


def mvn_to_gaussian(d: Distribution) -> Gaussian:
    """
    Convert a Gaussian distribution to a normalized :class:`Gaussian` factor.

    Parameters
    ----------
    d : Distribution
        ``MultivariateNormal`` or ``Independent(Normal, 1)``, possibly wrapped
        in ``ExpandedDistribution``.

    Returns
    -------
    Gaussian
        Factor with ``batch_shape == d.batch_shape`` whose ``log_density``
        equals ``d.log_prob``.
    """
    if _is_diag_normal(d):
        loc, scale = _diag_normal_params(d)
        scale_tril = jnp.eye(scale.shape[-1], dtype=scale.dtype) * scale[..., None]
    else:
        loc, scale_tril = _mvn_params(d)
    eye = jnp.broadcast_to(jnp.eye(loc.shape[-1], dtype=loc.dtype), scale_tril.shape)
    return _sqrt_form(scale_tril, loc, eye)


def matrix_and_gaussian_to_gaussian(matrix: Array, y_gaussian: Gaussian) -> Gaussian:
    """
    Joint factor over ``(x, y)`` for ``y - matrix @ x ~ y_gaussian``.

    Parameters
    ----------
    matrix : Array
        Shape ``(..., y_dim, x_dim)``.
    y_gaussian : Gaussian
        Factor over ``y`` with ``dim == y_dim``.
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

    Parameters
    ----------
    matrix : Array
        Shape ``(..., y_dim, x_dim)``.
    d : Distribution
        Noise distribution with ``event_shape == (y_dim,)``;
        ``MultivariateNormal`` or ``Independent(Normal, 1)``.

    Returns
    -------
    Gaussian or AffineNormal
        :class:`AffineNormal` for diagonal-normal noise, otherwise a
        :class:`Gaussian` in square-root form.
    """
    y_dim, x_dim = matrix.shape[-2:]
    if d.event_shape != (y_dim,):
        raise ValueError(
            f"noise event_shape {d.event_shape} does not match matrix rows {y_dim}"
        )
    batch_shape = lax.broadcast_shapes(matrix.shape[:-2], d.batch_shape)
    matrix = jnp.broadcast_to(matrix, batch_shape + (y_dim, x_dim))
    if _is_diag_normal(d):
        loc, scale = _diag_normal_params(d)
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

    Parameters
    ----------
    x : Gaussian
        Factor over ``(a, b)`` with ``b`` the trailing ``dims`` coordinates.
    y : Gaussian
        Factor over ``(b, c)`` with ``b`` the leading ``dims`` coordinates.
    dims : int
        Number of shared coordinates.

    Returns
    -------
    Gaussian
        Factor over ``(a, c)`` with the broadcast batch shape of ``x`` and ``y``.
    """
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
        B = jnp.pad(Pba, [(0, 0)] * (Pba.ndim - 1) + [(0, nc)]) + jnp.pad(
            Qbc, [(0, 0)] * (Qbc.ndim - 1) + [(na, 0)]
        )
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
            - jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)).sum(-1)
        )
    return Gaussian(log_normalizer, info_vec, precision)._broadcast()


def sequential_gaussian_tensordot(gaussian: Gaussian) -> Gaussian:
    """
    Reduce a time series of pairwise factors to one factor over ``(z_0, z_T)``.

    Parameters
    ----------
    gaussian : Gaussian
        Factors over ``(z_{t-1}, z_t)`` with ``dim == 2 * state_dim`` and time
        on the last batch axis.

    Returns
    -------
    Gaussian
        Factor over ``(z_0, z_T)`` with batch shape ``gaussian.batch_shape[:-1]``,
        computed with ``log2(T)`` batched contractions.
    """
    state_dim = gaussian.dim // 2
    while gaussian.batch_shape[-1] > 1:
        num_steps = gaussian.batch_shape[-1]
        even = num_steps // 2 * 2
        contracted = gaussian_tensordot(
            gaussian[..., 0:even:2], gaussian[..., 1:even:2], state_dim
        )
        if num_steps > even:
            contracted = Gaussian.cat([contracted, gaussian[..., -1:]], axis=-1)
        gaussian = contracted
    return gaussian[..., 0]


def sequential_gaussian_filter_sample(
    key: Optional[Array],
    init: Gaussian,
    trans: Gaussian,
    sample_shape: tuple[int, ...] = (),
    noise: Optional[Array] = None,
) -> Array:
    """
    Sample state paths from a chain of pairwise factors with ``O(log T)`` parallel depth.

    Parameters
    ----------
    key : Array, optional
        PRNG key; required when ``noise`` is ``None``.
    init : Gaussian
        Factor over ``z_0``.
    trans : Gaussian
        Factors over ``(z_{t-1}, z_t)`` with time on the last batch axis (``T`` steps).
    sample_shape : tuple[int, ...]
        Leading sample dimensions.
    noise : Array, optional
        Standard normal draws of shape ``sample_shape + batch_shape + (T + 1, state_dim)``.
        ``zeros`` yields the posterior mean and ``[n, 0, -n]`` an antithetic triple;
        ``sample(key)`` equals ``sample(noise=random.normal(key, ...))``.

    Returns
    -------
    Array
        Shape ``sample_shape + batch_shape + (T + 1, state_dim)`` including ``z_0``.
    """
    state_dim = init.dim
    num_steps = trans.batch_shape[-1]
    batch_shape = lax.broadcast_shapes(trans.batch_shape[:-1], init.batch_shape)
    trans = trans.expand(batch_shape + (num_steps,))
    perm = jnp.concatenate(
        [
            jnp.arange(state_dim, 2 * state_dim),
            jnp.arange(state_dim),
            jnp.arange(2 * state_dim, 3 * state_dim),
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
    if noise is None:
        noise = random.normal(key, shape, init.precision.dtype)
    noise = noise.reshape(shape)

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

    Parameters
    ----------
    info_vec : Array
        Shape ``(..., dim)``.
    precision : Array
        Shape ``(..., dim, dim)``, positive definite.

    Returns
    -------
    tuple[Array, Array]
        ``loc = precision^-1 info_vec`` and the lower Cholesky factor of
        ``precision^-1``, computed with one factorization of the jittered
        precision.
    """
    scale_tril = cholesky_of_inverse(relative_jitter(precision))
    return _mv(scale_tril, _mv(_mt(scale_tril), info_vec)), scale_tril


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AffineNormal:
    """
    Conditional ``y | x ~ Normal(matrix @ x + loc, scale)`` standing in for a
    joint factor over ``(x, y)``.

    Parameters
    ----------
    matrix : Array
        Shape ``batch_shape + (y_dim, x_dim)``.
    loc, scale : Array
        Shape ``batch_shape + (y_dim,)``.
    """

    matrix: Array
    loc: Array
    scale: Array

    @property
    def dim(self) -> int:
        return self.matrix.shape[-1] + self.matrix.shape[-2]

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.matrix.shape[:-2]

    def _map(self, fn) -> AffineNormal:
        return AffineNormal(fn(self.matrix, 2), fn(self.loc, 1), fn(self.scale, 1))

    def expand(self, batch_shape: Sequence[int]) -> AffineNormal:
        """Broadcast every field to ``batch_shape``."""
        return self._map(lambda x, k: _with_batch(x, k, tuple(batch_shape)))

    def reshape(self, batch_shape: Sequence[int]) -> AffineNormal:
        """Reshape the batch dimensions of every field to ``batch_shape``."""
        return self._map(
            lambda x, k: x.reshape(tuple(batch_shape) + x.shape[x.ndim - k :])
        )

    def __getitem__(self, index: Union[int, slice, tuple]) -> AffineNormal:
        index = index if isinstance(index, tuple) else (index,)
        return self._map(lambda x, k: x[index + (slice(None),) * k])

    def to_gaussian(self) -> Gaussian:
        """Promote to a full :class:`Gaussian` over ``(x, y)``."""
        noise = Independent(Normal(self.loc, self.scale), 1)
        return matrix_and_gaussian_to_gaussian(self.matrix, mvn_to_gaussian(noise))

    def condition(self, value: Array) -> Gaussian:
        """
        Condition on ``y`` (``value.shape[-1] == y_dim``) without a Cholesky
        factorization; other block sizes go through :meth:`to_gaussian`.
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
        Condition on ``x`` (``value.shape[-1] == x_dim``); returns an
        :class:`AffineNormal` with no remaining inputs.
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

        Parameters
        ----------
        key : Array, optional
            PRNG key; required when ``noise`` is ``None``.
        sample_shape : tuple[int, ...]
            Leading sample dimensions.
        noise : Array, optional
            Standard normal draws of shape ``sample_shape + batch_shape + (y_dim,)``.
        """
        if self.matrix.shape[-1] != 0:
            raise NotImplementedError(
                "AffineNormal.sample requires all inputs to be conditioned"
            )
        shape = tuple(sample_shape) + self.loc.shape
        if noise is None:
            noise = random.normal(key, shape, self.loc.dtype)
        return self.loc + noise.reshape(shape) * self.scale

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

    def event_permute(self, perm: Array) -> Gaussian:
        """Permute event coordinates."""
        return self.to_gaussian().event_permute(perm)

    def log_density(self, value: Array) -> Array:
        """Evaluate the factor at ``value`` of shape ``(..., x_dim + y_dim)``."""
        return self.to_gaussian().log_density(value)

    def __add__(self, other: Union[Gaussian, AffineNormal, Array, float]) -> Gaussian:
        return self.to_gaussian() + other
