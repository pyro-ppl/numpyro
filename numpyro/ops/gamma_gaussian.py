# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Gamma-Gaussian factors: Gaussian factors whose precision is scaled by a shared
Gamma-distributed variable.

A :class:`GammaGaussian` represents
``log p(x, s) = log_normalizer + alpha log s + s (x . info_vec - 0.5 x^T precision x - beta)``.
Fixing ``s`` gives a :class:`~numpyro.ops.gaussian.Gaussian`; integrating ``x``
gives a :class:`GammaFactor` over ``s``; integrating both gives a multivariate
Student-t. The same pairwise reduction as :mod:`numpyro.ops.gaussian` applies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar, Sequence, Union

import jax
from jax import Array, lax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import gammaln

from numpyro.distributions.continuous import Gamma, MultivariateStudentT
from numpyro.distributions.distribution import Distribution, ExpandedDistribution
from numpyro.distributions.util import safe_cholesky
from numpyro.ops.gaussian import (
    _LOG_2PI,
    Gaussian,
    _mt,
    _mv,
    _pad_event,
    _with_batch,
    loc_and_scale_tril,
    matrix_and_mvn_to_gaussian,
    mvn_to_gaussian,
)

__all__ = [
    "GammaFactor",
    "GammaGaussian",
    "gamma_and_mvn_to_gamma_gaussian",
    "gamma_gaussian_tensordot",
    "matrix_and_mvn_to_gamma_gaussian",
    "sequential_gamma_gaussian_tensordot",
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GammaFactor:
    """
    Unnormalized Gamma log-density
    ``log_normalizer + (concentration - 1) log s - rate s``.

    Parameters
    ----------
    log_normalizer, concentration, rate : Array
        Shape ``batch_shape``.
    """

    log_normalizer: Array
    concentration: Array
    rate: Array

    def log_density(self, s: Array) -> Array:
        """Evaluate the factor at scale ``s``."""
        return (
            self.log_normalizer + (self.concentration - 1) * jnp.log(s) - self.rate * s
        )

    def logsumexp(self) -> Array:
        """Integrate the factor over ``s > 0``."""
        return (
            self.log_normalizer
            + gammaln(self.concentration)
            - self.concentration * jnp.log(self.rate)
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GammaGaussian:
    """
    Factor ``log_normalizer + alpha log s + s (x . info_vec - 0.5 x^T precision x - beta)``
    over ``(x, s)``.

    Parameters
    ----------
    log_normalizer, alpha, beta : Array
        Shape ``batch_shape``.
    info_vec : Array
        Shape ``batch_shape + (dim,)``.
    precision : Array
        Shape ``batch_shape + (dim, dim)``.
    """

    log_normalizer: Array
    info_vec: Array
    precision: Array
    alpha: Array
    beta: Array

    event_ndims: ClassVar[tuple[int, ...]] = (0, 1, 2, 0, 0)

    @property
    def dim(self) -> int:
        return self.info_vec.shape[-1]

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return lax.broadcast_shapes(
            self.log_normalizer.shape,
            self.info_vec.shape[:-1],
            self.precision.shape[:-2],
            self.alpha.shape,
            self.beta.shape,
        )

    def _fields(self) -> tuple[Array, ...]:
        return (
            self.log_normalizer,
            self.info_vec,
            self.precision,
            self.alpha,
            self.beta,
        )

    def _map(self, fn: Callable[[Array, int], Array]) -> GammaGaussian:
        return GammaGaussian(
            *(fn(x, k) for x, k in zip(self._fields(), self.event_ndims))
        )

    def _broadcast(self) -> GammaGaussian:
        return self.expand(self.batch_shape)

    def expand(self, batch_shape: Sequence[int]) -> GammaGaussian:
        """Broadcast every field to ``batch_shape``."""
        return self._map(lambda x, k: _with_batch(x, k, tuple(batch_shape)))

    def reshape(self, batch_shape: Sequence[int]) -> GammaGaussian:
        """
        Reshape the batch dimensions of every field to ``batch_shape``.

        All fields must share one batch shape (the module invariant).
        """
        return self._map(
            lambda x, k: x.reshape(tuple(batch_shape) + x.shape[x.ndim - k :])
        )

    def __getitem__(self, index: Union[int, slice, tuple]) -> GammaGaussian:
        index = index if isinstance(index, tuple) else (index,)
        return self._map(lambda x, k: x[index + (slice(None),) * k])

    @staticmethod
    def cat(parts: Sequence[GammaGaussian], axis: int = 0) -> GammaGaussian:
        """
        Concatenate factors along a batch axis.

        All fields of every part must share one batch shape (the module
        invariant).
        """
        axis = axis % len(parts[0].batch_shape)
        return GammaGaussian(
            *(jnp.concatenate([p._fields()[i] for p in parts], axis) for i in range(5))
        )

    def event_pad(self, left: int = 0, right: int = 0) -> GammaGaussian:
        """Embed the factor into a larger event space with zero coupling."""
        return GammaGaussian(
            self.log_normalizer,
            _pad_event(self.info_vec, 1, left, right),
            _pad_event(self.precision, 2, left, right),
            self.alpha,
            self.beta,
        )

    def event_permute(self, perm: Array) -> GammaGaussian:
        """Permute event coordinates."""
        return GammaGaussian(
            self.log_normalizer,
            self.info_vec[..., perm],
            self.precision[..., perm, :][..., :, perm],
            self.alpha,
            self.beta,
        )

    def __add__(self, other: GammaGaussian) -> GammaGaussian:
        if not isinstance(other, GammaGaussian):
            raise TypeError(f"cannot add {type(other).__name__} to GammaGaussian")
        return GammaGaussian(
            *(a + b for a, b in zip(self._fields(), other._fields()))
        )._broadcast()

    def log_density(self, value: Array, s: Array) -> Array:
        """Evaluate the factor at ``value`` of shape ``(..., dim)`` and scale ``s``."""
        scale_term = self.alpha * jnp.log(s) - self.beta * s
        if self.dim == 0:
            return scale_term + self.log_normalizer
        quadratic = (value * (-0.5 * _mv(self.precision, value) + self.info_vec)).sum(
            -1
        )
        return self.log_normalizer + scale_term + s * quadratic

    def condition(self, value: Array) -> GammaGaussian:
        """
        Condition on the trailing block of coordinates.

        Parameters
        ----------
        value : Array
            Shape ``batch_shape + (right,)`` with ``right <= dim``.

        Returns
        -------
        GammaGaussian
            Factor over the leading ``dim - right`` coordinates with the
            conditioned quadratic folded into ``beta``, so
            ``g.log_density(concat([a, b]), s) == g.condition(b).log_density(a, s)``.
        """
        n = self.dim - value.shape[-1]
        info_a, info_b = self.info_vec[..., :n], self.info_vec[..., n:]
        P_aa = self.precision[..., :n, :n]
        P_ab = self.precision[..., :n, n:]
        P_bb = self.precision[..., n:, n:]
        beta = (
            self.beta
            + 0.5 * (value * _mv(P_bb, value)).sum(-1)
            - (value * info_b).sum(-1)
        )
        return GammaGaussian(
            self.log_normalizer,
            info_a - _mv(P_ab, value),
            P_aa,
            self.alpha,
            beta,
        )._broadcast()

    def marginalize(self, left: int = 0, right: int = 0) -> GammaGaussian:
        """
        Integrate out ``left`` leading and ``right`` trailing coordinates.

        Returns
        -------
        GammaGaussian
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
        n_b = left + right
        log_normalizer = (
            self.log_normalizer
            + 0.5 * n_b * _LOG_2PI
            - jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)).sum(-1)
        )
        return GammaGaussian(
            log_normalizer,
            self.info_vec[..., keep] - _mv(_mt(P_a), b_tmp),
            P_aa - _mt(P_a) @ P_a,
            self.alpha - 0.5 * n_b,
            self.beta - 0.5 * (b_tmp * b_tmp).sum(-1),
        )._broadcast()

    def event_logsumexp(self) -> GammaFactor:
        """
        Integrate the factor over ``x``; requires positive-definite precision.

        Returns
        -------
        GammaFactor
            The remaining factor over ``s``.
        """
        chol = safe_cholesky(self.precision)
        u = solve_triangular(chol, self.info_vec[..., None], lower=True)[..., 0]
        return GammaFactor(
            self.log_normalizer
            + 0.5 * self.dim * _LOG_2PI
            - jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)).sum(-1),
            self.alpha - 0.5 * self.dim + 1,
            self.beta - 0.5 * (u * u).sum(-1),
        )

    def compound(self) -> MultivariateStudentT:
        """
        Marginal over ``s`` of the normalized joint.

        Returns
        -------
        MultivariateStudentT
            Student-t with ``2 (alpha - dim / 2 + 1)`` degrees of freedom.
        """
        concentration = self.alpha - 0.5 * self.dim + 1
        loc, scale_tril = loc_and_scale_tril(self.info_vec, self.precision)
        rate = self.beta - 0.5 * (self.info_vec * loc).sum(-1)
        scale = jnp.sqrt(rate / concentration)
        return MultivariateStudentT(
            2 * concentration, loc, scale_tril * scale[..., None, None]
        )


def gamma_and_mvn_to_gamma_gaussian(
    gamma: Distribution, mvn: Distribution
) -> GammaGaussian:
    """
    Joint factor over ``(x, s)`` for ``s ~ gamma`` and
    ``x | s ~ MultivariateNormal(loc, precision = s * P)``.

    :param Distribution gamma: ``Gamma`` prior over the scale ``s``, possibly
        wrapped in ``ExpandedDistribution``.
    :param Distribution mvn: ``MultivariateNormal`` or ``Independent(Normal, 1)``
        with precision ``P`` and mean ``loc``, possibly wrapped in
        ``ExpandedDistribution``.
    :return: normalized factor whose ``log_density(x, s)`` equals
        ``gamma.log_prob(s) + MultivariateNormal(loc, precision=s * P).log_prob(x)``.
    :rtype: GammaGaussian
    :raises TypeError: if ``gamma`` is not a ``Gamma``.
    """
    base = gamma.base_dist if isinstance(gamma, ExpandedDistribution) else gamma
    if not isinstance(base, Gamma):
        raise TypeError(f"gamma must be a Gamma, got {type(gamma).__name__}")
    g = mvn_to_gaussian(mvn)
    loc = jnp.broadcast_to(mvn.mean, g.info_vec.shape)
    batch_shape = lax.broadcast_shapes(gamma.batch_shape, g.batch_shape)
    concentration = jnp.broadcast_to(base.concentration, batch_shape)
    rate = jnp.broadcast_to(base.rate, batch_shape)
    half_quadratic = 0.5 * (g.info_vec * loc).sum(-1)
    gaussian_logsumexp = -g.log_normalizer - half_quadratic
    log_normalizer = -GammaFactor(gaussian_logsumexp, concentration, rate).logsumexp()
    return GammaGaussian(
        log_normalizer,
        g.info_vec,
        g.precision,
        concentration + 0.5 * g.dim - 1,
        rate + half_quadratic,
    )._broadcast()


def matrix_and_mvn_to_gamma_gaussian(matrix: Array, mvn: Distribution) -> GammaGaussian:
    """
    Factor over ``(x, y, s)`` for ``y = matrix @ x + noise`` with
    ``noise ~ MultivariateNormal(loc, precision = s * P)``.

    Parameters
    ----------
    matrix : Array
        Shape ``(..., y_dim, x_dim)``.
    mvn : Distribution
        ``MultivariateNormal`` noise with ``event_shape == (y_dim,)``.

    Returns
    -------
    GammaGaussian
        Factor over ``concat([x, y])`` with ``alpha == y_dim / 2`` and the
        quadratic ``0.5 loc^T P loc`` tracked in ``beta``.

    Raises
    ------
    TypeError
        If ``mvn`` is a diagonal normal, for which
        :func:`~numpyro.ops.gaussian.matrix_and_mvn_to_gaussian` returns an
        :class:`~numpyro.ops.gaussian.AffineNormal`.
    """
    g = matrix_and_mvn_to_gaussian(matrix, mvn)
    if not isinstance(g, Gaussian):
        raise TypeError(
            "matrix_and_mvn_to_gamma_gaussian requires a MultivariateNormal noise "
            "distribution"
        )
    y_dim, x_dim = matrix.shape[-2:]
    loc = jnp.broadcast_to(mvn.mean, g.info_vec.shape[:-1] + (y_dim,))
    half_quadratic = 0.5 * (g.info_vec[..., x_dim:] * loc).sum(-1)
    alpha = jnp.full(g.batch_shape, 0.5 * y_dim, g.log_normalizer.dtype)
    return GammaGaussian(
        g.log_normalizer + half_quadratic,
        g.info_vec,
        g.precision,
        alpha,
        half_quadratic,
    )


def gamma_gaussian_tensordot(
    x: GammaGaussian, y: GammaGaussian, dims: int = 0
) -> GammaGaussian:
    """
    Contract two factors over ``dims`` shared coordinates:
    ``(x @ y)(a, c, s) = log int exp(x(a, b, s) + y(b, c, s)) db``.

    Parameters
    ----------
    x : GammaGaussian
        Factor over ``(a, b)`` with ``b`` the trailing ``dims`` coordinates.
    y : GammaGaussian
        Factor over ``(b, c)`` with ``b`` the leading ``dims`` coordinates.
    dims : int
        Number of shared coordinates.

    Returns
    -------
    GammaGaussian
        Factor over ``(a, c)`` with the broadcast batch shape of ``x`` and ``y``.
    """
    na, nb, nc = x.dim - dims, dims, y.dim - dims
    if na < 0 or nc < 0:
        raise ValueError("dims exceeds the event dimension of a factor")
    perm = jnp.concatenate(
        [jnp.arange(na), jnp.arange(x.dim, x.dim + nc), jnp.arange(na, x.dim)]
    )
    joint = x.event_pad(right=nc) + y.event_pad(left=na)
    return joint.event_permute(perm).marginalize(right=nb)


def sequential_gamma_gaussian_tensordot(gaussian: GammaGaussian) -> GammaGaussian:
    """
    Reduce pairwise factors over time to one factor over ``(z_0, z_T, s)``.

    Parameters
    ----------
    gaussian : GammaGaussian
        Batched factor whose trailing batch dimension indexes time and whose
        event dimension is ``2 * state_dim``.

    Returns
    -------
    GammaGaussian
        The contraction ``g[..., 0] @ g[..., 1] @ ... @ g[..., T - 1]`` over
        each intermediate state, computed in ``log2(T)`` batched steps.
    """
    state_dim = gaussian.dim // 2
    while gaussian.batch_shape[-1] > 1:
        num_steps = gaussian.batch_shape[-1]
        even = num_steps // 2 * 2
        contracted = gamma_gaussian_tensordot(
            gaussian[..., 0:even:2], gaussian[..., 1:even:2], state_dim
        )
        if num_steps > even:
            contracted = GammaGaussian.cat([contracted, gaussian[..., -1:]], axis=-1)
        gaussian = contracted
    return gaussian[..., 0]
