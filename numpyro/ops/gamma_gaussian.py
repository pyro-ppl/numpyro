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
from typing import ClassVar, Union

import numpy as np

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
    AffineNormal,
    Gaussian,
    _FactorShapeOps,
    _log_diag_sum,
    _mv,
    _pad_event,
    _schur_marginalize,
    _sequential_tensordot,
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
@dataclass(frozen=True, eq=False)
class GammaFactor:
    """
    Unnormalized Gamma log-density
    ``log_normalizer + (concentration - 1) log s - rate s``.

    :param Array log_normalizer: shape ``batch_shape``.
    :param Array concentration: shape ``batch_shape``.
    :param Array rate: shape ``batch_shape``.
    :raises ValueError: if the field shapes do not broadcast against each other.
    """

    log_normalizer: Array
    concentration: Array
    rate: Array

    def __post_init__(self) -> None:
        fields = (self.log_normalizer, self.concentration, self.rate)
        if not all(hasattr(x, "shape") for x in fields):
            return
        shapes = tuple(x.shape for x in fields)
        try:
            jnp.broadcast_shapes(*shapes)
        except ValueError as e:
            raise ValueError(
                f"GammaFactor fields must broadcast, got shapes {shapes}"
            ) from e

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
@dataclass(frozen=True, eq=False)
class GammaGaussian(_FactorShapeOps):
    """
    Factor ``log_normalizer + alpha log s + s (x . info_vec - 0.5 x^T precision x - beta)``
    over ``(x, s)``.

    For fixed ``s`` the factor is a :class:`~numpyro.ops.gaussian.Gaussian`
    with information vector ``s info_vec`` and precision ``s precision``, that
    is ``p(x | s) = N(s info_vec, s precision)`` in information form. A
    normalized joint ``s ~ Gamma(concentration, rate)``,
    ``x | s ~ N(loc, precision = s P)`` has ``alpha = concentration + dim / 2 - 1``
    and ``beta = rate + 0.5 info_vec^T P^-1 info_vec``, so that integrating
    ``x`` leaves a :class:`GammaFactor` with the prior ``concentration`` and
    ``rate``.

    :param Array log_normalizer: shape ``batch_shape``.
    :param Array info_vec: shape ``batch_shape + (dim,)``.
    :param Array precision: shape ``batch_shape + (dim, dim)``.
    :param Array alpha: shape ``batch_shape``.
    :param Array beta: shape ``batch_shape``.
    :raises ValueError: if the trailing shapes of ``info_vec`` and
        ``precision`` disagree, or if ``alpha`` and ``beta`` do not broadcast
        with the batch shape.
    """

    log_normalizer: Array
    info_vec: Array
    precision: Array
    alpha: Array
    beta: Array

    event_ndims: ClassVar[tuple[int, ...]] = (0, 1, 2, 0, 0)

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
        if hasattr(self.alpha, "shape") and hasattr(self.beta, "shape"):
            try:
                jnp.broadcast_shapes(
                    self.alpha.shape, self.beta.shape, self.info_vec.shape[:-1]
                )
            except ValueError as e:
                raise ValueError(
                    "alpha and beta must broadcast with the batch shape, got "
                    f"{self.alpha.shape}, {self.beta.shape} and "
                    f"{self.info_vec.shape[:-1]}"
                ) from e

    @property
    def dim(self) -> int:
        return self.info_vec.shape[-1]

    def _fields(self) -> tuple[Array, ...]:
        return (
            self.log_normalizer,
            self.info_vec,
            self.precision,
            self.alpha,
            self.beta,
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

    def event_permute(self, perm: Union[Array, np.ndarray]) -> GammaGaussian:
        """
        Permute event coordinates; a static ``numpy`` permutation lowers to
        slices (see :meth:`Gaussian.event_permute`).
        """
        g = Gaussian(self.log_normalizer, self.info_vec, self.precision)
        g = g.event_permute(perm)
        return GammaGaussian(
            g.log_normalizer, g.info_vec, g.precision, self.alpha, self.beta
        )

    def __add__(self, other: GammaGaussian) -> GammaGaussian:
        if not isinstance(other, GammaGaussian):
            raise TypeError(f"cannot add {type(other).__name__} to GammaGaussian")
        return GammaGaussian(
            *(a + b for a, b in zip(self._fields(), other._fields()))
        )._broadcast()

    def log_density(self, value: Array, s: Array) -> Array:
        """
        Evaluate the factor at ``value`` and scale ``s``.

        :param Array value: shape ``(..., dim)``.
        :param Array s: scale, broadcastable with ``batch_shape``.
        :return: log density with the broadcast shape of ``value.shape[:-1]``,
            ``batch_shape`` and ``s.shape``.
        :rtype: Array
        """
        scale_term = self.alpha * jnp.log(s) - self.beta * s
        if self.dim == 0:
            return jnp.broadcast_to(
                scale_term + self.log_normalizer,
                lax.broadcast_shapes(value.shape[:-1], self.batch_shape, jnp.shape(s)),
            )
        quadratic = (value * (-0.5 * _mv(self.precision, value) + self.info_vec)).sum(
            -1
        )
        return self.log_normalizer + scale_term + s * quadratic

    def condition(self, value: Array) -> GammaGaussian:
        """
        Condition on the trailing block of coordinates.

        :param Array value: shape ``(..., right)`` with ``right <= dim``;
            leading dimensions broadcast against ``batch_shape``.
        :return: factor over the leading ``dim - right`` coordinates with the
            conditioned quadratic folded into ``beta``, so
            ``g.log_density(concat([a, b]), s) == g.condition(b).log_density(a, s)``.
        :rtype: GammaGaussian
        :raises ValueError: if ``value`` conditions more than ``dim``
            coordinates.
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

        :param int left: number of leading coordinates to integrate.
        :param int right: number of trailing coordinates to integrate.
        :return: factor over the remaining coordinates with
            ``event_logsumexp`` preserved; the integrated block must have
            positive-definite precision.
        :rtype: GammaGaussian
        """
        if left == 0 and right == 0:
            return self
        precision, info_vec, b_tmp, log_diag = _schur_marginalize(
            self.precision, self.info_vec, left, right
        )
        n_b = left + right
        return GammaGaussian(
            self.log_normalizer + 0.5 * n_b * _LOG_2PI - log_diag,
            info_vec,
            precision,
            self.alpha - 0.5 * n_b,
            self.beta - 0.5 * (b_tmp * b_tmp).sum(-1),
        )._broadcast()

    def event_logsumexp(self) -> GammaFactor:
        """
        Integrate the factor over ``x``; requires positive-definite precision.

        :return: the remaining factor over ``s``.
        :rtype: GammaFactor
        """
        chol = safe_cholesky(self.precision)
        u = solve_triangular(chol, self.info_vec[..., None], lower=True)[..., 0]
        return GammaFactor(
            self.log_normalizer + 0.5 * self.dim * _LOG_2PI - _log_diag_sum(chol),
            self.alpha - 0.5 * self.dim + 1,
            self.beta - 0.5 * (u * u).sum(-1),
        )

    def compound(self) -> MultivariateStudentT:
        """
        Marginal over ``s`` of the normalized joint.

        The moments come from :func:`~numpyro.ops.gaussian.loc_and_scale_tril`,
        which factorizes a positive-definite ``precision`` exactly and adds the
        rounding-level diagonal jitter of
        :func:`~numpyro.distributions.util.jitter_if_singular` only when the
        plain factorization fails.

        :return: Student-t with ``2 (alpha - dim / 2 + 1)`` degrees of freedom.
        :rtype: MultivariateStudentT
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

    :param Array matrix: shape ``(..., y_dim, x_dim)``.
    :param Distribution mvn: ``MultivariateNormal`` or ``Independent(Normal, 1)``
        noise with ``event_shape == (y_dim,)``, possibly wrapped in
        ``ExpandedDistribution``.
    :return: factor over ``concat([x, y])`` with ``alpha == y_dim / 2`` and the
        quadratic ``0.5 loc^T P loc`` tracked in ``beta``.
    :rtype: GammaGaussian
    :raises TypeError: if ``mvn`` is not a supported Gaussian distribution.
    :raises ValueError: if ``mvn.event_shape`` does not match the rows of
        ``matrix``.
    """
    g = matrix_and_mvn_to_gaussian(matrix, mvn)
    if isinstance(g, AffineNormal):
        g = g.to_gaussian()
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

    :param GammaGaussian x: factor over ``(a, b)`` with ``b`` the trailing
        ``dims`` coordinates.
    :param GammaGaussian y: factor over ``(b, c)`` with ``b`` the leading
        ``dims`` coordinates.
    :param int dims: number of shared coordinates.
    :return: factor over ``(a, c)`` with the broadcast batch shape of ``x``
        and ``y``.
    :rtype: GammaGaussian
    :raises ValueError: if ``dims`` is negative or exceeds the event dimension
        of a factor.
    """
    if dims < 0:
        raise ValueError(f"dims must be non-negative, got {dims}")
    na, nb, nc = x.dim - dims, dims, y.dim - dims
    if na < 0 or nc < 0:
        raise ValueError("dims exceeds the event dimension of a factor")
    perm = np.concatenate(
        [np.arange(na), np.arange(x.dim, x.dim + nc), np.arange(na, x.dim)]
    )
    joint = x.event_pad(right=nc) + y.event_pad(left=na)
    return joint.event_permute(perm).marginalize(right=nb)


def sequential_gamma_gaussian_tensordot(gaussian: GammaGaussian) -> GammaGaussian:
    """
    Reduce pairwise factors over time to one factor over ``(z_0, z_T, s)``.

    :param GammaGaussian gaussian: batched factor whose trailing batch
        dimension indexes time and whose event dimension is ``2 * state_dim``.
    :return: the contraction ``g[..., 0] @ g[..., 1] @ ... @ g[..., T - 1]``
        over each intermediate state, computed in ``log2(T)`` batched steps.
    :rtype: GammaGaussian
    :raises ValueError: if the time axis is empty or the event dimension is odd.
    """
    return _sequential_tensordot(gaussian, gamma_gaussian_tensordot)
