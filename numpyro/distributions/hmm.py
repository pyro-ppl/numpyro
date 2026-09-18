# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import operator
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Optional,
    Protocol,
    Self,
    Sequence,
    TypeVar,
    Union,
)

import jax
from jax import Array, lax, random
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, solve_triangular
from jax.typing import ArrayLike

from numpyro.distributions import constraints
from numpyro.distributions.continuous import Gamma, MultivariateNormal
from numpyro.distributions.distribution import (
    Distribution,
    ExpandedDistribution,
    Independent,
    TransformedDistribution,
    _peel_event,
)
from numpyro.distributions.transforms import Transform
from numpyro.distributions.util import validate_sample
from numpyro.ops.gamma_gaussian import (
    GammaGaussian,
    gamma_and_mvn_to_gamma_gaussian,
    gamma_gaussian_tensordot,
    matrix_and_mvn_to_gamma_gaussian,
    sequential_gamma_gaussian_tensordot,
)
from numpyro.ops.gaussian import (
    AffineNormal,
    Gaussian,
    _type_name,
    gaussian_tensordot,
    loc_and_scale_tril,
    matrix_and_mvn_to_gaussian,
    mvn_moments,
    mvn_to_gaussian,
    sequential_gaussian_filter_sample,
    sequential_gaussian_tensordot,
)

__all__ = [
    "GammaGaussianHMM",
    "GaussianHMM",
    "GaussianMRF",
    "HiddenMarkovModel",
    "IndependentHMM",
    "LinearHMM",
]

Factor = Union[Gaussian, AffineNormal]


class _Shaped(Protocol):
    """Shape operations shared by every factor type."""

    @property
    def dim(self) -> int: ...
    @property
    def batch_shape(self) -> tuple[int, ...]: ...
    def expand(self, batch_shape: Sequence[int]) -> Self: ...
    def reshape(self, batch_shape: Sequence[int]) -> Self: ...


class _Reducible(_Shaped, Protocol):
    """Factor type a :class:`HiddenMarkovModel` reduces over."""

    def event_pad(self, left: int = ..., right: int = ...) -> Self: ...


F = TypeVar("F", bound=_Shaped)
S = TypeVar("S", bound=_Shaped)
Z = TypeVar("Z", bound=_Reducible)


class _Step(_Shaped, Protocol[Z]):
    """Per-step factor whose sum with a ``Z`` factor is a ``Z`` factor."""

    def __add__(self, other: Z) -> Z: ...


def _with_batch_rank(factor: F, rank: int) -> F:
    missing = rank - len(factor.batch_shape)
    return (
        factor.reshape((1,) * missing + factor.batch_shape) if missing > 0 else factor
    )


def _align(init: F, trans: S, obs: S) -> tuple[F, S, S]:
    """
    Insert leading singleton batch axes so ``init`` has rank ``r`` and the
    per-step factors rank ``r + 1``.
    """
    rank = max(
        len(init.batch_shape), len(trans.batch_shape) - 1, len(obs.batch_shape) - 1
    )
    return (
        _with_batch_rank(init, rank),
        _with_batch_rank(trans, rank + 1),
        _with_batch_rank(obs, rank + 1),
    )


def _vmap_leading(fn: Callable, ndim: int) -> Callable:
    for _ in range(ndim):
        fn = jax.vmap(fn)
    return fn


def _as_float(matrix: ArrayLike) -> Array:
    """
    Convert ``matrix`` to an array, promoting integer input to the default
    float dtype and leaving float dtypes untouched.
    """
    matrix = jnp.asarray(matrix)
    return matrix.astype(jnp.result_type(matrix.dtype, float))


def _time_shape(*shapes: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    try:
        shape = lax.broadcast_shapes(*shapes)
    except ValueError as e:
        raise ValueError(
            f"parameter batch shapes {list(shapes)} do not broadcast; "
            "the per-step time axis sizes are "
            f"{[s[-1] if s else 1 for s in shapes]}"
        ) from e
    return shape[:-1], shape[-1]


def _resolve_num_steps(time: int, num_steps: Optional[int]) -> int:
    if num_steps is not None:
        num_steps = operator.index(num_steps)
    if time == 1:
        if num_steps is None:
            raise ValueError(
                "num_steps is required when all parameters are time-homogeneous"
            )
    elif num_steps is not None and num_steps != time:
        raise ValueError(
            f"num_steps={num_steps} conflicts with the parameters' time axis "
            f"of size {time}"
        )
    else:
        num_steps = time
    if num_steps < 1:
        raise ValueError("num_steps must be a positive integer")
    return num_steps


def _check_event_shapes(
    initial_dist: Distribution,
    transition_matrix: Array,
    transition_dist: Distribution,
    observation_matrix: Array,
    observation_dist: Distribution,
) -> None:
    """
    Check that the matrices and noise distributions agree on ``hidden_dim``
    and ``obs_dim``.

    :param Distribution initial_dist: distribution over ``z_0``.
    :param Array transition_matrix: shape ``(..., hidden_dim, hidden_dim)``.
    :param Distribution transition_dist: process noise with
        ``event_shape == (hidden_dim,)``.
    :param Array observation_matrix: shape ``(..., obs_dim, hidden_dim)``.
    :param Distribution observation_dist: observation noise with
        ``event_shape == (obs_dim,)``.
    :raises ValueError: if any shape disagrees with ``observation_matrix``.
    """
    obs_dim, hidden_dim = observation_matrix.shape[-2:]
    if transition_matrix.shape[-2:] != (hidden_dim, hidden_dim):
        raise ValueError(
            "transition_matrix must have shape (..., hidden_dim, hidden_dim)"
        )
    for name, d, expected in (
        ("initial_dist", initial_dist, (hidden_dim,)),
        ("transition_dist", transition_dist, (hidden_dim,)),
        ("observation_dist", observation_dist, (obs_dim,)),
    ):
        if tuple(d.event_shape) != expected:
            raise ValueError(
                f"{name} must have event_shape {expected}, got {tuple(d.event_shape)}"
            )


def _resolve_layout(
    initial_dist: Distribution,
    transition_matrix: Array,
    transition_dist: Distribution,
    observation_matrix: Array,
    observation_dist: Distribution,
    num_steps: Optional[int],
    *,
    extra_batch_shapes: Sequence[tuple[int, ...]] = (),
) -> tuple[tuple[int, ...], int, int]:
    """
    Validate event shapes and broadcast the parameters' batch shapes.

    :param Distribution initial_dist: distribution over ``z_0``.
    :param Array transition_matrix: shape ``(..., hidden_dim, hidden_dim)``.
    :param Distribution transition_dist: process noise with
        ``event_shape == (hidden_dim,)``.
    :param Array observation_matrix: shape ``(..., obs_dim, hidden_dim)``.
    :param Distribution observation_dist: observation noise with
        ``event_shape == (obs_dim,)``.
    :param Optional[int] num_steps: requested length of the time axis.
    :param Sequence[tuple[int, ...]] extra_batch_shapes: batch shapes of
        additional time-homogeneous parameters (such as a shared scale prior)
        that must broadcast with the others.
    :return: ``(batch_shape, time, num_steps)`` where ``time`` is the size of
        the parameters' time axis (1 when every parameter is homogeneous).
    :rtype: tuple[tuple[int, ...], int, int]
    :raises ValueError: if event shapes disagree, the batch shapes do not
        broadcast, or ``num_steps`` is missing or conflicts with ``time``.
    """
    _check_event_shapes(
        initial_dist,
        transition_matrix,
        transition_dist,
        observation_matrix,
        observation_dist,
    )
    batch_shape, time = _time_shape(
        tuple(initial_dist.batch_shape) + (1,),
        transition_matrix.shape[:-2],
        tuple(transition_dist.batch_shape),
        observation_matrix.shape[:-2],
        tuple(observation_dist.batch_shape),
        *(tuple(shape) + (1,) for shape in extra_batch_shapes),
    )
    return batch_shape, time, _resolve_num_steps(time, num_steps)


def _check_expand(old: tuple[int, ...], new: Sequence[int]) -> tuple[int, ...]:
    """Return ``new`` as a tuple if ``old`` broadcasts to exactly it."""
    new = tuple(new)
    try:
        full = lax.broadcast_shapes(old, new)
    except ValueError:
        full = None
    if full != new:
        raise ValueError(f"Cannot broadcast distribution of shape {old} to shape {new}")
    return new


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _Moments:
    """
    Moment-form parameters of a :class:`GaussianHMM` for the sequential path.

    ``loc0``/``cov0`` have shape ``batch_shape + (hidden_dim,)`` /
    ``+ (hidden_dim, hidden_dim)``; the per-step fields carry a time axis of
    size 1 or ``num_steps`` after the batch dimensions.
    """

    loc0: Array
    cov0: Array
    A: Array
    b: Array
    Q: Array
    C: Array
    d: Array
    R: Array

    _event_ndims: ClassVar[tuple[int, ...]] = (1, 2, 3, 2, 3, 3, 2, 3)

    def _fields(self) -> tuple[Array, ...]:
        return (self.loc0, self.cov0, self.A, self.b, self.Q, self.C, self.d, self.R)

    def _map(self, fn: Callable[[Array, int], Array]) -> _Moments:
        return _Moments(*(fn(x, k) for x, k in zip(self._fields(), self._event_ndims)))

    @property
    def num_time(self) -> int:
        return self.A.shape[-3]

    def expand(self, batch_shape: tuple[int, ...]) -> _Moments:
        """
        Broadcast the batch dimensions (everything before the time axis for
        per-step fields) to ``batch_shape``.
        """
        return self._map(
            lambda x, k: jnp.broadcast_to(x, tuple(batch_shape) + x.shape[x.ndim - k :])
        )

    def reshape(self, batch_shape: tuple[int, ...]) -> _Moments:
        return self._map(
            lambda x, k: x.reshape(tuple(batch_shape) + x.shape[x.ndim - k :])
        )

    def time_slice(self, start: int, stop: int) -> _Moments:
        """
        Slice the time axis of the per-step fields; a size-1 time axis is kept
        as is.
        """
        if self.num_time == 1:
            return self
        sl = slice(start, stop)
        return _Moments(
            self.loc0,
            self.cov0,
            self.A[..., sl, :, :],
            self.b[..., sl, :],
            self.Q[..., sl, :, :],
            self.C[..., sl, :, :],
            self.d[..., sl, :],
            self.R[..., sl, :, :],
        )


def _kalman_filter(
    m: _Moments, value: Array, num_steps: int
) -> tuple[Array, Array, Array]:
    """
    Covariance-form Kalman filter over ``value`` of shape
    ``batch_shape + (num_steps, obs_dim)``.

    :return: ``(log_prob, loc_T, cov_T)`` with shapes ``batch_shape``,
        ``batch_shape + (hidden_dim,)`` and
        ``batch_shape + (hidden_dim, hidden_dim)``.
    :rtype: tuple[Array, Array, Array]
    """
    batch_shape = value.shape[:-2]
    obs_dim = value.shape[-1]
    hidden_dim = m.loc0.shape[-1]

    def per_step(x: Array, k: int) -> Array:
        x = jnp.broadcast_to(x, batch_shape + (num_steps,) + x.shape[x.ndim - k + 1 :])
        return jnp.moveaxis(x, len(batch_shape), 0)

    mv = lambda M, v: jnp.einsum("...ij,...j->...i", M, v)  # noqa: E731
    mm = lambda X, Y: jnp.matmul(X, Y)  # noqa: E731
    mt = lambda X: jnp.swapaxes(X, -1, -2)  # noqa: E731
    xs = (
        per_step(m.A, 3),
        per_step(m.b, 2),
        per_step(m.Q, 3),
        per_step(m.C, 3),
        per_step(m.d, 2),
        per_step(m.R, 3),
        jnp.moveaxis(value, len(batch_shape), 0),
    )
    loc0 = jnp.broadcast_to(m.loc0, batch_shape + (hidden_dim,))
    cov0 = jnp.broadcast_to(m.cov0, batch_shape + (hidden_dim, hidden_dim))

    def step(
        carry: tuple[Array, Array], inputs: tuple[Array, ...]
    ) -> tuple[tuple[Array, Array], Array]:
        loc, cov = carry
        A, b, Q, C, d, R, x = inputs
        loc_pred = mv(A, loc) + b
        cov_pred = mm(mm(A, cov), mt(A)) + Q
        S = mm(mm(C, cov_pred), mt(C)) + R
        L = jnp.linalg.cholesky(S)
        r = x - mv(C, loc_pred) - d
        u = solve_triangular(L, r[..., None], lower=True)[..., 0]
        ll = (
            -0.5 * (u * u).sum(-1)
            - jnp.log(jnp.einsum("...ii->...i", L)).sum(-1)
            - 0.5 * obs_dim * math.log(2 * math.pi)
        )
        K = mt(cho_solve((L, True), mm(C, cov_pred)))
        loc_new = loc_pred + mv(K, r)
        # Joseph form: stays positive definite in float32 where
        # cov_pred - K S K^T does not.
        I_KC = jnp.eye(hidden_dim, dtype=cov_pred.dtype) - mm(K, C)
        cov_new = mm(mm(I_KC, cov_pred), mt(I_KC)) + mm(mm(K, R), mt(K))
        cov_new = 0.5 * (cov_new + mt(cov_new))
        return (loc_new, cov_new), ll

    (loc_T, cov_T), lls = lax.scan(step, (loc0, cov0), xs)
    return lls.sum(0), loc_T, cov_T


def _peel_observation(d: Distribution) -> tuple[Distribution, list[Transform]]:
    """
    Strip ``Independent``, ``ExpandedDistribution`` and ``TransformedDistribution``
    wrappers, returning the base noise with ``event_dim == 1`` and the transforms.
    """
    shape = tuple(d.batch_shape) + tuple(d.event_shape)
    transforms: list[Transform] = []
    while True:
        if isinstance(d, (Independent, ExpandedDistribution)):
            d = d.base_dist
        elif isinstance(d, TransformedDistribution):
            transforms = list(d.transforms) + transforms
            d = d.base_dist
        else:
            break
    d = d.expand(shape[: len(shape) - d.event_dim])
    if d.event_dim == 0:
        d = d.to_event(1)
    return d, transforms


class HiddenMarkovModel(Distribution, Generic[Z]):
    """
    Base class for distributions over observation sequences with the latent
    chain marginalized by factor reduction.

    Subclasses store three factors: ``_init`` over ``z_0``, ``_trans`` over
    ``(z_{t-1}, z_t)`` and ``_obs`` over ``(z_t, x_t)``. ``_obs`` is always a
    normalized conditional over ``x_t`` given ``z_t``; any factor over ``z_t``
    alone (such as the marginal of a conjugate likelihood) lives in ``_trans``.
    Time is the rightmost batch axis of the per-step factors (size 1 when
    time-homogeneous).
    :meth:`Distribution.__init__` is not called because ``batch_shape`` and
    ``event_shape`` are properties derived from the factor leaves instead of
    stored metadata, so instances built under :func:`jax.vmap` or carried
    through :func:`jax.lax.scan` report the mapped batch shape (the lazy shape
    model proposed in https://github.com/pyro-ppl/numpyro/issues/2271).

    The class is generic in the factor type ``Z`` that ``_init`` and the
    reduction results have (:class:`~numpyro.ops.gaussian.Gaussian` or
    :class:`~numpyro.ops.gamma_gaussian.GammaGaussian`); the per-step factors
    may be any type whose sum with a ``Z`` factor is a ``Z`` factor, such as
    :class:`~numpyro.ops.gaussian.AffineNormal`. Subclasses set
    ``_sequential`` and ``_tensordot`` to the factor type's sequential
    reduction and pairwise contraction.

    .. note:: Matrices act on the left and the time axis is static; see the
        note in :class:`GaussianHMM` for the differences from Pyro.

    :param Z init: factor over ``z_0``.
    :param trans: per-step factors over ``(z_{t-1}, z_t)``.
    :param obs: per-step factors over ``(z_t, x_t)``.
    :param int num_steps: length of the time axis.
    """

    _init: Z
    _trans: _Step[Z]
    _obs: _Step[Z]
    _moments: Optional[_Moments]

    arg_constraints = {}
    support = constraints.real_matrix
    pytree_data_fields = ("_init", "_trans", "_obs", "_moments")
    pytree_aux_fields = ("num_steps",)

    @staticmethod
    def _sequential(factor: Z) -> Z:
        raise NotImplementedError(
            "HiddenMarkovModel is abstract; use a subclass such as GaussianHMM"
        )

    @staticmethod
    def _tensordot(x: Z, y: Z, dims: int) -> Z:
        raise NotImplementedError(
            "HiddenMarkovModel is abstract; use a subclass such as GaussianHMM"
        )

    def log_prob(
        self, value: ArrayLike, intermediates: Optional[list[Any]] = None
    ) -> Array:
        raise NotImplementedError(
            "HiddenMarkovModel is abstract; use a subclass such as GaussianHMM"
        )

    def sample(self, key: Optional[Array], sample_shape: tuple[int, ...] = ()) -> Array:
        raise NotImplementedError(
            "HiddenMarkovModel is abstract; use a subclass such as GaussianHMM"
        )

    def __init__(
        self,
        init: Z,
        trans: _Step[Z],
        obs: _Step[Z],
        num_steps: int,
        *,
        validate_args: Optional[bool] = None,
    ) -> None:
        self._moments = None
        self._init, self._trans, self._obs = _align(init, trans, obs)
        self.num_steps = num_steps
        if validate_args is not None:
            self._validate_args = validate_args

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return lax.broadcast_shapes(
            self._init.batch_shape,
            self._trans.batch_shape[:-1],
            self._obs.batch_shape[:-1],
        )

    @property
    def event_shape(self) -> tuple[int, ...]:
        return (self.num_steps, self.obs_dim)

    @property
    def hidden_dim(self) -> int:
        return self._init.dim

    @property
    def obs_dim(self) -> int:
        return self._obs.dim - self._init.dim

    def _replace(self, **fields) -> Self:
        new = copy.copy(self)
        for name, value in fields.items():
            setattr(new, name, value)
        new._init, new._trans, new._obs = _align(new._init, new._trans, new._obs)
        return new

    def expand(self, batch_shape: Sequence[int]) -> Self:
        """
        Broadcast the distribution to ``batch_shape``.

        Only the initial factor is expanded; the per-step factors keep their
        shapes and broadcast at reduction time. The result is an instance of
        the same class, not an :class:`ExpandedDistribution`.

        :param tuple batch_shape: batch shape to expand to; must be the
            broadcast of itself and the current ``batch_shape``.
        :return: a copy with ``batch_shape`` expanded.
        :rtype: HiddenMarkovModel
        :raises ValueError: if the current batch shape does not broadcast to
            ``batch_shape``.
        """
        batch_shape = _check_expand(self.batch_shape, batch_shape)
        moments = None if self._moments is None else self._moments.expand(batch_shape)
        return self._replace(_init=self._init.expand(batch_shape), _moments=moments)

    def reshape_batch(self, batch_shape: Sequence[int]) -> Self:
        """
        Reshape the batch dimensions to ``batch_shape`` with the same number
        of elements, e.g. to append a singleton batch axis before wrapping in
        :class:`IndependentHMM` or when a forecasting model has to line up
        batch axes of several models.

        :param tuple batch_shape: new batch shape.
        :return: a copy whose factors are broadcast to the current
            ``batch_shape`` and then reshaped.
        :rtype: HiddenMarkovModel
        """
        batch_shape = tuple(batch_shape)
        full = self.batch_shape
        trans_time = self._trans.batch_shape[-1:]
        obs_time = self._obs.batch_shape[-1:]
        moments = (
            None
            if self._moments is None
            else self._moments.expand(full).reshape(batch_shape)
        )
        return self._replace(
            _init=self._init.expand(full).reshape(batch_shape),
            _trans=self._trans.expand(full + trans_time).reshape(
                batch_shape + trans_time
            ),
            _obs=self._obs.expand(full + obs_time).reshape(batch_shape + obs_time),
            _moments=moments,
        )

    def _time_expanded(self, factor: F) -> F:
        return factor.expand(factor.batch_shape[:-1] + (self.num_steps,))

    def _reduce(self, z_factor: Z) -> Z:
        """
        Contract ``_init`` with the per-step factors ``_trans + z_factor`` over
        every step, returning a factor over ``z_T``.

        :param Z z_factor: Per-step factor over ``z_t`` (time as the rightmost
            batch axis, size 1 when homogeneous).
        :rtype: Z
        """
        logp = self._trans + z_factor.event_pad(left=self.hidden_dim)
        logp = self._sequential(self._time_expanded(logp))
        return self._tensordot(self._init, logp, self.hidden_dim)

    def _lead_and_extra(self, value: Array) -> tuple[Array, int]:
        if value.shape[-2:] != self.event_shape:
            raise ValueError(
                f"value must have trailing shape {self.event_shape}, "
                f"got {value.shape[-2:]}"
            )
        try:
            lead = lax.broadcast_shapes(value.shape[: value.ndim - 2], self.batch_shape)
        except ValueError as e:
            raise ValueError(
                f"value batch shape {value.shape[: value.ndim - 2]} does not "
                f"broadcast with batch_shape {self.batch_shape}"
            ) from e
        extra = len(lead) - len(self.batch_shape)
        return jnp.broadcast_to(value, lead + value.shape[-2:]), extra


class GaussianHMM(HiddenMarkovModel[Gaussian]):
    r"""
    Hidden Markov model with linear-Gaussian dynamics and observations, with
    the latent states marginalized out exactly.

    Generative model (matrices act on the left, as in
    :class:`~numpyro.distributions.GaussianStateSpace`)::

        z_0 ~ initial_dist
        z_t = transition_matrix[t] @ z_{t-1} + transition_dist[t].sample()
        x_t = observation_matrix[t] @ z_t + observation_dist[t].sample()

    ``event_shape == (num_steps, obs_dim)``. Per-step parameters carry time as
    their rightmost batch dimension; size 1 (or no batch dimensions) means
    time-homogeneous, in which case ``num_steps`` is required. ``log_prob``,
    :meth:`filter` and sampling run in ``O(log num_steps)`` parallel depth,
    following Sarkka and Garcia-Fernandez, "Temporal parallelization of
    Bayesian smoothers" (IEEE TAC 2021, arXiv:1905.13002).

    ``mean`` and ``variance`` are not implemented; the marginal moments of
    ``x_{1:T}`` follow from :meth:`sample` with zero noise and from the
    recursion in :class:`~numpyro.distributions.GaussianStateSpace`. This is a
    deliberate omission to keep the class focused on marginal likelihoods and
    filtering.

    .. note:: This class deviates from Pyro's ``GaussianHMM`` in two ways.

        Matrices act on the left. Pyro computes ``z @ transition_matrix`` and
        ``z @ observation_matrix`` with ``observation_matrix`` of shape
        ``(hidden_dim, obs_dim)``; here ``transition_matrix @ z`` and
        ``observation_matrix @ z`` with ``observation_matrix`` of shape
        ``(obs_dim, hidden_dim)``. To port a Pyro model pass
        ``jnp.swapaxes(transition_matrix, -1, -2)`` and
        ``jnp.swapaxes(observation_matrix, -1, -2)`` (transposing the trailing
        two axes). A square ``transition_matrix`` passed without the transpose
        silently defines a different model.

        The time axis is static. Pyro's ``duration=None`` mode, where
        homogeneous parameters give ``event_shape == (1, obs_dim)`` and
        ``log_prob`` accepts any length, does not exist: a time-homogeneous
        model needs ``num_steps``, and ``log_prob`` raises ``ValueError``
        unless ``value`` has the exact trailing shape ``(num_steps, obs_dim)``.

    Precision: the information form subtracts large numbers when a noise
    precision is much larger than the others, so in float32 ``log_prob`` and
    its gradient lose accuracy as the ratio between the largest and smallest
    precision entries and the series length grow. Measured on a two-state
    model with unit process noise and float64 data: with observation standard
    deviation 1.0 the float32 error is 1e-4 at ``num_steps=64``; with 0.01 it
    is 1e-2; with 0.001 it is about 2 nats at 64 steps and 10 nats at 256
    steps, and the gradient with respect to the observation scale is wrong by
    orders of magnitude. For such models pass ``sequential=True``
    (covariance-form Kalman filter, accurate to 1e-5 in float32 at the cost of
    ``O(num_steps)`` depth) or call :func:`numpyro.enable_x64`. The parallel
    reductions factorize their blocks with
    :func:`~numpyro.distributions.util.safe_cholesky`, which is exact for
    positive-definite blocks and retries with a rounding-level diagonal jitter
    only when a factorization fails; :meth:`filter` and
    :func:`~numpyro.ops.gaussian.loc_and_scale_tril` apply the same
    retry-on-failure jitter and then factorize through
    :func:`~numpyro.distributions.util.cholesky_of_inverse`. The sequential
    path instead factorizes the innovation covariance directly and yields
    ``nan`` when it is not positive definite.

    :param Distribution initial_dist: ``MultivariateNormal`` or
        ``Independent(Normal, 1)`` over ``z_0`` with
        ``event_shape == (hidden_dim,)``.
    :param Array transition_matrix: shape broadcastable to
        ``batch_shape + (num_steps, hidden_dim, hidden_dim)``.
    :param Distribution transition_dist: process noise with
        ``event_shape == (hidden_dim,)``.
    :param Array observation_matrix: shape broadcastable to
        ``batch_shape + (num_steps, obs_dim, hidden_dim)``.
    :param Distribution observation_dist: observation noise with
        ``event_shape == (obs_dim,)``.
    :param Optional[int] num_steps: length of the time axis; required when
        every per-step parameter is time-homogeneous.
    :param bool sequential: use a covariance-form Kalman filter with
        ``O(num_steps)`` depth for ``log_prob`` and :meth:`filter` instead of
        the ``O(log num_steps)`` information-form reduction. Numerically
        robust in float32 when noise precisions differ by orders of magnitude;
        see the precision note. Sampling always uses the information form, and
        :meth:`conjugate_update` is not available.
    :raises ValueError: if event shapes disagree, the batch shapes do not
        broadcast, or ``num_steps`` is missing or conflicts with the
        parameters' time axis.
    :raises TypeError: if a noise distribution is not ``MultivariateNormal``
        or ``Independent(Normal, 1)``.
    """

    _trans: Factor
    _obs: Factor
    _sequential = staticmethod(sequential_gaussian_tensordot)
    _tensordot = staticmethod(gaussian_tensordot)

    def __init__(
        self,
        initial_dist: Distribution,
        transition_matrix: Array,
        transition_dist: Distribution,
        observation_matrix: Array,
        observation_dist: Distribution,
        *,
        num_steps: Optional[int] = None,
        sequential: bool = False,
        validate_args: Optional[bool] = None,
    ) -> None:
        transition_matrix = _as_float(transition_matrix)
        observation_matrix = _as_float(observation_matrix)
        batch_shape, time, num_steps = _resolve_layout(
            initial_dist,
            transition_matrix,
            transition_dist,
            observation_matrix,
            observation_dist,
            num_steps,
        )
        super().__init__(
            mvn_to_gaussian(initial_dist),
            matrix_and_mvn_to_gaussian(transition_matrix, transition_dist),
            matrix_and_mvn_to_gaussian(observation_matrix, observation_dist),
            num_steps,
            validate_args=validate_args,
        )
        if sequential:
            hidden_dim, obs_dim = (
                transition_matrix.shape[-1],
                observation_matrix.shape[-2],
            )
            loc0, cov0 = mvn_moments(initial_dist)
            b, Q = mvn_moments(transition_dist)
            d, R = mvn_moments(observation_dist)
            self._moments = _Moments(
                jnp.broadcast_to(loc0, batch_shape + (hidden_dim,)),
                jnp.broadcast_to(cov0, batch_shape + (hidden_dim, hidden_dim)),
                jnp.broadcast_to(
                    transition_matrix, batch_shape + (time, hidden_dim, hidden_dim)
                ),
                jnp.broadcast_to(b, batch_shape + (time, hidden_dim)),
                jnp.broadcast_to(Q, batch_shape + (time, hidden_dim, hidden_dim)),
                jnp.broadcast_to(
                    observation_matrix, batch_shape + (time, obs_dim, hidden_dim)
                ),
                jnp.broadcast_to(d, batch_shape + (time, obs_dim)),
                jnp.broadcast_to(R, batch_shape + (time, obs_dim, obs_dim)),
            )

    @property
    def sequential(self) -> bool:
        """
        Whether ``log_prob`` and :meth:`filter` use the covariance-form
        sequential Kalman filter.
        """
        return self._moments is not None

    @property
    def has_rsample(self) -> bool:
        return True

    def _posterior(self, value: Array) -> Gaussian:
        """Factor over ``z_T`` given ``value`` of shape ``batch_shape + (num_steps, obs_dim)``."""
        return self._reduce(self._obs.condition(value))

    def _normalized_posterior(self, value: Array) -> Gaussian:
        """
        Normalized factor over ``z_T`` given ``value`` of exactly
        ``batch_shape + (num_steps, obs_dim)``.
        """
        g = self._posterior(value)
        return g - g.event_logsumexp()

    @validate_sample
    def log_prob(self, value: Array) -> Array:
        """
        Marginal log density of an observation sequence.

        :param Array value: observations of shape ``(..., num_steps, obs_dim)``;
            the result has shape equal to the broadcast of the leading
            dimensions with ``batch_shape``; dimensions beyond ``batch_shape``
            are mapped with :func:`jax.vmap`. With ``sequential=True`` the
            result comes from a covariance-form Kalman filter with
            ``O(num_steps)`` depth.
        :return: log density with the broadcast shape.
        :rtype: Array
        :raises ValueError: if ``value`` does not have trailing shape
            ``(num_steps, obs_dim)``.
        """
        value, extra = self._lead_and_extra(value)
        if self._moments is not None:
            m = self._moments
            return _vmap_leading(
                lambda v: _kalman_filter(m, v, self.num_steps)[0], extra
            )(value)
        return _vmap_leading(lambda v: self._posterior(v).event_logsumexp(), extra)(
            value
        )

    def filter(self, value: Array) -> MultivariateNormal:
        """
        Posterior over the final state ``z_T`` given the full observation sequence.

        :param Array value: observations of shape ``(..., num_steps, obs_dim)``;
            the result has batch shape equal to the broadcast of the leading
            dimensions with ``batch_shape``; dimensions beyond ``batch_shape``
            are mapped with :func:`jax.vmap`. With ``sequential=True`` the
            result comes from a covariance-form Kalman filter with
            ``O(num_steps)`` depth.
        :return: posterior with the broadcast batch shape; usable as
            ``initial_dist`` of a follow-on model.
        :rtype: MultivariateNormal
        :raises ValueError: if ``value`` does not have trailing shape
            ``(num_steps, obs_dim)``.
        """
        value, extra = self._lead_and_extra(value)
        if self._moments is not None:
            m = self._moments
            loc, cov = _vmap_leading(
                lambda v: _kalman_filter(m, v, self.num_steps)[1:], extra
            )(value)
            return MultivariateNormal(
                loc, covariance_matrix=cov, validate_args=self._validate_args
            )

        def moments(v: Array) -> tuple[Array, Array]:
            g = self._posterior(v)
            return loc_and_scale_tril(g.info_vec, g.precision)

        loc, scale_tril = _vmap_leading(moments, extra)(value)
        return MultivariateNormal(
            loc, scale_tril=scale_tril, validate_args=self._validate_args
        )

    def conjugate_update(self, other: Distribution) -> tuple[GaussianHMM, Array]:
        """
        Multiply by a Gaussian likelihood over the observations.

        :param Distribution other: ``Independent(Normal, 2)`` or
            ``Independent(MultivariateNormal, 1)`` (possibly expanded) with
            ``event_shape == (num_steps, obs_dim)``.
        :return: ``(updated, log_normalizer)`` such that
            ``self.log_prob(x) + other.log_prob(x) == updated.log_prob(x) + log_normalizer``.
        :rtype: tuple[GaussianHMM, Array]
        :raises ValueError: if ``other.event_shape`` differs from
            ``event_shape``.
        :raises NotImplementedError: if the model was built with
            ``sequential=True``.
        """
        if self._moments is not None:
            raise NotImplementedError(
                "conjugate_update is not available for sequential=True; "
                "construct the model with sequential=False"
            )
        if tuple(other.event_shape) != self.event_shape:
            raise ValueError(
                f"other must have event_shape {self.event_shape}, "
                f"got {tuple(other.event_shape)}"
            )
        per_step = mvn_to_gaussian(_peel_event(other, 1))
        full = self._obs + per_step.event_pad(left=self.hidden_dim)
        r = full.marginalize(right=self.obs_dim)
        new = self._replace(
            _trans=self._trans + r.event_pad(left=self.hidden_dim),
            _obs=full - r.event_pad(right=self.obs_dim),
        )
        log_normalizer = new._reduce(
            new._obs.marginalize(right=self.obs_dim)
        ).event_logsumexp()
        return new._replace(_init=new._init - log_normalizer), log_normalizer

    def prefix_condition(self, data: Array) -> GaussianHMM:
        """
        Condition on the first ``t < num_steps`` observations and return the
        model over the remaining steps.

        :param Array data: shape ``(..., t, obs_dim)`` with
            ``0 < t < num_steps``.
        :return: model over ``num_steps - t`` steps whose initial distribution
            is the filtered posterior. Leading dimensions of ``data`` beyond
            ``batch_shape`` become batch dimensions of the returned model.
        :rtype: GaussianHMM
        :raises ValueError: if ``t`` is not in ``(0, num_steps)``.
        """
        t = data.shape[-2]
        if not 0 < t < self.num_steps:
            raise ValueError(f"prefix length must be in (0, {self.num_steps}), got {t}")

        def split(factor: Factor) -> tuple[Factor, Factor]:
            if factor.batch_shape[-1] == 1:
                return factor, factor
            return factor[..., :t], factor[..., t:]

        trans_head, trans_tail = split(self._trans)
        obs_head, obs_tail = split(self._obs)
        head_moments = None
        if self._moments is not None:
            head_moments = self._moments.time_slice(0, t)
        head = self._replace(
            _trans=trans_head, _obs=obs_head, _moments=head_moments, num_steps=t
        )
        moments = None
        if self._moments is not None:
            posterior = head.filter(data)
            init = mvn_to_gaussian(posterior)
            tail_moments = self._moments.time_slice(t, self.num_steps)
            lead = posterior.batch_shape
            moments = _Moments(
                posterior.mean,
                jnp.asarray(posterior.covariance_matrix),
                *(
                    jnp.broadcast_to(x, lead + x.shape[x.ndim - k :])
                    for x, k in zip(
                        tail_moments._fields()[2:], _Moments._event_ndims[2:]
                    )
                ),
            )
        else:
            value, extra = head._lead_and_extra(data)
            init = _vmap_leading(head._normalized_posterior, extra)(value)
        return self._replace(
            _init=init,
            _trans=trans_tail,
            _obs=obs_tail,
            _moments=moments,
            num_steps=self.num_steps - t,
        )

    def _sample_states(
        self,
        key: Array,
        obs_factor: Optional[Gaussian] = None,
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        # Every stored ``_obs`` is a normalized conditional over ``x_t`` given
        # ``z_t`` (``conjugate_update`` folds the likelihood's ``z_t`` marginal
        # into ``_trans``), so Pyro's ``_obs.marginalize(right=obs_dim)`` term
        # is exactly zero and omitted.
        trans = self._trans
        if obs_factor is not None:
            trans = trans + obs_factor.event_pad(left=self.hidden_dim)
        elif isinstance(trans, AffineNormal):
            trans = trans.to_gaussian()
        trans = self._time_expanded(trans)
        return sequential_gaussian_filter_sample(key, self._init, trans, sample_shape)[
            ..., 1:, :
        ]

    def sample(self, key: Optional[Array], sample_shape: tuple[int, ...] = ()) -> Array:
        """
        Sample observation sequences ``x_{1:T}`` with the latent states integrated out.

        :param Optional[Array] key: PRNG key; ``None`` raises ``ValueError``.
        :param tuple sample_shape: leading sample dimensions.
        :return: draws of shape
            ``sample_shape + batch_shape + (num_steps, obs_dim)``.
        :rtype: Array
        """
        if key is None:
            raise ValueError("GaussianHMM.sample requires a PRNG key")
        key_z, key_x = random.split(key)
        z = self._sample_states(key_z, sample_shape=sample_shape)
        keys = random.split(key_x, sample_shape) if sample_shape else key_x
        emit = _vmap_leading(
            lambda states, k: self._obs.left_condition(states).sample(k),
            len(sample_shape),
        )
        return emit(z, keys)

    def sample_posterior(
        self, key: Array, value: Array, sample_shape: tuple[int, ...] = ()
    ) -> Array:
        """
        Sample latent paths ``z_{1:T}`` given observations.

        :param Array key: PRNG key.
        :param Array value: observations of shape ``(..., num_steps, obs_dim)``;
            dimensions beyond ``batch_shape`` are mapped with :func:`jax.vmap`.
        :param tuple sample_shape: leading sample dimensions.
        :return: latent paths of shape
            ``sample_shape + lead + (num_steps, hidden_dim)`` where ``lead`` is
            the broadcast of the leading dimensions of ``value`` with
            ``batch_shape``.
        :rtype: Array
        :raises ValueError: if ``value`` does not have trailing shape
            ``(num_steps, obs_dim)``.
        """
        value, extra = self._lead_and_extra(value)
        keys = random.split(key, value.shape[:extra]) if extra else key
        draw = _vmap_leading(
            lambda k, v: self._sample_states(k, self._obs.condition(v), sample_shape),
            extra,
        )
        z = draw(keys, value)
        sample_axes = tuple(range(extra, extra + len(sample_shape)))
        return jnp.moveaxis(z, sample_axes, tuple(range(len(sample_shape))))


class GammaGaussianHMM(HiddenMarkovModel[GammaGaussian]):
    r"""
    Hidden Markov model whose Gaussian noise covariances are all divided by a
    shared ``Gamma`` variable, giving a multivariate Student-t marginal over
    observations.

    Generative model::

        s ~ scale_dist
        z_0 ~ scale(initial_dist, s)
        z_t = transition_matrix[t] @ z_{t-1} + scale(transition_dist[t], s).sample()
        x_t = observation_matrix[t] @ z_t + scale(observation_dist[t], s).sample()

    where ``scale(mvn, s)`` multiplies the precision by ``s``. Only
    ``log_prob`` and :meth:`filter` are provided.

    Precision: the same float32 caveats as :class:`GaussianHMM` apply; there is
    no sequential covariance-form path for this class, so use
    :func:`numpyro.enable_x64` when noise precisions differ by orders of
    magnitude.

    .. note:: Matrices act on the left and ``num_steps`` is required for
        time-homogeneous parameters; see the note in :class:`GaussianHMM`.

    :param Distribution scale_dist: ``Gamma`` prior over the shared precision
        scale, possibly wrapped in ``ExpandedDistribution``.
    :param Distribution initial_dist: ``MultivariateNormal`` or
        ``Independent(Normal, 1)`` over ``z_0`` with
        ``event_shape == (hidden_dim,)``.
    :param Array transition_matrix: as in :class:`GaussianHMM`.
    :param Distribution transition_dist: process noise with
        ``event_shape == (hidden_dim,)``.
    :param Array observation_matrix: as in :class:`GaussianHMM`.
    :param Distribution observation_dist: observation noise with
        ``event_shape == (obs_dim,)``.
    :param Optional[int] num_steps: required when every per-step parameter is
        time-homogeneous.
    :raises TypeError: if ``scale_dist`` is not a ``Gamma`` or a noise
        distribution is not ``MultivariateNormal`` or
        ``Independent(Normal, 1)``.
    """

    _trans: GammaGaussian
    _obs: GammaGaussian
    _sequential = staticmethod(sequential_gamma_gaussian_tensordot)
    _tensordot = staticmethod(gamma_gaussian_tensordot)

    def __init__(
        self,
        scale_dist: Distribution,
        initial_dist: Distribution,
        transition_matrix: Array,
        transition_dist: Distribution,
        observation_matrix: Array,
        observation_dist: Distribution,
        *,
        num_steps: Optional[int] = None,
        validate_args: Optional[bool] = None,
    ) -> None:
        base_scale = (
            scale_dist.base_dist
            if isinstance(scale_dist, ExpandedDistribution)
            else scale_dist
        )
        if not isinstance(base_scale, Gamma):
            raise TypeError(f"scale_dist must be a Gamma, got {_type_name(scale_dist)}")
        transition_matrix = jnp.asarray(transition_matrix)
        observation_matrix = jnp.asarray(observation_matrix)
        _, _, num_steps = _resolve_layout(
            initial_dist,
            transition_matrix,
            transition_dist,
            observation_matrix,
            observation_dist,
            num_steps,
            extra_batch_shapes=(tuple(scale_dist.batch_shape),),
        )
        super().__init__(
            gamma_and_mvn_to_gamma_gaussian(scale_dist, initial_dist),
            matrix_and_mvn_to_gamma_gaussian(transition_matrix, transition_dist),
            matrix_and_mvn_to_gamma_gaussian(observation_matrix, observation_dist),
            num_steps,
            validate_args=validate_args,
        )

    @property
    def has_rsample(self) -> bool:
        return False

    def _posterior(self, value: Array) -> GammaGaussian:
        """Factor over ``(z_T, s)`` given ``value`` of shape ``batch_shape + (num_steps, obs_dim)``."""
        return self._reduce(self._obs.condition(value))

    @validate_sample
    def log_prob(self, value: Array) -> Array:
        """
        Marginal log density of an observation sequence with ``s`` and the
        hidden states integrated out.

        :param Array value: observations of shape ``(..., num_steps, obs_dim)``;
            leading dimensions broadcast with ``batch_shape`` as in
            :meth:`GaussianHMM.log_prob`.
        :return: log density with the broadcast shape.
        :rtype: Array
        :raises ValueError: if ``value`` does not have trailing shape
            ``(num_steps, obs_dim)``.
        """
        value, extra = self._lead_and_extra(value)
        return _vmap_leading(
            lambda v: self._posterior(v).event_logsumexp().logsumexp(), extra
        )(value)

    def filter(self, value: Array) -> tuple[Gamma, MultivariateNormal]:
        """
        Posterior over the precision scale and over the final state.

        :param Array value: observations of shape ``(..., num_steps, obs_dim)``;
            leading dimensions broadcast with ``batch_shape`` as in
            :meth:`GaussianHMM.filter`.
        :return: ``p(s | x)`` and ``p(z_T | x, s = 1)``, the latter a
            Gaussian with the posterior precision.
        :rtype: tuple[Gamma, MultivariateNormal]
        :raises ValueError: if ``value`` does not have trailing shape
            ``(num_steps, obs_dim)``.
        """
        value, extra = self._lead_and_extra(value)

        def moments(v: Array) -> tuple[Array, Array, Array, Array]:
            g = self._posterior(v)
            factor = g.event_logsumexp()
            loc, scale_tril = loc_and_scale_tril(g.info_vec, g.precision)
            return factor.concentration, factor.rate, loc, scale_tril

        concentration, rate, loc, scale_tril = _vmap_leading(moments, extra)(value)
        return (
            Gamma(concentration, rate, validate_args=self._validate_args),
            MultivariateNormal(
                loc, scale_tril=scale_tril, validate_args=self._validate_args
            ),
        )

    def sample(self, key: Optional[Array], sample_shape: tuple[int, ...] = ()) -> Array:
        raise NotImplementedError(f"{type(self).__name__} does not support sampling")


class GaussianMRF(HiddenMarkovModel[Gaussian]):
    r"""
    Temporal Markov random field with Gaussian pairwise factors, marginalizing
    the hidden chain.

    ``transition_dist`` is a joint Gaussian over ``(z_{t-1}, z_t)`` and
    ``observation_dist`` a joint Gaussian over ``(z_t, x_t)``;
    ``initial_dist`` is over ``z_0``. ``log_prob`` is
    ``log p(x) = log \int f(z, x) dz - log \int\int f(z, x) dz dx``. Only
    ``log_prob`` is provided; sampling is not supported.

    Precision: the same float32 caveats as :class:`GaussianHMM` apply; there is
    no sequential covariance-form path for this class, so use
    :func:`numpyro.enable_x64` when noise precisions differ by orders of
    magnitude. ``log_prob`` is the difference of two reductions, so
    cancellation is more severe than for :class:`GaussianHMM`.

    :param Distribution initial_dist: ``MultivariateNormal`` or
        ``Independent(Normal, 1)`` with ``event_shape == (hidden_dim,)``.
    :param Distribution transition_dist: joint over ``(z_{t-1}, z_t)`` with
        ``event_shape == (2 * hidden_dim,)``; time on the rightmost batch
        axis.
    :param Distribution observation_dist: joint over ``(z_t, x_t)`` with
        ``event_shape == (hidden_dim + obs_dim,)``; time on the rightmost
        batch axis.
    :param Optional[int] num_steps: required when both per-step distributions
        are time-homogeneous.
    :raises ValueError: if the event shapes disagree, the batch shapes do not
        broadcast, or ``num_steps`` is missing or conflicts with the time axis.
    :raises TypeError: if a distribution is not ``MultivariateNormal`` or
        ``Independent(Normal, 1)``.
    """

    _trans: Gaussian
    _obs: Gaussian
    _sequential = staticmethod(sequential_gaussian_tensordot)
    _tensordot = staticmethod(gaussian_tensordot)

    def __init__(
        self,
        initial_dist: Distribution,
        transition_dist: Distribution,
        observation_dist: Distribution,
        *,
        num_steps: Optional[int] = None,
        validate_args: Optional[bool] = None,
    ) -> None:
        hidden_dim = initial_dist.event_shape[0]
        if tuple(transition_dist.event_shape) != (2 * hidden_dim,):
            raise ValueError(
                f"transition_dist must have event_shape {(2 * hidden_dim,)}"
            )
        if observation_dist.event_shape[0] <= hidden_dim:
            raise ValueError(
                "observation_dist must be a joint over (hidden, observed) coordinates"
            )
        _, time = _time_shape(
            tuple(initial_dist.batch_shape) + (1,),
            tuple(transition_dist.batch_shape),
            tuple(observation_dist.batch_shape),
        )
        super().__init__(
            mvn_to_gaussian(initial_dist),
            mvn_to_gaussian(transition_dist),
            mvn_to_gaussian(observation_dist),
            _resolve_num_steps(time, num_steps),
            validate_args=validate_args,
        )

    @property
    def has_rsample(self) -> bool:
        return False

    @validate_sample
    def log_prob(self, value: Array) -> Array:
        r"""
        Marginal log density of an observation sequence.

        The normalizer ``log \int\int f(z, x) dz dx`` does not depend on
        ``value`` and is computed once; only the conditioned reduction is
        mapped over the leading dimensions of ``value``.

        :param Array value: observations of shape ``(..., num_steps, obs_dim)``;
            leading dimensions broadcast with ``batch_shape`` as in
            :meth:`GaussianHMM.log_prob`.
        :return: log density with the broadcast shape.
        :rtype: Array
        :raises ValueError: if ``value`` does not have trailing shape
            ``(num_steps, obs_dim)``.
        """
        value, extra = self._lead_and_extra(value)
        log_normalizer = self._reduce(
            self._obs.marginalize(right=self.obs_dim)
        ).event_logsumexp()
        log_joint = _vmap_leading(
            lambda v: self._reduce(self._obs.condition(v)).event_logsumexp(), extra
        )(value)
        return log_joint - log_normalizer

    def sample(self, key: Optional[Array], sample_shape: tuple[int, ...] = ()) -> Array:
        raise NotImplementedError(f"{type(self).__name__} does not support sampling")


class IndependentHMM(Distribution):
    """
    Wrap a batch of independent single-observation HMMs into one distribution
    over vector observations.

    The base distribution has ``event_shape == (num_steps, 1)`` and batch shape
    ``shape + (obs_dim,)``; the result has ``batch_shape == shape`` and
    ``event_shape == (num_steps, obs_dim)``. :meth:`reshape_batch` requires a
    :class:`HiddenMarkovModel` base and :meth:`prefix_condition` a
    :class:`GaussianHMM` base.

    Unlike Pyro, this class is not a :class:`HiddenMarkovModel` subclass: it
    stores a base distribution rather than factors, so
    ``isinstance(d, HiddenMarkovModel)`` is ``False`` and nesting
    ``IndependentHMM(IndependentHMM(...))`` is not supported by
    :meth:`reshape_batch`.

    :param Distribution base_dist: batched distribution with a trailing batch
        dimension of size ``obs_dim`` and ``event_shape == (num_steps, 1)``.
    :raises ValueError: if ``base_dist`` is unbatched or its event shape is
        not ``(num_steps, 1)``.
    """

    arg_constraints = {}
    pytree_data_fields = ("base_dist",)
    base_dist: Distribution

    def __init__(
        self, base_dist: Distribution, *, validate_args: Optional[bool] = None
    ) -> None:
        if (
            not base_dist.batch_shape
            or len(base_dist.event_shape) != 2
            or base_dist.event_shape[-1] != 1
        ):
            raise ValueError(
                "base_dist must be batched with event_shape (num_steps, 1)"
            )
        self.base_dist = base_dist
        if validate_args is not None:
            self._validate_args = validate_args

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.base_dist.batch_shape[:-1]

    @property
    def event_shape(self) -> tuple[int, ...]:
        return tuple(self.base_dist.event_shape)[:-1] + self.base_dist.batch_shape[-1:]

    @constraints.dependent_property(event_dim=2)
    def support(self) -> Optional[constraints.Constraint]:
        return self.base_dist.support

    @property
    def has_rsample(self) -> bool:
        return self.base_dist.has_rsample

    @property
    def num_steps(self) -> int:
        return self.base_dist.event_shape[0]

    def sample(self, key: Optional[Array], sample_shape: tuple[int, ...] = ()) -> Array:
        """
        Sample from the base distribution and move the observation axis last.

        :param Optional[Array] key: PRNG key, passed to ``base_dist.sample``;
            ``None`` raises ``ValueError``.
        :param tuple sample_shape: leading sample dimensions.
        :return: draws of shape
            ``sample_shape + batch_shape + (num_steps, obs_dim)``.
        :rtype: Array
        """
        if key is None:
            raise ValueError("IndependentHMM.sample requires a PRNG key")
        x = jnp.asarray(self.base_dist.sample(key, sample_shape))
        return jnp.swapaxes(x[..., 0], -1, -2)

    @validate_sample
    def log_prob(self, value: Array) -> Array:
        """
        Sum the base log densities over the observation axis.

        :param Array value: shape ``(..., num_steps, obs_dim)``.
        :return: log density of shape ``(...)`` broadcast with ``batch_shape``.
        :rtype: Array
        """
        value = jnp.swapaxes(value, -1, -2)[..., None]
        return jnp.asarray(self.base_dist.log_prob(value)).sum(-1)

    def _rewrap(self, base: Distribution) -> IndependentHMM:
        """
        Wrap ``base`` with this instance's ``validate_args`` setting.

        :param Distribution base: replacement base distribution.
        :rtype: IndependentHMM
        """
        return IndependentHMM(base, validate_args=self.__dict__.get("_validate_args"))

    def expand(self, batch_shape: Sequence[int]) -> IndependentHMM:
        """
        Broadcast the distribution to ``batch_shape`` by expanding the base to
        ``batch_shape + (obs_dim,)``.

        :param tuple batch_shape: batch shape to expand to; must be the
            broadcast of itself and the current ``batch_shape``.
        :return: a wrapper around the expanded base.
        :rtype: IndependentHMM
        :raises ValueError: if the current batch shape does not broadcast to
            ``batch_shape``.
        """
        batch_shape = _check_expand(self.batch_shape, batch_shape)
        obs = self.base_dist.batch_shape[-1:]
        return self._rewrap(self.base_dist.expand(batch_shape + obs))

    def reshape_batch(self, batch_shape: Sequence[int]) -> IndependentHMM:
        """
        Reshape the batch dimensions with the same number of elements (see
        :meth:`HiddenMarkovModel.reshape_batch`).

        :param tuple batch_shape: new batch shape, without the trailing
            ``obs_dim`` axis of the base.
        :return: a wrapper around the reshaped base.
        :rtype: IndependentHMM
        :raises TypeError: if the base is not a :class:`HiddenMarkovModel`.
        """
        base = self.base_dist
        if not isinstance(base, HiddenMarkovModel):
            raise TypeError(
                "reshape_batch requires a HiddenMarkovModel base distribution"
            )
        obs = base.batch_shape[-1:]
        return self._rewrap(base.reshape_batch(tuple(batch_shape) + obs))

    def prefix_condition(self, data: Array) -> IndependentHMM:
        """
        Condition on a prefix of observations (see
        :meth:`GaussianHMM.prefix_condition`).

        :param Array data: shape ``(..., t, obs_dim)`` with
            ``0 < t < num_steps``.
        :return: wrapper around the base model over the remaining
            ``num_steps - t`` steps.
        :rtype: IndependentHMM
        :raises TypeError: if the base is not a :class:`GaussianHMM`.
        :raises ValueError: if ``t`` is not in ``(0, num_steps)``.
        """
        base = self.base_dist
        if not isinstance(base, GaussianHMM):
            raise TypeError("prefix_condition requires a GaussianHMM base distribution")
        prefix = jnp.swapaxes(data, -1, -2)[..., None]
        return self._rewrap(base.prefix_condition(prefix))


class LinearHMM(Distribution):
    r"""
    Hidden Markov model with linear dynamics and arbitrary reparameterized
    noise, supporting sampling only.

    Generative model::

        z_0 ~ initial_dist
        z_t = transition_matrix[t] @ z_{t-1} + transition_dist[t].sample()
        x_t = observation_matrix[t] @ z_t + observation_dist[t].sample()

    Components may be any distributions with ``has_rsample`` and
    ``event_dim == 1`` (for example ``Independent(StudentT, 1)``);
    ``TransformedDistribution`` observation noise such as ``LogNormal`` is
    split into a base noise and ``transforms`` applied to the observations.
    ``log_prob`` is not available; use
    :class:`~numpyro.infer.reparam.LinearHMMReparam` for inference.

    ``LinearHMM`` is not a :class:`HiddenMarkovModel` subclass: it stores
    component distributions rather than factors, so
    :meth:`IndependentHMM.reshape_batch` does not accept it as a base;
    :meth:`IndependentHMM.expand` does.

    .. note:: Matrices act on the left and ``num_steps`` is required for
        time-homogeneous parameters; see the note in :class:`GaussianHMM`.

    :param Distribution initial_dist: reparameterized distribution over
        ``z_0`` with ``event_shape == (hidden_dim,)``.
    :param Array transition_matrix: as in :class:`GaussianHMM`.
    :param Distribution transition_dist: reparameterized process noise with
        ``event_shape == (hidden_dim,)``.
    :param Array observation_matrix: as in :class:`GaussianHMM`.
    :param Distribution observation_dist: reparameterized observation noise
        with ``event_shape == (obs_dim,)``; observation transforms must
        preserve the event shape.
    :param Optional[int] num_steps: required when every per-step parameter is
        time-homogeneous.
    :raises TypeError: if a noise distribution is not reparameterized or does
        not have ``event_dim == 1``.
    :raises ValueError: if event shapes disagree, the batch shapes do not
        broadcast, ``num_steps`` is missing or conflicts with the parameters'
        time axis, or an observation transform changes the event shape.
    """

    arg_constraints = {}
    pytree_data_fields = (
        "initial_dist",
        "transition_matrix",
        "transition_dist",
        "observation_matrix",
        "observation_dist",
        "transforms",
    )
    pytree_aux_fields = ("num_steps",)

    def __init__(
        self,
        initial_dist: Distribution,
        transition_matrix: Array,
        transition_dist: Distribution,
        observation_matrix: Array,
        observation_dist: Distribution,
        *,
        num_steps: Optional[int] = None,
        validate_args: Optional[bool] = None,
    ) -> None:
        transition_matrix = jnp.asarray(transition_matrix)
        observation_matrix = jnp.asarray(observation_matrix)
        for name, d in (
            ("initial_dist", initial_dist),
            ("transition_dist", transition_dist),
            ("observation_dist", observation_dist),
        ):
            if d.event_dim != 1:
                raise TypeError(f"{name} must have event_dim == 1, got {d.event_dim}")
            if not d.has_rsample:
                raise TypeError(
                    f"{name} must be reparameterized (has_rsample), "
                    f"got {type(d).__name__}"
                )
        batch_shape, time, num_steps = _resolve_layout(
            initial_dist,
            transition_matrix,
            transition_dist,
            observation_matrix,
            observation_dist,
            num_steps,
        )
        obs_dim, hidden_dim = observation_matrix.shape[-2:]
        observation_dist, transforms = _peel_observation(observation_dist)
        if tuple(observation_dist.event_shape) != (obs_dim,):
            raise ValueError(
                f"observation noise must have event_shape {(obs_dim,)}, "
                f"got {tuple(observation_dist.event_shape)}"
            )
        for transform in transforms:
            if tuple(transform.forward_shape((obs_dim,))) != (obs_dim,):
                raise ValueError(
                    f"observation transform {type(transform).__name__} maps "
                    f"event_shape {(obs_dim,)} to "
                    f"{tuple(transform.forward_shape((obs_dim,)))}"
                )
        self.initial_dist = initial_dist.expand(batch_shape)
        self.transition_matrix = jnp.broadcast_to(
            transition_matrix, batch_shape + (time, hidden_dim, hidden_dim)
        )
        self.transition_dist = transition_dist.expand(batch_shape + (time,))
        self.observation_matrix = jnp.broadcast_to(
            observation_matrix, batch_shape + (time, obs_dim, hidden_dim)
        )
        self.observation_dist = observation_dist.expand(batch_shape + (time,))
        self.transforms = transforms
        self.num_steps = num_steps
        if validate_args is not None:
            self._validate_args = validate_args

    @property
    def has_rsample(self) -> bool:
        return True

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.transition_matrix.shape[:-3]

    @property
    def event_shape(self) -> tuple[int, ...]:
        return (self.num_steps, self.observation_matrix.shape[-2])

    @property
    def hidden_dim(self) -> int:
        return self.observation_matrix.shape[-1]

    @property
    def obs_dim(self) -> int:
        return self.observation_matrix.shape[-2]

    @constraints.dependent_property(event_dim=2)
    def support(self) -> Optional[constraints.Constraint]:
        support = (
            self.transforms[-1].codomain
            if self.transforms
            else self.observation_dist.support
        )
        if support is None:
            return None
        if support.event_dim > 2:
            raise ValueError(
                f"observation support must have event_dim <= 2, got {support.event_dim}"
            )
        return constraints.independent(support, 2 - support.event_dim)

    def sample(self, key: Optional[Array], sample_shape: tuple[int, ...] = ()) -> Array:
        """
        Draw observation sequences by simulating the generative model.

        :param Optional[Array] key: PRNG key.
        :param tuple sample_shape: leading sample dimensions, drawn by mapping
            over split keys so that the component distributions' static batch
            shapes need not know about batch axes added by :func:`jax.vmap`.
        :return: draws of shape
            ``sample_shape + batch_shape + (num_steps, obs_dim)``.
        :rtype: Array
        """
        assert key is not None
        if sample_shape:
            keys = random.split(key, math.prod(sample_shape))
            keys = keys.reshape(tuple(sample_shape) + keys.shape[1:])
            return _vmap_leading(self.sample, len(sample_shape))(keys)
        key_init, key_trans, key_obs = random.split(key, 3)
        time_shape = self.batch_shape + (self.num_steps,)
        z0 = jnp.asarray(self.initial_dist.expand(self.batch_shape).sample(key_init))
        eps = jnp.asarray(self.transition_dist.expand(time_shape).sample(key_trans))
        nu = jnp.asarray(self.observation_dist.expand(time_shape).sample(key_obs))
        A = jnp.moveaxis(
            jnp.broadcast_to(
                self.transition_matrix, time_shape + self.transition_matrix.shape[-2:]
            ),
            -3,
            0,
        )
        H = jnp.broadcast_to(
            self.observation_matrix, time_shape + self.observation_matrix.shape[-2:]
        )

        def step(z: Array, inputs: tuple[Array, Array]) -> tuple[Array, Array]:
            A_t, eps_t = inputs
            z = jnp.einsum("...ij,...j->...i", A_t, z) + eps_t
            return z, z

        _, z = lax.scan(step, z0, (A, jnp.moveaxis(eps, -2, 0)))
        x = jnp.einsum("...ij,...j->...i", H, jnp.moveaxis(z, 0, -2)) + nu
        for transform in self.transforms:
            x = jnp.asarray(transform(x))
        return x

    def log_prob(
        self, value: ArrayLike, intermediates: Optional[list[Any]] = None
    ) -> ArrayLike:
        raise NotImplementedError(
            "LinearHMM.log_prob is not implemented; use LinearHMMReparam"
        )

    def expand(self, batch_shape: Sequence[int]) -> LinearHMM:
        batch_shape = _check_expand(self.batch_shape, batch_shape)
        time_shape = batch_shape + (self.transition_dist.batch_shape[-1],)
        new = copy.copy(self)
        new.initial_dist = self.initial_dist.expand(batch_shape)
        new.transition_matrix = jnp.broadcast_to(
            self.transition_matrix, time_shape + self.transition_matrix.shape[-2:]
        )
        new.transition_dist = self.transition_dist.expand(time_shape)
        new.observation_matrix = jnp.broadcast_to(
            self.observation_matrix, time_shape + self.observation_matrix.shape[-2:]
        )
        new.observation_dist = self.observation_dist.expand(time_shape)
        return new
