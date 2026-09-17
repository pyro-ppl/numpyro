# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import operator
from typing import Callable, Optional, Protocol, Self, Sequence, TypeVar, Union

import jax
from jax import Array, lax, random
import jax.numpy as jnp

from numpyro.distributions import constraints
from numpyro.distributions.continuous import MultivariateNormal
from numpyro.distributions.distribution import (
    Distribution,
    ExpandedDistribution,
    Independent,
)
from numpyro.distributions.util import validate_sample
from numpyro.ops.gaussian import (
    AffineNormal,
    Gaussian,
    gaussian_tensordot,
    loc_and_scale_tril,
    matrix_and_mvn_to_gaussian,
    mvn_to_gaussian,
    sequential_gaussian_filter_sample,
    sequential_gaussian_tensordot,
)

__all__ = ["GaussianHMM", "HiddenMarkovModel", "IndependentHMM"]

Factor = Union[Gaussian, AffineNormal]


class _Shaped(Protocol):
    """Batch shape operations shared by every factor type."""

    @property
    def batch_shape(self) -> tuple[int, ...]: ...
    def expand(self, batch_shape: Sequence[int]) -> Self: ...
    def reshape(self, batch_shape: Sequence[int]) -> Self: ...


F = TypeVar("F", bound=_Shaped)


def _with_batch_rank(factor: F, rank: int) -> F:
    missing = rank - len(factor.batch_shape)
    return (
        factor.reshape((1,) * missing + factor.batch_shape) if missing > 0 else factor
    )


def _align(
    init: Gaussian, trans: Factor, obs: Factor
) -> tuple[Gaussian, Factor, Factor]:
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


def _peel_event(d: Distribution, n: int) -> Distribution:
    """
    Remove ``n`` reinterpreted batch dimensions from nested ``Independent``
    layers, looking through an outer ``ExpandedDistribution``.
    """
    if n == 0:
        return d
    if isinstance(d, ExpandedDistribution):
        base = _peel_event(d.base_dist, n)
        return base.expand(d.batch_shape + tuple(d.event_shape)[:n])
    if not isinstance(d, Independent):
        raise ValueError(f"cannot remove {n} event dimensions from {type(d).__name__}")
    k = min(n, d.reinterpreted_batch_ndims)
    remaining = d.reinterpreted_batch_ndims - k
    d = d.base_dist if remaining == 0 else Independent(d.base_dist, remaining)
    return _peel_event(d, n - k)


def _vmap_leading(fn: Callable, ndim: int) -> Callable:
    for _ in range(ndim):
        fn = jax.vmap(fn)
    return fn


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


class HiddenMarkovModel(Distribution):
    """
    Base class for distributions over observation sequences with the latent
    chain marginalized by factor reduction.

    Subclasses store three factors: ``_init`` over ``z_0``, ``_trans`` over
    ``(z_{t-1}, z_t)`` and ``_obs`` over ``(z_t, x_t)``. Time is the rightmost
    batch axis of the per-step factors (size 1 when time-homogeneous).
    :meth:`Distribution.__init__` is not called because ``batch_shape`` and
    ``event_shape`` are properties derived from the factor leaves instead of
    stored metadata, so instances built under :func:`jax.vmap` or carried
    through :func:`jax.lax.scan` report the mapped batch shape (the lazy shape
    model proposed in issue #2271).

    Subclasses set ``_sequential`` and ``_tensordot`` to the factor type's
    sequential reduction and pairwise contraction.

    .. note:: Matrices act on the left and the time axis is static; see the
        note in :class:`GaussianHMM` for the differences from Pyro.

    :param init: factor over ``z_0``.
    :type init: :class:`~numpyro.ops.gaussian.Gaussian`
    :param trans: per-step factors over ``(z_{t-1}, z_t)``.
    :type trans: :class:`~numpyro.ops.gaussian.Gaussian` or
        :class:`~numpyro.ops.gaussian.AffineNormal`
    :param obs: per-step factors over ``(z_t, x_t)``.
    :type obs: :class:`~numpyro.ops.gaussian.Gaussian` or
        :class:`~numpyro.ops.gaussian.AffineNormal`
    :param int num_steps: length of the time axis.
    """

    _sequential: Callable[[Gaussian], Gaussian]
    _tensordot: Callable[[Gaussian, Gaussian, int], Gaussian]

    arg_constraints = {}
    support = constraints.real_matrix
    pytree_data_fields = ("_init", "_trans", "_obs")
    pytree_aux_fields = ("num_steps",)

    def __init__(
        self,
        init: Gaussian,
        trans: Factor,
        obs: Factor,
        num_steps: int,
        *,
        validate_args: Optional[bool] = None,
    ) -> None:
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
            object.__setattr__(new, name, value)
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
        return self._replace(_init=self._init.expand(batch_shape))

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
        return self._replace(
            _init=self._init.expand(full).reshape(batch_shape),
            _trans=self._trans.expand(full + trans_time).reshape(
                batch_shape + trans_time
            ),
            _obs=self._obs.expand(full + obs_time).reshape(batch_shape + obs_time),
        )

    def _time_expanded(self, factor: F) -> F:
        return factor.expand(factor.batch_shape[:-1] + (self.num_steps,))

    def _reduce(self, z_factor: Gaussian) -> Gaussian:
        """
        Contract ``_init`` with the per-step factors ``_trans + z_factor`` over
        every step, returning a factor over ``z_T``.

        :param Gaussian z_factor: Per-step factor over ``z_t`` (time as the
            rightmost batch axis, size 1 when homogeneous).
        :rtype: Gaussian
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
        lead = lax.broadcast_shapes(value.shape[: value.ndim - 2], self.batch_shape)
        extra = len(lead) - len(self.batch_shape)
        return jnp.broadcast_to(value, lead + value.shape[-2:]), extra


class GaussianHMM(HiddenMarkovModel):
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

    .. note:: This class deviates from Pyro's ``GaussianHMM`` in two ways.

        Matrices act on the left. Pyro computes ``z @ transition_matrix`` and
        ``z @ observation_matrix`` with ``observation_matrix`` of shape
        ``(hidden_dim, obs_dim)``; here ``transition_matrix @ z`` and
        ``observation_matrix @ z`` with ``observation_matrix`` of shape
        ``(obs_dim, hidden_dim)``. To port a Pyro model pass
        ``transition_matrix.T`` and ``observation_matrix.T`` (transposing the
        trailing two axes). A square ``transition_matrix`` passed without the
        transpose silently defines a different model.

        The time axis is static. Pyro's ``duration=None`` mode, where
        homogeneous parameters give ``event_shape == (1, obs_dim)`` and
        ``log_prob`` accepts any length, does not exist: a time-homogeneous
        model needs ``num_steps``, and ``log_prob`` raises ``ValueError``
        unless ``value`` has the exact trailing shape ``(num_steps, obs_dim)``.

    Precision: in float32 the information form loses accuracy when observation
    magnitudes are large relative to the noise scales or when the noise
    variances differ by several orders of magnitude, and the error grows with
    the series length. Center and scale the data, and call
    :func:`numpyro.enable_x64` for long or large-magnitude series. Every
    Cholesky factorization adds a gradient-free jitter of
    ``CHOLESKY_RELATIVE_JITTER * eps * abs(diagonal)`` to the precision
    diagonal (see :func:`~numpyro.distributions.util.relative_jitter`), which
    is at rounding level for well-posed problems; the factorizations inside
    the reductions go through :func:`~numpyro.distributions.util.safe_cholesky`,
    which clamps 1x1 blocks at the smallest positive float instead of
    jittering them.

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
    :raises ValueError: if event shapes disagree, the batch shapes do not
        broadcast, or ``num_steps`` is missing or conflicts with the
        parameters' time axis.
    :raises TypeError: if a noise distribution is not ``MultivariateNormal``
        or ``Independent(Normal, 1)``.
    """

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
        validate_args: Optional[bool] = None,
    ) -> None:
        transition_matrix = jnp.asarray(transition_matrix)
        observation_matrix = jnp.asarray(observation_matrix)
        _, _, num_steps = _resolve_layout(
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

    @property
    def has_rsample(self) -> bool:
        return True

    def _posterior(self, value: Array) -> Gaussian:
        """Factor over ``z_T`` given ``value`` of shape ``batch_shape + (num_steps, obs_dim)``."""
        return self._reduce(self._obs.condition(value))

    @validate_sample
    def log_prob(self, value: Array) -> Array:
        """
        Marginal log density of an observation sequence.

        :param Array value: observations of shape ``(..., num_steps, obs_dim)``;
            the result has shape equal to the broadcast of the leading
            dimensions with ``batch_shape``; dimensions beyond ``batch_shape``
            are mapped with :func:`jax.vmap`.
        :return: log density with the broadcast shape.
        :rtype: Array
        :raises ValueError: if ``value`` does not have trailing shape
            ``(num_steps, obs_dim)``.
        """
        value, extra = self._lead_and_extra(value)
        return _vmap_leading(lambda v: self._posterior(v).event_logsumexp(), extra)(
            value
        )

    def filter(self, value: Array) -> MultivariateNormal:
        """
        Posterior over the final state ``z_T`` given the full observation sequence.

        :param Array value: observations of shape ``(..., num_steps, obs_dim)``;
            the result has batch shape equal to the broadcast of the leading
            dimensions with ``batch_shape``; dimensions beyond ``batch_shape``
            are mapped with :func:`jax.vmap`.
        :return: posterior with the broadcast batch shape; usable as
            ``initial_dist`` of a follow-on model.
        :rtype: MultivariateNormal
        :raises ValueError: if ``value`` does not have trailing shape
            ``(num_steps, obs_dim)``.
        """
        value, extra = self._lead_and_extra(value)

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
        """
        if tuple(other.event_shape) != self.event_shape:
            raise ValueError(
                f"other must have event_shape {self.event_shape}, "
                f"got {tuple(other.event_shape)}"
            )
        per_step = mvn_to_gaussian(_peel_event(other, 1))
        new = self._replace(_obs=self._obs + per_step.event_pad(left=self.hidden_dim))
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
        head = self._replace(_trans=trans_head, _obs=obs_head, num_steps=t)
        return self._replace(
            _init=mvn_to_gaussian(head.filter(data)),
            _trans=trans_tail,
            _obs=obs_tail,
            num_steps=self.num_steps - t,
        )

    def _sample_states(
        self,
        key: Array,
        obs_factor: Optional[Gaussian] = None,
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
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

        :param Optional[Array] key: PRNG key. The annotation follows
            :meth:`Distribution.sample`, but a key is required; ``None`` fails
            an assertion.
        :param tuple sample_shape: leading sample dimensions.
        :return: draws of shape
            ``sample_shape + batch_shape + (num_steps, obs_dim)``.
        :rtype: Array
        """
        assert key is not None
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


class IndependentHMM(Distribution):
    """
    Wrap a batch of independent single-observation HMMs into one distribution
    over vector observations.

    The base distribution has ``event_shape == (num_steps, 1)`` and batch shape
    ``shape + (obs_dim,)``; the result has ``batch_shape == shape`` and
    ``event_shape == (num_steps, obs_dim)``. :meth:`reshape_batch` requires a
    :class:`HiddenMarkovModel` base and :meth:`prefix_condition` a
    :class:`GaussianHMM` base.

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

        :param Optional[Array] key: PRNG key, passed to ``base_dist.sample``.
        :param tuple sample_shape: leading sample dimensions.
        :return: draws of shape
            ``sample_shape + batch_shape + (num_steps, obs_dim)``.
        :rtype: Array
        """
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
