# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from typing import Callable, Optional, Self, Sequence, Union, overload

import jax
from jax import Array, lax, random
import jax.numpy as jnp

from numpyro.distributions import constraints
from numpyro.distributions.continuous import MultivariateNormal
from numpyro.distributions.distribution import Distribution, Independent
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


@overload
def _with_batch_rank(factor: Gaussian, rank: int) -> Gaussian: ...
@overload
def _with_batch_rank(factor: AffineNormal, rank: int) -> AffineNormal: ...
def _with_batch_rank(factor: Factor, rank: int) -> Factor:
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
    """Remove ``n`` reinterpreted batch dimensions from nested ``Independent`` layers."""
    while n > 0:
        if not isinstance(d, Independent):
            raise ValueError(
                f"cannot remove {n} event dimensions from {type(d).__name__}"
            )
        k = min(n, d.reinterpreted_batch_ndims)
        remaining = d.reinterpreted_batch_ndims - k
        d = d.base_dist if remaining == 0 else Independent(d.base_dist, remaining)
        n -= k
    return d


def _vmap_leading(fn: Callable, ndim: int) -> Callable:
    for _ in range(ndim):
        fn = jax.vmap(fn)
    return fn


def _time_shape(*shapes: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    shape = lax.broadcast_shapes(*shapes)
    return shape[:-1], shape[-1]


def _resolve_num_steps(time: int, num_steps: Optional[int]) -> int:
    if time == 1:
        if num_steps is None:
            raise ValueError(
                "num_steps is required when all parameters are time-homogeneous"
            )
        return int(num_steps)
    if num_steps is not None and num_steps != time:
        raise ValueError(
            f"num_steps={num_steps} conflicts with the parameters' time axis "
            f"of size {time}"
        )
    return time


class HiddenMarkovModel(Distribution):
    """
    Base class for distributions over observation sequences with the latent
    chain marginalized by factor reduction.

    Subclasses store three factors: ``_init`` over ``z_0``, ``_trans`` over
    ``(z_{t-1}, z_t)`` and ``_obs`` over ``(z_t, x_t)``. Time is the rightmost
    batch axis of the per-step factors (size 1 when time-homogeneous). Shapes
    are derived from the factors, so instances built under :func:`jax.vmap` or
    carried through :func:`jax.lax.scan` report the mapped batch shape.
    """

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
        batch_shape = lax.broadcast_shapes(self.batch_shape, tuple(batch_shape))
        return self._replace(_init=self._init.expand(batch_shape))

    def reshape_batch(self, batch_shape: Sequence[int]) -> Self:
        """
        Reshape the batch dimensions (same number of elements), e.g. to append
        a singleton batch axis.
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

    def _time_expanded(self, factor: Gaussian) -> Gaussian:
        return factor.expand(factor.batch_shape[:-1] + (self.num_steps,))

    def _lead_and_extra(self, value: Array) -> tuple[Array, int]:
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
    :meth:`filter` and sampling run in ``O(log num_steps)`` parallel depth.

    Precision: the information form loses accuracy when the ratio between the
    largest and smallest noise precision within a step is large. float32 is
    adequate for ratios below about ``1e3``; otherwise call
    :func:`numpyro.enable_x64`.

    Parameters
    ----------
    initial_dist : Distribution
        ``MultivariateNormal`` or ``Independent(Normal, 1)`` over ``z_0`` with
        ``event_shape == (hidden_dim,)``.
    transition_matrix : Array
        Shape broadcastable to ``batch_shape + (num_steps, hidden_dim, hidden_dim)``.
    transition_dist : Distribution
        Process noise with ``event_shape == (hidden_dim,)``.
    observation_matrix : Array
        Shape broadcastable to ``batch_shape + (num_steps, obs_dim, hidden_dim)``.
    observation_dist : Distribution
        Observation noise with ``event_shape == (obs_dim,)``.
    num_steps : int, optional
        Length of the time axis; required when every per-step parameter is
        time-homogeneous.
    """

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
                    f"{name} must have event_shape {expected}, "
                    f"got {tuple(d.event_shape)}"
                )
        batch_shape, time = _time_shape(
            tuple(initial_dist.batch_shape) + (1,),
            transition_matrix.shape[:-2],
            tuple(transition_dist.batch_shape),
            observation_matrix.shape[:-2],
            tuple(observation_dist.batch_shape),
        )
        super().__init__(
            mvn_to_gaussian(initial_dist),
            matrix_and_mvn_to_gaussian(transition_matrix, transition_dist),
            matrix_and_mvn_to_gaussian(observation_matrix, observation_dist),
            _resolve_num_steps(time, num_steps),
            validate_args=validate_args,
        )

    def _posterior(self, value: Array) -> Gaussian:
        """Factor over ``z_T`` given ``value`` of shape ``batch_shape + (num_steps, obs_dim)``."""
        logp = self._trans + self._obs.condition(value).event_pad(left=self.hidden_dim)
        logp = sequential_gaussian_tensordot(self._time_expanded(logp))
        return gaussian_tensordot(self._init, logp, self.hidden_dim)

    @validate_sample
    def log_prob(self, value: Array) -> Array:
        value, extra = self._lead_and_extra(value)
        return _vmap_leading(lambda v: self._posterior(v).event_logsumexp(), extra)(
            value
        )

    def filter(self, value: Array) -> MultivariateNormal:
        """
        Posterior over the final state ``z_T`` given the full observation sequence.

        Returns
        -------
        MultivariateNormal
            Batch shape broadcast of ``value`` and ``batch_shape``; usable as
            ``initial_dist`` of a follow-on model.
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

        Parameters
        ----------
        other : Distribution
            ``Independent(Normal, 2)`` or ``Independent(MultivariateNormal, 1)``
            (possibly expanded) with ``event_shape == (num_steps, obs_dim)``.

        Returns
        -------
        tuple[GaussianHMM, Array]
            ``(updated, log_normalizer)`` such that
            ``self.log_prob(x) + other.log_prob(x) == updated.log_prob(x) + log_normalizer``.
        """
        if tuple(other.event_shape) != self.event_shape:
            raise ValueError(
                f"other must have event_shape {self.event_shape}, "
                f"got {tuple(other.event_shape)}"
            )
        per_step = mvn_to_gaussian(_peel_event(other, 1))
        new = self._replace(_obs=self._obs + per_step.event_pad(left=self.hidden_dim))
        logp = new._trans + new._obs.marginalize(right=self.obs_dim).event_pad(
            left=self.hidden_dim
        )
        logp = sequential_gaussian_tensordot(new._time_expanded(logp))
        log_normalizer = gaussian_tensordot(
            new._init, logp, self.hidden_dim
        ).event_logsumexp()
        return new._replace(_init=new._init - log_normalizer), log_normalizer

    def prefix_condition(self, data: Array) -> GaussianHMM:
        """
        Condition on the first ``t < num_steps`` observations and return the
        model over the remaining steps.

        Parameters
        ----------
        data : Array
            Shape ``(..., t, obs_dim)`` with ``0 < t < num_steps``.

        Returns
        -------
        GaussianHMM
            Model over ``num_steps - t`` steps whose initial distribution is
            the filtered posterior.
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
        self, key: Array, obs_factor: Gaussian, sample_shape: tuple[int, ...]
    ) -> Array:
        trans = self._time_expanded(
            self._trans + obs_factor.event_pad(left=self.hidden_dim)
        )
        return sequential_gaussian_filter_sample(key, self._init, trans, sample_shape)[
            ..., 1:, :
        ]

    def sample(self, key: Optional[Array], sample_shape: tuple[int, ...] = ()) -> Array:
        """
        Sample observation sequences ``x_{1:T}`` with the latent states integrated out.

        Parameters
        ----------
        key : Array
            PRNG key.
        sample_shape : tuple[int, ...]
            Leading sample dimensions.

        Returns
        -------
        Array
            Shape ``sample_shape + batch_shape + (num_steps, obs_dim)``.
        """
        assert key is not None
        key_z, key_x = random.split(key)
        z = self._sample_states(
            key_z, self._obs.marginalize(right=self.obs_dim), sample_shape
        )
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

        Parameters
        ----------
        key : Array
            PRNG key.
        value : Array
            Observations of shape ``lead + (num_steps, obs_dim)``.
        sample_shape : tuple[int, ...]
            Leading sample dimensions.

        Returns
        -------
        Array
            Shape ``sample_shape + lead + (num_steps, hidden_dim)`` where
            ``lead`` broadcasts ``value`` against ``batch_shape``.
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
    ``event_shape == (num_steps, obs_dim)``.

    Parameters
    ----------
    base_dist : GaussianHMM
        Batched model with a trailing batch dimension of size ``obs_dim`` and a
        unit observation dimension.
    """

    arg_constraints = {}
    pytree_data_fields = ("base_dist",)

    def __init__(
        self, base_dist: GaussianHMM, *, validate_args: Optional[bool] = None
    ) -> None:
        if not base_dist.batch_shape or tuple(base_dist.event_shape)[-1:] != (1,):
            raise ValueError(
                "base_dist must be batched and have a unit observation dimension"
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
    def support(self) -> constraints.Constraint:
        return self.base_dist.support

    @property
    def has_rsample(self) -> bool:
        return self.base_dist.has_rsample

    @property
    def num_steps(self) -> int:
        return self.base_dist.num_steps

    def sample(self, key: Optional[Array], sample_shape: tuple[int, ...] = ()) -> Array:
        return jnp.swapaxes(self.base_dist.sample(key, sample_shape)[..., 0], -1, -2)

    @validate_sample
    def log_prob(self, value: Array) -> Array:
        return self.base_dist.log_prob(jnp.swapaxes(value, -1, -2)[..., None]).sum(-1)

    def expand(self, batch_shape: Sequence[int]) -> IndependentHMM:
        obs = self.base_dist.batch_shape[-1:]
        return IndependentHMM(self.base_dist.expand(tuple(batch_shape) + obs))

    def reshape_batch(self, batch_shape: Sequence[int]) -> IndependentHMM:
        obs = self.base_dist.batch_shape[-1:]
        return IndependentHMM(self.base_dist.reshape_batch(tuple(batch_shape) + obs))

    def prefix_condition(self, data: Array) -> IndependentHMM:
        """
        Condition on a prefix of observations (see
        :meth:`GaussianHMM.prefix_condition`).
        """
        prefix = jnp.swapaxes(data, -1, -2)[..., None]
        return IndependentHMM(self.base_dist.prefix_condition(prefix))
