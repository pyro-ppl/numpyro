# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0
from collections import namedtuple
from typing import Any, Callable, Optional

import jax
from jax import random
import jax.numpy as jnp

from numpyro.infer.initialization import init_to_uniform
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import initialize_model
from numpyro.util import identity, is_prng_key

__all__ = ["ExternalKernel", "ExternalKernelState"]


ExternalKernelState = namedtuple(
    "ExternalKernelState", ["position", "inner", "info", "rng_key"]
)
"""
Wrapper state used by :class:`ExternalKernel`.

:ivar position: the constrained-space-bound *unconstrained* position pytree
    (``dict`` keyed by site name). Exposed at the top level so
    :class:`~numpyro.infer.MCMC` can read it via ``sample_field == "position"``.
:ivar inner: the external sampler's own state object (e.g. a Blackjax
    ``MCLMCState`` or ``HMCState``). Accessible via dotted ``extra_fields``,
    e.g. ``extra_fields=("inner.logdensity",)``.
:ivar info: the most recent ``info`` returned by the external sampler's
    ``step`` function. Accessible via dotted ``extra_fields``, e.g.
    ``extra_fields=("info.is_divergent",)`` for NUTS.
:ivar rng_key: PRNG key carried through the chain so each :meth:`sample`
    call can draw a fresh sub-key.
"""


PositionDict = dict
"""Type alias for an unconstrained position dict keyed by site name."""

StepFn = Callable[[jax.Array, Any], Any]
"""External sampler step: ``(rng_key, inner_state) -> (new_inner_state, info)``."""

GetPositionFn = Callable[[Any], PositionDict]
"""Extract the position pytree from the external sampler's inner state."""

BuildKernelFn = Callable[
    [jax.Array, Callable[[PositionDict], jax.Array], PositionDict],
    Any,
]
"""``build_kernel`` callback. See :class:`ExternalKernel` for the contract."""


class ExternalKernel(MCMCKernel):
    """
    (EXPERIMENTAL INTERFACE) Generic :class:`~numpyro.infer.mcmc.MCMCKernel`
    adapter that drives an arbitrary external sampler against a NumPyro model
    without NumPyro depending on the external library.

    The caller supplies a single ``build_kernel`` callback which constructs
    the sampler-specific state and step function from a JAX log-density and
    (optionally) runs any offline warmup/adaptation. NumPyro contributes the
    rest of the pipeline: ``initialize_model``, potential-to-log-density
    negation, progress bar, postprocess back to constrained space (including
    ``deterministic`` sites), and :class:`~numpyro.infer.util.Predictive`
    integration.

    ``build_kernel`` contract::

        build_kernel(rng_key, logdensity_fn, init_position)
            -> (inner_state, step_fn, get_position)

        step_fn:      (rng_key, inner_state) -> (new_inner_state, info)
                      Matches the Blackjax ``kernel.step`` convention.
        get_position: (inner_state) -> position pytree

    Warmup / adaptation must live **inside** ``build_kernel`` (close over
    ``num_warmup`` and any other adaptation budget). When wiring up MCMC, pass
    ``num_warmup=0`` to :class:`~numpyro.infer.MCMC` — a non-zero value will
    raise ``ValueError`` from :meth:`init`.

    .. note:: Multi-chain support: ``chain_method="parallel"`` and
        ``chain_method="sequential"`` are supported because NumPyro splits the
        PRNG key and dispatches per-chain — each chain runs its own ``init``
        with an un-batched key. ``chain_method="vectorized"`` is not supported
        in this release: it would require extracting a vmappable ``step_fn``
        out of a vmap'd ``build_kernel`` call, which JAX closures don't allow
        cleanly. Vectorized callers receive a ``NotImplementedError`` with a
        pointer to the alternative chain methods.

    :param model: a NumPyro model. Either ``model`` or ``potential_fn`` is
        required, not both.
    :type model: Callable, optional
    :param potential_fn: pre-built potential energy. When supplied, the caller
        must pass ``init_params`` to :meth:`MCMC.run() <numpyro.infer.MCMC.run>`.
    :type potential_fn: Callable, optional
    :param BuildKernelFn build_kernel: required callback documented above.
    :param Callable init_strategy: per-site init strategy for the
        unconstrained starting position.
    :param diagnostics_fn: optional ``(state) -> str`` used to populate the
        progress-bar diagnostics line (mirrors :class:`~numpyro.infer.HMC`'s
        ``acc. prob`` line). Defaults to no diagnostics.
    :type diagnostics_fn: Callable, optional

    **Example** (Blackjax MCLMC end-to-end)::

        import blackjax
        from numpyro.infer import MCMC, ExternalKernel

        def build_mclmc(rng_key, logdensity_fn, init_position):
            init = blackjax.mcmc.mclmc.init(
                position=init_position,
                logdensity_fn=logdensity_fn,
                random_generator_arg=rng_key,
            )
            base = blackjax.mclmc(logdensity_fn, step_size=0.1, L=0.1)
            (state, params), _ = blackjax.mclmc_find_L_and_step_size(
                mclmc_kernel=base.step,
                num_steps=2_000,
                state=init,
                rng_key=rng_key,
            )
            final = blackjax.mclmc(
                logdensity_fn, step_size=params.step_size, L=params.L
            )
            return state, final.step, lambda s: s.position

        mcmc = MCMC(
            ExternalKernel(model, build_kernel=build_mclmc),
            num_warmup=0,
            num_samples=5_000,
            num_chains=4,
            chain_method="parallel",
        )
        mcmc.run(rng_key, x, y)
        samples = mcmc.get_samples()
    """

    def __init__(
        self,
        model: Optional[Callable] = None,
        *,
        potential_fn: Optional[Callable[[PositionDict], jax.Array]] = None,
        build_kernel: BuildKernelFn,
        init_strategy: Callable = init_to_uniform,
        diagnostics_fn: Optional[Callable[[ExternalKernelState], str]] = None,
    ) -> None:
        if (model is None) == (potential_fn is None):
            raise ValueError(
                "Exactly one of `model` or `potential_fn` must be supplied to "
                "ExternalKernel."
            )
        self._model = model
        self._potential_fn = potential_fn
        self._build_kernel = build_kernel
        self._init_strategy = init_strategy
        self._diagnostics_fn = diagnostics_fn
        self._postprocess_fn: Optional[Callable] = None
        self._get_position: Optional[GetPositionFn] = None
        # The cached step function. MCMC's _single_chain_mcmc checks
        # `getattr(self, "_sample_fn", None) is None` to decide whether to
        # re-initialize — keep the attribute named exactly this.
        self._sample_fn: Optional[StepFn] = None

    @property
    def model(self):
        """The underlying NumPyro model (``None`` if ``potential_fn`` was used)."""
        return self._model

    @property
    def sample_field(self) -> str:
        """Field of :class:`ExternalKernelState` that holds the MCMC sample."""
        return "position"

    @property
    def default_fields(self) -> tuple:
        """Fields collected by default during :meth:`MCMC.run()`."""
        return ("position",)

    def get_diagnostics_str(self, state: ExternalKernelState) -> str:
        """Forward to the user-supplied ``diagnostics_fn`` (default: empty)."""
        if self._diagnostics_fn is None:
            return ""
        return self._diagnostics_fn(state)

    def postprocess_fn(self, args, kwargs) -> Callable[[PositionDict], PositionDict]:
        """Return a single-position postprocess callable bound to model args.

        Called by :class:`~numpyro.infer.MCMC` per-sample to convert
        unconstrained positions back to the constrained space and include
        ``deterministic`` sites.
        """
        if self._postprocess_fn is None:
            return identity
        return self._postprocess_fn(*args, **kwargs)

    def init(
        self,
        rng_key: jax.Array,
        num_warmup: int,
        init_params: Optional[Any],
        model_args: tuple,
        model_kwargs: dict,
    ) -> ExternalKernelState:
        """
        Build the external sampler (running any offline warmup inside the
        user's ``build_kernel`` closure) and return the initial wrapper state.

        :raises ValueError: if ``num_warmup > 0``. Warmup must be handled
            inside ``build_kernel``; pass ``num_warmup=0`` to :class:`MCMC`.
        :raises NotImplementedError: if ``rng_key`` is batched (vectorized
            chain method). Use ``chain_method="parallel"`` or
            ``chain_method="sequential"`` instead.
        """
        if num_warmup > 0:
            raise ValueError(
                "ExternalKernel performs all warmup/adaptation inside the "
                "user-supplied `build_kernel` callback. Pass `num_warmup=0` "
                "to MCMC and bake the warmup budget into the closure of "
                "`build_kernel`."
            )
        if not is_prng_key(rng_key):
            raise NotImplementedError(
                "ExternalKernel does not support `chain_method='vectorized'` "
                "in this release. Use `chain_method='parallel'` or "
                "`chain_method='sequential'` for multi-chain sampling."
            )

        key_model, key_build, key_eval, key_state = random.split(rng_key, 4)

        if self._model is not None:
            model_info = initialize_model(
                key_model,
                self._model,
                init_strategy=self._init_strategy,
                dynamic_args=True,
                model_args=model_args,
                model_kwargs=model_kwargs,
            )
            init_position = (
                init_params if init_params is not None else model_info.param_info.z
            )
            bound_potential = model_info.potential_fn(*model_args, **model_kwargs)
            self._postprocess_fn = model_info.postprocess_fn
        else:
            if init_params is None:
                raise ValueError(
                    "`init_params` is required when ExternalKernel is "
                    "constructed with `potential_fn`."
                )
            init_position = init_params
            bound_potential = self._potential_fn
            self._postprocess_fn = None

        def logdensity_fn(position: PositionDict) -> jax.Array:
            return -bound_potential(position)

        inner, step_fn, get_position = self._build_kernel(
            key_build, logdensity_fn, init_position
        )
        self._sample_fn = step_fn
        self._get_position = get_position

        info_shape = jax.eval_shape(step_fn, key_eval, inner)[1]
        info = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), info_shape)

        return ExternalKernelState(
            position=get_position(inner),
            inner=inner,
            info=info,
            rng_key=key_state,
        )

    def sample(
        self,
        state: ExternalKernelState,
        model_args: tuple,
        model_kwargs: dict,
    ) -> ExternalKernelState:
        """Advance the chain by one step of the external sampler."""
        rng_key, step_key = random.split(state.rng_key)
        new_inner, info = self._sample_fn(step_key, state.inner)
        assert self._get_position is not None
        return ExternalKernelState(
            position=self._get_position(new_inner),
            inner=new_inner,
            info=info,
            rng_key=rng_key,
        )
