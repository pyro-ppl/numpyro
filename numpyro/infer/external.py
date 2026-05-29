# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, NamedTuple, Optional

import jax
from jax import random
import jax.numpy as jnp

from numpyro.infer.initialization import init_to_uniform
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import ParamInfo, get_log_density_fn
from numpyro.util import identity, is_prng_key

__all__ = ["ExternalKernel", "ExternalKernelState"]


PositionDict = dict[str, jax.Array]
"""Type alias for an unconstrained position dict keyed by site name."""

StepFn = Callable[[jax.Array, Any], Any]
"""External sampler step: ``(rng_key, inner_state) -> (new_inner_state, info)``."""

GetPositionFn = Callable[[Any], PositionDict]
"""Extract the position pytree from the external sampler's inner state."""

BuildKernelFn = Callable[
    [jax.Array, Callable[[PositionDict], jax.Array], PositionDict],
    tuple[Any, StepFn, GetPositionFn],
]
"""``build_kernel`` callback. See :class:`ExternalKernel` for the contract."""


class ExternalKernelState(NamedTuple):
    """
    Wrapper state used by :class:`ExternalKernel`.

    :ivar position: unconstrained position pytree emitted by the external
        sampler (``dict`` keyed by site name). Exposed at the top level so
        :class:`~numpyro.infer.MCMC` can read it via
        ``sample_field == "position"``; :class:`~numpyro.infer.MCMC` later
        applies ``postprocess_fn`` to recover constrained samples plus
        ``deterministic`` sites.
    :ivar inner: the external sampler's own state object (e.g. a Blackjax
        ``MCLMCState`` or ``HMCState``). Accessible via dotted
        ``extra_fields``, e.g. ``extra_fields=("inner.logdensity",)``.
    :ivar info: the most recent ``info`` returned by the external sampler's
        ``step`` function. Accessible via dotted ``extra_fields``, e.g.
        ``extra_fields=("info.is_divergent",)`` for NUTS.
    :ivar rng_key: PRNG key carried through the chain so each :meth:`sample`
        call can draw a fresh sub-key.
    """

    position: PositionDict
    inner: Any
    info: Any
    rng_key: jax.Array


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
        from blackjax.mcmc.integrators import isokinetic_mclachlan
        from jax import random
        from numpyro.infer import MCMC, ExternalKernel

        def build_mclmc(rng_key, logdensity_fn, init_position):
            key_init, key_tune = random.split(rng_key)
            init_state = blackjax.mcmc.mclmc.init(
                position=init_position,
                logdensity_fn=logdensity_fn,
                rng_key=key_init,
            )

            def kernel_factory(inverse_mass_matrix):
                return blackjax.mcmc.mclmc.build_kernel(
                    logdensity_fn=logdensity_fn,
                    inverse_mass_matrix=inverse_mass_matrix,
                    integrator=isokinetic_mclachlan,
                )

            state, params, _ = blackjax.mclmc_find_L_and_step_size(
                mclmc_kernel=kernel_factory,
                num_steps=2_000,
                state=init_state,
                rng_key=key_tune,
            )
            final_kernel = kernel_factory(params.inverse_mass_matrix)

            def step_fn(rng_key, state):
                return final_kernel(rng_key, state, params.L, params.step_size)

            return state, step_fn, lambda s: s.position

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
        init_params: Any,
        model_args: tuple,
        model_kwargs: dict,
    ) -> ExternalKernelState:
        """
        Build the external sampler (running any offline warmup inside the
        user's ``build_kernel`` closure) and return the initial wrapper state.

        ``model_args`` / ``model_kwargs`` are forwarded by
        :class:`~numpyro.infer.MCMC`; they are bound into the log-density
        function passed to ``build_kernel``.

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

        key_model, key_build, key_state = random.split(rng_key, 3)
        init_position, logdensity_fn = self._setup_logdensity(
            key_model, init_params, model_args, model_kwargs
        )
        inner, step_fn, get_position = self._call_build_kernel(
            key_build, logdensity_fn, init_position
        )
        self._sample_fn = step_fn
        self._get_position = get_position
        info = self._zero_info_placeholder(step_fn, inner)

        return ExternalKernelState(
            position=get_position(inner),
            inner=inner,
            info=info,
            rng_key=key_state,
        )

    def _setup_logdensity(
        self,
        rng_key: jax.Array,
        init_params: Any,
        model_args: tuple,
        model_kwargs: dict,
    ) -> tuple[PositionDict, Callable[[PositionDict], jax.Array]]:
        """Resolve the unconstrained initial position and bind ``logdensity_fn``.

        For the ``model`` branch this delegates to :func:`get_log_density_fn`
        so the binding logic stays in one place. For the ``potential_fn``
        branch the user-supplied function is used verbatim, with ``identity``
        as the postprocess.
        """
        if self._model is not None:
            info = get_log_density_fn(
                rng_key,
                self._model,
                model_args=model_args,
                model_kwargs=model_kwargs,
                init_strategy=self._init_strategy,
            )
            self._postprocess_fn = info.model_info.postprocess_fn
            init_position = self._resolve_init_position(
                init_params, default=info.init_position
            )
            return init_position, info.logdensity_fn

        if init_params is None:
            raise ValueError(
                "`init_params` is required when ExternalKernel is "
                "constructed with `potential_fn`."
            )
        self._postprocess_fn = None
        init_position = self._resolve_init_position(init_params, default=None)
        potential_fn = self._potential_fn
        assert potential_fn is not None  # guaranteed by __init__

        def logdensity_fn(position: PositionDict) -> jax.Array:
            return -potential_fn(position)

        return init_position, logdensity_fn

    @staticmethod
    def _resolve_init_position(
        init_params: Any, *, default: Optional[PositionDict]
    ) -> PositionDict:
        """Unwrap a :class:`ParamInfo` or pass through a raw dict.

        Mirrors :class:`~numpyro.infer.HMC`'s convention at
        ``numpyro/infer/hmc.py:763`` so users passing the output of
        :func:`initialize_model` work seamlessly.
        """
        if init_params is None:
            if default is None:
                raise ValueError("`init_params` cannot be None.")
            return default
        if isinstance(init_params, ParamInfo):
            return init_params[0]
        return init_params

    def _call_build_kernel(
        self,
        rng_key: jax.Array,
        logdensity_fn: Callable[[PositionDict], jax.Array],
        init_position: PositionDict,
    ) -> tuple[Any, StepFn, GetPositionFn]:
        """Invoke the user callback and surface a clear error on shape mismatch."""
        result = self._build_kernel(rng_key, logdensity_fn, init_position)
        try:
            inner, step_fn, get_position = result
        except (TypeError, ValueError) as e:
            raise TypeError(
                "`build_kernel` must return a 3-tuple "
                "`(inner_state, step_fn, get_position)`; got "
                f"{type(result).__name__!s} with len="
                f"{len(result) if hasattr(result, '__len__') else 'N/A'}."
            ) from e
        return inner, step_fn, get_position

    @staticmethod
    def _zero_info_placeholder(step_fn: StepFn, inner: Any) -> Any:
        """Allocate a zero-filled ``info`` pytree with the right structure.

        ``fori_collect`` sizes its collection buffer from the *initial* state
        passed to it, so the placeholder needs matching shape/dtype but never
        appears in the collected output (the first stored sample is post-step).
        ``jax.eval_shape`` abstracts ``rng_key`` and ``inner``, so the key
        value does not matter; we pass ``random.key(0)`` as a dummy.
        """
        info_shape = jax.eval_shape(step_fn, random.key(0), inner)[1]
        return jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), info_shape)

    def sample(
        self,
        state: ExternalKernelState,
        model_args: tuple,
        model_kwargs: dict,
    ) -> ExternalKernelState:
        """Advance the chain by one step of the external sampler.

        ``model_args`` / ``model_kwargs`` are accepted for
        :class:`~numpyro.infer.mcmc.MCMCKernel` interface compatibility but
        unused — the log-density was bound at :meth:`init` time.
        """
        rng_key, step_key = random.split(state.rng_key)
        new_inner, info = self._sample_fn(step_key, state.inner)
        return ExternalKernelState(
            position=self._get_position(new_inner),
            inner=new_inner,
            info=info,
            rng_key=rng_key,
        )
