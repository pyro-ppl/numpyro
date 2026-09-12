# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Composable Gibbs kernels: a composite :class:`Gibbs` kernel that updates the latent sites of a
model block by block, the generic block kernels :class:`CustomGibbs` and :class:`DiscreteGibbs`,
and the pure helpers they share with the HMC-within-Gibbs kernels
(:mod:`numpyro.infer.hmc_gibbs`, :mod:`numpyro.infer.mixed_hmc`).
"""

from collections import OrderedDict
from collections.abc import Callable, Sequence
import copy
from functools import partial, reduce
from typing import Any, NamedTuple, Protocol, TypeAlias

import numpy as np

import jax
from jax import random
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
from jax.scipy.special import expit

from numpyro._typing import (
    ConstrainFn,
    ModelArgs,
    ModelKwargs,
    ModelT,
    PotentialFn,
    PyTree,
    SiteValues,
    TraceT,
)
from numpyro.handlers import condition, seed, substitute, trace
from numpyro.infer.hmc import HMC
from numpyro.infer.initialization import init_to_sample
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import (
    _prepare_model_for_potential,
    _transforms_from_trace,
    potential_energy,
    transform_fn,
)
from numpyro.util import cond, fori_loop, identity, is_prng_key

__all__ = [
    "CustomGibbs",
    "CustomGibbsState",
    "DiscreteGibbs",
    "DiscreteGibbsState",
    "GIBBS_SITES_KWARG",
    "Gibbs",
    "GibbsState",
    "conditioned",
    "discrete_latent_sites",
    "with_conditioning",
]

GIBBS_SITES_KWARG: str = "_gibbs_sites"
"""Reserved model keyword through which conditioning values travel."""

ModelWrapper: TypeAlias = Callable[[ModelT], ModelT]
"""Maps a model to a model with the same call signature (conditioning, likelihood estimation)."""

SiteSelector: TypeAlias = Callable[[TraceT], Sequence[str]]
"""Picks site names from a prototype trace, e.g. :func:`discrete_latent_sites`."""

SitesSpec: TypeAlias = Sequence[str] | SiteSelector | None
"""How a block declares its sites: explicit names, a selector, or `None` for the remainder."""


class GibbsUpdateFn(Protocol):
    """
    Signature of the user callable of :class:`~numpyro.infer.gibbs.CustomGibbs` /
    :class:`~numpyro.infer.hmc_gibbs.HMCGibbs`. Called with keywords only; `hmc_sites` holds
    the constrained values of every conditioning site (the name is kept for compatibility).
    """

    def __call__(
        self, *, rng_key: jax.Array, gibbs_sites: SiteValues, hmc_sites: SiteValues
    ) -> SiteValues: ...


def _conditioned_model(model: ModelT, *args: Any, **kwargs: Any) -> Any:
    """Module-level target of :func:`conditioned` (kept module-level so kernels pickle)."""
    values = kwargs.pop(GIBBS_SITES_KWARG, {})
    with condition(data=values), substitute(data=values):
        return model(*args, **kwargs)


def conditioned(model: ModelT) -> ModelT:
    """
    Return a model that pops :data:`GIBBS_SITES_KWARG` from its keyword arguments and runs
    `model` under `condition(data=values)` and `substitute(data=values)`. Idempotent: wrapping
    an already conditioned model returns it unchanged.

    :param model: the model.
    :return: the conditioned model.
    """
    if isinstance(model, partial) and model.func is _conditioned_model:
        return model
    return partial(_conditioned_model, model)


def with_conditioning(
    model_kwargs: ModelKwargs | None,
    values: SiteValues,
    *,
    allowed: frozenset[str] | None = None,
) -> ModelKwargs:
    """
    Return a copy of `model_kwargs` whose :data:`GIBBS_SITES_KWARG` entry is the existing
    entry (if any) extended by `values`. Extending rather than replacing is what makes nested
    composites correct: at any depth a block sees exactly the sites it does not own.

    :param model_kwargs: keyword arguments of the model.
    :param values: conditioning values to add.
    :param allowed: when given, raise if a key of `values` is outside this set.
    :return: a new keyword argument dict.
    """
    if allowed is not None:
        extra = set(values) - allowed
        if extra:
            raise ValueError(f"Cannot condition on non-latent sites {sorted(extra)}.")
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    model_kwargs[GIBBS_SITES_KWARG] = {
        **model_kwargs.get(GIBBS_SITES_KWARG, {}),
        **values,
    }
    return model_kwargs


def prototype_trace(
    model: ModelT,
    rng_key: jax.Array,
    model_args: ModelArgs,
    model_kwargs: ModelKwargs | None,
) -> TraceT:
    """
    Trace `model` once with values drawn by :func:`~numpyro.infer.initialization.init_to_sample`
    (which also handles sites without a `sample` method, such as `ImproperUniform`). Callers
    must not store the returned trace on a kernel object (it may hold tracers under
    `pmap`/`vmap`); store only static metadata derived from it.

    :param model: the model.
    :param rng_key: random key used to draw the prototype values.
    :param tuple model_args: arguments provided to the model.
    :param dict model_kwargs: keyword arguments provided to the model.
    :return: the trace.
    """
    model_kwargs = {} if model_kwargs is None else model_kwargs
    return trace(
        substitute(seed(model, rng_key), substitute_fn=init_to_sample)
    ).get_trace(*model_args, **model_kwargs)


def latent_sample_sites(model_trace: TraceT) -> tuple[str, ...]:
    """Names of unobserved `sample` sites in trace order."""
    return tuple(
        name
        for name, site in model_trace.items()
        if site["type"] == "sample" and not site["is_observed"]
    )


def discrete_latent_sites(model_trace: TraceT) -> tuple[str, ...]:
    """
    Unobserved sample sites whose distribution has enumerate support and which are not
    marked `infer={"enumerate": "parallel"}`. Usable as a :data:`SitesSpec` selector.
    """
    return tuple(
        name
        for name, site in model_trace.items()
        if site["type"] == "sample"
        and not site["is_observed"]
        and site["fn"].has_enumerate_support
        and site["infer"].get("enumerate", "") != "parallel"
    )


def discrete_support_sizes(
    model_trace: TraceT, sites: Sequence[str]
) -> dict[str, np.ndarray]:
    """Per-site support sizes broadcast to the site's shape, as static `numpy` arrays."""
    return {
        name: np.broadcast_to(
            model_trace[name]["fn"].enumerate_support(False).shape[0],
            jnp.shape(model_trace[name]["value"]),
        )
        for name in sites
    }


def _flat_support_sizes(model_trace: TraceT, sites: Sequence[str]) -> np.ndarray:
    """Support sizes flattened in :func:`ravel_pytree` leaf order, as a static numpy array.

    Built with pure numpy so that it stays concrete when `init` runs under a staging
    trace such as :func:`jax.pmap` (`jnp` operations would produce tracers there).
    """
    sizes = discrete_support_sizes(model_trace, sites)
    return np.concatenate([np.ravel(leaf) for leaf in jax.tree.leaves(sizes)])


def subsample_plate_sizes(model_trace: TraceT) -> dict[str, tuple[int, int]]:
    """`{plate_name: (size, subsample_size)}` for plates with `size > subsample_size`."""
    return {
        name: site["args"]
        for name, site in model_trace.items()
        if site["type"] == "plate"
        and (site["args"][1] is not None)
        and site["args"][0] > site["args"][1]
    }


def any_changed(old: PyTree, new: PyTree) -> jax.Array:
    """Scalar boolean: whether any leaf of two pytrees with the same structure differs."""
    flags = [
        jnp.any(a != b)
        for a, b in zip(jax.tree.leaves(old), jax.tree.leaves(new), strict=True)
    ]
    if not flags:
        return jnp.array(False)
    return reduce(jnp.logical_or, flags)


class GibbsState(NamedTuple):
    """
    - **z** - dict of the current values of all latent sites, each in the native representation
      of the block that owns it (unconstrained for HMC blocks, constrained for
      :class:`DiscreteGibbs` and :class:`CustomGibbs` blocks). Written once per step from the
      block states; never read back by :meth:`Gibbs.sample`.
    - **block_states** - tuple with one state pytree per block, in block order. Source of truth
      during a sweep. Addressable from `extra_fields` as `"block_states.<i>.<field>"`.
    - **rng_key** - random key for the next step.
    """

    z: SiteValues
    block_states: tuple[PyTree, ...]
    rng_key: jax.Array


def _as_arrays(values: SiteValues) -> SiteValues:
    """Canonicalize values to arrays so that `cond` branches carrying them agree on dtypes."""
    return {k: jnp.asarray(v) for k, v in values.items()}


def _or(flags: Sequence[Any]) -> Any:
    """
    Logical or of Python bools and traced booleans. Returns a Python bool when the result is
    statically known, so that callers can skip emitting a `cond`.
    """
    traced = [f for f in flags if not isinstance(f, bool)]
    if any(f is True for f in flags):
        return True
    if not traced:
        return False
    return reduce(jnp.logical_or, traced)


def _maybe_refresh(
    kernel: MCMCKernel,
    state: PyTree,
    changed: Any,
    model_args: ModelArgs,
    model_kwargs: ModelKwargs,
) -> PyTree:
    if changed is False:
        return state
    if changed is True:
        return kernel.refresh(state, model_args, model_kwargs)
    return cond(
        changed,
        state,
        lambda s: kernel.refresh(s, model_args, model_kwargs),
        state,
        identity,
    )


def _has_model(kernel: MCMCKernel) -> bool:
    return getattr(kernel, "model", None) is not None


class Gibbs(MCMCKernel):
    """
    Composite kernel that updates the latent sites of a model block by block. Each block is a
    `(kernel, sites)` pair; the block kernel is run on the model conditioned on the current
    values of every site it does not own. Blocks are visited in order once per MCMC step.

    :param blocks: sequence of `(kernel, sites)`. `kernel` is any
        :class:`~numpyro.infer.mcmc.MCMCKernel` that implements
        :meth:`~numpyro.infer.mcmc.MCMCKernel.refresh` (and
        :meth:`~numpyro.infer.mcmc.MCMCKernel.wrap_model` if it holds a model). `sites` is a
        sequence of site names, a callable mapping a prototype trace to site names (for example
        :func:`~numpyro.infer.gibbs.discrete_latent_sites`), or `None` for "all latent
        sample sites not owned by another block" (allowed for at most one block). All
        model-based blocks must be built on the same model callable.

    Validation at construction: at least one block; kernels override `refresh`; model-based
    kernels share one model; each model-based kernel is rebound once with
    ``kernel.wrap_model(conditioned)``.

    Validation at `init` (errors, never warnings, because the fallbacks inside
    :func:`~numpyro.infer.util.initialize_model` are silent): the union of block sites equals
    the set of unobserved sample sites minus those marked `enumerate="parallel"`; no discrete
    latent is left to an HMC block; blocks are disjoint and non-empty; no site is already
    conditioned by an enclosing composite; `init_params` keys lie in the union (unconstrained
    values for HMC blocks, constrained values otherwise); the model has no subsample plates
    (use :class:`~numpyro.infer.hmc_gibbs.HMCECS`); HMC blocks have no value-dependent
    supports.

    .. note:: Each HMC block adapts its step size and mass matrix on its own conditional, so
        blocks should leave `find_heuristic_step_size=False` (the heuristic binds the potential
        to the initial conditioning). Strongly correlated continuous sites belong in one HMC
        block.

    **Example**

    .. doctest::

        >>> from jax import random
        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.infer import MCMC, NUTS, Gibbs, DiscreteGibbs, CustomGibbs
        >>> from numpyro.infer.gibbs import discrete_latent_sites
        ...
        >>> def model(probs, locs):
        ...     c = numpyro.sample("c", dist.Categorical(probs))
        ...     x = numpyro.sample("x", dist.Normal(locs[c], 0.5))
        ...     y = numpyro.sample("y", dist.Normal(0.0, 2.0))
        ...     numpyro.sample("obs", dist.Normal(x + y, 1.0), obs=jnp.array([1.0]))
        ...
        >>> def gibbs_fn(rng_key, gibbs_sites, hmc_sites):
        ...     x = hmc_sites["x"]
        ...     return {"y": dist.Normal(0.8 * (1 - x), jnp.sqrt(0.8)).sample(rng_key)}
        ...
        >>> kernel = Gibbs([
        ...     (DiscreteGibbs(model), discrete_latent_sites),
        ...     (CustomGibbs(gibbs_fn), ["y"]),
        ...     (NUTS(model), None),
        ... ])
        >>> mcmc = MCMC(kernel, num_warmup=100, num_samples=100, progress_bar=False)
        >>> mcmc.run(random.key(0), jnp.array([0.15, 0.3, 0.3, 0.25]), jnp.array([-2.0, 0.0, 2.0, 4.0]))
        >>> mcmc.print_summary()  # doctest: +SKIP
    """

    _state_cls: type[GibbsState] = GibbsState
    sample_field: str = "z"

    def __init__(self, blocks: Sequence[tuple[MCMCKernel, SitesSpec]]) -> None:
        blocks = list(blocks)
        if not blocks:
            raise ValueError("Gibbs requires at least one block.")
        kernels, specs = [], []
        for block in blocks:
            if not (isinstance(block, tuple) and len(block) == 2):
                raise ValueError("Each block must be a `(kernel, sites)` pair.")
            kernel, spec = block
            if not isinstance(kernel, MCMCKernel):
                raise ValueError(f"{kernel!r} is not an MCMCKernel.")
            if type(kernel).refresh is MCMCKernel.refresh:
                raise ValueError(
                    f"{type(kernel).__name__} does not implement `refresh` and cannot be "
                    "used as a block of Gibbs."
                )
            if hasattr(kernel, "model") and kernel.model is None:
                raise ValueError(
                    "Kernels built from a potential function cannot be blocks of Gibbs."
                )
            if spec is not None and not callable(spec):
                spec = tuple(spec)
                if not spec or not all(isinstance(s, str) for s in spec):
                    raise ValueError(
                        "`sites` must be a non-empty sequence of site names, a callable "
                        "or None."
                    )
            kernels.append(kernel)
            specs.append(spec)
        if sum(spec is None for spec in specs) > 1:
            raise ValueError("At most one block can use `None` for its sites.")
        models = [kernel.model for kernel in kernels if _has_model(kernel)]
        if not models:
            raise ValueError("Gibbs requires at least one block built on a model.")
        if any(model != models[0] for model in models[1:]):
            raise ValueError("All model-based blocks must share the same model.")
        self._model = models[0]
        self._kernels = tuple(
            kernel.wrap_model(conditioned) if _has_model(kernel) else kernel
            for kernel in kernels
        )
        self._specs = tuple(specs)
        # static metadata resolved at `init`
        self._sites: tuple[tuple[str, ...], ...] = ()
        self._has_deterministic = False
        self._sample_fn = None

    @property
    def model(self) -> ModelT | None:
        """The shared, unwrapped model of the model-based blocks."""
        return self._model

    @property
    def blocks(self) -> tuple[tuple[MCMCKernel, SitesSpec], ...]:
        """The `(kernel, sites)` pairs; kernels are the conditioned copies actually run."""
        return tuple(zip(self._kernels, self._specs))

    @property
    def default_fields(self) -> tuple[str, ...]:
        return ("z",)

    def get_diagnostics_str(self, state: GibbsState) -> str:
        parts = [
            kernel.get_diagnostics_str(block_state)
            for kernel, block_state in zip(self._kernels, state.block_states)
        ]
        return " | ".join(part for part in parts if part)

    def _resolve_partition(
        self, model_trace: Any, enclosing: frozenset[str]
    ) -> tuple[tuple[str, ...], ...]:
        latent = tuple(
            name
            for name in latent_sample_sites(model_trace)
            if model_trace[name]["infer"].get("enumerate", "") != "parallel"
        )
        latent_set = frozenset(latent)
        sites: list[tuple[str, ...] | None] = []
        for spec in self._specs:
            if spec is None:
                sites.append(None)
                continue
            names = tuple(spec(model_trace)) if callable(spec) else spec
            unknown = [name for name in names if name not in latent_set]
            if unknown:
                hint = (
                    " (already conditioned by an enclosing kernel)"
                    if any(name in enclosing for name in unknown)
                    else ""
                )
                raise ValueError(
                    f"Sites {unknown} are not latent sample sites of the model{hint}."
                )
            sites.append(names)
        owned = [name for names in sites if names is not None for name in names]
        duplicates = sorted({name for name in owned if owned.count(name) > 1})
        if duplicates:
            raise ValueError(f"Sites {duplicates} are owned by more than one block.")
        remainder = tuple(name for name in latent if name not in owned)
        # the remainder block may own no sites (for example an HMC block on a model whose
        # latent sites are all discrete); explicit blocks may not
        if None in sites:
            sites[sites.index(None)] = remainder
        elif remainder:
            raise ValueError(
                f"Latent sites {list(remainder)} are not owned by any block; add a block "
                "or use `None` as the sites of one block."
            )
        resolved = tuple(names if names is not None else () for names in sites)
        for kernel, spec, names in zip(self._kernels, self._specs, resolved):
            if not names and spec is not None:
                raise ValueError(f"Block {type(kernel).__name__} owns no sites.")
            if isinstance(kernel, HMC):
                discrete = [
                    name
                    for name in names
                    if model_trace[name]["fn"].support.is_discrete
                ]
                if discrete:
                    raise ValueError(
                        f"Discrete latent sites {discrete} cannot be sampled by an HMC "
                        "block; use DiscreteGibbs or mark them with "
                        "`infer={'enumerate': 'parallel'}`."
                    )
        return resolved

    @staticmethod
    def _block_trace(model_trace: Any, names: Sequence[str]) -> Any:
        """The trace restricted to the block's sample sites (other site types are kept)."""
        return OrderedDict(
            (name, site)
            for name, site in model_trace.items()
            if site["type"] != "sample" or name in names
        )

    def _split_init_params(
        self, init_params: SiteValues | None
    ) -> tuple[SiteValues | None, ...]:
        if not init_params:
            return tuple(None for _ in self._kernels)
        owned = {name for names in self._sites for name in names}
        unknown = sorted(set(init_params) - owned)
        if unknown:
            raise ValueError(f"`init_params` has unknown sites {unknown}.")
        return tuple(
            {name: init_params[name] for name in names if name in init_params} or None
            for names in self._sites
        )

    def init(
        self,
        rng_key: jax.Array,
        num_warmup: int,
        init_params: SiteValues | None,
        model_args: ModelArgs,
        model_kwargs: ModelKwargs | None,
    ) -> GibbsState:
        if not is_prng_key(rng_key):
            raise ValueError(
                "Gibbs only supports a single random key; for multiple chains use "
                '`chain_method="parallel"`, `chain_method="sequential"` or a callable '
                "chain method such as `jax.vmap`."
            )
        model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        enclosing = frozenset(model_kwargs.get(GIBBS_SITES_KWARG, {}))
        rng_key, key_trace = random.split(rng_key)
        model_trace = prototype_trace(
            conditioned(self._model), key_trace, model_args, model_kwargs
        )
        if subsample_plate_sizes(model_trace):
            raise ValueError(
                "Gibbs does not support models with subsample plates; use HMCECS."
            )
        self._sites = self._resolve_partition(model_trace, enclosing)
        self._has_deterministic = any(
            site["type"] == "deterministic" for site in model_trace.values()
        )
        block_init_params = self._split_init_params(init_params)

        # constrained initial values of every latent site, used to condition the blocks
        constrained = {
            name: model_trace[name]["value"] for names in self._sites for name in names
        }
        for kernel, names, params in zip(self._kernels, self._sites, block_init_params):
            if isinstance(kernel, HMC):
                transforms = _transforms_from_trace(
                    self._block_trace(model_trace, names), raise_warnings=False
                )
                if transforms.dynamic_support:
                    raise ValueError(
                        f"The supports of sites {list(names)} depend on other sites; "
                        "Gibbs does not support value-dependent supports across blocks."
                    )
                if params:
                    constrained.update(transform_fn(transforms.inv_transforms, params))
            elif params:
                constrained.update(params)

        block_states = []
        for i, (kernel, names, params) in enumerate(
            zip(self._kernels, self._sites, block_init_params)
        ):
            siblings = {k: v for k, v in constrained.items() if k not in names}
            kwargs_i = with_conditioning(model_kwargs, siblings)
            if not isinstance(kernel, HMC):
                params = {**{k: constrained[k] for k in names}, **(params or {})}
            rng_key, key_i = random.split(rng_key)
            state_i = kernel.init(key_i, num_warmup, params, model_args, kwargs_i)
            block_states.append(state_i)
            z_i = getattr(state_i, kernel.sample_field)
            constrained_i = kernel.get_constrain_fn(model_args, kwargs_i)(z_i)
            constrained.update({k: constrained_i[k] for k in names})

        # blocks were initialized against prototype values of later blocks; refresh them
        for i in range(len(self._kernels) - 1):
            siblings = {k: v for k, v in constrained.items() if k not in self._sites[i]}
            kwargs_i = with_conditioning(model_kwargs, siblings)
            block_states[i] = self._kernels[i].refresh(
                block_states[i], model_args, kwargs_i
            )

        z = self._merge_z(block_states)
        self._sample_fn = self._sample_one
        return self._state_cls(z, tuple(block_states), rng_key)

    def _merge_z(self, block_states: Sequence[PyTree]) -> SiteValues:
        z = {}
        for kernel, names, block_state in zip(self._kernels, self._sites, block_states):
            z_i = getattr(block_state, kernel.sample_field)
            z.update({k: z_i[k] for k in names})
        return z

    def _sample_one(
        self,
        state: GibbsState,
        model_args: ModelArgs,
        model_kwargs: ModelKwargs | None,
    ) -> GibbsState:
        model_kwargs = {} if model_kwargs is None else model_kwargs
        num_blocks = len(self._kernels)
        block_states = list(state.block_states)
        z_blocks = [
            getattr(block_state, kernel.sample_field)
            for kernel, block_state in zip(self._kernels, block_states)
        ]
        # constrained values of each block's own sites, for conditioning its siblings
        constrained = [
            {
                k: v
                for k, v in kernel.get_constrain_fn(model_args, model_kwargs)(
                    z_i
                ).items()
                if k in names
            }
            for kernel, names, z_i in zip(self._kernels, self._sites, z_blocks)
        ]

        def kwargs_for(i: int) -> ModelKwargs:
            siblings = {}
            for j in range(num_blocks):
                if j != i:
                    siblings.update(constrained[j])
            return with_conditioning(model_kwargs, siblings)

        moved: list[Any] = [False] * num_blocks
        for i, (kernel, names) in enumerate(zip(self._kernels, self._sites)):
            kwargs_i = kwargs_for(i)
            # siblings visited earlier in this sweep may have moved since the last refresh
            block_states[i] = _maybe_refresh(
                kernel, block_states[i], _or(moved[:i]), model_args, kwargs_i
            )
            block_states[i] = kernel.sample(block_states[i], model_args, kwargs_i)
            z_new = getattr(block_states[i], kernel.sample_field)
            moved[i] = any_changed(z_blocks[i], z_new)
            constrained[i] = {
                k: v
                for k, v in kernel.get_constrain_fn(model_args, kwargs_i)(z_new).items()
                if k in names
            }
        # siblings visited later in this sweep may have moved; refresh against the final values
        for i in range(num_blocks - 1):
            block_states[i] = _maybe_refresh(
                self._kernels[i],
                block_states[i],
                _or(moved[i + 1 :]),
                model_args,
                kwargs_for(i),
            )

        rng_key, _ = random.split(state.rng_key)
        return state._replace(
            z=self._merge_z(block_states),
            block_states=tuple(block_states),
            rng_key=rng_key,
        )

    def sample(
        self,
        state: GibbsState,
        model_args: ModelArgs,
        model_kwargs: ModelKwargs | None,
    ) -> GibbsState:
        """
        Run one sweep over the blocks from the given :class:`GibbsState` and return the
        resulting :class:`GibbsState`.

        :param GibbsState state: the current state.
        :param tuple model_args: arguments provided to the model.
        :param dict model_kwargs: keyword arguments provided to the model.
        :return: the next state.
        """
        assert self._sample_fn is not None, "`init` must be called before `sample`."
        return self._sample_fn(state, model_args, model_kwargs)

    def refresh(
        self,
        state: GibbsState,
        model_args: ModelArgs,
        model_kwargs: ModelKwargs | None,
    ) -> GibbsState:
        """Identity: :meth:`sample` refreshes each block against the current conditioning."""
        return state

    def wrap_model(self, wrapper: ModelWrapper) -> "Gibbs":
        """New composite with `wrapper` applied to every model-based block (nesting)."""
        kernel = copy.copy(self)
        kernel._model = wrapper(self._model)
        kernel._kernels = tuple(
            block.wrap_model(wrapper) if _has_model(block) else block
            for block in self._kernels
        )
        kernel._sites = ()
        kernel._sample_fn = None
        return kernel

    def get_constrain_fn(
        self,
        model_args: ModelArgs,
        model_kwargs: ModelKwargs | None,
    ) -> ConstrainFn:
        """Constrain each block's sites with the block's own constrain function (no replay)."""
        if not self._sites:
            return identity

        def fn(z: SiteValues) -> SiteValues:
            out = {}
            for kernel, names in zip(self._kernels, self._sites):
                z_i = {k: z[k] for k in names}
                constrained_i = kernel.get_constrain_fn(model_args, model_kwargs)(z_i)
                out.update({k: constrained_i[k] for k in names})
            return out

        return fn

    def postprocess_fn(
        self,
        model_args: ModelArgs,
        model_kwargs: ModelKwargs | None,
    ) -> ConstrainFn:
        """
        Constrain each block's sites with the block's own constrain function, then, only if
        the model has `deterministic` sites, replay the model once with all constrained values
        substituted to collect them.
        """
        if not self._sites:
            return identity
        constrain = self.get_constrain_fn(model_args, model_kwargs)
        if not self._has_deterministic:
            return constrain
        model_kwargs = {} if model_kwargs is None else model_kwargs

        def fn(z: SiteValues) -> SiteValues:
            constrained = constrain(z)
            model = substitute(
                seed(conditioned(self._model), random.key(0)), data=constrained
            )
            model_trace = trace(model).get_trace(*model_args, **model_kwargs)
            deterministic = {
                name: site["value"]
                for name, site in model_trace.items()
                if site["type"] == "deterministic"
            }
            return {**constrained, **deterministic}

        return fn

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_sample_fn"] = None
        return state


class CustomGibbsState(NamedTuple):
    """
    - **z** - dict of the block's current (constrained) values.
    - **rng_key** - random key for the next step.
    """

    z: SiteValues
    rng_key: jax.Array


class CustomGibbs(MCMCKernel):
    """
    Block kernel that delegates the update to a user callable. The callable receives the
    current values of the block's sites and the constrained values of all conditioning sites
    and returns new values for the block's sites (it must sample from the conditional;
    correctness is the user's responsibility, as with
    :class:`~numpyro.infer.hmc_gibbs.HMCGibbs`). Only usable as a block of :class:`Gibbs`.

    :param gibbs_fn: called as `gibbs_fn(rng_key=..., gibbs_sites=..., hmc_sites=...)`; the
        keyword names are kept for compatibility with existing `HMCGibbs` users, and
        `hmc_sites` holds all conditioning sites (not only HMC ones). The returned dict must
        have exactly the block's site names as keys.
    """

    sample_field: str = "z"

    def __init__(self, gibbs_fn: GibbsUpdateFn) -> None:
        if not callable(gibbs_fn):
            raise ValueError("gibbs_fn must be a callable")
        self._gibbs_fn = gibbs_fn

    def init(
        self,
        rng_key: jax.Array,
        num_warmup: int,
        init_params: SiteValues | None,
        model_args: ModelArgs,
        model_kwargs: ModelKwargs | None,
    ) -> CustomGibbsState:
        """`init_params` are the block's initial (constrained) values, supplied by the composite."""
        if not init_params:
            raise ValueError(
                "CustomGibbs requires initial values; use it as a block of Gibbs."
            )
        return CustomGibbsState(_as_arrays(init_params), rng_key)

    def sample(
        self,
        state: CustomGibbsState,
        model_args: ModelArgs,
        model_kwargs: ModelKwargs | None,
    ) -> CustomGibbsState:
        """Reads the conditioning values from `model_kwargs[GIBBS_SITES_KWARG]`."""
        model_kwargs = {} if model_kwargs is None else model_kwargs
        rng_key, key_update = random.split(state.rng_key)
        z_new = self._gibbs_fn(
            rng_key=key_update,
            gibbs_sites=state.z,
            hmc_sites=model_kwargs.get(GIBBS_SITES_KWARG, {}),
        )
        if set(z_new) != set(state.z):
            raise ValueError(
                f"gibbs_fn must return values for exactly the sites {sorted(state.z)}, "
                f"got {sorted(z_new)}."
            )
        return state._replace(z=_as_arrays(z_new), rng_key=rng_key)

    def refresh(
        self,
        state: CustomGibbsState,
        model_args: ModelArgs,
        model_kwargs: ModelKwargs | None,
    ) -> CustomGibbsState:
        """Identity: nothing is cached."""
        return state

    def wrap_model(self, wrapper: ModelWrapper) -> "CustomGibbs":
        """Returns `self`: there is no model."""
        return self

    def get_constrain_fn(
        self,
        model_args: ModelArgs,
        model_kwargs: ModelKwargs | None,
    ) -> Callable:
        """Identity: the block's values are already constrained."""
        return identity


# Discrete proposals. Each is called as
# `(rng_key, z, pe, potential_fn, idx, support_size) -> (rng_key, z_new, pe_new, log_accept_ratio)`
# where `idx` is the flat coordinate of `z` to update and `support_size` its support size.

ProposalFn: TypeAlias = Callable[
    [jax.Array, SiteValues, jax.Array, PotentialFn, jax.Array, jax.Array],
    tuple[jax.Array, SiteValues, jax.Array, jax.Array],
]
"""`(rng_key, z, pe, potential_fn, idx, support_size) -> (rng_key, z_new, pe_new, log_accept_ratio)`."""


def _discrete_gibbs_proposal_body_fn(
    z_init_flat, unravel_fn, pe_init, potential_fn, idx, i, val
):
    rng_key, z, pe, log_weight_sum = val
    rng_key, rng_transition = random.split(rng_key)
    proposal = jnp.where(i >= z_init_flat[idx], i + 1, i)
    z_new_flat = z_init_flat.at[idx].set(proposal)
    z_new = unravel_fn(z_new_flat)
    pe_new = potential_fn(z_new)
    log_weight_new = pe_init - pe_new
    # Handles the NaN case...
    log_weight_new = jnp.where(jnp.isfinite(log_weight_new), log_weight_new, -jnp.inf)
    # transition_prob = e^weight_new / (e^weight_logsumexp + e^weight_new)
    transition_prob = expit(log_weight_new - log_weight_sum)
    z, pe = cond(
        random.bernoulli(rng_transition, transition_prob),
        (z_new, pe_new),
        identity,
        (z, pe),
        identity,
    )
    log_weight_sum = jnp.logaddexp(log_weight_new, log_weight_sum)
    return rng_key, z, pe, log_weight_sum


def _discrete_gibbs_proposal(
    rng_key: jax.Array,
    z_discrete: SiteValues,
    pe: jax.Array,
    potential_fn: PotentialFn,
    idx: jax.Array,
    support_size: jax.Array,
) -> tuple[jax.Array, SiteValues, jax.Array, jax.Array]:
    # idx: current index of `z_discrete_flat` to update
    # support_size: support size of z_discrete at the index idx

    z_discrete_flat, unravel_fn = ravel_pytree(z_discrete)
    # Here we loop over the support of z_flat[idx] to get z_new
    # Note: we can't vmap potential_fn over all proposals and sample from the conditional
    # categorical distribution because support_size is a traced value, i.e. its value
    # might change across different discrete variables;
    # so here we will loop over all proposals and use an online scheme to sample from
    # the conditional categorical distribution
    body_fn = partial(
        _discrete_gibbs_proposal_body_fn,
        z_discrete_flat,
        unravel_fn,
        pe,
        potential_fn,
        idx,
    )
    init_val = (rng_key, z_discrete, pe, jnp.array(0.0))
    rng_key, z_new, pe_new, _ = fori_loop(0, support_size - 1, body_fn, init_val)
    log_accept_ratio = jnp.array(0.0)
    return rng_key, z_new, pe_new, log_accept_ratio


def _discrete_modified_gibbs_proposal(
    rng_key: jax.Array,
    z_discrete: SiteValues,
    pe: jax.Array,
    potential_fn: PotentialFn,
    idx: jax.Array,
    support_size: jax.Array,
    stay_prob: float = 0.0,
) -> tuple[jax.Array, SiteValues, jax.Array, jax.Array]:
    assert isinstance(stay_prob, float) and stay_prob >= 0.0 and stay_prob < 1
    z_discrete_flat, unravel_fn = ravel_pytree(z_discrete)
    body_fn = partial(
        _discrete_gibbs_proposal_body_fn,
        z_discrete_flat,
        unravel_fn,
        pe,
        potential_fn,
        idx,
    )
    # like gibbs_step but here, weight of the current value is 0
    init_val = (rng_key, z_discrete, pe, jnp.array(-jnp.inf))
    rng_key, z_new, pe_new, log_weight_sum = fori_loop(
        0, support_size - 1, body_fn, init_val
    )
    rng_key, rng_stay = random.split(rng_key)
    z_new, pe_new = cond(
        random.bernoulli(rng_stay, stay_prob),
        (z_discrete, pe),
        identity,
        (z_new, pe_new),
        identity,
    )
    # here we calculate the MH correction: (1 - P(z)) / (1 - P(z_new))
    # where 1 - P(z) ~ weight_sum
    # and 1 - P(z_new) ~ 1 + weight_sum - z_new_weight
    log_accept_ratio = log_weight_sum - jnp.log(
        jnp.exp(log_weight_sum) - jnp.expm1(pe - pe_new)
    )
    return rng_key, z_new, pe_new, log_accept_ratio


def _discrete_rw_proposal(
    rng_key: jax.Array,
    z_discrete: SiteValues,
    pe: jax.Array,
    potential_fn: PotentialFn,
    idx: jax.Array,
    support_size: jax.Array,
) -> tuple[jax.Array, SiteValues, jax.Array, jax.Array]:
    rng_key, rng_proposal = random.split(rng_key, 2)
    z_discrete_flat, unravel_fn = ravel_pytree(z_discrete)

    proposal = random.randint(rng_proposal, (), minval=0, maxval=support_size)
    z_new_flat = z_discrete_flat.at[idx].set(proposal)
    z_new = unravel_fn(z_new_flat)
    pe_new = potential_fn(z_new)
    log_accept_ratio = pe - pe_new
    return rng_key, z_new, pe_new, log_accept_ratio


def _discrete_modified_rw_proposal(
    rng_key: jax.Array,
    z_discrete: SiteValues,
    pe: jax.Array,
    potential_fn: PotentialFn,
    idx: jax.Array,
    support_size: jax.Array,
    stay_prob: float = 0.0,
) -> tuple[jax.Array, SiteValues, jax.Array, jax.Array]:
    assert isinstance(stay_prob, float) and stay_prob >= 0.0 and stay_prob < 1
    rng_key, rng_proposal, rng_stay = random.split(rng_key, 3)
    z_discrete_flat, unravel_fn = ravel_pytree(z_discrete)

    i = random.randint(rng_proposal, (), minval=0, maxval=support_size - 1)
    proposal = jnp.where(i >= z_discrete_flat[idx], i + 1, i)
    proposal = jnp.where(
        random.bernoulli(rng_stay, stay_prob), z_discrete_flat[idx], proposal
    )
    z_new_flat = z_discrete_flat.at[idx].set(proposal)
    z_new = unravel_fn(z_new_flat)
    pe_new = potential_fn(z_new)
    log_accept_ratio = pe - pe_new
    return rng_key, z_new, pe_new, log_accept_ratio


def select_discrete_proposal(random_walk: bool, modified: bool) -> ProposalFn:
    """
    Pick the discrete proposal: the exact conditional (Gibbs) or a uniform random walk, each
    optionally in Liu's modified form that never proposes the current value.

    :param bool random_walk: uniform proposals over the support instead of the conditional.
    :param bool modified: use the modified (Metropolised) proposal.
    """
    if random_walk:
        if modified:
            return partial(_discrete_modified_rw_proposal, stay_prob=0.0)
        return _discrete_rw_proposal
    if modified:
        return partial(_discrete_modified_gibbs_proposal, stay_prob=0.0)
    return _discrete_gibbs_proposal


def discrete_gibbs_sweep(
    rng_key: jax.Array,
    z: SiteValues,
    potential_energy: jax.Array,
    potential_fn: PotentialFn,
    support_sizes_flat: jax.Array,
    proposal_fn: ProposalFn,
) -> tuple[SiteValues, jax.Array]:
    """
    One sweep over the flat discrete coordinates of `z` in a random order, each coordinate
    updated with `proposal_fn` and Metropolis corrected.

    :param rng_key: random key.
    :param z: current discrete values.
    :param potential_energy: potential energy at `z`.
    :param potential_fn: potential energy as a function of the discrete values.
    :param support_sizes_flat: support size of each flat coordinate, in `ravel_pytree` order.
    :param proposal_fn: a discrete proposal, see :func:`select_discrete_proposal`.
    :return: the new values and their potential energy.
    """
    num_discretes = support_sizes_flat.shape[0]
    rng_key, rng_permute = random.split(rng_key)
    idxs = random.permutation(rng_permute, jnp.arange(num_discretes))

    def body_fn(i, val):
        idx = idxs[i]
        support_size = support_sizes_flat[idx]
        rng_key, z, pe = val
        rng_key, z_new, pe_new, log_accept_ratio = proposal_fn(
            rng_key, z, pe, potential_fn, idx, support_size
        )
        rng_key, rng_accept = random.split(rng_key)
        # u ~ Uniform(0, 1), u < accept_ratio => -log(u) > -log_accept_ratio
        # and -log(u) ~ exponential(1)
        z, pe = cond(
            random.exponential(rng_accept) > -log_accept_ratio,
            (z_new, pe_new),
            identity,
            (z, pe),
            identity,
        )
        return rng_key, z, pe

    init_val = (rng_key, z, potential_energy)
    _, z, pe = fori_loop(0, num_discretes, body_fn, init_val)
    return z, pe


class DiscreteGibbsState(NamedTuple):
    """
    - **z** - dict of the current discrete values.
    - **potential_energy** - potential energy at `z` under the current conditioning (the block's
      own potential, which differs from an HMC block's by a constant and cannot be handed
      across blocks).
    - **rng_key** - random key for the next step.
    """

    z: SiteValues
    potential_energy: jax.Array
    rng_key: jax.Array


class DiscreteGibbs(MCMCKernel):
    """
    Metropolis / Gibbs updates of the discrete latent sites of a model, one flat coordinate at
    a time in a random order. Usable standalone on a purely discrete model, or as a block of
    :class:`Gibbs`.

    The potential is the negative log joint of the model at the block's (constrained) values,
    with every other latent site fixed to the conditioning values passed through the model
    keyword arguments; sites marked `infer={"enumerate": "parallel"}` are marginalized exactly
    as in HMC.

    :param model: the model.
    :param bool random_walk: uniform proposals over the support instead of the exact
        conditional.
    :param bool modified: Liu's modified (Metropolised) proposal that never proposes the
        current value.

    **References:**

    1. *Peskun's theorem and a modified discrete-state Gibbs sampler*, Liu, J. S. (1996)
    """

    sample_field: str = "z"

    def __init__(
        self, model: ModelT, *, random_walk: bool = False, modified: bool = False
    ) -> None:
        self._model = model
        self._random_walk = random_walk
        self._modified = modified
        self._proposal_fn = select_discrete_proposal(random_walk, modified)
        # static metadata resolved at `init`
        self._sites: tuple[str, ...] | None = None
        self._support_sizes_flat: np.ndarray | None = None
        self._enum = False
        # closes over trace values, rebuilt at every `init`
        self._prepared_model: ModelT | None = None

    @property
    def model(self) -> ModelT:
        return self._model

    def get_potential_fn(
        self,
        model_args: ModelArgs = (),
        model_kwargs: ModelKwargs | None = None,
    ) -> PotentialFn:
        """Potential over the block's discrete values for the given arguments and conditioning."""
        if self._prepared_model is None:
            raise RuntimeError(
                "`get_potential_fn` requires the kernel to be initialized; run `init` first."
            )
        prepared_model, enum = self._prepared_model, self._enum

        def potential_fn(z: SiteValues) -> jax.Array:
            return potential_energy(
                prepared_model,
                model_args,
                with_conditioning(model_kwargs, z),
                {},
                enum=enum,
            )

        return potential_fn

    def init(
        self,
        rng_key: jax.Array,
        num_warmup: int,
        init_params: SiteValues | None,
        model_args: ModelArgs,
        model_kwargs: ModelKwargs | None,
    ) -> DiscreteGibbsState:
        model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        rng_key, key_trace = random.split(rng_key)
        model_trace = prototype_trace(
            conditioned(self._model), key_trace, model_args, model_kwargs
        )
        sites = discrete_latent_sites(model_trace)
        if not sites:
            raise ValueError(
                "Cannot detect any discrete latent variables in the model."
            )
        others = [
            name
            for name in latent_sample_sites(model_trace)
            if name not in sites
            and model_trace[name]["infer"].get("enumerate", "") != "parallel"
        ]
        if others:
            raise ValueError(
                f"DiscreteGibbs cannot sample the latent sites {others}; condition on them "
                "or use DiscreteGibbs as a block of Gibbs."
            )
        self._sites = sites
        self._support_sizes_flat = _flat_support_sizes(model_trace, sites)
        self._enum = any(
            site["type"] == "sample"
            and not site["is_observed"]
            and site["infer"].get("enumerate", "") == "parallel"
            for site in model_trace.values()
        )
        self._prepared_model = _prepare_model_for_potential(
            conditioned(self._model), model_trace, enum=self._enum
        )
        init_params = {} if init_params is None else init_params
        z = _as_arrays(
            {name: init_params.get(name, model_trace[name]["value"]) for name in sites}
        )
        pe = self.get_potential_fn(model_args, model_kwargs)(z)
        return DiscreteGibbsState(z, jnp.asarray(pe), rng_key)

    def sample(
        self,
        state: DiscreteGibbsState,
        model_args: ModelArgs,
        model_kwargs: ModelKwargs | None,
    ) -> DiscreteGibbsState:
        """One :func:`~numpyro.infer.gibbs.discrete_gibbs_sweep` with the selected proposal."""
        rng_key, key_sweep = random.split(state.rng_key)
        z, pe = discrete_gibbs_sweep(
            key_sweep,
            state.z,
            state.potential_energy,
            self.get_potential_fn(model_args, model_kwargs),
            jnp.asarray(self._support_sizes_flat),
            self._proposal_fn,
        )
        return state._replace(z=z, potential_energy=pe, rng_key=rng_key)

    def refresh(
        self,
        state: DiscreteGibbsState,
        model_args: ModelArgs,
        model_kwargs: ModelKwargs | None,
    ) -> DiscreteGibbsState:
        """Recompute `potential_energy` at `state.z` (one model evaluation)."""
        pe = self.get_potential_fn(model_args, model_kwargs)(state.z)
        return state._replace(potential_energy=jnp.asarray(pe))

    def wrap_model(self, wrapper: ModelWrapper) -> "DiscreteGibbs":
        kernel = copy.copy(self)
        kernel._model = wrapper(self._model)
        kernel._sites = None
        kernel._support_sizes_flat = None
        kernel._prepared_model = None
        return kernel

    def get_constrain_fn(
        self,
        model_args: ModelArgs,
        model_kwargs: ModelKwargs | None,
    ) -> Callable:
        """Identity: discrete values are already constrained."""
        return identity

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_prepared_model"] = None
        return state
