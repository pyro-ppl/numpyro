# Design: composable Gibbs kernels (issue #898)

Status: scaffolding proposal for review, no implementation. Target: [pyro-ppl/numpyro#898](https://github.com/pyro-ppl/numpyro/issues/898), "Make Gibbs kernels composable". All `file:line` references are against the current `master` (`26cc211a`).

## 1. Problem

NumPyro ships four kernels that combine an HMC/NUTS move on the continuous sites with some other update on the remaining sites: `HMCGibbs` (a user supplied `gibbs_fn`), `DiscreteHMCGibbs` (Metropolis/Gibbs on discrete latents), `HMCECS` (pseudo-marginal update of subsample indices), and `MixedHMC` (discrete updates interleaved inside the HMC trajectory). They live in `numpyro/infer/hmc_gibbs.py` and `numpyro/infer/mixed_hmc.py` and are all subclasses of `HMCGibbs`.

The issue asks for a redesign in which "each kernel is applied to a subset of variables and MCMC still runs", i.e. a user should be able to write something like `Gibbs([(NUTS(model), ["mu", "sigma"]), (DiscreteGibbs(model), ["c"]), (CustomGibbs(fn), ["beta"])])` and get a valid sampler, and the maintainers should be able to add new block samplers without touching HMC internals.

Today this is impossible because the design hard-wires exactly two blocks (the Gibbs sites and the HMC sites), the inner kernel type, and the shape of the state. Concretely:

- The inner kernel must be an `HMC` instance (`hmc_gibbs.py:88`); `MixedHMC` additionally checks a private `_algo` string (`mixed_hmc.py:80`).
- The inner kernel is shallow-copied and its private `_model` is replaced by a wrapper (`hmc_gibbs.py:96-97`); `HMCECS` stacks two more wrappers, one at construction (`:562`) and one at every `init` (`:617-619`). Because `HMC` builds its `_init_fn`/`_sample_fn` closures only once (`hmc.py:706-711`) and `MCMC` re-runs `init` on every `run()` for kernels without a `_sample_fn` attribute (`mcmc.py:475`), calling `mcmc.warmup(...)` followed by `mcmc.run(...)` on `HMCECS(NUTS(model), proxy=...)` fails with `all sites must have unique names but got '_biased_corrected_log_likelihood' duplicated` (verified on master).
- Every subclass reads private HMC state and attributes: `inner_kernel._potential_fn_gen`, `_forward_mode_differentiation`, `hmc_state.z`, `z_grad`, `potential_energy` (`hmc_gibbs.py:157-179`, `:454-480`, `:644-674`; `mixed_hmc.py:120-165`), and `MixedHMC` rebuilds the warmup adapter from five private attributes (`mixed_hmc.py:98-105`).
- The Gibbs/HMC partition has two sources of truth: `sample` splits `state.z` by membership in `state.hmc_state.z` (`hmc_gibbs.py:162-163`), `postprocess_fn` splits by `self._gibbs_sites` (`:115-116`).
- `DiscreteHMCGibbs` and `HMCECS` pass `identity` and `None` to the parent constructor to satisfy validation and then never use `_gibbs_fn` (`:401`, `:560`); each re-implements `sample` from scratch. The `potential_fn(z_gibbs, z_hmc)` closure and the "recompute potential energy and gradient, then `_replace` the HMC state" step are copy-pasted four times.
- The prototype trace is cached on `self` (`:125-131`). Under `chain_method="parallel"` (`pmap`) or a callable chain method (`test/infer/test_hmc_gibbs.py:463-486` uses `chain_method=vmap`) the cached trace holds tracers, and a second `run()` reuses them.
- `estimate_likelihood` locates the unconstrained parameters by scanning `numpyro.primitives._PYRO_STACK` for a `substitute` whose `substitute_fn` is a `partial` of `_unconstrain_reparam` (`:706-714`), and the proxy state reaches it through a fake `"_gibbs_state"` message type pushed with `apply_stack` (`:495-499`).
- Smaller defects: `init_params` is mutated by `pop` (`:139`); RNG keys are split and not used (`:314-315`, `:642`; `mixed_hmc.py:221`); `_discrete_modified_rw_proposal` writes `idx` (the coordinate) instead of `z_discrete_flat[idx]` (the current value) into the stay branch (`:300`, dead code since `stay_prob=0.0`); no `chain_method="vectorized"` support (unconditional `random.split` on the key).

Consequences: no N-block composition, no non-HMC block, no nesting, block logic cannot be tested in isolation, HMC internals cannot change without auditing four files, and the same bookkeeping is maintained in four places.

## 2. Current status

| Kernel | Base | State (namedtuple) | Conditioning | Reads from inner kernel | `default_fields` | Tests |
|---|---|---|---|---|---|---|
| `HMCGibbs` | `MCMCKernel` | `HMCGibbsState(z, hmc_state, rng_key)` | `_gibbs_sites` kwarg popped by `_wrap_model` and applied with `condition` + `substitute` (`hmc_gibbs.py:32-35`) | `_model`, `_potential_fn_gen`, `_forward_mode_differentiation`, `postprocess_fn`, `hmc_state.{z,z_grad,potential_energy}` | `("z",)` (inherited) | `test_hmc_gibbs.py:41-172, 463-486` |
| `DiscreteHMCGibbs` | `HMCGibbs` | `HMCGibbsState` | same, discrete sites discovered from a prototype trace (`:428-447`) | same, plus `hmc_state.potential_energy` fed into the sweep | `("z",)` | `test_hmc_gibbs.py:175-272`, `test_pickle.py:118` |
| `HMCECS` | `HMCGibbs` | `HMCECSState(z, hmc_state, rng_key, gibbs_state, accept_prob)` | same, plus `_gibbs_state` kwarg turned into a message (`:495-499`) and `estimate_likelihood` (`:692-770`) | same, plus `_model` re-wrapped at `init` | `("z",)` | `test_hmc_gibbs.py:274-461`, `test_pickle.py:128`, `test/contrib/test_esc_proxies.py` |
| `MixedHMC` | `DiscreteHMCGibbs` | `MixedHMCState(z, hmc_state, rng_key, accept_prob)` | same | `_potential_fn_gen`, `_forward_mode_differentiation`, `_adapt_*`, `_dense_mass`, `_target_accept_prob`, `_algo`, `hmc_state.{r, i, adapt_state, trajectory_length, num_steps}` | `("z",)` | `test_hmc_gibbs.py:175-272`, `test_pickle.py:118` |

Data flow today, one MCMC step of `HMCGibbs.sample` (`hmc_gibbs.py:153-186`): split `state.z` into `z_gibbs` and `z_hmc`; constrain `z_hmc` with the inner `postprocess_fn`; call `gibbs_fn`; recompute `(pe, z_grad)` at the unchanged `hmc_state.z` under the new `z_gibbs` with `value_and_grad` (or `jacfwd`); `hmc_state._replace(z_grad=..., potential_energy=...)`; call `inner_kernel.sample` with `_gibbs_sites` in `model_kwargs`; merge. The mechanism that makes this work without recompilation is that `hmc()` rebuilds `pe_fn = potential_fn_gen(*model_args, **model_kwargs)` on every call (`hmc.py:373-377`, `:425-429`), so the conditioning values are ordinary traced kwargs.

What is worth keeping: the kwarg conditioning channel (cheap, JIT friendly, composes with enumeration), the discrete proposal functions (`hmc_gibbs.py:194-345`), the ECS proxies (`numpyro/contrib/ecs_proxies.py`), and the tests.

## 3. Objective and non-goals

Objective. Introduce a composite kernel `Gibbs` that runs an ordered list of block kernels, each owning a subset of the model's latent sample sites and each conditioned on the current values of all other sites, such that:

1. `MCMC(Gibbs([(k_1, sites_1), ..., (k_N, sites_N)]))` works with any mix of `HMC`/`NUTS`, `DiscreteGibbs`, `CustomGibbs`, and nested `Gibbs` blocks (`SA` and `BarkerMH` are follow-ups, see section 8).
2. `HMCGibbs`, `DiscreteHMCGibbs`, `HMCECS`, `MixedHMC` keep their constructors and public behavior and become thin layers over shared machinery.
3. No block accesses private attributes of another kernel; the two things a composite needs from a block are formal `MCMCKernel` methods.
4. Per-step cost is unchanged for `HMCGibbs`, `HMCECS`, `MixedHMC`, and within one model evaluation of today for `DiscreteHMCGibbs` (section 5 has the exact table).
5. States are pytrees with a stable treedef; nothing traced is ever stored on kernel objects; pickling works.
6. `chain_method` `"parallel"`, `"sequential"` and callable work in v1; `"vectorized"` is a documented follow-up.

Non-goals (v1): changing the HMC integrator or adaptation; ensemble kernels as blocks; `HMCECS` as a sibling block inside `Gibbs` (it stays a wrapper around an HMC kernel; the correct composition is `HMCECS(Gibbs(...))`, section 8); value-dependent supports that cross blocks; deprecating any public class.

## 4. Scaffolding design

#### 4.0 Typing vocabulary

Rule: no bare `Any` in a public signature; every recurring shape gets a named alias so the signature reads as documentation even where the alias itself must be permissive (model inputs are arbitrary by design). Package-wide aliases live in `numpyro/_typing.py` next to the existing `ModelT`, `TraceT`, `PyTree`; Gibbs-specific ones in `numpyro/infer/gibbs_util.py`. New state containers are `typing.NamedTuple` classes (typed fields, still pytrees, still `_replace`/`_fields`/`_asdict`; precedent `numpyro/optim.py:256`). Imports: `from collections.abc import Callable, Sequence`; `from typing import Any, NamedTuple, Protocol, TypeAlias, TypeVar`; `jax.Array` for arrays and PRNG keys.

```python
# numpyro/_typing.py (additions)
ModelArgs: TypeAlias = tuple[Any, ...]
"""Positional arguments of a model, as passed to `MCMC.run(rng_key, *args)`."""

ModelKwargs: TypeAlias = dict[str, Any]
"""Keyword arguments of a model; may carry reserved keys such as `GIBBS_SITES_KWARG`."""

SiteValues: TypeAlias = dict[str, jax.Array]
"""Values keyed by site name (a sample, a set of init params, a conditioning set)."""

PotentialFn: TypeAlias = Callable[[SiteValues], jax.Array]
"""Negative log joint as a function of (unconstrained) site values."""

ConstrainFn: TypeAlias = Callable[[SiteValues], SiteValues]
"""Maps site values to site values (constrain / postprocess)."""

StateT = TypeVar("StateT")
"""A kernel state pytree; used where a method returns the same state type it received."""
```

```python
# numpyro/infer/gibbs_util.py (aliases)
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

    def __call__(self, *, rng_key: jax.Array, gibbs_sites: SiteValues, hmc_sites: SiteValues) -> SiteValues: ...


LikelihoodEstimator: TypeAlias = Callable[[dict[str, tuple], SiteValues, PyTree], jax.Array]
"""`(likelihoods, unconstrained_params, gibbs_state) -> log-likelihood estimate`; see `perturbed_method` in `numpyro/contrib/ecs_proxies.py:23`."""

ProxyConstructor: TypeAlias = Callable[..., tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]]
"""`(prototype_trace, subsample_plate_sizes, model, model_args, model_kwargs, num_blocks) -> (proxy_fn, gibbs_init, gibbs_update)`; the protocol pinned by `test_taylor_proxy_norm`."""
```

| Alias | Stands for | Residual `Any` |
|---|---|---|
| `ModelArgs`, `ModelKwargs` | model inputs | yes, inherent |
| `SiteValues` | `dict[str, jax.Array]` | no |
| `PotentialFn`, `ConstrainFn` | the two function shapes every kernel exposes | no |
| `StateT` | "same state type in and out" (`refresh`) | no (TypeVar) |
| `ModelWrapper`, `SiteSelector`, `SitesSpec` | Gibbs plumbing | no |
| `GibbsUpdateFn` | keyword contract of the user callback | no |
| `LikelihoodEstimator`, `ProxyConstructor` | ECS proxy protocol (existing, untyped today) | inside the proxy triple only |
| `PyTree` (existing) | opaque block states, proxy state | yes, by definition |

The only remaining bare `Any` below are `*args: Any, **kwargs: Any` in model wrappers (they forward whatever the model takes) and `__getstate__ -> dict[str, Any]`.

### 4.1 `numpyro/infer/mcmc.py`: two opt-in hooks on `MCMCKernel`

```python
class MCMCKernel(ABC):
    # existing: postprocess_fn, init, sample, sample_field, default_fields, is_ensemble_kernel, get_diagnostics_str

    def refresh(
        self,
        state: StateT,
        model_args: ModelArgs,
        model_kwargs: ModelKwargs | None,
    ) -> StateT:
        """
        Recompute every value cached in `state` that depends on `(model_args, model_kwargs)`
        without advancing the chain, for example the potential energy and its gradient at the
        current sample. Composite kernels call this before :meth:`sample` whenever the values
        the target is conditioned on have changed. Kernels that cache nothing return `state`.

        The default raises `NotImplementedError`; only kernels that override it can be used as
        blocks of :class:`~numpyro.infer.gibbs.Gibbs`. This is deliberate: a kernel that binds
        its potential at `init` (e.g. :class:`~numpyro.infer.barker.BarkerMH`) would silently
        target the wrong conditional if the default were the identity.

        :param state: current kernel state.
        :param tuple model_args: arguments provided to the model.
        :param dict model_kwargs: keyword arguments provided to the model, including any
            conditioning values.
        :return: a state of the same type with refreshed cached values.
        """
        ...

    def wrap_model(self, wrapper: ModelWrapper) -> "MCMCKernel":
        """
        Return a copy of this kernel whose model is `wrapper(self.model)`. Kernels that hold
        other kernels apply the wrapper recursively; kernels without a model return `self`.
        Any function built lazily from the old model (potential, postprocess, sampler closures)
        must be reset in the copy. The default raises `NotImplementedError`.

        :param wrapper: callable mapping a model to a model with the same call signature.
        :return: a new kernel bound to the wrapped model.
        """
        ...
```

### 4.2 `numpyro/infer/hmc.py`: public accessors on `HMC`

```python
class HMC(MCMCKernel):
    def get_potential_fn(
        self,
        model_args: ModelArgs = (),
        model_kwargs: ModelKwargs | None = None,
    ) -> PotentialFn:
        """
        Return the potential energy function (negative log joint in unconstrained space) for the
        given model arguments; today's private `_potential_fn_gen(*model_args, **model_kwargs)`.
        Requires :meth:`init` to have run. Raises `RuntimeError` otherwise.
        """
        ...

    def get_constrain_fn(
        self,
        model_args: ModelArgs = (),
        model_kwargs: ModelKwargs | None = None,
        *,
        return_deterministic: bool = False,
    ) -> ConstrainFn:
        """
        Return a function mapping unconstrained sample values to constrained values. When
        `return_deterministic=False` and the model has no value-dependent supports, this is a
        transform-only function (:func:`~numpyro.infer.util.transform_fn`) that never runs the
        model; otherwise it replays the model (:func:`~numpyro.infer.util.constrain_fn`).
        Composite kernels use the transform-only form to condition sibling blocks.
        """
        ...

    def refresh(
        self,
        state: HMCState,
        model_args: ModelArgs,
        model_kwargs: ModelKwargs | None,
    ) -> HMCState:
        """
        Recompute `potential_energy` and `z_grad` at `state.z` with the potential for the given
        arguments (via :func:`~numpyro.infer.hmc_util._value_and_grad`, honoring
        `forward_mode_differentiation`). `energy` is left as is: `sample` recomputes it from the
        potential and the fresh momentum (`hmc.py:396-401`).
        """
        ...

    def wrap_model(self, wrapper: ModelWrapper) -> "HMC":
        """
        Shallow copy with `_model = wrapper(self._model)` and `_init_fn`, `_sample_fn`,
        `_potential_fn_gen`, `_postprocess_fn` reset to `None`, so the copy rebuilds its closures
        against the wrapped model on its next `init` (see `hmc.py:706-712`). Raises `ValueError`
        for kernels constructed with `potential_fn`.
        """
        ...
```

`NUTS` inherits all four. `MixedHMC` uses `isinstance(inner_kernel, NUTS)` instead of `_algo`.

### 4.3 `numpyro/infer/util.py` and `numpyro/util.py`: small shared helpers

```python
class _unconstrain_params(substitute):
    """
    The handler :func:`potential_energy` uses to substitute unconstrained `params` through
    :func:`_unconstrain_reparam`. Exposes `.params` so that model wrappers that need the current
    unconstrained values (:class:`~numpyro.infer.hmc_gibbs.estimate_likelihood`) can find it with
    `isinstance` instead of inspecting `substitute_fn.func`.
    """

    params: SiteValues


def _prepare_model_for_potential(model: ModelT, model_trace: TraceT) -> ModelT:
    """
    The model preparation :func:`initialize_model` performs before building a potential
    (`util.py:731-761`): substitute `param`/`mutable` values from the trace, add a default PRNG
    key, wrap with `enum(config_enumerate(...))` when the trace contains discrete latent sites
    marked `enumerate="parallel"`, and validate plates. Extracted so that
    :class:`~numpyro.infer.gibbs.DiscreteGibbs` builds its potential from the same prepared
    model as HMC.
    """
    ...


def _get_model_transforms(model, model_args=(), model_kwargs=None):
    """
    Unchanged signature. Internally distinguishes `has_deterministic` from `dynamic_support`;
    `replay_model = has_deterministic or dynamic_support` as before, and the two flags are
    returned in the trace metadata so :meth:`HMC.get_constrain_fn` can pick the transform-only
    path when only deterministic sites are present.
    """
    ...
```

```python
# numpyro/util.py
def _get_nested_attr(obj: PyTree, field: str) -> PyTree:
    """
    As today, plus: when `obj` is a tuple or list and `attr` is a decimal string, index by
    `int(attr)`. Enables `extra_fields=("block_states.1.diverging",)` for composite kernels.
    """
    ...
```

### 4.4 New module `numpyro/infer/gibbs_util.py` (pure functions, no kernel classes)

```python
GIBBS_SITES_KWARG: str = "_gibbs_sites"
"""Reserved model keyword through which conditioning values travel. Documented as reserved."""


def conditioned(model: ModelT) -> ModelT:
    """
    Return a model that pops `GIBBS_SITES_KWARG` from its keyword arguments and runs `model`
    under `condition(data=values)` and `substitute(data=values)`. Idempotent: wrapping an already
    conditioned model returns it unchanged. This is today's `_wrap_model` (`hmc_gibbs.py:32-35`)
    given a name and an idempotency guard.
    """
    ...


def _conditioned_model(model: ModelT, *args: Any, **kwargs: Any) -> Any:
    """Module-level target of :func:`conditioned` (kept module-level so kernels pickle)."""
    ...


def with_conditioning(
    model_kwargs: ModelKwargs | None,
    values: SiteValues,
    *,
    allowed: frozenset[str] | None = None,
) -> ModelKwargs:
    """
    Return a copy of `model_kwargs` whose `GIBBS_SITES_KWARG` entry is the existing entry (if
    any) extended by `values`. Extend-not-replace is what makes nested composites correct: at any
    depth a block sees exactly the sites it does not own. When `allowed` is given, raise if a key
    of `values` is outside it (guards against leaking deterministic sites into the model).
    """
    ...


def prototype_trace(
    model: ModelT,
    rng_key: jax.Array,
    model_args: ModelArgs,
    model_kwargs: ModelKwargs | None,
) -> TraceT:
    """
    `trace(substitute(seed(model, rng_key), substitute_fn=init_to_sample)).get_trace(...)`,
    the idiom currently repeated at `hmc_gibbs.py:129-131, 424-426, 582-584`. Callers must not
    store the returned trace on a kernel object (it may hold tracers under `pmap`/`vmap`);
    store only static metadata derived from it.
    """
    ...


def latent_sample_sites(trace: TraceT) -> tuple[str, ...]:
    """Names of unobserved `sample` sites in trace order."""
    ...


def discrete_latent_sites(trace: TraceT) -> tuple[str, ...]:
    """
    Unobserved sample sites whose distribution has enumerate support and which are not marked
    `infer={"enumerate": "parallel"}` (`hmc_gibbs.py:437-444`). Usable as a `SitesSpec`
    selector.
    """
    ...


def discrete_support_sizes(trace: TraceT, sites: Sequence[str]) -> dict[str, np.ndarray]:
    """Per-site support sizes broadcast to the site's shape, as `numpy` arrays (static)."""
    ...


def subsample_plate_sizes(trace: TraceT) -> dict[str, tuple[int, int]]:
    """`{plate_name: (size, subsample_size)}` for plates with `size > subsample_size`."""
    ...


def any_changed(old: PyTree, new: PyTree) -> jax.Array:
    """Scalar boolean: whether any leaf of two pytrees with the same structure differs."""
    ...


# discrete proposals, moved from hmc_gibbs.py:194-345 with typed signatures. Two fixes on move:
# use the split key for the permutation (:314-315), and write the current value, not the index,
# in the modified random-walk stay branch (:300).

ProposalFn: TypeAlias = Callable[
    [jax.Array, SiteValues, jax.Array, PotentialFn, jax.Array, jax.Array],
    tuple[jax.Array, SiteValues, jax.Array, jax.Array],
]
"""`(rng_key, z, pe, potential_fn, idx, support_size) -> (rng_key, z_new, pe_new, log_accept_ratio)`."""


def _discrete_gibbs_proposal(rng_key, z, pe, potential_fn, idx, support_size): ...
def _discrete_modified_gibbs_proposal(rng_key, z, pe, potential_fn, idx, support_size, stay_prob=0.0): ...
def _discrete_rw_proposal(rng_key, z, pe, potential_fn, idx, support_size): ...
def _discrete_modified_rw_proposal(rng_key, z, pe, potential_fn, idx, support_size, stay_prob=0.0): ...


def select_discrete_proposal(random_walk: bool, modified: bool) -> ProposalFn:
    """The dispatch table at `hmc_gibbs.py:404-417`."""
    ...


def discrete_gibbs_sweep(
    rng_key: jax.Array,
    z: SiteValues,
    potential_energy: jax.Array,
    potential_fn: PotentialFn,
    support_sizes_flat: jax.Array,
    proposal_fn: ProposalFn,
) -> tuple[SiteValues, jax.Array]:
    """
    One sweep over the flat discrete coordinates in a random permutation, each coordinate updated
    with `proposal_fn` and Metropolis corrected (`fori_loop` + `cond`, today's `_discrete_gibbs_fn`).
    Returns the new values and their potential energy.
    """
    ...
```

### 4.5 New module `numpyro/infer/gibbs.py`: the composite and the two generic block kernels

```python
from numpyro.infer.gibbs_util import GIBBS_SITES_KWARG, GibbsUpdateFn, ModelWrapper, SitesSpec, conditioned, with_conditioning


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


class Gibbs(MCMCKernel):
    """
    Composite kernel that updates the latent sites of a model block by block. Each block is a
    `(kernel, sites)` pair; the block kernel is run on the model conditioned on the current
    values of every site it does not own. Blocks are visited in order once per MCMC step.

    :param blocks: sequence of `(kernel, sites)`. `kernel` is any :class:`~numpyro.infer.mcmc.MCMCKernel`
        that implements :meth:`~numpyro.infer.mcmc.MCMCKernel.refresh` (and
        :meth:`~numpyro.infer.mcmc.MCMCKernel.wrap_model` if it holds a model). `sites` is a
        sequence of site names, a callable mapping a prototype trace to site names (for example
        :func:`~numpyro.infer.gibbs_util.discrete_latent_sites`), or `None` for "all latent
        sample sites not owned by another block" (allowed for at most one block). All model-based
        blocks must be built on the same model callable.

    Validation at construction: at least one block; kernels override `refresh`; model-based
    kernels share one model; each model-based kernel is rebound once with
    ``kernel.wrap_model(conditioned)``.

    Validation at `init` (errors, never warnings, because the fallbacks inside
    :func:`~numpyro.infer.util.initialize_model` are silent): the union of block sites equals the
    set of unobserved sample sites minus those marked `enumerate="parallel"`; no discrete latent
    is left to an HMC block; blocks are disjoint; no site is already conditioned by an enclosing
    composite; `init_params` keys lie in the union (unconstrained values for HMC blocks,
    constrained values otherwise); the model has no subsample plates (use :class:`~numpyro.infer.hmc_gibbs.HMCECS`).

    **Example**

    .. doctest::

        >>> from jax import random
        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.infer import MCMC, NUTS, Gibbs, DiscreteGibbs, CustomGibbs
        >>> from numpyro.infer.gibbs_util import discrete_latent_sites
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

    def __init__(self, blocks: Sequence[tuple[MCMCKernel, SitesSpec]]) -> None: ...

    @property
    def model(self) -> ModelT | None:
        """The shared, unwrapped model of the model-based blocks (`None` if there is none)."""
        ...

    @property
    def default_fields(self) -> tuple[str, ...]:
        """`("z",)`. Static so that :class:`~numpyro.infer.mcmc.MCMC` can snapshot it."""
        ...

    def init(
        self,
        rng_key: jax.Array,
        num_warmup: int,
        init_params: SiteValues | None,
        model_args: ModelArgs,
        model_kwargs: ModelKwargs | None,
    ) -> GibbsState:
        """
        Single-key only in v1 (`is_prng_key(rng_key)` asserted; use `chain_method="parallel"`,
        `"sequential"` or a callable). Steps: prototype trace (local, not stored); resolve and
        validate the partition; split `init_params` by block without mutating the caller's dict;
        initialize blocks in order, each with `model_kwargs` extended by the constrained initial
        values of the other blocks; run one :meth:`refresh` pass so every block's cached
        potential/gradient is consistent with the final initial values; set `self._sample_fn` so
        :class:`~numpyro.infer.mcmc.MCMC` does not re-run `init` on every `run()`
        (`mcmc.py:475`); return `self._state_cls(z, block_states, rng_key)`.
        """
        ...

    def sample(
        self,
        state: GibbsState,
        model_args: ModelArgs,
        model_kwargs: ModelKwargs | None,
    ) -> GibbsState:
        """
        One sweep. Python loop over blocks, unrolled inside the enclosing `jit`. Keeps a local
        dict of constrained sibling values (computed once at the start of the sweep with each
        block's transform-only constrain function, then updated only for the block that just
        moved). For block `i`: `kwargs_i = with_conditioning(model_kwargs, siblings_i)`;
        `state_i = cond(changed_i, lambda s: kernel_i.refresh(s, args, kwargs_i), identity, state_i)`
        where `changed_i` records whether any sibling moved since block `i` was last refreshed;
        `state_i = kernel_i.sample(state_i, args, kwargs_i)`. Returns `state._replace(...)` so the
        concrete state class (and treedef) is preserved.
        """
        ...

    def refresh(self, state: GibbsState, model_args: ModelArgs, model_kwargs: ModelKwargs | None) -> GibbsState:
        """Identity: :meth:`sample` refreshes each block against the current conditioning."""
        ...

    def wrap_model(self, wrapper: ModelWrapper) -> "Gibbs":
        """New composite with `wrapper` applied to every block (nesting)."""
        ...

    def postprocess_fn(
        self,
        model_args: ModelArgs,
        model_kwargs: ModelKwargs | None,
    ) -> ConstrainFn:
        """
        Two phases: (1) per block, constrain that block's own sites with the block's transform-only
        constrain function; (2) only if the prototype trace has `deterministic` sites, replay the
        unwrapped model once with all constrained values substituted to collect them (the
        :class:`~numpyro.infer.Predictive` idiom). Phase 1 never replays the model, which avoids
        the chicken-and-egg problem of replaying block A's model before block B is constrained.
        """
        ...

    def get_diagnostics_str(self, state: GibbsState) -> str:
        """Non-empty block diagnostics joined with `" | "`."""
        ...

    def __getstate__(self) -> dict[str, Any]:
        """Drops `_sample_fn`; nothing traced is ever stored on the kernel."""
        ...

    # private helpers
    def _resolve_partition(self, trace: TraceT, conditioned: frozenset[str]) -> tuple[tuple[str, ...], ...]: ...
    def _split_init_params(self, init_params: SiteValues | None) -> tuple[SiteValues | None, ...]: ...
    def _sample_one(self, state: GibbsState, model_args: ModelArgs, model_kwargs: ModelKwargs | None) -> GibbsState: ...


class CustomGibbsState(NamedTuple):
    """
    - **z** - dict of the block's current (constrained) values.
    - **rng_key** - random key for the next step.
    """

    z: SiteValues
    rng_key: jax.Array


class CustomGibbs(MCMCKernel):
    """
    Block kernel that delegates the update to a user callable. The callable receives the current
    values of the block's sites and the constrained values of all conditioning sites and returns
    new values for the block's sites (it must sample from the conditional; correctness is the
    user's responsibility, as with :class:`~numpyro.infer.hmc_gibbs.HMCGibbs` today).

    :param gibbs_fn: called as `gibbs_fn(rng_key=..., gibbs_sites=..., hmc_sites=...)`; the
        keyword names are kept for compatibility with existing `HMCGibbs` users, and `hmc_sites`
        holds all conditioning sites (not only HMC ones). The returned dict must have exactly the
        block's site names as keys (validated at trace time).
    """

    sample_field: str = "z"

    def __init__(self, gibbs_fn: GibbsUpdateFn) -> None: ...

    def init(self, rng_key: jax.Array, num_warmup: int, init_params: SiteValues | None, model_args: ModelArgs, model_kwargs: ModelKwargs | None) -> CustomGibbsState:
        """`init_params` are the block's initial values (constrained); the composite supplies prototype values when absent."""
        ...

    def sample(self, state: CustomGibbsState, model_args: ModelArgs, model_kwargs: ModelKwargs | None) -> CustomGibbsState:
        """Reads conditioning values from `model_kwargs[GIBBS_SITES_KWARG]`."""
        ...

    def refresh(self, state: CustomGibbsState, model_args: ModelArgs, model_kwargs: ModelKwargs | None) -> CustomGibbsState:
        """Identity: nothing is cached."""
        ...

    def wrap_model(self, wrapper: ModelWrapper) -> "CustomGibbs":
        """Returns `self`: there is no model."""
        ...


class DiscreteGibbsState(NamedTuple):
    """
    - **z** - dict of the current discrete values.
    - **potential_energy** - potential energy at `z` under the current conditioning (the block's
      own potential; see section 5 for why it is not interchangeable with HMC's).
    - **rng_key** - random key for the next step.
    """

    z: SiteValues
    potential_energy: jax.Array
    rng_key: jax.Array


class DiscreteGibbs(MCMCKernel):
    """
    Metropolis / Gibbs updates of the discrete latent sites of a model, one flat coordinate at a
    time in a random order (today's `DiscreteHMCGibbs` update, `hmc_gibbs.py:308-345`). Usable
    standalone on a purely discrete model, or as a block of :class:`Gibbs`.

    The potential is `potential_energy(prepared_model, model_args, kwargs, params={})` where
    `prepared_model` comes from :func:`~numpyro.infer.util._prepare_model_for_potential` (so
    `enumerate="parallel"` sites are marginalized exactly as in HMC) and the block's own values
    are injected through the conditioning channel (`with_conditioning(kwargs, z)`); values cannot
    be passed as `params` because :func:`~numpyro.infer.util._unconstrain_reparam` has no
    bijector for discrete supports (`util.py:305-308`).

    :param model: the model.
    :param bool random_walk: uniform proposals over the support instead of the exact conditional.
    :param bool modified: Liu's modified (Metropolised) proposal that never proposes the current value.

    **References:**

    1. *Peskun's theorem and a modified discrete-state Gibbs sampler*, Liu, J. S. (1996)
    """

    sample_field: str = "z"

    def __init__(self, model: ModelT, *, random_walk: bool = False, modified: bool = False) -> None: ...

    @property
    def model(self) -> ModelT: ...

    def get_potential_fn(
        self,
        model_args: ModelArgs = (),
        model_kwargs: ModelKwargs | None = None,
    ) -> PotentialFn:
        """Potential over the block's discrete values for the given arguments and conditioning."""
        ...

    def init(self, rng_key: jax.Array, num_warmup: int, init_params: SiteValues | None, model_args: ModelArgs, model_kwargs: ModelKwargs | None) -> DiscreteGibbsState:
        """
        Prototype trace (local); sites = :func:`discrete_latent_sites`; support sizes stored as
        `numpy` (static); raise if unconditioned continuous latents remain (standalone use on a
        mixed model); initial `potential_energy` at the initial values.
        """
        ...

    def sample(self, state: DiscreteGibbsState, model_args: ModelArgs, model_kwargs: ModelKwargs | None) -> DiscreteGibbsState:
        """:func:`~numpyro.infer.gibbs_util.discrete_gibbs_sweep` with the selected proposal."""
        ...

    def refresh(self, state: DiscreteGibbsState, model_args: ModelArgs, model_kwargs: ModelKwargs | None) -> DiscreteGibbsState:
        """Recompute `potential_energy` at `state.z` (one model evaluation)."""
        ...

    def wrap_model(self, wrapper: ModelWrapper) -> "DiscreteGibbs": ...

    def __getstate__(self) -> dict[str, Any]: ...
```

### 4.6 `numpyro/infer/hmc_gibbs.py`: facades, `HMCECS` as a wrapper kernel

Every name importable today from `numpyro.infer.hmc_gibbs` stays importable (`HMCGibbs`, `DiscreteHMCGibbs`, `HMCECS`, `HMCGibbsState`, `HMCECSState`, `estimate_likelihood`, `taylor_proxy`), which keeps the `docs/source/mcmc.rst:62-136` anchors valid.

```python
class HMCGibbsState(GibbsState):
    """
    :class:`~numpyro.infer.gibbs.GibbsState` for the two-block facades, with an `hmc_state`
    property returning the HMC block's :data:`~numpyro.infer.hmc.HMCState` (`block_states[-1]`)
    so `extra_fields=["hmc_state.potential_energy"]` keeps working (`nested_attrgetter` uses
    `getattr`, `numpyro/util.py:850-859`). Minor break: positional construction and tuple
    unpacking of the old three-field namedtuple.
    """

    __slots__ = ()

    @property
    def hmc_state(self) -> HMCState: ...


class HMCGibbs(Gibbs):
    """
    [EXPERIMENTAL INTERFACE] HMC-within-Gibbs with a user supplied `gibbs_fn`. Facade over
    ``Gibbs([(CustomGibbs(gibbs_fn), gibbs_sites), (inner_kernel, None)])``. Constructor,
    docstring example and behavior unchanged; `inner_kernel` remains an attribute.
    """

    _state_cls = HMCGibbsState

    def __init__(self, inner_kernel: HMC, gibbs_fn: GibbsUpdateFn, gibbs_sites: Sequence[str]) -> None: ...


class DiscreteHMCGibbs(Gibbs):
    """
    [EXPERIMENTAL INTERFACE] Facade over
    ``Gibbs([(DiscreteGibbs(inner_kernel.model, random_walk=..., modified=...), discrete_latent_sites), (inner_kernel, None)])``.
    """

    _state_cls = HMCGibbsState

    def __init__(self, inner_kernel: HMC, *, random_walk: bool = False, modified: bool = False) -> None: ...


HMCECSState = namedtuple("HMCECSState", "z, hmc_state, rng_key, gibbs_state, accept_prob")  # unchanged


def _ecs_model(model: ModelT, estimator: "estimate_likelihood", *args: Any, **kwargs: Any) -> Any:
    """
    Wrapper installed once by :class:`HMCECS`: pops `_gibbs_state` from the keyword arguments,
    hands it to `estimator`, and runs the (already conditioned) model under `estimator`.
    Replaces `_wrap_gibbs_state` and the fake `"_gibbs_state"` message (`hmc_gibbs.py:495-499`).
    """
    ...


class estimate_likelihood(numpyro.primitives.Messenger):
    """
    As today (`hmc_gibbs.py:692-770`) with two changes: `method` and `gibbs_state` are set late
    (at `HMCECS.init` and per call, respectively) so the messenger can be created once at
    construction; the unconstrained parameters are found with
    `isinstance(handler, _unconstrain_params)` instead of inspecting `substitute_fn.func`.
    """

    def __init__(self, fn: ModelT | None = None, method: LikelihoodEstimator | None = None) -> None: ...


class HMCECS(MCMCKernel):
    """
    [EXPERIMENTAL INTERFACE] HMC with Energy Conserving Subsampling. A wrapper around an HMC
    kernel (not a :class:`Gibbs` block): it changes the inner target (likelihood estimator) and
    performs the pseudo-marginal Metropolis update of the subsample indices, using only the
    public HMC accessors (:meth:`~numpyro.infer.hmc.HMC.get_potential_fn`,
    :meth:`~numpyro.infer.hmc.HMC.refresh`, `sample_field`). Constructor, references, example,
    `taylor_proxy` and `HMCECSState` unchanged. Wraps the inner kernel exactly once, at
    construction, with ``inner.wrap_model(lambda m: partial(_ecs_model, conditioned(m), self._estimator))``
    written as a module-level function; `init` only binds the estimator's `method` and the proxy
    state, so `warmup()` followed by `run()` is safe.

    Refresh of the HMC state after an accepted subsample stays inside `cond(transition, ...)`, so
    the per-step cost is unchanged. `init` raises if any conditioned site (from an enclosing
    composite) sits inside a subsample plate.
    """

    sample_field: str = "z"

    def __init__(self, inner_kernel: HMC, *, num_blocks: int = 1, proxy: ProxyConstructor | None = None) -> None: ...

    @property
    def model(self) -> ModelT: ...

    def init(self, rng_key: jax.Array, num_warmup: int, init_params: SiteValues | None, model_args: ModelArgs, model_kwargs: ModelKwargs | None) -> HMCECSState: ...
    def sample(self, state: HMCECSState, model_args: ModelArgs, model_kwargs: ModelKwargs | None) -> HMCECSState: ...
    def refresh(self, state: HMCECSState, model_args: ModelArgs, model_kwargs: ModelKwargs | None) -> HMCECSState:
        """Delegates to the inner kernel with the current subsample indices and proxy state in the kwargs."""
        ...
    def wrap_model(self, wrapper: ModelWrapper) -> "HMCECS": ...
    def postprocess_fn(self, model_args: ModelArgs, model_kwargs: ModelKwargs | None) -> ConstrainFn:
        """Inner postprocess on the HMC sites; subsample indices are dropped (as today)."""
        ...
    def get_diagnostics_str(self, state: HMCECSState) -> str: ...

    @staticmethod
    def taylor_proxy(reference_params: SiteValues, degree: int = 2) -> ProxyConstructor: ...

    def __getstate__(self) -> dict[str, Any]: ...
```

### 4.7 `numpyro/infer/mixed_hmc.py`

```python
class MixedHMC(MCMCKernel):
    """
    Mixed HMC (Zhou 2020). No longer a subclass of `DiscreteHMCGibbs`: it is an HMC-family
    wrapper that interleaves discrete updates inside the trajectory, so it cannot be expressed as
    sequential blocks. Constructor, `MixedHMCState`, references and example unchanged. Uses
    :func:`~numpyro.infer.gibbs_util.discrete_latent_sites`,
    :func:`~numpyro.infer.gibbs_util.discrete_support_sizes`,
    :func:`~numpyro.infer.gibbs_util.select_discrete_proposal`,
    :meth:`~numpyro.infer.hmc.HMC.get_potential_fn` and :meth:`~numpyro.infer.hmc.HMC.refresh`.
    It keeps building its own warmup adapter from the inner kernel's adaptation settings
    (documented intra-package access to `_adapt_step_size`, `_adapt_mass_matrix`, `_dense_mass`,
    `_target_accept_prob`; a public accessor is not worth adding for one caller). Requires an
    :class:`~numpyro.infer.hmc.HMC` inner kernel that is not :class:`~numpyro.infer.hmc.NUTS`.
    Implements `refresh` and `wrap_model` so it can be a block of :class:`~numpyro.infer.gibbs.Gibbs`.
    """

    sample_field: str = "z"

    def __init__(self, inner_kernel: HMC, *, num_discrete_updates: int | None = None, random_walk: bool = False, modified: bool = False) -> None: ...
    def init(self, rng_key: jax.Array, num_warmup: int, init_params: SiteValues | None, model_args: ModelArgs, model_kwargs: ModelKwargs | None) -> MixedHMCState: ...
    def sample(self, state: MixedHMCState, model_args: ModelArgs, model_kwargs: ModelKwargs | None) -> MixedHMCState: ...
    def refresh(self, state: MixedHMCState, model_args: ModelArgs, model_kwargs: ModelKwargs | None) -> MixedHMCState: ...
    def wrap_model(self, wrapper: ModelWrapper) -> "MixedHMC": ...
    def postprocess_fn(self, model_args: ModelArgs, model_kwargs: ModelKwargs | None) -> ConstrainFn: ...
    def get_diagnostics_str(self, state: MixedHMCState) -> str: ...
    def __getstate__(self) -> dict[str, Any]: ...
```

### 4.8 Exports and docs

`numpyro/infer/__init__.py` adds `Gibbs`, `CustomGibbs`, `DiscreteGibbs`. `docs/source/mcmc.rst` adds `autoclass` entries for the three, changes the `HMCGibbsState` `autodata` to `autoclass`, adds `GibbsState`, and adds one overview bullet for `Gibbs`.

## 5. Data flow, performance and JAX considerations

### 5.1 One MCMC step of `Gibbs`

1. `siblings = {i: transform_only_constrain_i(z_i) restricted to sites_i}` for every block, computed once at the start of the sweep (or carried from the previous sweep in a follow-up).
2. For block `i` in order: `kwargs_i = with_conditioning(model_kwargs, union of siblings[j] for j != i)`; `state_i = cond(changed_i, refresh_i, identity)`; `state_i = kernel_i.sample(state_i, args, kwargs_i)`; `siblings[i]` recomputed; `changed_j |= any_changed(old_i, new_i)` for `j != i`.
3. `z = merge of block sample fields`; return `state._replace(z=z, block_states=..., rng_key=...)`.

Everything runs inside the single `jit` of `fori_collect`; the block loop is unrolled at trace time; conditioning values are traced kwargs so nothing recompiles across steps or `run()` calls (`fori_collect` caches its body by `(body_fun, transform)` identity, `numpyro/util.py:384-386`, and `MCMC._get_cached_fns` keys on the argument objects, `mcmc.py:401-459`; new dicts and partials created inside `Gibbs.sample` live inside that one trace). Compile time is O(number of blocks) since each HMC block traces its own leapfrog/tree loop over the full model.

### 5.2 Cost per MCMC step

Notation: E = one model log-density evaluation; VG = value and gradient (reverse mode; `grad` alone costs the same); T = transform-only constrain (elementwise bijectors, negligible); R = model replay for deterministic sites; L = leapfrog steps or tree proposals; S = discrete-sweep evaluations (sum over coordinates of `support_size - 1`, or the number of coordinates for random-walk proposals); MCMC's own postprocess of collected samples is identical before and after and omitted.

| Kernel | Today | Proposed |
|---|---|---|
| `HMCGibbs` | T/R + `gibbs_fn` + 1 VG + L VG (`hmc_gibbs.py:166-182`) | T + `gibbs_fn` + 1 VG + L VG |
| `DiscreteHMCGibbs` | S E + 1 VG + L VG (`:466-483`; no constrain, `pe` handed from the sweep to HMC) | T + 1 E + S E + 1 VG + L VG |
| `HMCECS` | 1 E + [1 VG on accept] + L VG (`:653-678`) | identical (wrapper keeps the `cond`) |
| `MixedHMC` | per discrete update: proposal E's + [1 VG on refract] + sub-trajectory | identical |
| N HMC blocks | n/a | N T + N VG + sum of L_i VG |

Notes.

- The `DiscreteHMCGibbs` regression is exactly +1 E: the discrete block's own potential is `-log p` at constrained values with `params={}`, whereas HMC's potential includes the `_{name}_log_det` factors from `_unconstrain_reparam` (`util.py:329`) for the HMC sites; the difference is constant in the discrete values, so the sweep is correct, but the absolute values differ and cannot be handed across blocks. Relative size is 1 E / (S E + (L + 1) VG), roughly 5 to 8 percent for the doctest GMM (K = 4, one discrete site, NUTS depth 2 to 3) and well under 1 percent for many discrete coordinates. Partial offset: each of the S sweep evaluations no longer pays the bijectors and log-det terms of the continuous sites.
- The `changed`-gated refresh (`cond`, executed one branch under `jit`) saves the 1 VG whenever a discrete sweep changed nothing (sticky assignments in a well separated mixture) and the 1 E whenever HMC rejected; under `vmap` both branches run and it costs nothing extra. It is the pattern `HMCECS.sample` (`hmc_gibbs.py:663-672`) and `MixedHMC` (`mixed_hmc.py:147-160`) already use.
- Refresh cannot be folded into the first leapfrog step: `sample_kernel` never calls `velocity_verlet.init_fn`; it builds `IntegratorState(z, r, potential_energy, z_grad)` from the state (`hmc.py:482-484`) and needs `U(z0)` and `grad U(z0)` under the new conditioning before `z` moves. In plain HMC that gradient is free because the previous step's last leapfrog gradient is reused; conditioning changes invalidate it, so exactly 1 VG per HMC block per sweep is intrinsic to HMC-within-Gibbs (today's code pays it at `hmc_gibbs.py:176`, `:479`). `None`/NaN sentinels in `HMCState` are not an option: the state is a `fori_loop`/`while_loop` carry and must keep a fixed treedef.
- Sibling constrained values with the transform-only `get_constrain_fn` cost N T per sweep (incremental local dict); the naive "postprocess every sibling for every block" would be N(N-1) and, worse, would replay the model whenever it has deterministic sites (today's `HMCGibbs.sample:166` pays that R every step).

### 5.3 State, pytrees, RNG

- `GibbsState` and the block states are `NamedTuple` pytrees; `Gibbs.sample` returns `state._replace(...)`, never a fresh `GibbsState(...)`, so the facade subclass survives the loop carry.
- The duplicate copy of `z` in the loop carry is negligible (the carry is never donated, `numpyro/util.py:384, 412-416`); collected samples hold `z` once.
- Side benefit: dense mass matrices become block diagonal per HMC block, and NUTS checkpoints shrink accordingly.
- Static metadata only on `self` (site tuples, `numpy` support sizes, python-int plate sizes); traces are local to `init`.
- Keys are split explicitly for every consumer (fixing the three unused-key sites); fixed-seed regression numbers will move.

### 5.4 Warmup and adaptation

Per-block adaptation on the moving conditional is the right target for HMC-within-Gibbs (conditional covariance, step size averaged over conditionals); a joint warmup would adapt to the marginal geometry. Blocks share `num_warmup`, so windows align. Two caveats to document: `find_heuristic_step_size=True` binds `pe_fn` at init (`hmc.py:316-323`) and re-runs it at every window end (`hmc_util.py:618-621`) against the initial conditioning, so it should stay `False` for blocks (letting `warmup_adapter` receive `pe_fn` per call is a small HMC-internal follow-up); mixing degrades with cross-block posterior correlation, so correlated continuous sites belong in one HMC block.

### 5.5 Vectorized chains (follow-up)

`HMC.init` self-vmaps `_sample_fn` with `in_axes=(0, None, None)` (`hmc.py:796-798`), i.e. kwargs are shared across chains, so per-chain conditioning cannot flow through a self-vmapped block. Therefore blocks must stay single-chain and the composite must own the `vmap`: `vmap(_init_one)` and `self._sample_fn = vmap(_sample_one, in_axes=(0, None, None))`, exactly what `chain_method=jax.vmap` already does to today's `HMCGibbs`. Pitfalls to handle then: the prototype trace must be taken with `rng_key[0]` outside `vmap`; `initialize_model` substitutes `param`/`mutable` values from the trace (`util.py:734-741`), which are tracers under `vmap`; `find_valid_initial_params` cannot raise under trace. v1 asserts a single key with a message pointing to `"parallel"`, `"sequential"` or a callable chain method (as `BarkerMH`, `barker.py:173-177`).

### 5.6 Optional optimization: vmapped discrete sweep

The comment at `hmc_gibbs.py:225-229` ("we can't vmap ... support_size is a traced value") holds only for the per-coordinate value; the set of distinct support sizes is static at `init`. With `lax.switch` over the distinct sizes at each traced coordinate, each branch can `vmap(potential_fn)` over its K-1 proposals and sample with `random.categorical`, replacing K-1 sequential model evaluations per coordinate by one batched evaluation (the modified proposals batch the same way; random-walk proposals have nothing to batch). For a GMM with K = 4 and 100 discrete coordinates this turns 300 sequential evaluations per sweep into 100 batched ones, roughly 2 to 3x on the sweep on GPU where small evaluations are launch bound, more for larger K. It also lets the +1 E above ride along with the first coordinate's batch. `MixedHMC` keeps the traced-support path (its coordinate comes from `argmin` of arrival times). Recorded as a follow-up in `gibbs_util.py`; RNG consumption changes, tests are statistical. Separately, when discrete sites are conditionally independent given the continuous block, a `CustomGibbs` block that computes the vectorized conditional does the whole update in O(1) evaluations, which is exactly what the composable API enables.

## 6. Migration and PR sequence (tests green at every step)

1. Hooks with no behavior change: `MCMCKernel.refresh`/`wrap_model` (raising defaults); `HMC.get_potential_fn`/`get_constrain_fn`/`refresh`/`wrap_model`; `_unconstrain_params`; `_prepare_model_for_potential`; `has_deterministic`/`dynamic_support` split; `_get_nested_attr` integer index. Rewrite the four existing kernels to use the accessors instead of `_potential_fn_gen`/`_replace(z_grad=...)`. `HMCECS` wraps once with the late-bound estimator; regression tests for `warmup()` then `run()`, two `run()` calls, and reusing an already-run `NUTS` instance in a Gibbs kernel. RNG hygiene fixes.
2. `gibbs_util.py` and `gibbs.py` with `Gibbs` and `CustomGibbs`; tests mirroring `test_linear_model_*` and `test_gaussian_model` through `Gibbs([(CustomGibbs(fn), sites), (NUTS(model), None)])`; nested composite smoke test; pickle; `jit_model_args=True`; `chain_method=jax.vmap`; a `scan` model. `HMCGibbs` untouched in this PR.
3. `DiscreteGibbs` and the proposal move; standalone tests (`MCMC(DiscreteGibbs(model))` on a purely discrete model); `Gibbs([(DiscreteGibbs(model), discrete_latent_sites), (NUTS(model), None)])` on the existing bernoulli, GMM, enumeration and multi-site cases.
4. Facades `HMCGibbs`/`DiscreteHMCGibbs` over `Gibbs` with `HMCGibbsState`; `MixedHMC` decoupled from `DiscreteHMCGibbs`; docs.
5. `HMCECS` rewritten as a wrapper on the public accessors; explicit error for conditioned sites inside subsample plates.
6. Follow-ups: vectorized chains (5.5); vmapped discrete sweep (5.6); `SA.refresh` (recompute `adapt_state.pes`, `sa.py:183`) and `BarkerMH` (rebuild its potential per call instead of at init, `barker.py:167`) so both can be blocks; `MCMCKernel.is_initialized` to replace the `_sample_fn` duck typing at `mcmc.py:475`; `HMCECS(Gibbs(...))`.

## 7. Test plan

- Unit tests per block in isolation: `CustomGibbs` (returned keys validated), `DiscreteGibbs` standalone (support sizes, permutation, `refresh`), `HMC.refresh` equals `value_and_grad` of `get_potential_fn`, `HMC.get_constrain_fn` transform-only vs replay, `wrap_model` resets closures.
- Invariance: `Gibbs([(CustomGibbs(fn), sites), (NUTS(model), None)])` produces the same samples as `HMCGibbs(NUTS(model), fn, sites)` for the same key.
- Statistical tests reusing the existing models in `test/infer/test_hmc_gibbs.py`, plus a three-block model and a nested composite.
- Error paths: overlapping blocks, uncovered site, leftover discrete site, `init_params` with unknown keys, subsample plate inside a composite, block kernel without `refresh`, two different models.
- Lifecycle: `warmup()` then `run()`; second `run()` after `chain_method="parallel"`; pickle then `run()` from `post_warmup_state`; `extra_fields=("block_states.1.diverging", "hmc_state.potential_energy")`.

## 8. Alternatives considered and open questions

Alternatives considered (rejected, with the reason):

- `sites=` argument on every kernel: the partition belongs to the composite, not to a kernel; a kernel does not know its siblings and `NUTS(model, sites=[...])` alone means nothing.
- An explicit `Block` protocol (`init_block`, `sample_block(state, z_rest, ...)`) with adapters per kernel family: the only per-step input channel `MCMCKernel.sample` offers is `model_kwargs`, so adapters would put `z_rest` there anyway; one more class per family and a second `sample`-like method for the same mechanism.
- A `Conditioning`/`Target` object passed to blocks: needs a new signature or the same kwargs smuggling.
- Native `condition=` in `initialize_model`/`potential_energy`: every model call site (`_get_model_transforms`, `find_valid_initial_params`, `log_density`, `constrain_fn`, funsor `log_density`, `Predictive`) would have to strip a reserved kwarg; wrapping the model once touches one place and composes with enumeration for free.
- A `Messenger` subclass for the conditioning wrapper: must mutate handler state per call to pop the kwarg; the stateless `partial` is three lines and equally picklable. A messenger is used only where the state is per call anyway (`estimate_likelihood`).
- `z` as a property computed from `block_states`: the property would need each block's `sample_field`, which cannot live in a pytree; keeping `z` as a real field costs one copy in the loop carry.
- `block_states` as a name-keyed dict: self-describing, but positional matches the `blocks` list and a three-line `_get_nested_attr` tweak gives `extra_fields` access.
- Facades as factory functions: break `isinstance`, `autoclass` and `mcmc.sampler.inner_kernel`.
- Identity default for `refresh`: silently wrong for `BarkerMH`/`SA` blocks.
- `HMCECS` as a sibling block: siblings cannot evaluate a model with subsample plates without the indices, and with a proxy their target differs from HMC's estimated target; the sound composition is `HMCECS(Gibbs(...))`.
- Reusing `initialize_model` for the discrete potential: it auto-enumerates unconditioned discrete latents (`util.py:753-759`); the shared preparation helper gives the same enumeration handling without that.
- MCMC-owned `vmap` of `sampler.sample` for vectorized chains: cleaner long term but would double-vmap `HMC`/`SA` unless gated; out of scope.

Open questions for reviewers:

1. Name of the user-callable block: `CustomGibbs` (matches the "custom Gibbs samplers" wording of the `HMCGibbs` docstring) vs `GibbsUpdate` vs `GibbsFn`.
2. Whether to add `MCMCKernel.is_initialized` now (replacing the `_sample_fn` duck typing at `mcmc.py:475`) or keep following the `_sample_fn` convention as `HMC`/`SA` do.
3. Whether the `changed`-gated refresh ships in v1 or as a follow-up (it is a few lines and the pattern already exists in `HMCECS`/`MixedHMC`).
4. Whether v1 should attempt vectorized chains for `Gibbs` or leave it as the follow-up described in 5.5.
5. Whether `HMCGibbsState` should keep the `hmc_state` property permanently or only for a deprecation cycle.
