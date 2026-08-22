# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Pure helpers shared by the composable Gibbs kernels (:mod:`numpyro.infer.gibbs`) and the
HMC-within-Gibbs kernels (:mod:`numpyro.infer.hmc_gibbs`, :mod:`numpyro.infer.mixed_hmc`).
"""

from collections.abc import Callable, Sequence
from functools import partial, reduce
from typing import Any, Protocol, TypeAlias

import numpy as np

import jax
from jax import random
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
from jax.scipy.special import expit

from numpyro._typing import (
    ModelArgs,
    ModelKwargs,
    ModelT,
    PotentialFn,
    PyTree,
    SiteValues,
    TraceT,
)
from numpyro.handlers import condition, seed, substitute, trace
from numpyro.infer.initialization import init_to_sample
from numpyro.util import cond, fori_loop, identity

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
