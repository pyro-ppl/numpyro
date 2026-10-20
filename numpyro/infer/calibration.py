# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Variational Simulation-Based Calibration (VSBC) diagnostic.

Implements the calibration diagnostic from:
    Yao, Y., Vehtari, A., Simpson, D., and Gelman, A. (2018).
    Yes, but Did It Work?: Evaluating Variational Inference.
    International Conference on Machine Learning.
"""

from __future__ import annotations

from collections import namedtuple
from collections.abc import Mapping
import inspect
import warnings

import numpy as np

import jax
from jax import pmap, random, vmap
import jax.numpy as jnp

from numpyro.handlers import condition, seed, trace
from numpyro.infer.svi import SVI
from numpyro.infer.util import Predictive

__all__ = ["vsbc_diagnostic"]


VSBCResult = namedtuple(
    "VSBCResult",
    ["probabilities", "ks_stats", "ks_pvalues", "ranks", "param_names"],
)
r"""
A :func:`~collections.namedtuple` with fields:
 - **probabilities** - dict mapping parameter names to arrays of
   calibration probability estimates, each of shape
   ``(num_simulations, *site_shape)``. These are Monte Carlo estimates of
   :math:`p_{ij} = \Pr_q(\theta_i < \theta_i^*)`, computed as the
   proportion of posterior guide draws strictly less than the true latent
   value.
 - **ks_stats** - dict mapping parameter names to KS test statistics,
   each of shape ``site_shape``.
 - **ks_pvalues** - dict mapping parameter names to KS test p-values,
   each of shape ``site_shape``.
 - **ranks** - dict mapping parameter names to arrays of rank statistics,
   each of shape ``(num_simulations, *site_shape)``. This is retained as
   a secondary diagnostic and as the Monte Carlo support statistic for
   ``probabilities``.
 - **param_names** - tuple of parameter names.
"""


_MISSING = object()
_SUPPORTED_ARG_KINDS = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
)
_INFER_ARGS_ERROR = (
    "Unable to infer how simulated observations should be passed to model "
    "arguments. Use argument names matching observed_sites, leave direct "
    "observed-data arguments unset/None, or pass simulated_data_to_args "
    "explicitly."
)


def _prior_predictive_sample(rng_key, model, *args, **kwargs):
    """Draw a single joint sample from the prior predictive distribution.

    Returns dict mapping all sample site names to their values.
    """
    model_trace = trace(seed(model, rng_key)).get_trace(*args, **kwargs)
    return {
        name: site["value"]
        for name, site in model_trace.items()
        if site["type"] == "sample"
    }


def _resolve_observed_sites(model_trace, observed_sites):
    sample_names = [
        name for name, site in model_trace.items() if site["type"] == "sample"
    ]
    sample_name_set = set(sample_names)

    if observed_sites is None:
        observed_names = [
            name
            for name, site in model_trace.items()
            if site["type"] == "sample" and site.get("is_observed", False)
        ]
        if not observed_names:
            raise ValueError(
                "Unable to infer observed sites from a generative model trace. "
                "Please specify observed_sites explicitly."
            )
    else:
        observed_names = list(dict.fromkeys(observed_sites))
        unknown_sites = [name for name in observed_names if name not in sample_name_set]
        if unknown_sites:
            raise ValueError(
                f"Unknown observed sites: {unknown_sites}. Expected a subset of "
                f"{sample_names}."
            )

    if not observed_names:
        raise ValueError("observed_sites must contain at least one sample site.")

    latent_names = sorted(set(sample_names) - set(observed_names))
    if not latent_names:
        raise ValueError("No latent parameters found. All sample sites are observed.")

    return observed_names, latent_names


def _validate_latent_sites(model_trace, latent_names):
    discrete_latents = [
        name for name in latent_names if model_trace[name]["fn"].support.is_discrete
    ]
    if discrete_latents:
        raise ValueError(
            "vsbc_diagnostic currently supports only continuous latent sample "
            f"sites. Found discrete latents: {discrete_latents}."
        )


def _get_observed_arg_injector(fn, observed_names, args, kwargs):
    signature = inspect.signature(fn)
    bound = signature.bind_partial(*args, **kwargs)

    assignments = []

    for site_name in observed_names:
        param = signature.parameters.get(site_name)
        current = bound.arguments.get(site_name, _MISSING)
        if (
            param is not None
            and param.kind in _SUPPORTED_ARG_KINDS
            and (current is _MISSING or current is None)
        ):
            assignments.append((site_name, site_name))
    remaining_sites = [name for name in observed_names if name not in dict(assignments)]

    if remaining_sites:
        candidate_params = []
        bound_conflicts = []
        for name, param in signature.parameters.items():
            if (
                name in dict(assignments)
                or param.kind not in _SUPPORTED_ARG_KINDS
                or name in observed_names
            ):
                continue
            current = bound.arguments.get(name, _MISSING)
            if (
                current is not _MISSING
                and current is not None
                and param.default is None
            ):
                bound_conflicts.append(name)
            if current is None or (current is _MISSING and param.default is None):
                candidate_params.append(name)

        if bound_conflicts:
            raise ValueError(
                f"{_INFER_ARGS_ERROR} Conflicting bound arguments: {bound_conflicts}."
            )

        if len(remaining_sites) == 1 and len(candidate_params) == 1:
            assignments.append((candidate_params[0], remaining_sites[0]))
        else:
            raise ValueError(_INFER_ARGS_ERROR)

    def inject(sim_data):
        if not assignments:
            return args, kwargs
        call_bound = signature.bind_partial(*args, **kwargs)
        for param_name, site_name in assignments:
            if site_name in sim_data:
                call_bound.arguments[param_name] = sim_data[site_name]
        return call_bound.args, call_bound.kwargs

    return inject


def _get_simulated_data_injector(
    model, observed_names, args, kwargs, simulated_data_to_args
):
    if simulated_data_to_args is None:
        return _get_observed_arg_injector(model, observed_names, args, kwargs)

    if not callable(simulated_data_to_args):
        raise ValueError("simulated_data_to_args must be callable.")

    def inject(sim_data):
        injected = simulated_data_to_args(sim_data, *args, **kwargs)
        if not isinstance(injected, tuple) or len(injected) != 2:
            raise ValueError(
                "simulated_data_to_args must return a pair (args, kwargs)."
            )

        sim_args, sim_kwargs = injected
        if not isinstance(sim_args, (tuple, list)):
            raise ValueError(
                "simulated_data_to_args must return a tuple or list for args."
            )
        if not isinstance(sim_kwargs, Mapping):
            raise ValueError("simulated_data_to_args must return a mapping for kwargs.")

        return tuple(sim_args), dict(sim_kwargs)

    return inject


def _warmup_guide(
    model,
    guide,
    optim,
    loss,
    observed_names,
    simulated_data_injector,
    *args,
    **kwargs,
):
    """Run one SVI init to populate the guide's prototype trace.

    This must happen before ``jax.lax.map`` because auto-guide prototype
    setup uses ``while_loop`` which leaks tracers inside ``lax.map``.
    """
    dummy_data = {}
    model_trace = trace(seed(model, random.PRNGKey(0))).get_trace(*args, **kwargs)
    for name in observed_names:
        if name in model_trace and model_trace[name]["type"] == "sample":
            dummy_data[name] = model_trace[name]["value"]

    dummy_model = condition(model, dummy_data)
    warmup_args, warmup_kwargs = simulated_data_injector(dummy_data)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found non-auxiliary vars")
        svi = SVI(dummy_model, guide, optim, loss)
        svi.init(random.PRNGKey(0), *warmup_args, **warmup_kwargs)


def vsbc_diagnostic(
    rng_key,
    model,
    guide,
    optim,
    loss,
    *args,
    observed_sites=None,
    simulated_data_to_args=None,
    num_simulations=100,
    num_svi_steps=1000,
    num_samples=1000,
    chain_method="sequential",
    **kwargs,
):
    r"""Compute the VSBC diagnostic for a model/guide pair.

    Variational Simulation-Based Calibration checks whether a variational
    approximation produces unbiased marginal point estimates on average
    across simulated datasets. This implementation estimates the
    calibration probabilities
    :math:`p_{ij} = \Pr_q(\theta_i < \theta_i^*)` from posterior guide
    draws and applies a two-sample Kolmogorov-Smirnov test to compare
    :math:`p_{i:}` with :math:`1 - p_{i:}`. For a VSBC-calibrated
    approximation, those marginal probabilities should be symmetric
    around ``0.5``.

    For tensor-valued latent sites, calibration probabilities, ranks, and
    KS tests are computed component-wise. No multiple-comparison correction
    is applied across components. Because the Monte Carlo probabilities lie
    on a discrete grid with denominator ``num_samples``, the symmetry test
    uses the asymptotic two-sample KS approximation.

    The model must be generative when observed data is cleared: it should
    sample both latent parameters and data from the prior predictive
    distribution. The current implementation supports only continuous
    latent sample sites.

    Independent SVI fits are dispatched according to ``chain_method``
    (following the same convention as :class:`~numpyro.infer.MCMC`):

    - ``"sequential"`` (default): :func:`jax.lax.map` — JIT-compiled loop,
      one simulation at a time. Memory-efficient; best for CPU.
    - ``"vectorized"``: :func:`jax.vmap` — all simulations batched on one
      device. Higher throughput on GPU but requires M× memory.
    - ``"parallel"``: :func:`jax.pmap` — one simulation per device.
      Requires ``num_simulations`` devices; falls back to ``"sequential"``
      if insufficient devices are available.

    **Example**::

        >>> from jax import random
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.infer import Trace_ELBO
        >>> from numpyro.infer.autoguide import AutoNormal
        >>> from numpyro.infer.calibration import vsbc_diagnostic

        >>> def model(obs=None):
        ...     loc = numpyro.sample("loc", dist.Normal(0.0, 1.0))
        ...     numpyro.sample("obs", dist.Normal(loc, 1.0), obs=obs)

        >>> guide = AutoNormal(model)
        >>> result = vsbc_diagnostic(
        ...     random.PRNGKey(0), model, guide,
        ...     numpyro.optim.Adam(0.01), Trace_ELBO(),
        ...     observed_sites=["obs"],
        ...     num_simulations=50, num_svi_steps=500, num_samples=500,
        ... )

    This diagnostic is computationally expensive because it runs
    ``num_simulations`` independent SVI fits.

    :param jax.random.PRNGKey rng_key: random number generator key.
    :param Callable model: NumPyro model. Must be generative: when called
        without observed data, it samples from the prior predictive.
    :param Callable guide: NumPyro guide (unfitted). Will be re-initialized
        for each simulation.
    :param optim: a NumPyro optimizer (e.g. ``numpyro.optim.Adam``).
    :param loss: an ELBO loss instance (e.g. ``Trace_ELBO()``).
    :param args: positional arguments to model and guide. For direct
        observed-data arguments, leave them unset or pass ``None`` so they
        can be filled from ``observed_sites``. Passing concrete observed
        data through direct arguments during VSBC is unsupported unless
        ``simulated_data_to_args`` rewrites those arguments explicitly. If
        observed-site names do not match argument names, automatic alias
        inference is only supported when there is exactly one unset direct
        observed-data argument.
    :param list observed_sites: names of observed (data) sample sites in
        the model. If None, observed sites are auto-detected from the
        initial model trace when possible; otherwise a ``ValueError`` is
        raised and sites must be provided explicitly.
    :param callable simulated_data_to_args: optional callback to map a
        simulated observation dict into model/guide call arguments. It is
        called as ``simulated_data_to_args(sim_data, *args, **kwargs)``
        and must return ``(sim_args, sim_kwargs)``. This is required when
        observed data is passed through a container-style argument such as
        ``data={"y": obs}`` rather than direct ``obs=...`` parameters.
        Because it is executed inside the mapped simulation loop, this
        callback should be a pure structural rewrite that is compatible
        with JAX transformations such as ``lax.map``, ``vmap``, and
        ``pmap``. If observed data is already bound in ``args`` or
        ``kwargs``, the same callback is also used to clear those values
        during discovery and prior-predictive simulation.
    :param int num_simulations: number of prior predictive simulations (M
        in Algorithm 2). Default 100.
    :param int num_svi_steps: number of SVI optimization steps per
        simulation. Default 1000.
    :param int num_samples: number of posterior samples (L) to draw from
        the fitted guide for Monte Carlo estimation of calibration
        probabilities. Default 1000.
    :param str chain_method: a callable JAX transform like :func:`jax.vmap`
        or one of ``"sequential"`` (default), ``"vectorized"``, or
        ``"parallel"``. See above for details.
    :param kwargs: keyword arguments to model and guide.
    :return: a :data:`VSBCResult` namedtuple containing calibration
        probability estimates, symmetry-test results, and raw ranks.
    :rtype: VSBCResult

    **References**

    - Yao, Y., Vehtari, A., Simpson, D., and Gelman, A. (2018).
      *Yes, but Did It Work?: Evaluating Variational Inference.*
    - Talts, S., Betancourt, M., Simpson, D., Vehtari, A., and Gelman, A.
      (2018). *Validating Bayesian Inference Algorithms with
      Simulation-Based Calibration.*
    """
    if num_simulations < 2:
        raise ValueError("num_simulations must be at least 2.")
    if num_svi_steps < 1:
        raise ValueError("num_svi_steps must be at least 1.")
    if num_samples < 2:
        raise ValueError("num_samples must be at least 2.")
    if not callable(chain_method) and chain_method not in (
        "sequential",
        "vectorized",
        "parallel",
    ):
        raise ValueError(
            "Only supporting the following methods to dispatch simulations:"
            ' "sequential", "parallel", or "vectorized"'
        )

    if chain_method == "parallel" and jax.local_device_count() < num_simulations:
        chain_method = "sequential"
        warnings.warn(
            f"Not enough devices for parallel execution: expected "
            f"{num_simulations} but got {jax.local_device_count()}."
            f" Falling back to sequential. If running on CPU, consider"
            f" using numpyro.set_host_device_count({num_simulations}).",
            stacklevel=2,
        )

    rng_key, key_discover = random.split(rng_key)

    if observed_sites is None:
        model_trace = trace(seed(model, key_discover)).get_trace(*args, **kwargs)
        observed_names, latent_names = _resolve_observed_sites(
            model_trace, observed_sites
        )
        simulated_data_injector = _get_simulated_data_injector(
            model, observed_names, args, kwargs, simulated_data_to_args
        )
        cleared_args, cleared_kwargs = args, kwargs
    else:
        observed_names = list(dict.fromkeys(observed_sites))
        simulated_data_injector = _get_simulated_data_injector(
            model, observed_names, args, kwargs, simulated_data_to_args
        )
        cleared_args, cleared_kwargs = simulated_data_injector(
            {name: None for name in observed_names}
        )
        model_trace = trace(seed(model, key_discover)).get_trace(
            *cleared_args, **cleared_kwargs
        )
        observed_names, latent_names = _resolve_observed_sites(
            model_trace, observed_sites
        )
        simulated_data_injector = _get_simulated_data_injector(
            model, observed_names, args, kwargs, simulated_data_to_args
        )

    _validate_latent_sites(model_trace, latent_names)

    _warmup_guide(
        model,
        guide,
        optim,
        loss,
        observed_names,
        simulated_data_injector,
        *args,
        **kwargs,
    )

    # Draw prior predictive samples (Python loop — model tracing cannot
    # be JIT-compiled, but this is fast relative to the SVI fits below).
    rng_key, key_prior = random.split(rng_key)
    prior_keys = random.split(key_prior, num_simulations)
    prior_latents = []
    sim_data = []
    for i in range(num_simulations):
        samples = _prior_predictive_sample(
            prior_keys[i], model, *cleared_args, **cleared_kwargs
        )
        prior_latents.append({n: samples[n] for n in latent_names if n in samples})
        sim_data.append({n: samples[n] for n in observed_names if n in samples})

    sim_data_stacked = {
        n: jnp.stack([sim_data[i][n] for i in range(num_simulations)])
        for n in observed_names
    }
    theta_true_stacked = {
        n: jnp.stack([prior_latents[i][n] for i in range(num_simulations)])
        for n in latent_names
    }

    # Run M independent SVI fits. Handlers (condition, seed) are applied
    # inside the mapped function so each iteration gets fresh state.
    rng_key, key_sim = random.split(rng_key)
    sim_keys = random.split(key_sim, num_simulations * 2).reshape(
        num_simulations, 2, -1
    )

    def _single_simulation(carry):
        key_svi, key_post, sim_data, theta_true = carry
        conditioned_model = condition(model, sim_data)
        sim_args, sim_kwargs = simulated_data_injector(sim_data)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Found non-auxiliary vars")
            svi = SVI(conditioned_model, guide, optim, loss)
            svi_state = svi.init(key_svi, *sim_args, **sim_kwargs)

            def body_fn(svi_state, _):
                svi_state, svi_loss = svi.update(svi_state, *sim_args, **sim_kwargs)
                return svi_state, svi_loss

            svi_state, _ = jax.lax.scan(body_fn, svi_state, None, length=num_svi_steps)
            params = svi.get_params(svi_state)

        predictive = Predictive(guide, params=params, num_samples=num_samples)
        posterior_samples = predictive(key_post, *sim_args, **sim_kwargs)

        rank_dict = {
            name: jnp.sum(posterior_samples[name] < theta_true[name], axis=0)
            for name in latent_names
        }
        return rank_dict

    map_args = (
        sim_keys[:, 0],
        sim_keys[:, 1],
        sim_data_stacked,
        theta_true_stacked,
    )

    if chain_method == "sequential":
        rank_results = jax.lax.map(_single_simulation, map_args)
    elif chain_method == "parallel":
        rank_results = pmap(_single_simulation)(map_args)
    elif callable(chain_method):
        rank_results = chain_method(_single_simulation)(map_args)
    else:
        assert chain_method == "vectorized"
        rank_results = vmap(_single_simulation)(map_args)

    all_ranks = {
        name: np.asarray(jax.device_get(rank_results[name])) for name in latent_names
    }

    from scipy.stats import ks_2samp

    probabilities = {}
    ranks = {}
    ks_stats = {}
    ks_pvalues = {}
    param_names = tuple(sorted(all_ranks.keys()))

    for name in param_names:
        ranks[name] = all_ranks[name]
        probabilities[name] = np.asarray(all_ranks[name] / num_samples)
        flat_probabilities = probabilities[name].reshape((num_simulations, -1))
        flat_stats = np.empty(flat_probabilities.shape[1], dtype=float)
        flat_pvalues = np.empty(flat_probabilities.shape[1], dtype=float)
        for i in range(flat_probabilities.shape[1]):
            ks_stat, ks_pvalue = ks_2samp(
                flat_probabilities[:, i],
                1.0 - flat_probabilities[:, i],
                method="asymp",
            )
            flat_stats[i] = ks_stat
            flat_pvalues[i] = ks_pvalue

        site_shape = probabilities[name].shape[1:]
        if site_shape:
            ks_stats[name] = flat_stats.reshape(site_shape)
            ks_pvalues[name] = flat_pvalues.reshape(site_shape)
        else:
            ks_stats[name] = float(flat_stats[0])
            ks_pvalues[name] = float(flat_pvalues[0])

    return VSBCResult(
        probabilities=probabilities,
        ks_stats=ks_stats,
        ks_pvalues=ks_pvalues,
        ranks=ranks,
        param_names=param_names,
    )
