# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Pareto Smoothed Importance Sampling (PSIS) diagnostics for variational inference.

Implements the k-hat diagnostic from:
    Yao, Y., Vehtari, A., Simpson, D., and Gelman, A. (2018).
    Yes, but Did It Work?: Evaluating Variational Inference.
    International Conference on Machine Learning.

    Vehtari, A., Simpson, D., Gelman, A., Yao, Y., and Gabry, J. (2024).
    Pareto smoothed importance sampling.
    Journal of Machine Learning Research, 25(72):1-58.
"""

from __future__ import annotations

from collections.abc import Callable
import math
import warnings

import numpy as np

import jax
from jax import device_get, random

from numpyro.handlers import seed
from numpyro.infer.elbo import get_importance_log_probs

__all__ = ["psis_diagnostic"]


def _fit_generalized_pareto(x: np.ndarray) -> tuple[float, float]:
    """Estimate parameters of the Generalized Pareto Distribution (GPD).

    Returns empirical Bayes estimates for the shape (k) and scale (sigma)
    parameters of the two-parameter GPD, using the method of Zhang and
    Stephens (2009) with the prior regularization from Vehtari et al. (2024).

    References:
        Zhang, J. and Stephens, M.A. (2009). A new and efficient estimation
        method for the generalized Pareto distribution. Technometrics,
        51(3):316-325.

        Vehtari, A., Simpson, D., Gelman, A., Yao, Y., and Gabry, J. (2024).
        Pareto smoothed importance sampling. Journal of Machine Learning
        Research, 25(72):1-58.

    :param numpy.ndarray x: one-dimensional array of positive exceedances (tail samples).
    :return: tuple of (k, sigma) where k is the shape parameter and sigma
        is the scale parameter.
    """
    if x.ndim != 1 or len(x) <= 1:
        raise ValueError(
            f"Expected 1-D array with at least 2 elements, got shape {x.shape}."
        )

    # Broad errstate is needed because degenerate inputs (e.g. zeros or
    # identical values) cause cascading numerical issues at multiple points:
    #   divide: 1/x[quartile], 1/x[-1], -k/b when tail values are zero
    #   over:   exp(L - L') when profile log-likelihood differences are large
    #   invalid: downstream ops on nan/inf from earlier divide-by-zero
    # The resulting nan/inf propagate correctly through the algorithm,
    # matching the reference implementation behavior.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return _fit_generalized_pareto_impl(x)


def _fit_generalized_pareto_impl(x: np.ndarray) -> tuple[float, float]:
    x = np.sort(x)
    n = len(x)
    PRIOR = 3
    m = 30 + int(np.sqrt(n))

    # Candidate shape parameters (Zhang & Stephens grid)
    bs = np.arange(1, m + 1, dtype=float)
    bs -= 0.5
    np.divide(m, bs, out=bs)
    np.sqrt(bs, out=bs)
    np.subtract(1, bs, out=bs)
    bs /= PRIOR * x[int(n / 4 + 0.5) - 1]
    bs += 1 / x[-1]

    # Profile log-likelihood for each candidate
    ks = np.negative(bs)
    temp = ks[:, None] * x
    np.log1p(temp, out=temp)
    np.mean(temp, axis=1, out=ks)

    L = bs / ks
    np.negative(L, out=L)
    np.log(L, out=L)
    L -= ks
    L -= 1
    L *= n

    # Posterior weights (overflow in exp is expected and harmless;
    # overflowed values get negligible weight after normalization)
    temp = L - L[:, None]
    np.exp(temp, out=temp)
    w = np.sum(temp, axis=1)
    np.divide(1, w, out=w)

    # Remove negligible weights
    dii = w >= 10 * np.finfo(float).eps
    if not np.all(dii):
        w = w[dii]
        bs = bs[dii]
    w /= w.sum()

    # Posterior mean for b
    b = np.sum(bs * w)

    # Estimate for k (note: negated relative to Zhang & Stephens)
    temp = (-b) * x
    np.log1p(temp, out=temp)
    k = np.mean(temp)

    # Estimate for sigma
    sigma = -k / b

    # Weakly informative prior for k (Vehtari et al. 2024, Appendix G)
    # Prior: mean=0.5, effective sample size a=10
    a = 10
    k = k * n / (n + a) + a * 0.5 / (n + a)

    return float(k), float(sigma)


def _compute_log_weights(
    rng_key: jax.Array,
    param_map: dict[str, jax.Array],
    model: Callable,
    guide: Callable,
    args: tuple,
    kwargs: dict,
) -> jax.Array:
    """Compute log importance weight log p(x,z) - log q(z) for a single particle."""
    # Separate seeds: guide needs its own randomness for sampling latent sites;
    # model gets an independent seed in case it has stochastic structure beyond
    # the latent sites replayed from the guide (e.g. stochastic control flow).
    model_seed, guide_seed = random.split(rng_key)
    seeded_model = seed(model, model_seed)
    seeded_guide = seed(guide, guide_seed)
    model_log_probs, guide_log_probs = get_importance_log_probs(
        seeded_model, seeded_guide, args, kwargs, param_map
    )
    log_model = sum(v.sum() for v in model_log_probs.values())
    log_guide = sum(v.sum() for v in guide_log_probs.values())
    return log_model - log_guide


def _psis_khat(log_weights: np.ndarray) -> float:
    """Compute PSIS k-hat from an array of raw log importance weights."""
    log_weights = log_weights.copy()
    log_weights -= log_weights.max()
    log_weights = np.sort(log_weights)

    # S matches notation in Vehtari et al. (2024), Algorithm 1
    S = len(log_weights)

    # Tail extraction (Vehtari et al. 2024, Algorithm 1)
    M = math.ceil(min(0.2 * S, 3 * math.sqrt(S)))
    cutoff_ind = -(M + 1)
    lw_cutoff = max(np.log(np.finfo(float).tiny), log_weights[cutoff_ind])

    lw_tail = log_weights[log_weights > lw_cutoff]

    if len(lw_tail) < 5:
        warnings.warn(
            "Not enough tail samples for reliable PSIS diagnostic.",
            stacklevel=3,
        )
        return float("inf")

    # Shift to exceedances
    tail = np.exp(lw_tail) - np.exp(lw_cutoff)

    # Fit GPD to the tail
    k, sigma = _fit_generalized_pareto(tail)

    return float(k)


def psis_diagnostic(
    rng_key: jax.Array,
    param_map: dict[str, jax.Array],
    model: Callable,
    guide: Callable,
    *args,
    num_particles: int = 1000,
    chunk_size: int | None = None,
    **kwargs,
) -> float:
    r"""Compute the PSIS k-hat diagnostic for a model/guide pair.

    The k-hat statistic measures the reliability of importance sampling
    estimates. It is the shape parameter of a Generalized Pareto Distribution
    (GPD) fitted to the upper tail of the importance weights.

    Interpretation (Vehtari et al. 2024):

    - k < 0.5: finite variance, classical CLT applies
    - 0.5 <= k < 0.7: finite mean, generalized CLT may apply
    - k >= 0.7: unreliable importance sampling estimates

    **Example usage**::

        >>> from jax import random
        >>> from numpyro.infer import SVI, Trace_ELBO, psis_diagnostic
        >>> svi = SVI(model, guide, optimizer, Trace_ELBO())
        >>> svi_result = svi.run(random.PRNGKey(0), num_steps, *args)
        >>> khat = psis_diagnostic(
        ...     random.PRNGKey(1), svi_result.params, model, guide, *args
        ... )

    .. note::

        For reliable results, use at least several hundred particles
        (the default of 1000 is usually sufficient). Very few particles
        may not provide enough tail samples for GPD fitting.

    :param jax.random.PRNGKey rng_key: random number generator key.
    :param dict param_map: dictionary of current parameter values
        (e.g. ``svi_result.params``).
    :param Callable model: NumPyro model.
    :param Callable guide: NumPyro guide.
    :param args: positional arguments to model and guide.
    :param int num_particles: number of importance weight samples to draw.
    :param int chunk_size: maximum particles to evaluate at once (for memory
        control). If None, all particles are evaluated together.
    :param kwargs: keyword arguments to model and guide.
    :return: the estimated k-hat statistic.
    :rtype: float
    """
    if num_particles < 2:
        raise ValueError("num_particles must be at least 2.")

    if chunk_size is None:
        chunk_size = num_particles

    rng_keys = random.split(rng_key, num_particles)

    # Compute log weights in batches
    def compute_fn(key):
        return _compute_log_weights(key, param_map, model, guide, args, kwargs)

    log_weights_list = []
    for batch_start in range(0, num_particles, chunk_size):
        batch_keys = rng_keys[batch_start : batch_start + chunk_size]
        batch_lw = jax.vmap(compute_fn)(batch_keys)
        log_weights_list.append(batch_lw)

    log_weights = np.concatenate(
        [np.asarray(device_get(lw)) for lw in log_weights_list]
    )

    return _psis_khat(log_weights)
