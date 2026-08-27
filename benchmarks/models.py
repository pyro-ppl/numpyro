# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""Models and synthetic datasets shared by the benchmark suites.

Data is generated from a fixed NumPy seed rather than downloaded, so a
benchmark run never depends on the network and always sees identical inputs.
Everything here must stay on public NumPyro API that has been stable for a
while: these models are executed against both the feature branch *and* the base
branch, so anything freshly added would make the base side fail to run.
"""

from typing import Any, Optional

import numpy as np

import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

__all__ = [
    "eight_schools",
    "eight_schools_data",
    "hierarchical_glm",
    "hierarchical_glm_data",
    "logistic_regression",
    "logistic_regression_data",
    "neals_funnel",
]


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


def logistic_regression(
    features: jnp.ndarray, labels: Optional[jnp.ndarray] = None
) -> None:
    """Flat Bayesian logistic regression -- a dense, well-conditioned target."""
    coefs = numpyro.sample(
        "coefs", dist.Normal(0.0, 1.0).expand([features.shape[1]]).to_event(1)
    )
    intercept = numpyro.sample("intercept", dist.Normal(0.0, 5.0))
    logits = features @ coefs + intercept
    numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=labels)


def eight_schools(sigma: jnp.ndarray, y: Optional[jnp.ndarray] = None) -> None:
    """Non-centred eight schools -- small, but exercises deterministic sites."""
    mu = numpyro.sample("mu", dist.Normal(0.0, 5.0))
    tau = numpyro.sample("tau", dist.HalfCauchy(5.0))
    with numpyro.plate("J", sigma.shape[0]):
        theta_base = numpyro.sample("theta_base", dist.Normal(0.0, 1.0))
        theta = numpyro.deterministic("theta", mu + tau * theta_base)
        numpyro.sample("obs", dist.Normal(theta, sigma), obs=y)


def hierarchical_glm(
    group_idx: jnp.ndarray,
    features: jnp.ndarray,
    n_groups: int,
    y: Optional[jnp.ndarray] = None,
) -> None:
    """Partially pooled linear regression -- nested plates and broadcasting."""
    n_features = features.shape[1]
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0).expand([n_features]).to_event(1))
    tau = numpyro.sample("tau", dist.HalfNormal(1.0).expand([n_features]).to_event(1))
    with numpyro.plate("groups", n_groups):
        offset = numpyro.sample(
            "offset", dist.Normal(0.0, 1.0).expand([n_features]).to_event(1)
        )
    beta = mu + tau * offset
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    loc = jnp.sum(features * beta[group_idx], axis=-1)
    numpyro.sample("obs", dist.Normal(loc, sigma), obs=y)


def neals_funnel(dim: int = 10) -> None:
    """Neal's funnel -- pathological geometry, stresses the NUTS trajectory."""
    y = numpyro.sample("y", dist.Normal(0.0, 3.0))
    numpyro.sample("x", dist.Normal(jnp.zeros(dim), jnp.exp(y / 2)).to_event(1))


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


def logistic_regression_data(
    n_rows: int = 1000, n_features: int = 10, seed: int = 0
) -> dict[str, Any]:
    """Design matrix and Bernoulli labels for :func:`logistic_regression`."""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n_rows, n_features))
    coefs = rng.normal(size=n_features)
    probs = 1.0 / (1.0 + np.exp(-features @ coefs))
    labels = rng.binomial(1, probs)
    return {
        "features": jnp.asarray(features, dtype=jnp.float32),
        "labels": jnp.asarray(labels, dtype=jnp.int32),
    }


def eight_schools_data() -> dict[str, Any]:
    """The canonical eight schools treatment effects and standard errors."""
    return {
        "sigma": jnp.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]),
        "y": jnp.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]),
    }


def hierarchical_glm_data(
    n_rows: int = 2000, n_features: int = 5, n_groups: int = 20, seed: int = 0
) -> dict[str, Any]:
    """Grouped regression data for :func:`hierarchical_glm`."""
    rng = np.random.default_rng(seed)
    group_idx = rng.integers(0, n_groups, size=n_rows)
    features = rng.normal(size=(n_rows, n_features))
    beta = rng.normal(size=(n_groups, n_features))
    y = np.sum(features * beta[group_idx], axis=-1) + rng.normal(scale=0.5, size=n_rows)
    return {
        "group_idx": jnp.asarray(group_idx, dtype=jnp.int32),
        "features": jnp.asarray(features, dtype=jnp.float32),
        "n_groups": n_groups,
        "y": jnp.asarray(y, dtype=jnp.float32),
    }
