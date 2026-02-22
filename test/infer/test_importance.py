# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.stats import genpareto

from jax import random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.importance import (
    _fit_generalized_pareto,
    _psis_khat,
    psis_diagnostic,
)

# --- Model/guide helpers ---


def normal_model(zdim):
    numpyro.sample("z", dist.Normal(jnp.zeros(zdim), 1.0).to_event(1))


def normal_guide_factory(zdim, scale):
    def guide(d):
        numpyro.sample(
            "z",
            dist.Normal(jnp.zeros(d), scale * jnp.ones(d)).to_event(1),
        )

    return guide


def _synthetic_log_weights(scale, zdim=3, n=50_000, seed=42):
    """Generate synthetic log importance weights from Normal model/guide."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, scale, size=(n, zdim))
    log_model = -0.5 * np.sum(z**2, axis=1) - 0.5 * zdim * np.log(2 * np.pi)
    log_guide = -0.5 * np.sum(z**2 / scale**2, axis=1) - 0.5 * zdim * np.log(
        2 * np.pi * scale**2
    )
    return log_model - log_guide


# --- Model/guide diagnostic tests ---


@pytest.mark.parametrize("zdim", [1, 5])
@pytest.mark.parametrize(
    "scale, khat_range",
    [
        (0.5, (0.4, 1.2)),
        (0.95, (-0.1, 0.35)),
    ],
)
def test_psis_diagnostic_normal_model(zdim, scale, khat_range):
    """PSIS k-hat should reflect guide quality for Normal model/guide pairs."""
    guide = normal_guide_factory(zdim, scale)
    khat = psis_diagnostic(
        random.PRNGKey(0),
        {},
        normal_model,
        guide,
        zdim,
        num_particles=20_000,
    )
    lo, hi = khat_range
    assert lo < khat < hi, f"k-hat={khat:.3f} not in ({lo}, {hi})"


# --- Cross-implementation comparison via precomputed reference values ---

# Reference k-hat values computed using both Vehtari's gpdfitnew (BSD-3, psis.py)
# and Pyro 1.9.1 psis_diagnostic pipeline on identical synthetic log weights.
# Both agree with our implementation to ~1e-15.
_PIPELINE_REFERENCE_KHAT = {
    # scale: k_hat (from Vehtari reference; Pyro agrees to ~1e-15)
    0.5: 0.9080815512205151,
    0.7: 0.6371459500343336,
    0.9: 0.27494481323913184,
    0.99: 0.08018355166699344,
}


@pytest.mark.parametrize("scale", [0.5, 0.7, 0.9, 0.99])
def test_psis_khat_matches_reference_pipeline(scale):
    """Full k-hat pipeline matches precomputed reference from Vehtari/Pyro."""
    lw = _synthetic_log_weights(scale=scale, n=50_000)
    k_ours = _psis_khat(lw)
    k_ref = _PIPELINE_REFERENCE_KHAT[scale]
    assert_allclose(k_ours, k_ref, atol=1e-10)


# --- Paper-aligned validation ---


@pytest.mark.parametrize(
    "scale, expected_regime",
    [
        (0.3, "unreliable"),  # k >= 0.7
        (0.5, "unreliable"),  # k >= 0.7 (narrow guide, high dim mismatch)
        (0.7, "marginal"),  # 0.5 <= k < 0.7
        (0.95, "good"),  # k < 0.5
        (0.99, "good"),  # k < 0.5
    ],
)
def test_psis_khat_regime_classification(scale, expected_regime):
    """Verify k-hat regimes match paper thresholds (Vehtari et al. 2024).

    Uses deterministic synthetic log weights (large n) so regime boundaries
    are stable without needing expensive model/guide evaluation.
    """
    lw = _synthetic_log_weights(scale=scale, zdim=3, n=100_000)
    khat = _psis_khat(lw)

    if expected_regime == "good":
        assert khat < 0.5, f"Expected good (k<0.5), got k={khat:.3f}"
    elif expected_regime == "marginal":
        assert 0.5 <= khat < 0.7, f"Expected marginal (0.5<=k<0.7), got k={khat:.3f}"
    elif expected_regime == "unreliable":
        assert khat >= 0.7, f"Expected unreliable (k>=0.7), got k={khat:.3f}"


# --- Batching tests ---


def test_psis_diagnostic_batching():
    """chunk_size should produce the same k-hat as no batching."""
    zdim = 2
    scale = 0.8
    guide = normal_guide_factory(zdim, scale)

    khat_full = psis_diagnostic(
        random.PRNGKey(42),
        {},
        normal_model,
        guide,
        zdim,
        num_particles=10_000,
    )
    khat_batched = psis_diagnostic(
        random.PRNGKey(42),
        {},
        normal_model,
        guide,
        zdim,
        num_particles=10_000,
        chunk_size=2_000,
    )
    assert_allclose(khat_full, khat_batched, atol=1e-6)


def test_psis_diagnostic_batching_non_divisible():
    """Non-evenly-divisible batch size should produce the same k-hat."""
    zdim = 2
    scale = 0.8
    guide = normal_guide_factory(zdim, scale)

    khat_full = psis_diagnostic(
        random.PRNGKey(42),
        {},
        normal_model,
        guide,
        zdim,
        num_particles=1_000,
    )
    khat_batched = psis_diagnostic(
        random.PRNGKey(42),
        {},
        normal_model,
        guide,
        zdim,
        num_particles=1_000,
        chunk_size=300,
    )
    assert_allclose(khat_full, khat_batched, atol=1e-6)


# --- Input validation tests ---


@pytest.mark.parametrize("num_particles", [0, 1])
def test_psis_diagnostic_rejects_num_particles_below_2(num_particles):
    """num_particles < 2 should raise ValueError."""
    guide = normal_guide_factory(1, 0.5)
    with pytest.raises(ValueError, match="num_particles must be at least 2"):
        psis_diagnostic(
            random.PRNGKey(0), {}, normal_model, guide, 1, num_particles=num_particles
        )


# --- Degenerate tail tests ---


def test_psis_diagnostic_perfect_guide_returns_inf():
    """A perfect guide (model == guide) returns inf (no tail to fit).

    This matches both Vehtari's reference and Pyro's behavior: when all
    importance weights are identical, there is no tail variance and
    the GPD fit is undefined.
    """

    def model():
        numpyro.sample("z", dist.Normal(0.0, 1.0))

    def guide():
        numpyro.sample("z", dist.Normal(0.0, 1.0))

    with pytest.warns(match="Not enough tail samples"):
        khat = psis_diagnostic(random.PRNGKey(0), {}, model, guide, num_particles=1_000)
    assert khat == float("inf")


# --- Model with observations and trained params ---


def test_psis_diagnostic_with_observations():
    """psis_diagnostic works with models that have observed data."""

    def model(x):
        loc = numpyro.sample("loc", dist.Normal(0.0, 10.0))
        numpyro.sample("obs", dist.Normal(loc, 1.0), obs=x)

    def guide(x):
        loc_mean = numpyro.param("loc_mean", 0.0)
        loc_scale = numpyro.param(
            "loc_scale", 1.0, constraint=dist.constraints.positive
        )
        numpyro.sample("loc", dist.Normal(loc_mean, loc_scale))

    x_obs = jnp.array(3.0)

    # Train briefly so param_map is non-empty
    optimizer = numpyro.optim.Adam(0.1)
    svi = SVI(model, guide, optimizer, Trace_ELBO())
    svi_result = svi.run(random.PRNGKey(0), 500, x_obs)
    params = svi_result.params

    khat = psis_diagnostic(
        random.PRNGKey(1),
        params,
        model,
        guide,
        x_obs,
        num_particles=5_000,
    )
    assert np.isfinite(khat)
    # Trained guide should be reasonable
    assert khat < 1.0


# --- GPD fitting tests ---


@pytest.mark.parametrize("k", [0.1, 0.5, 1.0])
def test_fit_generalized_pareto_recovers_shape(k):
    """Fit GPD to synthetic samples and verify recovered shape parameter."""
    rng = np.random.default_rng(42)
    x = genpareto.rvs(c=k, scale=1.0, size=5000, random_state=rng)
    k_hat, sigma_hat = _fit_generalized_pareto(x)
    assert_allclose(k_hat, k, atol=0.15)


@pytest.mark.parametrize("k", [0.1, 0.5, 1.0])
def test_fit_generalized_pareto_recovers_scale(k):
    """Fit GPD to synthetic samples and verify recovered scale parameter."""
    rng = np.random.default_rng(42)
    x = genpareto.rvs(c=k, scale=2.0, size=5000, random_state=rng)
    k_hat, sigma_hat = _fit_generalized_pareto(x)
    assert_allclose(sigma_hat, 2.0, atol=0.5)


# Reference values (precomputed from Vehtari gpdfitnew + Pyro 1.9.1)
_REFERENCE_VALUES = {
    # k_true: (k_ref, sigma_ref) — independent rng(123) per k
    0.1: (0.11030538299874189, 0.9689311464171625),
    0.5: (0.5035203992870904, 0.966373900167903),
    1.0: (0.9891947607352517, 0.9688794540950301),
}


@pytest.mark.parametrize("k_true", [0.1, 0.5, 1.0])
def test_fit_generalized_pareto_matches_reference(k_true):
    """Cross-check against precomputed reference values from Vehtari's gpdfitnew."""
    rng = np.random.default_rng(123)
    x = genpareto.rvs(c=k_true, scale=1.0, size=3000, random_state=rng)
    k_ours, sigma_ours = _fit_generalized_pareto(x)
    k_ref, sigma_ref = _REFERENCE_VALUES[k_true]
    assert_allclose(k_ours, k_ref, atol=1e-10)
    assert_allclose(sigma_ours, sigma_ref, atol=1e-10)


def test_fit_generalized_pareto_rejects_2d():
    with pytest.raises(ValueError, match="Expected 1-D array with at least 2 elements"):
        _fit_generalized_pareto(np.ones((5, 2)))


def test_fit_generalized_pareto_rejects_empty():
    with pytest.raises(ValueError, match="Expected 1-D array with at least 2 elements"):
        _fit_generalized_pareto(np.array([]))


def test_fit_generalized_pareto_rejects_scalar():
    with pytest.raises(ValueError, match="Expected 1-D array with at least 2 elements"):
        _fit_generalized_pareto(np.array(1.0))


def test_fit_generalized_pareto_rejects_length_one():
    with pytest.raises(ValueError, match="Expected 1-D array with at least 2 elements"):
        _fit_generalized_pareto(np.array([1.0]))


def test_fit_generalized_pareto_n2():
    """n=2 is the minimum valid input; should produce finite results."""
    k, sigma = _fit_generalized_pareto(np.array([1.0, 2.0]))
    assert np.isfinite(k)
    assert np.isfinite(sigma)


def test_fit_generalized_pareto_small_sample():
    """n=20 should produce finite results."""
    rng = np.random.default_rng(42)
    x = genpareto.rvs(c=0.5, scale=1.0, size=20, random_state=rng)
    k_hat, sigma_hat = _fit_generalized_pareto(x)
    assert np.isfinite(k_hat)
    assert np.isfinite(sigma_hat)
    assert sigma_hat > 0


def test_fit_generalized_pareto_zeros_in_input():
    """Input containing zeros produces nan sigma (matching reference behavior)."""
    k, sigma = _fit_generalized_pareto(np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    # k is still finite (prior regularization), sigma is nan (division by zero
    # in the internal computation). This matches Vehtari's gpdfitnew exactly.
    assert np.isfinite(k)
    assert np.isnan(sigma)


def test_fit_generalized_pareto_all_identical():
    """All-identical positive values produce nan sigma (matching reference)."""
    k, sigma = _fit_generalized_pareto(np.ones(100))
    assert np.isfinite(k)
    assert np.isnan(sigma)
