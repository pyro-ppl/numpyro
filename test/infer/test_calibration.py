# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the VSBC diagnostic (numpyro.infer.calibration).

These tests exercise the probability-based VSBC implementation against
known Gaussian cases, regression scenarios, and API edge cases.
"""

import numpy as np
import pytest
from scipy import stats

from jax import random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.calibration import vsbc_diagnostic


def _normal_model(x=None):
    loc = numpyro.sample("loc", dist.Normal(0.0, 1.0))
    numpyro.sample("obs", dist.Normal(loc, 1.0), obs=x)


def _fixed_unbiased_guide(x=None):
    numpyro.sample("loc", dist.Normal(0.0, 1.0))


def _fixed_underdispersed_guide(x=None):
    numpyro.sample("loc", dist.Normal(0.0, 0.1))


def _fixed_biased_guide(x=None):
    numpyro.sample("loc", dist.Normal(1.0, 1.0))


def _vector_obs_model(x=None):
    loc = numpyro.sample("loc", dist.Normal(0.0, 1.0))
    numpyro.sample("obs", dist.Normal(loc, 1.0).expand([3]), obs=x)


def _data_dependent_guide(x=None):
    if x is None:
        raise RuntimeError("guide received x=None")
    loc_mean = numpyro.param("loc_mean", 0.0)
    numpyro.sample("loc", dist.Normal(loc_mean + x.mean(), 1.0))


def _fixed_data_dependent_guide(x=None):
    if x is None:
        raise RuntimeError("guide received x=None")
    numpyro.sample("loc", dist.Normal(x.mean(), 0.2))


def _misrouting_model(x=None, mask=None):
    loc = numpyro.sample("loc", dist.Normal(0.0, 1.0))
    numpyro.sample("obs", dist.Normal(loc, 1.0), obs=x)


def _misrouting_guide(x=None, mask=None):
    if mask is not None:
        raise RuntimeError("mask was injected")
    if x is None:
        raise RuntimeError("guide received x=None")
    numpyro.sample("loc", dist.Normal(x.mean(), 1.0))


def _scaled_obs_model(x=None, scale=2.0):
    loc = numpyro.sample("loc", dist.Normal(0.0, 1.0))
    numpyro.sample("obs", dist.Normal(loc, scale), obs=x)


def _scaled_data_guide(x=None, scale=2.0):
    if x is None:
        raise RuntimeError("guide received x=None")
    numpyro.sample("loc", dist.Normal(x.mean() / scale, 1.0))


def _container_model(data=None):
    loc = numpyro.sample("loc", dist.Normal(0.0, 1.0))
    scale = numpyro.sample("scale", dist.LogNormal(0.0, 0.3))
    obs_loc = None if data is None else data["obs_loc"]
    obs_scale = None if data is None else data["obs_scale"]
    numpyro.sample("obs_loc", dist.Normal(loc, 1.0), obs=obs_loc)
    numpyro.sample("obs_scale", dist.Normal(scale, 0.5), obs=obs_scale)


def _container_data_guide(data=None):
    if data is None:
        raise RuntimeError("guide received data=None")
    numpyro.sample("loc", dist.Normal(data["obs_loc"], 1.0))
    numpyro.sample("scale", dist.LogNormal(data["obs_scale"], 0.2))


def _simulated_data_to_container_args(sim_data, *args, **kwargs):
    sim_kwargs = dict(kwargs)
    sim_kwargs["data"] = sim_data
    return args, sim_kwargs


def _simulated_data_to_x_args(sim_data, *args, **kwargs):
    return (sim_data["obs"],), kwargs


def _single_obs_container_model(data=None):
    loc = numpyro.sample("loc", dist.Normal(0.0, 1.0))
    obs = None if data is None else data["obs"]
    numpyro.sample("obs", dist.Normal(loc, 1.0), obs=obs)


def _single_obs_container_guide(data=None):
    if data is None:
        raise RuntimeError("guide received data=None")
    numpyro.sample("loc", dist.Normal(data["obs"], 0.2))


def _simulated_data_to_single_obs_container_args(sim_data, *args, **kwargs):
    sim_kwargs = dict(kwargs)
    sim_kwargs["data"] = {"obs": sim_data["obs"]}
    return args, sim_kwargs


def _discrete_model(x=None):
    z = numpyro.sample("z", dist.Bernoulli(0.5))
    numpyro.sample("obs", dist.Normal(z, 1.0), obs=x)


def _discrete_guide(x=None):
    numpyro.sample("z", dist.Bernoulli(0.5))


class TestVSBCDiagnostic:
    """End-to-end tests for vsbc_diagnostic."""

    def test_output_structure(self):
        """Return value has correct shape, bounds, and param names."""
        guide = AutoNormal(_normal_model)
        num_samples = 100
        result = vsbc_diagnostic(
            random.PRNGKey(0),
            _normal_model,
            guide,
            numpyro.optim.Adam(0.01),
            Trace_ELBO(),
            observed_sites=["obs"],
            num_simulations=5,
            num_svi_steps=100,
            num_samples=num_samples,
        )
        assert result.param_names == ("loc",)
        assert "obs" not in result.param_names
        assert result.probabilities["loc"].shape == (5,)
        assert np.all(result.probabilities["loc"] >= 0)
        assert np.all(result.probabilities["loc"] <= 1)
        assert result.ranks["loc"].shape == (5,)
        assert np.all(result.ranks["loc"] >= 0)
        assert np.all(result.ranks["loc"] <= num_samples)
        assert 0 <= result.ks_stats["loc"] <= 1
        assert 0 <= result.ks_pvalues["loc"] <= 1

    def test_probabilities_match_analytic_gaussian_cdf(self):
        """Monte Carlo probabilities should match the guide CDF in a Gaussian case."""
        rng_key = random.PRNGKey(0)
        num_simulations = 6
        result = vsbc_diagnostic(
            rng_key,
            _normal_model,
            _fixed_unbiased_guide,
            numpyro.optim.Adam(0.01),
            Trace_ELBO(),
            observed_sites=["obs"],
            num_simulations=num_simulations,
            num_svi_steps=5,
            num_samples=4000,
        )
        rng_key, _ = random.split(rng_key)
        _, key_prior = random.split(rng_key)
        theta_true = np.array(
            Predictive(_normal_model, num_samples=num_simulations)(key_prior)["loc"]
        )
        expected = stats.norm.cdf(theta_true, loc=0.0, scale=1.0)
        np.testing.assert_allclose(result.probabilities["loc"], expected, atol=0.03)

    def test_underdispersed_but_unbiased_guide_passes_symmetry_test(self):
        """VSBC should track symmetry, not uniformity, for unbiased point estimates."""
        result = vsbc_diagnostic(
            random.PRNGKey(42),
            _normal_model,
            _fixed_underdispersed_guide,
            numpyro.optim.Adam(0.01),
            Trace_ELBO(),
            observed_sites=["obs"],
            num_simulations=150,
            num_svi_steps=5,
            num_samples=2000,
        )
        assert result.ks_pvalues["loc"] > 0.05
        assert stats.kstest(result.probabilities["loc"], "uniform").pvalue < 1e-6

    def test_biased_guide_fails_symmetry_test(self):
        """A biased variational center should fail the VSBC symmetry check."""
        result = vsbc_diagnostic(
            random.PRNGKey(123),
            _normal_model,
            _fixed_biased_guide,
            numpyro.optim.Adam(0.01),
            Trace_ELBO(),
            observed_sites=["obs"],
            num_simulations=150,
            num_svi_steps=5,
            num_samples=2000,
        )
        assert result.ks_pvalues["loc"] < 0.05

    def test_multiple_latent_parameters(self):
        """Each latent parameter gets its own probabilities, ranks, and KS test."""

        def multi_model(x=None):
            loc = numpyro.sample("loc", dist.Normal(0.0, 1.0))
            scale = numpyro.sample("scale", dist.HalfNormal(1.0))
            numpyro.sample("obs", dist.Normal(loc, scale), obs=x)

        guide = AutoNormal(multi_model)
        result = vsbc_diagnostic(
            random.PRNGKey(0),
            multi_model,
            guide,
            numpyro.optim.Adam(0.01),
            Trace_ELBO(),
            observed_sites=["obs"],
            num_simulations=5,
            num_svi_steps=100,
            num_samples=100,
        )
        assert result.param_names == ("loc", "scale")
        assert result.probabilities["loc"].shape == (5,)
        assert result.probabilities["scale"].shape == (5,)
        assert result.ranks["loc"].shape == (5,)
        assert result.ranks["scale"].shape == (5,)

    def test_data_dependent_guide_uses_simulated_observations(self):
        """Direct observed-data args should receive the simulated observations."""
        rng_key = random.PRNGKey(0)
        num_simulations = 4
        result = vsbc_diagnostic(
            rng_key,
            _vector_obs_model,
            _fixed_data_dependent_guide,
            numpyro.optim.Adam(0.01),
            Trace_ELBO(),
            observed_sites=["obs"],
            num_simulations=num_simulations,
            num_svi_steps=5,
            num_samples=4000,
        )
        rng_key, _ = random.split(rng_key)
        _, key_prior = random.split(rng_key)
        prior_samples = Predictive(_vector_obs_model, num_samples=num_simulations)(key_prior)
        expected = stats.norm.cdf(
            np.array(prior_samples["loc"]),
            loc=np.array(prior_samples["obs"]).mean(axis=-1),
            scale=0.2,
        )
        np.testing.assert_allclose(result.probabilities["loc"], expected, atol=0.03)

    def test_bound_direct_observed_args_require_explicit_mapper(self):
        """Bound direct observed args should fail fast instead of misrouting."""
        with pytest.raises(ValueError, match="Conflicting bound arguments"):
            vsbc_diagnostic(
                random.PRNGKey(0),
                _misrouting_model,
                _misrouting_guide,
                numpyro.optim.Adam(0.01),
                Trace_ELBO(),
                np.ones(3),
                observed_sites=["obs"],
                num_simulations=2,
                num_svi_steps=5,
                num_samples=10,
            )

    def test_alias_observed_arg_allows_other_fixed_hyperparameters(self):
        """Alias-style observed args should still work with fixed non-data kwargs."""
        result = vsbc_diagnostic(
            random.PRNGKey(0),
            _scaled_obs_model,
            _scaled_data_guide,
            numpyro.optim.Adam(0.01),
            Trace_ELBO(),
            observed_sites=["obs"],
            scale=2.0,
            num_simulations=4,
            num_svi_steps=10,
            num_samples=20,
        )
        assert result.param_names == ("loc",)
        assert result.probabilities["loc"].shape == (4,)

    def test_bound_direct_observed_data_is_cleared_with_explicit_mapper(self):
        """Bound direct observed data should be cleared when a mapper is supplied."""
        baseline = vsbc_diagnostic(
            random.PRNGKey(0),
            _vector_obs_model,
            _fixed_data_dependent_guide,
            numpyro.optim.Adam(0.01),
            Trace_ELBO(),
            observed_sites=["obs"],
            simulated_data_to_args=_simulated_data_to_x_args,
            num_simulations=4,
            num_svi_steps=5,
            num_samples=4000,
        )
        reused_data = vsbc_diagnostic(
            random.PRNGKey(0),
            _vector_obs_model,
            _fixed_data_dependent_guide,
            numpyro.optim.Adam(0.01),
            Trace_ELBO(),
            np.full(3, 10.0),
            observed_sites=["obs"],
            simulated_data_to_args=_simulated_data_to_x_args,
            num_simulations=4,
            num_svi_steps=5,
            num_samples=4000,
        )
        np.testing.assert_allclose(
            reused_data.probabilities["loc"], baseline.probabilities["loc"]
        )

    def test_bound_container_observed_data_is_cleared_for_prior_predictive(self):
        """Bound container data should not affect simulated datasets."""
        baseline = vsbc_diagnostic(
            random.PRNGKey(0),
            _single_obs_container_model,
            _single_obs_container_guide,
            numpyro.optim.Adam(0.01),
            Trace_ELBO(),
            observed_sites=["obs"],
            simulated_data_to_args=_simulated_data_to_single_obs_container_args,
            num_simulations=4,
            num_svi_steps=5,
            num_samples=4000,
        )
        reused_data = vsbc_diagnostic(
            random.PRNGKey(0),
            _single_obs_container_model,
            _single_obs_container_guide,
            numpyro.optim.Adam(0.01),
            Trace_ELBO(),
            observed_sites=["obs"],
            simulated_data_to_args=_simulated_data_to_single_obs_container_args,
            data={"obs": 10.0},
            num_simulations=4,
            num_svi_steps=5,
            num_samples=4000,
        )
        np.testing.assert_allclose(
            reused_data.probabilities["loc"], baseline.probabilities["loc"]
        )

    def test_container_observed_data_requires_explicit_mapper(self):
        """Container-style observed data should require simulated_data_to_args."""
        with pytest.raises(ValueError, match="simulated_data_to_args"):
            vsbc_diagnostic(
                random.PRNGKey(0),
                _container_model,
                _container_data_guide,
                numpyro.optim.Adam(0.01),
                Trace_ELBO(),
                observed_sites=["obs_loc", "obs_scale"],
                num_simulations=2,
                num_svi_steps=5,
                num_samples=10,
            )

    def test_container_observed_data_mapper_threads_multiple_sites(self):
        """Container mappers should route the intended simulated observations."""
        rng_key = random.PRNGKey(0)
        num_simulations = 4
        result = vsbc_diagnostic(
            rng_key,
            _container_model,
            _container_data_guide,
            numpyro.optim.Adam(0.01),
            Trace_ELBO(),
            observed_sites=["obs_loc", "obs_scale"],
            simulated_data_to_args=_simulated_data_to_container_args,
            num_simulations=num_simulations,
            num_svi_steps=5,
            num_samples=4000,
        )
        rng_key, _ = random.split(rng_key)
        _, key_prior = random.split(rng_key)
        prior_samples = Predictive(_container_model, num_samples=num_simulations)(key_prior)
        expected_loc = stats.norm.cdf(
            np.array(prior_samples["loc"]),
            loc=np.array(prior_samples["obs_loc"]),
            scale=1.0,
        )
        expected_scale = stats.lognorm.cdf(
            np.array(prior_samples["scale"]),
            s=0.2,
            scale=np.exp(np.array(prior_samples["obs_scale"])),
        )
        np.testing.assert_allclose(result.probabilities["loc"], expected_loc, atol=0.03)
        np.testing.assert_allclose(
            result.probabilities["scale"], expected_scale, atol=0.03
        )

    def test_container_observed_data_mapper_supports_vectorized_dispatch(self):
        """Vectorized dispatch should preserve container-mapper behavior."""
        sequential = vsbc_diagnostic(
            random.PRNGKey(0),
            _container_model,
            _container_data_guide,
            numpyro.optim.Adam(0.01),
            Trace_ELBO(),
            observed_sites=["obs_loc", "obs_scale"],
            simulated_data_to_args=_simulated_data_to_container_args,
            num_simulations=4,
            num_svi_steps=20,
            num_samples=20,
        )
        vectorized = vsbc_diagnostic(
            random.PRNGKey(0),
            _container_model,
            _container_data_guide,
            numpyro.optim.Adam(0.01),
            Trace_ELBO(),
            observed_sites=["obs_loc", "obs_scale"],
            simulated_data_to_args=_simulated_data_to_container_args,
            num_simulations=4,
            num_svi_steps=20,
            num_samples=20,
            chain_method="vectorized",
        )
        assert vectorized.param_names == sequential.param_names
        np.testing.assert_allclose(
            vectorized.probabilities["loc"], sequential.probabilities["loc"]
        )
        np.testing.assert_allclose(
            vectorized.probabilities["scale"], sequential.probabilities["scale"]
        )
        np.testing.assert_array_equal(vectorized.ranks["loc"], sequential.ranks["loc"])
        np.testing.assert_array_equal(
            vectorized.ranks["scale"], sequential.ranks["scale"]
        )

    @pytest.mark.parametrize(
        "simulated_data_to_args, match",
        [
            (0, "must be callable"),
            (lambda sim_data, *args, **kwargs: sim_data, "must return a pair"),
            (
                lambda sim_data, *args, **kwargs: (sim_data["obs_loc"], {}),
                "must return a tuple or list for args",
            ),
            (
                lambda sim_data, *args, **kwargs: (args, sim_data["obs_loc"]),
                "must return a mapping for kwargs",
            ),
        ],
    )
    def test_container_observed_data_mapper_validation(
        self, simulated_data_to_args, match
    ):
        """Container mappers should fail fast on invalid callback contracts."""
        with pytest.raises(ValueError, match=match):
            vsbc_diagnostic(
                random.PRNGKey(0),
                _container_model,
                _container_data_guide,
                numpyro.optim.Adam(0.01),
                Trace_ELBO(),
                observed_sites=["obs_loc", "obs_scale"],
                simulated_data_to_args=simulated_data_to_args,
                num_simulations=2,
                num_svi_steps=5,
                num_samples=10,
            )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(num_simulations=1),
        dict(num_svi_steps=0),
        dict(num_samples=1),
        dict(observed_sites=["missing"]),
        dict(observed_sites=["loc", "obs"]),
        dict(chain_method="invalid"),
    ],
)
def test_input_validation(kwargs):
    """Invalid inputs should raise ValueError."""
    defaults = dict(observed_sites=["obs"], num_simulations=2)
    defaults.update(kwargs)
    guide = AutoNormal(_normal_model)
    with pytest.raises(ValueError):
        vsbc_diagnostic(
            random.PRNGKey(0),
            _normal_model,
            guide,
            numpyro.optim.Adam(0.01),
            Trace_ELBO(),
            **defaults,
        )


def test_observed_sites_auto_detection_requires_explicit_sites_for_generative_models():
    """Generative models with obs=None should request observed_sites explicitly."""
    guide = AutoNormal(_normal_model)
    with pytest.raises(ValueError, match="Unable to infer observed sites"):
        vsbc_diagnostic(
            random.PRNGKey(0),
            _normal_model,
            guide,
            numpyro.optim.Adam(0.01),
            Trace_ELBO(),
            num_simulations=2,
            num_svi_steps=10,
            num_samples=10,
        )


def test_discrete_latents_raise_value_error():
    """VSBC currently supports only continuous latent sample sites."""
    with pytest.raises(ValueError, match="supports only continuous latent"):
        vsbc_diagnostic(
            random.PRNGKey(0),
            _discrete_model,
            _discrete_guide,
            numpyro.optim.Adam(0.01),
            Trace_ELBO(),
            observed_sites=["obs"],
            num_simulations=2,
            num_svi_steps=5,
            num_samples=10,
        )


@pytest.mark.parametrize("chain_method", ["sequential", "vectorized"])
def test_chain_methods(chain_method):
    """Both sequential and vectorized dispatch produce valid VSBC outputs."""
    guide = AutoNormal(_normal_model)
    num_samples = 100
    result = vsbc_diagnostic(
        random.PRNGKey(42),
        _normal_model,
        guide,
        numpyro.optim.Adam(0.01),
        Trace_ELBO(),
        observed_sites=["obs"],
        num_simulations=5,
        num_svi_steps=200,
        num_samples=num_samples,
        chain_method=chain_method,
    )
    assert result.probabilities["loc"].shape == (5,)
    assert np.all(result.probabilities["loc"] >= 0)
    assert np.all(result.probabilities["loc"] <= 1)
    assert result.ranks["loc"].shape == (5,)
    assert np.all(result.ranks["loc"] >= 0)
    assert np.all(result.ranks["loc"] <= num_samples)


def test_vector_latent_sites_return_marginal_probabilities():
    """Vector-valued latents should return per-component VSBC diagnostics."""

    def vector_latent_model(x=None):
        z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([2]))
        numpyro.sample("obs", dist.Normal(z.sum(), 1.0), obs=x)

    def vector_guide(x=None):
        numpyro.sample("z", dist.Normal(0.0, 1.0).expand([2]))

    result = vsbc_diagnostic(
        random.PRNGKey(0),
        vector_latent_model,
        vector_guide,
        numpyro.optim.Adam(0.01),
        Trace_ELBO(),
        observed_sites=["obs"],
        num_simulations=5,
        num_svi_steps=5,
        num_samples=200,
    )
    assert result.probabilities["z"].shape == (5, 2)
    assert result.ranks["z"].shape == (5, 2)
    assert result.ks_stats["z"].shape == (2,)
    assert result.ks_pvalues["z"].shape == (2,)


def test_parallel_fallback():
    """parallel falls back to sequential with a warning when not enough devices."""
    guide = AutoNormal(_normal_model)
    with pytest.warns(UserWarning, match="Not enough devices"):
        result = vsbc_diagnostic(
            random.PRNGKey(0),
            _normal_model,
            guide,
            numpyro.optim.Adam(0.01),
            Trace_ELBO(),
            observed_sites=["obs"],
            num_simulations=5,
            num_svi_steps=100,
            num_samples=50,
            chain_method="parallel",
        )
    assert result.ranks["loc"].shape == (5,)
