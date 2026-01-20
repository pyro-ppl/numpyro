# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from numpy.testing import assert_allclose
import pytest

from jax import random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC
from numpyro.infer.mclmc import MCLMC


def test_mclmc_model_required():
    """Test that ValueError is raised when model is None."""
    with pytest.raises(ValueError, match="Model must be specified"):
        MCLMC(model=None)


def test_mclmc_blackjax_not_installed(monkeypatch):
    """Test that ImportError is raised with informative message when blackjax is not installed."""
    import numpyro.infer.mclmc as mclmc_module

    # Temporarily set _BLACKJAX_AVAILABLE to False
    monkeypatch.setattr(mclmc_module, "_BLACKJAX_AVAILABLE", False)

    def dummy_model():
        numpyro.sample("x", dist.Normal(0, 1))

    with pytest.raises(ImportError, match="MCLMC requires the 'blackjax' package"):
        MCLMC(model=dummy_model)


def test_mclmc_normal():
    """Test MCLMC with a 2D normal distribution.

    Note: MCLMC requires at least 2 dimensions (blackjax limitation).
    """
    true_mean = jnp.array([1.0, 2.0])
    true_std = jnp.array([0.5, 1.0])
    num_warmup, num_samples = 1000, 2000

    def model():
        numpyro.sample("x", dist.Normal(true_mean, true_std).to_event(1))

    kernel = MCLMC(model=model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(0))
    samples = mcmc.get_samples()

    assert "x" in samples
    assert samples["x"].shape == (num_samples, 2)
    assert_allclose(jnp.mean(samples["x"], axis=0), true_mean, atol=0.1)
    assert_allclose(jnp.std(samples["x"], axis=0), true_std, atol=0.2)


def test_mclmc_gaussian_2d():
    """Test MCLMC with a 2D Gaussian model with observation."""
    num_warmup, num_samples = 1000, 1000

    def model():
        x = numpyro.sample("x", dist.Normal(0.0, 1.0))
        y = numpyro.sample("y", dist.Normal(0.0, 1.0))
        numpyro.sample("obs", dist.Normal(x + y, 0.5), obs=jnp.array(0.0))

    kernel = MCLMC(
        model=model,
        diagonal_preconditioning=True,
        desired_energy_var=5e-4,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(0))
    samples = mcmc.get_samples()

    assert "x" in samples
    assert "y" in samples
    assert samples["x"].shape == (num_samples,)
    assert samples["y"].shape == (num_samples,)
    # With obs=0, x+y should be close to 0, so means should be near 0
    assert_allclose(jnp.mean(samples["x"]) + jnp.mean(samples["y"]), 0.0, atol=0.2)


def test_mclmc_logistic_regression():
    """Test MCLMC with a logistic regression model.

    Note: MCLMC currently doesn't pass model_args, so we use a closure pattern.
    """
    N, dim = 1000, 3
    num_warmup, num_samples = 1000, 2000

    key1, key2, key3 = random.split(random.PRNGKey(0), 3)
    data = random.normal(key1, (N, dim))
    true_coefs = jnp.arange(1.0, dim + 1.0)
    logits = jnp.sum(true_coefs * data, axis=-1)
    labels = dist.Bernoulli(logits=logits).sample(key2)

    # Use closure pattern since MCLMC doesn't pass model_args
    def model():
        coefs = numpyro.sample("coefs", dist.Normal(jnp.zeros(dim), jnp.ones(dim)))
        logits = jnp.sum(coefs * data, axis=-1)
        numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=labels)

    kernel = MCLMC(model=model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(key3)
    samples = mcmc.get_samples()

    assert "coefs" in samples
    assert samples["coefs"].shape == (num_samples, dim)
    assert_allclose(jnp.mean(samples["coefs"], 0), true_coefs, atol=0.5)


def test_mclmc_sample_shape():
    """Test that MCLMC produces samples with expected shapes."""
    num_warmup, num_samples = 500, 500

    def model():
        numpyro.sample("a", dist.Normal(0, 1))
        numpyro.sample("b", dist.Normal(0, 1).expand([3]))
        numpyro.sample("c", dist.Normal(0, 1).expand([2, 4]))

    kernel = MCLMC(model=model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(0))
    samples = mcmc.get_samples()

    assert samples["a"].shape == (num_samples,)
    assert samples["b"].shape == (num_samples, 3)
    assert samples["c"].shape == (num_samples, 2, 4)
