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


def _two_dim_model():
    numpyro.sample("x", dist.Normal(jnp.array([0.0, 0.0]), 1.0).to_event(1))


def _model_with_args(loc, scale=1.0):
    numpyro.sample("x", dist.Normal(loc, scale).to_event(1))


def test_mclmc_model_required():
    """Test that ValueError is raised when model is None."""
    with pytest.raises(ValueError, match="Model must be specified"):
        MCLMC(model=None)


def test_mclmc_normal():
    """Test MCLMC with a 2D normal distribution."""
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
    """Test MCLMC with a logistic regression model."""
    N, dim = 1000, 3
    num_warmup, num_samples = 1000, 2000

    key1, key2, key3 = random.split(random.PRNGKey(0), 3)
    data = random.normal(key1, (N, dim))
    true_coefs = jnp.arange(1.0, dim + 1.0)
    logits = jnp.sum(true_coefs * data, axis=-1)
    labels = dist.Bernoulli(logits=logits).sample(key2)

    # Closure pattern is used here for compactness.
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


def test_mclmc_model_args_and_kwargs():
    """Test that model_args/model_kwargs are respected during inference."""
    true_mean = jnp.array([1.5, -0.5])
    true_scale = 0.8
    num_warmup, num_samples = 500, 1000

    kernel = MCLMC(model=_model_with_args)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(1), true_mean, scale=true_scale)
    samples = mcmc.get_samples()["x"]

    assert samples.shape == (num_samples, 2)
    assert_allclose(jnp.mean(samples, axis=0), true_mean, atol=0.2)
    assert_allclose(jnp.std(samples, axis=0), true_scale, atol=0.2)


def test_mclmc_rejects_one_dimensional_latent_space():
    """Test that MCLMC rejects models with fewer than 2 latent dimensions."""

    def one_dim_model():
        numpyro.sample("x", dist.Normal(0.0, 1.0))

    kernel = MCLMC(model=one_dim_model)
    mcmc = MCMC(
        kernel,
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        progress_bar=False,
    )
    with pytest.raises(
        ValueError,
        match="target distribution must have more than 1 dimension",
    ):
        mcmc.run(random.PRNGKey(0))


def test_mclmc_small_warmup_runs():
    """Test small warmup edge case where adaptation phases are tiny."""
    kernel = MCLMC(model=_two_dim_model)
    mcmc = MCMC(
        kernel,
        num_warmup=3,
        num_samples=20,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(2))
    samples = mcmc.get_samples()["x"]
    assert samples.shape == (20, 2)
