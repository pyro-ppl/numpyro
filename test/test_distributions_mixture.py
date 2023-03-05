# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest

import jax
import jax.numpy as jnp

import numpyro.distributions as dist

rng_key = jax.random.PRNGKey(42)


def get_normal(batch_shape):
    """Get parameterized Normal with given batch shape."""
    loc = jnp.zeros(batch_shape)
    scale = jnp.ones(batch_shape)
    normal = dist.Normal(loc=loc, scale=scale)
    return normal


def get_mvn(batch_shape):
    """Get parameterized MultivariateNormal with given batch shape."""
    dimensions = 2
    loc = jnp.zeros((*batch_shape, dimensions))
    cov_matrix = jnp.eye(dimensions, dimensions)
    for i, s in enumerate(batch_shape):
        loc = jnp.repeat(jnp.expand_dims(loc, i), s, axis=i)
        cov_matrix = jnp.repeat(jnp.expand_dims(cov_matrix, i), s, axis=i)
    mvn = dist.MultivariateNormal(loc=loc, covariance_matrix=cov_matrix)
    return mvn


@pytest.mark.parametrize("jax_dist_getter", [get_normal, get_mvn])
@pytest.mark.parametrize("nb_mixtures", [1, 3])
@pytest.mark.parametrize("batch_shape", [(), (1,), (7,), (2, 5)])
@pytest.mark.parametrize("same_family", [True, False])
def test_mixture_same_batch_shape(
    jax_dist_getter, nb_mixtures, batch_shape, same_family
):
    mixing_probabilities = jnp.ones(nb_mixtures) / nb_mixtures
    for i, s in enumerate(batch_shape):
        mixing_probabilities = jnp.repeat(
            jnp.expand_dims(mixing_probabilities, i), s, axis=i
        )
    assert jnp.allclose(mixing_probabilities.sum(axis=-1), 1.0)
    mixing_distribution = dist.Categorical(probs=mixing_probabilities)
    if same_family:
        component_distribution = jax_dist_getter((*batch_shape, nb_mixtures))
    else:
        component_distribution = [
            jax_dist_getter(batch_shape) for _ in range(nb_mixtures)
        ]
    _test_mixture(mixing_distribution, component_distribution)


@pytest.mark.parametrize("jax_dist_getter", [get_normal, get_mvn])
@pytest.mark.parametrize("nb_mixtures", [3])
@pytest.mark.parametrize("mixing_batch_shape, component_batch_shape", [[(2,), (7, 2)]])
@pytest.mark.parametrize("same_family", [True, False])
def test_mixture_broadcast_batch_shape(
    jax_dist_getter, nb_mixtures, mixing_batch_shape, component_batch_shape, same_family
):
    # Create mixture
    mixing_probabilities = jnp.ones(nb_mixtures) / nb_mixtures
    for i, s in enumerate(mixing_batch_shape):
        mixing_probabilities = jnp.repeat(
            jnp.expand_dims(mixing_probabilities, i), s, axis=i
        )
    assert jnp.allclose(mixing_probabilities.sum(axis=-1), 1.0)
    mixing_distribution = dist.Categorical(probs=mixing_probabilities)
    if same_family:
        component_distribution = jax_dist_getter((*component_batch_shape, nb_mixtures))
    else:
        component_distribution = [
            jax_dist_getter(component_batch_shape) for _ in range(nb_mixtures)
        ]
    _test_mixture(mixing_distribution, component_distribution)


def _test_mixture(mixing_distribution, component_distribution):
    # Create mixture
    mixture = dist.Mixture(
        mixing_distribution=mixing_distribution,
        component_distributions=component_distribution,
    )
    assert (
        mixture.mixture_size == mixing_distribution.probs.shape[-1]
    ), "Mixture size needs to be the size of the probability vector"

    if isinstance(component_distribution, dist.Distribution):
        assert (
            mixture.batch_shape == component_distribution.batch_shape[:-1]
        ), "Mixture batch shape needs to be the component batch shape without the mixture dimension."
    else:
        assert (
            mixture.batch_shape == component_distribution[0].batch_shape
        ), "Mixture batch shape needs to be the component batch shape."
    # Test samples
    sample_shape = (11,)
    # Samples from component distribution(s)
    component_samples = mixture.component_sample(rng_key, sample_shape)
    assert component_samples.shape == (
        *sample_shape,
        *mixture.batch_shape,
        mixture.mixture_size,
        *mixture.event_shape,
    )
    # Samples from mixture
    samples = mixture.sample(rng_key, sample_shape=sample_shape)
    assert samples.shape == (*sample_shape, *mixture.batch_shape, *mixture.event_shape)
    # Check log_prob
    lp = mixture.log_prob(samples)
    nb_value_dims = len(samples.shape) - mixture.event_dim
    expected_shape = samples.shape[:nb_value_dims]
    assert lp.shape == expected_shape
    # Samples with indices
    samples_, [indices] = mixture.sample_with_intermediates(
        rng_key, sample_shape=sample_shape
    )
    assert samples_.shape == samples.shape
    assert indices.shape == (*sample_shape, *mixture.batch_shape)
    assert jnp.issubdtype(indices.dtype, jnp.integer)
    assert (indices >= 0).all() and (indices < mixture.mixture_size).all()
    # Check mean
    mean = mixture.mean
    assert mean.shape == mixture.shape()
    # Check variance
    var = mixture.variance
    assert var.shape == mixture.shape()
    # Check cdf
    if mixture.event_shape == ():
        cdf = mixture.cdf(samples)
        assert cdf.shape == (*sample_shape, *mixture.shape())
