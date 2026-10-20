# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest
from scipy import stats
from scipy.special import logsumexp

import jax
import jax.numpy as jnp

import numpyro.distributions as dist


def get_normal(batch_shape):
    """Get parameterized Normal with given batch shape."""
    loc = jnp.zeros(batch_shape)
    scale = jnp.ones(batch_shape)
    normal = dist.Normal(loc=loc, scale=scale)
    return normal


def get_half_normal(batch_shape):
    """Get parameterized HalfNormal with given batch shape."""
    scale = jnp.ones(batch_shape)
    half_normal = dist.HalfNormal(scale=scale)
    return half_normal


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


@pytest.mark.parametrize("batch_shape", [(), (1,), (7,), (2, 5)])
@pytest.mark.filterwarnings(
    "ignore:Out-of-support values provided to log prob method."
    " The value argument should be within the support.:UserWarning"
)
def test_mixture_with_different_support(batch_shape):
    mixing_probabilities = jnp.ones(2) / 2
    mixing_distribution = dist.Categorical(probs=mixing_probabilities)
    component_distribution = [
        get_normal(batch_shape),
        get_half_normal(batch_shape),
    ]
    mixture = dist.MixtureGeneral(
        mixing_distribution=mixing_distribution,
        component_distributions=component_distribution,
        support=dist.constraints.real,
    )
    assert mixture.batch_shape == batch_shape
    sample_shape = (11,)
    component_distribution[0]._validate_args = True
    component_distribution[1]._validate_args = True
    xx = component_distribution[0].sample(jax.random.key(42), sample_shape)
    log_prob_0 = component_distribution[0].log_prob(xx)
    log_prob_1 = component_distribution[1].log_prob(xx)
    expected_log_prob = jax.scipy.special.logsumexp(
        jnp.stack(
            [
                log_prob_0 + jnp.log(mixing_probabilities[0]),
                log_prob_1 + jnp.log(mixing_probabilities[1]),
            ],
            axis=-1,
        ),
        axis=-1,
    )
    result = mixture.log_prob(xx)
    assert jnp.allclose(result, expected_log_prob)


def test_mixture_cdf_with_different_support():
    # a component contributes 0 to the cdf below its support and 1 above it
    value = np.array([-1.0, 0.0, 0.25, 0.5, 1.0, 1.5])

    def cdf(params):
        mixture = dist.MixtureGeneral(
            dist.Categorical(probs=np.ones(3) / 3),
            [
                dist.Normal(0.0, 1.0),
                dist.Gamma(params[0], params[1]),
                dist.Beta(params[2], params[3]),
            ],
            support=dist.constraints.real,
        )
        return mixture.cdf(value)

    params = np.array([2.0, 3.0, 2.0, 3.0])
    expected = (
        stats.norm.cdf(value)
        + stats.gamma.cdf(value, 2.0, scale=1 / 3)
        + stats.beta.cdf(value, 2.0, 3.0)
    ) / 3
    np.testing.assert_allclose(cdf(params), expected, rtol=1e-6, atol=1e-7)
    # evaluating inside the support (not only masking) keeps gradients finite
    assert np.isfinite(jax.grad(lambda p: cdf(p).sum())(params)).all()


def _test_mixture(mixing_distribution, component_distribution):
    # Create mixture
    mixture = dist.Mixture(
        mixing_distribution=mixing_distribution,
        component_distributions=component_distribution,
    )
    assert mixture.mixture_size == mixing_distribution.probs.shape[-1], (
        "Mixture size needs to be the size of the probability vector"
    )

    if isinstance(component_distribution, dist.Distribution):
        assert mixture.batch_shape == component_distribution.batch_shape[:-1], (
            "Mixture batch shape needs to be the component batch shape without the mixture dimension."
        )
    else:
        assert mixture.batch_shape == component_distribution[0].batch_shape, (
            "Mixture batch shape needs to be the component batch shape."
        )
    # Test samples
    sample_shape = (11,)
    # Samples from component distribution(s)
    component_samples = mixture.component_sample(jax.random.key(42), sample_shape)
    assert component_samples.shape == (
        *sample_shape,
        *mixture.batch_shape,
        mixture.mixture_size,
        *mixture.event_shape,
    )
    # Samples from mixture
    samples = mixture.sample(jax.random.key(42), sample_shape=sample_shape)
    assert samples.shape == (*sample_shape, *mixture.batch_shape, *mixture.event_shape)
    # Check log_prob
    lp = mixture.log_prob(samples)
    nb_value_dims = len(samples.shape) - mixture.event_dim
    expected_shape = samples.shape[:nb_value_dims]
    assert lp.shape == expected_shape
    # Samples with indices
    samples_, [indices] = mixture.sample_with_intermediates(
        jax.random.key(42), sample_shape=sample_shape
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


@pytest.mark.parametrize(
    "component_dist",
    [
        dist.Uniform(low=np.array([0.0, 5.0]), high=np.array([1.0, 10.0])),
        dist.TruncatedNormal(loc=np.zeros(2), scale=np.ones(2), low=0.0, high=1.0),
    ],
)
def test_mixture_rejects_parameter_dependent_components(component_dist):
    mixing_dist = dist.Categorical(probs=np.array([0.5, 0.5]))
    with pytest.raises(AssertionError, match="ParameterFreeConstraint, but found "):
        dist.MixtureSameFamily(mixing_dist, component_dist)


@pytest.mark.parametrize(
    "component_dist",
    [
        dist.Normal(np.zeros(2), np.ones(2)),
        dist.Exponential(np.ones(2)),
        dist.Bernoulli(probs=np.array([0.5, 0.5])),
    ],
)
def test_mixture_accepts_parameter_free_components(component_dist):
    mixing_dist = dist.Categorical(probs=np.array([0.3, 0.7]))
    dist.MixtureSameFamily(mixing_dist, component_dist)


def test_mixture_log_prob_grad_at_zero_weights():
    # regression test for #1874: gradients must not become NaN when some
    # mixing weights are exactly zero
    def loss(probs):
        mixture = dist.MixtureSameFamily(
            dist.Categorical(probs=probs),
            dist.Normal(jnp.arange(3.0), jnp.ones(3)),
        )
        return mixture.log_prob(jnp.array([0.5, 2.0])).sum()

    grad = jax.grad(loss)(jnp.array([0.5, 0.5, 0.0]))
    assert not jnp.isnan(grad).any()


def test_mixture_log_prob_grad_all_components_neg_inf():
    # if every component assigns -inf to a value, the log_prob is -inf and
    # gradients must not become NaN
    mixture = dist.MixtureGeneral(
        dist.Categorical(probs=jnp.array([0.5, 0.5])),
        [
            dist.HalfNormal(1.0, validate_args=False),
            dist.HalfNormal(2.0, validate_args=False),
        ],
        support=dist.constraints.real,
        validate_args=False,
    )
    value = jnp.array([-1.0, 1.0])
    assert jnp.isneginf(mixture.log_prob(value)[0])
    grad = jax.grad(lambda x: mixture.log_prob(x).sum())(value)
    assert not jnp.isnan(grad).any()


@pytest.mark.parametrize(
    "batch_shapes", [((), (3,)), ((2, 1), (1, 3)), ((2, 3), (2, 3))]
)
@pytest.mark.parametrize("event_shape", [(), (2,), (2, 2)])
@pytest.mark.parametrize("jit", [False, True])
def test_general_mixture_broadcast_components(batch_shapes, event_shape, jit):
    batch_shape = np.broadcast_shapes(*batch_shapes)
    locs = [
        np.full(shape + event_shape, loc)
        for shape, loc in zip(batch_shapes, [-1.0, 2.0])
    ]
    components = [
        dist.Normal(locs[0], 2.0).to_event(len(event_shape)),
        dist.Laplace(locs[1], 1.5).to_event(len(event_shape)),
    ]
    mixture = dist.MixtureGeneral(
        dist.Categorical(probs=np.array([0.3, 0.7])), components
    )
    mean = np.broadcast_to(0.3 * locs[0] + 0.7 * locs[1], batch_shape + event_shape)
    variance = 0.3 * (4.0 + (locs[0] - mean) ** 2) + 0.7 * (4.5 + (locs[1] - mean) ** 2)
    value = np.full(event_shape, 0.25)
    component_log_probs = [
        stats.norm.logpdf(value, loc=locs[0], scale=2.0),
        stats.laplace.logpdf(value, loc=locs[1], scale=1.5),
    ]
    if event_shape:
        axes = tuple(range(-len(event_shape), 0))
        component_log_probs = [p.sum(axis=axes) for p in component_log_probs]
    expected_log_prob = logsumexp(
        np.stack(np.broadcast_arrays(*component_log_probs), axis=-1)
        + np.log([0.3, 0.7]),
        axis=-1,
    )

    def evaluate(value):
        return mixture.mean, mixture.variance, mixture.log_prob(value)

    actual_mean, actual_variance, actual_log_prob = (
        jax.jit(evaluate) if jit else evaluate
    )(value)
    assert actual_mean.shape == batch_shape + event_shape
    assert actual_variance.shape == batch_shape + event_shape
    assert actual_log_prob.shape == batch_shape
    np.testing.assert_allclose(actual_mean, mean, rtol=1e-6)
    np.testing.assert_allclose(actual_variance, variance, rtol=1e-6)
    np.testing.assert_allclose(actual_log_prob, expected_log_prob, rtol=1e-6)
    if not event_shape:
        expected_cdf = 0.3 * stats.norm.cdf(value, loc=locs[0], scale=2.0)
        expected_cdf = expected_cdf + 0.7 * stats.laplace.cdf(
            value, loc=locs[1], scale=1.5
        )
        actual_cdf = (jax.jit(mixture.cdf) if jit else mixture.cdf)(value)
        np.testing.assert_allclose(actual_cdf, expected_cdf, rtol=1e-6)
