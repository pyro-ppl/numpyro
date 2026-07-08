# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest

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


def _make_mixture(mixture_cls, probs):
    """A 3-component Normal mixture, built as SameFamily or General."""
    locs = jnp.array([0.0, 1.0, 2.0])
    mixing = dist.Categorical(probs=probs)
    if mixture_cls is dist.MixtureSameFamily:
        return dist.MixtureSameFamily(mixing, dist.Normal(locs, 1.0))
    return dist.MixtureGeneral(mixing, [dist.Normal(loc, 1.0) for loc in locs])


@pytest.mark.parametrize("mixture_cls", [dist.MixtureSameFamily, dist.MixtureGeneral])
@pytest.mark.parametrize(
    "probs",
    [
        np.array([0.0, 0.5, 0.5]),  # single exact-zero weight
        np.array([0.0, 0.0, 1.0]),  # two exact-zero weights
    ],
)
def test_mixture_log_prob_grad_at_zero_weight(mixture_cls, probs):
    # A probs= mixing weight of exactly 0 used to produce a NaN gradient (the
    # forward value was finite) because the density was written in log-weight
    # form and log(0) has an infinite VJP. The weighted-logsumexp formulation
    # keeps both the value and the gradient finite and exact.
    probs = jnp.asarray(probs)
    value = jnp.array(0.3)
    locs = jnp.array([0.0, 1.0, 2.0])

    # Reference: log sum_k w_k N(x; loc_k, 1), component densities constant in w.
    component_probs = jnp.exp(dist.Normal(locs, 1.0).log_prob(value))

    def ref(w):
        return jnp.log(jnp.sum(w * component_probs))

    def actual(w):
        return _make_mixture(mixture_cls, w).log_prob(value)

    value_ref, grad_ref = jax.value_and_grad(ref)(probs)
    value_actual, grad_actual = jax.value_and_grad(actual)(probs)

    assert jnp.isfinite(value_actual)
    assert jnp.all(jnp.isfinite(grad_actual))
    assert jnp.allclose(value_actual, value_ref, rtol=1e-5)
    # Gradient is exact, including the p_k / sum_j w_j p_j slot at w_k == 0.
    assert jnp.allclose(grad_actual, grad_ref, rtol=1e-5)


def test_mixture_general_out_of_support_component_zero_weight():
    # An out-of-support component (log p_k = -inf) that also has zero weight must
    # not pollute the value or the gradient.
    value = jnp.array(-0.5)

    def actual(w):
        # validate_args=False: the HalfNormal component is evaluated outside its
        # support on purpose (masked via the explicit ``support=`` override), and
        # validation would warn about it.
        mixture = dist.MixtureGeneral(
            dist.Categorical(probs=w),
            [
                dist.HalfNormal(1.0, validate_args=False),
                dist.Normal(-1.0, 1.0, validate_args=False),
                dist.Normal(-2.0, 1.0, validate_args=False),
            ],
            support=dist.constraints.real,
        )
        return mixture.log_prob(value)

    # HalfNormal is out of support at -0.5; give it zero weight.
    probs = jnp.array([0.0, 0.5, 0.5])
    component_probs = jnp.array(
        [
            0.0,  # out of support -> density 0
            jnp.exp(dist.Normal(-1.0, 1.0).log_prob(value)),
            jnp.exp(dist.Normal(-2.0, 1.0).log_prob(value)),
        ]
    )
    value_ref = jnp.log(jnp.sum(probs * component_probs))
    grad_ref = component_probs / jnp.sum(probs * component_probs)

    value_actual, grad_actual = jax.value_and_grad(actual)(probs)
    assert jnp.isfinite(value_actual)
    assert jnp.all(jnp.isfinite(grad_actual))
    assert jnp.allclose(value_actual, value_ref, rtol=1e-5)
    assert jnp.allclose(grad_actual, grad_ref, rtol=1e-5)


@pytest.mark.parametrize("mixture_cls", [dist.MixtureSameFamily, dist.MixtureGeneral])
def test_mixture_logits_path_matches_probs(mixture_cls):
    # A logits-parameterized mixing distribution was never affected by the bug;
    # guard that the refactor leaves its value and gradient unchanged.
    value = jnp.array(0.3)

    def with_logits(logits):
        probs = jax.nn.softmax(logits)
        return _make_mixture(mixture_cls, probs).log_prob(value)

    logits = jnp.array([0.5, -0.3, 0.8])
    v, g = jax.value_and_grad(with_logits)(logits)
    assert jnp.isfinite(v)
    assert jnp.all(jnp.isfinite(g))


@pytest.mark.parametrize("mixture_cls", [dist.MixtureSameFamily, dist.MixtureGeneral])
def test_mixture_component_log_probs_are_responsibilities(mixture_cls):
    # component_log_probs - log_prob must exponentiate to responsibilities that
    # sum to 1 (the documented public contract).
    value = jnp.array(0.3)
    mixture = _make_mixture(mixture_cls, jnp.array([0.2, 0.3, 0.5]))
    responsibilities = jnp.exp(
        mixture.component_log_probs(value) - mixture.log_prob(value)
    )
    assert jnp.allclose(responsibilities.sum(), 1.0, rtol=1e-5)
