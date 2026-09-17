# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from numpy.testing import assert_allclose
import pytest

import jax
from jax import random
import jax.numpy as jnp

import numpyro.distributions as dist
from numpyro.ops.gaussian import (
    AffineNormal,
    Gaussian,
    _mv,
    matrix_and_gaussian_to_gaussian,
    matrix_and_mvn_to_gaussian,
    mvn_to_gaussian,
)


def random_gaussian(key, batch_shape, dim, rank=None):
    rank = 2 * dim if rank is None else rank
    k1, k2, k3 = random.split(key, 3)
    log_normalizer = random.normal(k1, batch_shape)
    info_vec = random.normal(k2, batch_shape + (dim,))
    factor = random.normal(k3, batch_shape + (dim, rank))
    return Gaussian(log_normalizer, info_vec, factor @ jnp.swapaxes(factor, -1, -2))


def random_mvn(key, batch_shape, dim):
    k1, k2 = random.split(key)
    factor = random.normal(k2, batch_shape + (dim, dim))
    return dist.MultivariateNormal(
        random.normal(k1, batch_shape + (dim,)),
        covariance_matrix=factor @ jnp.swapaxes(factor, -1, -2) + jnp.eye(dim),
    )


def assert_close_gaussian(actual, expected, rtol=1e-4, atol=1e-4):
    assert actual.dim == expected.dim
    assert actual.batch_shape == expected.batch_shape
    assert_allclose(
        actual.log_normalizer, expected.log_normalizer, rtol=rtol, atol=atol
    )
    assert_allclose(actual.info_vec, expected.info_vec, rtol=rtol, atol=atol)
    assert_allclose(actual.precision, expected.precision, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "old_shape,new_shape", [((), (4, 2)), ((3,), (2, 3)), ((1, 3), (5, 3))]
)
def test_expand(old_shape, new_shape):
    g = random_gaussian(random.key(0), old_shape, 2)
    expanded = g.expand(new_shape)
    assert expanded.batch_shape == new_shape
    assert expanded.precision.shape == new_shape + (2, 2)


def test_reshape_round_trip():
    g = random_gaussian(random.key(0), (6,), 3)
    assert_close_gaussian(g.reshape((2, 3)).reshape((6,)), g)


def test_getitem_and_cat():
    g = random_gaussian(random.key(0), (5, 4), 2)
    assert g[..., 1].batch_shape == (5,)
    assert_close_gaussian(Gaussian.cat([g[..., :2], g[..., 2:]], axis=-1), g)
    assert_close_gaussian(Gaussian.cat([g[:2], g[2:]], axis=0), g)
    assert_close_gaussian(
        Gaussian.cat([g[..., 0:4:2], g[..., 1:4:2]], axis=-1)[..., [0, 2, 1, 3]], g
    )


def test_pad_and_permute():
    g = random_gaussian(random.key(0), (3,), 2)
    padded = g.event_pad(left=1, right=2)
    assert padded.dim == 5
    assert_allclose(padded.precision[..., 1:3, 1:3], g.precision)
    assert_allclose(padded.info_vec[..., 0], 0.0)
    perm = jnp.array([1, 0])
    assert_close_gaussian(g.event_permute(perm).event_permute(perm), g)


def test_add_is_additive_in_log_density():
    x = random_gaussian(random.key(0), (3,), 2)
    y = random_gaussian(random.key(1), (), 2)
    value = random.normal(random.key(2), (3, 2))
    assert_allclose(
        (x + y).log_density(value),
        x.log_density(value) + y.log_density(value),
        rtol=1e-4,
    )
    assert_allclose((x + 1.5).log_density(value), x.log_density(value) + 1.5, rtol=1e-4)
    assert (x + y).batch_shape == (3,)
    assert (x + y).precision.shape == (3, 2, 2)


def test_vmap_over_factory_derives_batch_shape():
    def make(loc):
        return Gaussian(jnp.zeros(()), loc, jnp.eye(2))

    g = jax.vmap(make)(jnp.zeros((5, 2)))
    assert g.batch_shape == (5,)
    assert g.log_density(jnp.zeros((5, 2))).shape == (5,)


@pytest.mark.parametrize("left,right", [(1, 0), (0, 1), (2, 0), (0, 2), (1, 1)])
def test_marginalize_condition_identity(left, right):
    g = random_gaussian(random.key(0), (3,), 4)
    value = random.normal(random.key(1), (3, 4 - left - right))
    marginal = g.marginalize(left=left, right=right)
    assert marginal.batch_shape == (3,)
    assert marginal.precision.shape == (3, 4 - left - right, 4 - left - right)
    if right == 0:
        assert_allclose(
            marginal.log_density(value),
            g.condition(value).event_logsumexp(),
            rtol=1e-4,
        )
    assert_allclose(marginal.event_logsumexp(), g.event_logsumexp(), rtol=1e-4)


def test_condition_and_left_condition():
    g = random_gaussian(random.key(0), (3,), 5)
    a = random.normal(random.key(1), (3, 2))
    b = random.normal(random.key(2), (3, 3))
    ab = jnp.concatenate([a, b], -1)
    assert_allclose(g.condition(b).log_density(a), g.log_density(ab), rtol=1e-4)
    assert_allclose(g.left_condition(a).log_density(b), g.log_density(ab), rtol=1e-4)
    assert g.condition(b).batch_shape == (3,)


def test_event_logsumexp_against_monte_carlo():
    g = random_gaussian(random.key(0), (), 2)
    box = 6.0
    grid = jnp.linspace(-box, box, 400)
    xs = jnp.stack(jnp.meshgrid(grid, grid, indexing="ij"), -1).reshape(-1, 2)
    expected = jax.scipy.special.logsumexp(g.log_density(xs)) + 2 * jnp.log(
        grid[1] - grid[0]
    )
    assert_allclose(g.event_logsumexp(), expected, atol=1e-2)


def test_sample_moments_and_noise():
    g = random_gaussian(random.key(0), (2,), 3)
    samples = g.sample(random.key(1), (20000,))
    assert samples.shape == (20000, 2, 3)
    cov = jnp.linalg.inv(g.precision)
    mean = _mv(cov, g.info_vec)
    assert_allclose(samples.mean(0), mean, atol=0.05)
    assert_allclose(jax.vmap(lambda s: jnp.cov(s.T), in_axes=1)(samples), cov, atol=0.1)
    noise = random.normal(random.key(2), (4, 2, 3))
    assert_allclose(
        g.sample(sample_shape=(4,), noise=noise),
        g.sample(random.key(9), (4,), noise=noise),
    )
    assert_allclose(g.sample(noise=jnp.zeros((2, 3))), mean, rtol=1e-4)


@pytest.mark.parametrize(
    "make",
    [
        lambda: random_mvn(random.key(0), (3,), 2),
        lambda: random_mvn(random.key(0), (), 2).expand((3,)),
        lambda: dist.Normal(random.normal(random.key(0), (3, 2)), 0.7).to_event(1),
        lambda: dist.Normal(jnp.zeros(2), 1.0).to_event(1),
        lambda: dist.Normal(0.0, 1.0).expand((3, 2)).to_event(1),
    ],
)
def test_mvn_to_gaussian_matches_log_prob(make):
    d = make()
    value = random.normal(random.key(1), (3, 2))
    g = mvn_to_gaussian(d)
    assert g.batch_shape == d.batch_shape
    assert g.precision.shape == d.batch_shape + (2, 2)
    assert_allclose(g.log_density(value), d.log_prob(value), rtol=1e-4)


def test_mvn_to_gaussian_rejects_other_types():
    with pytest.raises(TypeError):
        mvn_to_gaussian(dist.StudentT(3.0, jnp.zeros(2), 1.0).to_event(1))


def test_matrix_and_gaussian_to_gaussian_broadcasts_batch():
    x_dim, y_dim = 3, 2
    matrix = random.normal(random.key(0), (4, y_dim, x_dim))
    y_gaussian = random_gaussian(random.key(1), (), y_dim)
    x = random.normal(random.key(2), (4, x_dim))
    y = random.normal(random.key(3), (4, y_dim))
    g = matrix_and_gaussian_to_gaussian(matrix, y_gaussian)
    assert g.batch_shape == (4,)
    assert g.precision.shape == (4, x_dim + y_dim, x_dim + y_dim)
    expected = y_gaussian.log_density(y - _mv(matrix, x))
    assert_allclose(g.log_density(jnp.concatenate([x, y], -1)), expected, rtol=1e-4)


@pytest.mark.parametrize("diag", [False, True])
def test_matrix_and_mvn_to_gaussian_density(diag):
    x_dim, y_dim = 3, 2
    matrix = random.normal(random.key(0), (4, y_dim, x_dim))
    noise = (
        dist.Normal(random.normal(random.key(1), (4, y_dim)), 0.5).to_event(1)
        if diag
        else random_mvn(random.key(1), (4,), y_dim)
    )
    x = random.normal(random.key(2), (4, x_dim))
    y = random.normal(random.key(3), (4, y_dim))
    g = matrix_and_mvn_to_gaussian(matrix, noise)
    assert isinstance(g, AffineNormal if diag else type(mvn_to_gaussian(noise)))
    assert g.dim == x_dim + y_dim
    expected = noise.log_prob(y - jnp.einsum("...ij,...j->...i", matrix, x))
    assert_allclose(g.log_density(jnp.concatenate([x, y], -1)), expected, rtol=1e-4)
    if diag:
        full = g.to_gaussian()
        assert_close_gaussian(g.condition(y), full.condition(y))
        assert isinstance(g.left_condition(x), AffineNormal)
        assert_allclose(g.left_condition(x).log_density(y), expected, rtol=1e-4)
        assert_allclose(g.marginalize(right=y_dim).precision, 0.0)
