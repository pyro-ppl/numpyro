# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from numpy.testing import assert_allclose
import pytest

import jax
from jax import random
import jax.numpy as jnp

import numpyro.distributions as dist
from numpyro.ops.gamma_gaussian import GammaFactor, GammaGaussian
from numpyro.ops.gaussian import Gaussian


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


def random_gamma_gaussian(key, batch_shape, dim):
    k1, k2, k3 = random.split(key, 3)
    g = random_gaussian(k1, batch_shape, dim)
    loc = random.normal(k2, batch_shape + (dim,))
    info_vec = jnp.einsum("...ij,...j->...i", g.precision, loc)
    alpha = jnp.exp(random.normal(k3, batch_shape)) + 0.5 * dim - 1
    beta = jnp.exp(random.normal(random.fold_in(k3, 1), batch_shape)) + 0.5 * (
        info_vec * loc
    ).sum(-1)
    return GammaGaussian(g.log_normalizer, info_vec, g.precision, alpha, beta)


def gaussian_at(gg, s):
    """The Gaussian factor obtained by fixing the scale variable ``s``."""
    s = jnp.asarray(s)
    return Gaussian(
        gg.log_normalizer + gg.alpha * jnp.log(s) - s * gg.beta,
        s[..., None] * gg.info_vec,
        s[..., None, None] * gg.precision,
    )


def assert_close_gamma_gaussian(actual, expected, rtol=1e-4, atol=1e-4):
    assert actual.dim == expected.dim and actual.batch_shape == expected.batch_shape
    for name in ("log_normalizer", "info_vec", "precision", "alpha", "beta"):
        assert_allclose(
            getattr(actual, name), getattr(expected, name), rtol=rtol, atol=atol
        )


def test_shape_ops():
    gg = random_gamma_gaussian(random.key(0), (6,), 2)
    assert gg.expand((3, 6)).batch_shape == (3, 6)
    assert_close_gamma_gaussian(gg.reshape((2, 3)).reshape((6,)), gg)
    assert_close_gamma_gaussian(GammaGaussian.cat([gg[:2], gg[2:]], axis=0), gg)
    assert gg.event_pad(left=1, right=1).dim == 4
    perm = jnp.array([1, 0])
    assert_close_gamma_gaussian(gg.event_permute(perm).event_permute(perm), gg)


def test_log_density_matches_fixed_scale_gaussian():
    gg = random_gamma_gaussian(random.key(0), (3,), 2)
    x = random.normal(random.key(1), (3, 2))
    s = jnp.exp(random.normal(random.key(2), (3,)))
    assert_allclose(gg.log_density(x, s), gaussian_at(gg, s).log_density(x), rtol=1e-4)
    added = gg + random_gamma_gaussian(random.key(3), (3,), 2)
    assert added.batch_shape == (3,)


@pytest.mark.parametrize("left,right", [(1, 0), (0, 1), (0, 2), (1, 1)])
def test_marginalize_and_condition(left, right):
    gg = random_gamma_gaussian(random.key(0), (3,), 4)
    s = jnp.exp(random.normal(random.key(2), (3,)))
    value = random.normal(random.key(1), (3, 4 - left - right))
    marginal = gg.marginalize(left=left, right=right)
    assert_allclose(
        marginal.log_density(value, s),
        gaussian_at(gg, s).marginalize(left=left, right=right).log_density(value),
        rtol=1e-4,
        atol=1e-4,
    )
    if right == 0:
        assert_allclose(
            marginal.log_density(value, s),
            gg.condition(value).event_logsumexp().log_density(s),
            rtol=1e-4,
            atol=1e-4,
        )


def test_event_logsumexp_and_compound():
    gg = random_gamma_gaussian(random.key(0), (3,), 2)
    s = jnp.exp(random.normal(random.key(2), (3,)))
    factor = gg.event_logsumexp()
    assert isinstance(factor, GammaFactor)
    assert_allclose(
        factor.log_density(s), gaussian_at(gg, s).event_logsumexp(), rtol=1e-4
    )
    x = random.normal(random.key(1), (3, 2))
    grid = jnp.linspace(1e-3, 40.0, 20000)
    log_dgrid = jnp.log(grid[1] - grid[0])
    expected = jax.vmap(
        lambda xi, ggi: (
            jax.scipy.special.logsumexp(gaussian_at(ggi, grid).log_density(xi))
            + log_dgrid
        )
    )(x, gg)
    assert_allclose(gg.compound().log_prob(x), expected - factor.logsumexp(), atol=2e-2)
    assert_allclose(
        factor.logsumexp(),
        jax.vmap(
            lambda f: jax.scipy.special.logsumexp(f.log_density(grid)) + log_dgrid
        )(factor),
        atol=2e-2,
    )
