# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from numpy.testing import assert_allclose
import pytest

import jax
from jax import random
import jax.numpy as jnp

from numpyro.ops.gaussian import Gaussian


def random_gaussian(key, batch_shape, dim, rank=None):
    rank = 2 * dim if rank is None else rank
    k1, k2, k3 = random.split(key, 3)
    log_normalizer = random.normal(k1, batch_shape)
    info_vec = random.normal(k2, batch_shape + (dim,))
    factor = random.normal(k3, batch_shape + (dim, rank))
    return Gaussian(log_normalizer, info_vec, factor @ jnp.swapaxes(factor, -1, -2))


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
