from functools import reduce
from operator import mul

import jax.numpy as np
import jax.random as random
import numpy as onp
import pytest
import scipy.stats as sp
from jax import lax, grad

import numpyro.distributions as dist


@pytest.mark.parametrize('loc, scale', [
    (1, 1),
    (1., np.array([1., 2.])),
])
@pytest.mark.parametrize('prepend_shape', [
    None,
    (),
    (2,),
    (2, 3),
])
def test_shape(loc, scale, prepend_shape):
    rng = random.PRNGKey(0)
    expected_shape = lax.broadcast_shapes(*[np.shape(loc), np.shape(scale)])
    assert np.shape(dist.norm.rvs(loc, scale, random_state=rng)) == expected_shape
    assert np.shape(dist.norm(loc, scale).rvs(random_state=rng)) == expected_shape
    if prepend_shape is not None:
        expected_shape = prepend_shape + lax.broadcast_shapes(*[np.shape(loc), np.shape(scale)])
        assert np.shape(dist.norm.rvs(loc, scale, random_state=rng, size=expected_shape)) == expected_shape
        assert np.shape(dist.norm(loc, scale).rvs(random_state=rng, size=expected_shape)) == expected_shape


@pytest.mark.parametrize('loc, scale', [
    (1., 1.),
    (1., np.array([1., 2.])),
])
def test_sample_gradient(loc, scale):
    rng = random.PRNGKey(0)
    expected_shape = lax.broadcast_shapes(*[np.shape(loc), np.shape(scale)])

    def fn(loc, scale):
        return dist.norm.rvs(loc, scale, random_state=rng).sum()

    assert grad(fn)(loc, scale) == loc * reduce(mul, expected_shape[:len(expected_shape) - len(np.shape(loc))], 1.)
    assert onp.allclose(grad(fn, 1)(loc, scale), random.normal(rng, shape=expected_shape))


@pytest.mark.parametrize("loc_scale", [
    (),
    (1,),
    (1, 1),
    (1., np.array([1., 2.])),
])
def test_normal_logprob(loc_scale):
    rng = random.PRNGKey(2)
    samples = dist.norm.rvs(*loc_scale, random_state=rng)
    assert np.allclose(dist.norm.logpdf(samples, *loc_scale), sp.norm.logpdf(samples, *loc_scale))
