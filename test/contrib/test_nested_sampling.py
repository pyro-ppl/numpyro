# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numpy.testing import assert_allclose
import pytest

from jax import random
import jax.numpy as jnp

import numpyro
from numpyro.contrib.nested_sampling import NestedSampler, UniformReparam
import numpyro.distributions as dist
from numpyro.distributions.transforms import AffineTransform, ExpTransform


# Test helper to extract a few central moments from samples.
def get_moments(x):
    m1 = jnp.mean(x, axis=0)
    x = x - m1
    xx = x * x
    xxx = x * xx
    xxxx = xx * xx
    m2 = jnp.mean(xx, axis=0)
    m3 = jnp.mean(xxx, axis=0) / m2 ** 1.5
    m4 = jnp.mean(xxxx, axis=0) / m2 ** 2
    return jnp.stack([m1, m2, m3, m4])


@pytest.mark.parametrize(
    "batch_shape,base_batch_shape",
    [
        ((), ()),
        ((4,), (4,)),
        ((2, 3), (2, 3)),
        ((2, 3), ()),
    ],
    ids=str,
)
@pytest.mark.parametrize("event_shape", [(), (5,)], ids=str)
def test_log_normal(batch_shape, base_batch_shape, event_shape):
    shape = batch_shape + event_shape
    base_shape = base_batch_shape + event_shape
    loc = np.random.rand(*base_shape) * 2 - 1
    scale = np.random.rand(*base_shape) + 0.5

    def model():
        fn = dist.TransformedDistribution(
            dist.Normal(jnp.zeros_like(loc), jnp.ones_like(scale)),
            [AffineTransform(loc, scale), ExpTransform()],
        ).expand(shape)
        if event_shape:
            fn = fn.to_event(len(event_shape)).expand_by([100000])
        with numpyro.plate_stack("plates", batch_shape):
            with numpyro.plate("particles", 100000):
                return numpyro.sample("x", fn)

    with numpyro.handlers.trace() as tr:
        value = numpyro.handlers.seed(model, 0)()
    expected_moments = get_moments(jnp.log(value))

    with numpyro.handlers.reparam(config={"x": UniformReparam()}):
        with numpyro.handlers.trace() as tr:
            value = numpyro.handlers.seed(model, 0)()
    assert tr["x"]["type"] == "deterministic"
    actual_moments = get_moments(jnp.log(value))
    assert_allclose(actual_moments, expected_moments, atol=0.05, rtol=0.01)


@pytest.mark.parametrize("rho", [-0.7, 0.8])
def test_dense_mass(rho):
    true_cov = jnp.array([[10.0, rho], [rho, 0.1]])

    def model():
        numpyro.sample(
            "x", dist.MultivariateNormal(jnp.zeros(2), covariance_matrix=true_cov)
        )

    ns = NestedSampler(model, num_live_points=10, max_samples=1000)
    ns.run(random.PRNGKey(0))

    samples = ns.get_samples(random.PRNGKey(1), 1000)["x"]
    assert_allclose(jnp.mean(samples[:, 0]), jnp.array(0.0), atol=0.50)
    assert_allclose(jnp.mean(samples[:, 1]), jnp.array(0.0), atol=0.05)
    assert_allclose(jnp.mean(samples[:, 0] * samples[:, 1]), jnp.array(rho), atol=0.20)
    assert_allclose(jnp.var(samples, axis=0), jnp.array([10.0, 0.1]), rtol=0.20)
