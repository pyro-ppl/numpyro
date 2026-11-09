# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose
import pytest

from jax import random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import AffineTransform, ExpTransform

_jaxns_available = True
try:
    if os.environ.get("JAX_ENABLE_X64"):
        from numpyro.contrib.nested_sampling import NestedSampler, UniformReparam

except ImportError:
    _jaxns_available = False


pytestmark = [
    pytest.mark.filterwarnings("ignore:jax.tree_.+ is deprecated:FutureWarning"),
    pytest.mark.filterwarnings("ignore:JAX x64"),
    pytest.mark.skipif(
        not os.environ.get("JAX_ENABLE_X64") or not _jaxns_available,
        reason="test suite for jaxns requires double precision",
    ),
]


# Test helper to extract a few central moments from samples.
def get_moments(x):
    m1 = jnp.mean(x, axis=0)
    x = x - m1
    xx = x * x
    xxx = x * xx
    xxxx = xx * xx
    m2 = jnp.mean(xx, axis=0)
    m3 = jnp.mean(xxx, axis=0) / m2**1.5
    m4 = jnp.mean(xxxx, axis=0) / m2**2
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
    num_samples = 1000000

    def model():
        fn = dist.TransformedDistribution(
            dist.Normal(jnp.zeros_like(loc), jnp.ones_like(scale)),
            [AffineTransform(loc, scale), ExpTransform()],
        ).expand(shape)
        if event_shape:
            fn = fn.to_event(len(event_shape)).expand_by([num_samples])
        with numpyro.plate_stack("plates", batch_shape):
            with numpyro.plate("particles", num_samples):
                return numpyro.sample("x", fn)

    with numpyro.handlers.trace() as tr:
        value = numpyro.handlers.seed(model, 0)()
    expected_moments = get_moments(jnp.log(value))

    with numpyro.handlers.trace() as tr:
        with numpyro.handlers.reparam(config={"x": UniformReparam()}):
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

    ns = NestedSampler(model, constructor_kwargs={"num_live_points": 200})
    ns.run(random.key(0))

    samples = ns.get_samples(random.key(1), 1000)["x"]
    assert_allclose(jnp.mean(samples[:, 0]), jnp.array(0.0), atol=0.5)
    assert_allclose(jnp.mean(samples[:, 1]), jnp.array(0.0), atol=0.1)
    assert_allclose(jnp.mean(samples[:, 0] * samples[:, 1]), jnp.array(rho), atol=0.20)
    assert_allclose(jnp.var(samples, axis=0), jnp.array([10.0, 0.1]), rtol=0.20)


def test_print_summary_weighted_samples(capsys):
    ns = NestedSampler(lambda: None)
    x = jnp.array([0.0, 10.0, 20.0, 1000.0])
    # Unequal weights distinguish the weighted mean (16) from the ordinary mean.
    # The outlier has zero weight and must not affect the summary.
    ns._results = SimpleNamespace(
        samples={"x": x, "y": jnp.stack((x, -x), axis=-1)},
        log_dp_mean=jnp.array([0.0, np.log(2.0), np.log(7.0), -np.inf]),
    )

    ns.print_summary()

    lines = capsys.readouterr().out.strip().splitlines()
    columns = lines[0].split()
    assert "n_eff" in columns and "r_hat" not in columns
    rows = {}
    for line in lines[1:]:
        name, *values = line.split()
        rows[name] = dict(zip(columns, map(float, values)))
    assert set(rows) == {"x", "y[0]", "y[1]"}
    for name, expected_mean in [("x", 16.0), ("y[0]", 16.0), ("y[1]", -16.0)]:
        assert rows[name]["mean"] == expected_mean
        # (1 + 2 + 7)**2 / (1**2 + 2**2 + 7**2), printed to two decimals.
        assert rows[name]["n_eff"] == 1.85


def test_print_summary_before_run():
    ns = NestedSampler(lambda: None)
    with pytest.raises(RuntimeError, match=r"NestedSampler\.run"):
        ns.print_summary()
