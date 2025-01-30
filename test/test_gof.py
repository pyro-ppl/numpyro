# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from test_distributions import CONTINUOUS, DIRECTIONAL

import jax.random as random

import numpyro.distributions as dist
from numpyro.distributions.gof import InvalidTest, auto_goodness_of_fit

TEST_FAILURE_RATE = 2e-5  # For all goodness-of-fit tests.


@pytest.mark.parametrize("jax_dist, sp_dist, params", CONTINUOUS + DIRECTIONAL)
def test_gof(jax_dist, sp_dist, params):
    if "Improper" in jax_dist.__name__:
        pytest.skip("distribution has improper .log_prob()")
    if "LKJ" in jax_dist.__name__ or "Wishart" in jax_dist.__name__:
        pytest.xfail("incorrect submanifold scaling")
    if jax_dist is dist.EulerMaruyama:
        d = jax_dist(*params)
        if d.event_dim > 1:
            pytest.skip("EulerMaruyama skip test when event shape is non-trivial.")
    if jax_dist is dist.ZeroSumNormal:
        pytest.skip("skip gof test for ZeroSumNormal")
    if jax_dist is dist.MatrixNormal:
        pytest.skip(
            "skip gof test for MatrixNormal, likely incorrect submanifold scaling"
        )

    num_samples = 10000
    if any(
        name in jax_dist.__name__
        for name in ["BetaProportion", "SineBivariateVonMises"]
    ):
        num_samples = 20000
    rng_key = random.PRNGKey(0)
    d = jax_dist(*params)
    samples = d.sample(key=rng_key, sample_shape=(num_samples,))
    probs = np.exp(d.log_prob(samples))

    dim = None
    if jax_dist is dist.ProjectedNormal:
        dim = samples.shape[-1] - 1

    # Test each batch independently.
    probs = probs.reshape(num_samples, -1)
    samples = samples.reshape(probs.shape + d.event_shape)
    if "Dirichlet" in jax_dist.__name__:
        # The Dirichlet density is over all but one of the probs.
        samples = samples[..., :-1]
    for b in range(probs.shape[1]):
        try:
            gof = auto_goodness_of_fit(samples[:, b], probs[:, b], dim=dim)
        except InvalidTest:
            pytest.skip("expensive test")
        else:
            assert gof > TEST_FAILURE_RATE
