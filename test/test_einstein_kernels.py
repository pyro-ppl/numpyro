from collections import namedtuple

import jax.numpy as jnp

import pytest
from numpyro.infer.einstein.kernels import (
    RBFKernel,
    RandomFeatureKernel,
    GraphicalKernel,
    IMQKernel,
    LinearKernel,
    MixtureKernel)

T = namedtuple("TestSteinKernel", ["kernel", "particles", "particle_info", 'loss_fn'])

TEST_CASES = []

TEST_IDS = [t[0].__class__.__name__ for t in TEST_CASES]


@pytest.mark.parametrize("kernel, particles, particle_info, loss_fn", TEST_CASES, ids=TEST_IDS)
def test_kernel_forward(kernel, particles):
    pass
