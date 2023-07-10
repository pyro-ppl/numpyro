# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from copy import copy

import numpy as np
from numpy.testing import assert_allclose
import pytest

from jax import numpy as jnp, random

from numpyro.contrib.einstein import SteinVI
from numpyro.contrib.einstein.stein_kernels import (
    GraphicalKernel,
    IMQKernel,
    LinearKernel,
    MixtureKernel,
    RandomFeatureKernel,
    RBFKernel,
)
from numpyro.optim import Adam

T = namedtuple("TestSteinKernel", ["kernel", "particle_info", "loss_fn", "kval"])

PARTICLES_2D = np.array([[1.0, 2.0], [-10.0, 10.0], [7.0, 3.0], [2.0, -1]])

TPARTICLES_2D = (np.array([1.0, 2.0]), np.array([10.0, 5.0]))  # transformed particles
TEST_CASES = [
    T(
        RBFKernel,
        lambda d: {},
        lambda x: x,
        {
            "norm": 0.040711474,
            "vector": np.array([0.056071877, 0.7260586]),
            "matrix": np.array([[0.040711474, 0.0], [0.0, 0.040711474]]),
        },
    ),
    T(RandomFeatureKernel, lambda d: {}, lambda x: x, {"norm": 15.173317}),
    T(
        IMQKernel,
        lambda d: {},
        lambda x: x,
        {"norm": 0.104828484, "vector": np.array([0.11043153, 0.31622776])},
    ),
    T(LinearKernel, lambda d: {}, lambda x: x, {"norm": 21.0}),
    T(
        lambda mode: MixtureKernel(
            mode=mode,
            ws=np.array([0.2, 0.8]),
            kernel_fns=[RBFKernel(mode), RBFKernel(mode)],
        ),
        lambda d: {},
        lambda x: x,
        {"matrix": np.array([[0.040711474, 0.0], [0.0, 0.040711474]])},
    ),
    T(
        lambda mode: GraphicalKernel(
            mode=mode, local_kernel_fns={"p1": RBFKernel("norm")}
        ),
        lambda d: {"p1": (0, d)},
        lambda x: x,
        {"matrix": np.array([[0.040711474, 0.0], [0.0, 0.040711474]])},
    ),
]

PARTICLES = [(PARTICLES_2D, TPARTICLES_2D)]

TEST_IDS = [t[0].__class__.__name__ for t in TEST_CASES]


@pytest.mark.parametrize(
    "kernel, particle_info, loss_fn, kval", TEST_CASES, ids=TEST_IDS
)
@pytest.mark.parametrize("particles, tparticles", PARTICLES)
@pytest.mark.parametrize("mode", ["norm", "vector", "matrix"])
def test_kernel_forward(
    kernel, particles, particle_info, loss_fn, tparticles, mode, kval
):
    if mode not in kval:
        return
    (d,) = tparticles[0].shape
    kernel = kernel(mode=mode)
    kernel.init(random.PRNGKey(0), particles.shape)
    kernel_fn = kernel.compute(particles, particle_info(d), loss_fn)
    value = kernel_fn(*tparticles)
    assert_allclose(value, jnp.array(kval[mode]), atol=1e-6)


@pytest.mark.parametrize(
    "kernel, particle_info, loss_fn, kval", TEST_CASES, ids=TEST_IDS
)
@pytest.mark.parametrize("mode", ["norm", "vector", "matrix"])
@pytest.mark.parametrize("particles, tparticles", PARTICLES)
def test_apply_kernel(
    kernel, particles, particle_info, loss_fn, tparticles, mode, kval
):
    if mode not in kval:
        pytest.skip()
    (d,) = tparticles[0].shape
    kernel_fn = kernel(mode=mode)
    kernel_fn.init(random.PRNGKey(0), particles.shape)
    kernel_fn = kernel_fn.compute(particles, particle_info(d), loss_fn)
    v = np.ones_like(kval[mode])
    stein = SteinVI(id, id, Adam(1.0), kernel(mode))
    value = stein._apply_kernel(kernel_fn, *tparticles, v)
    kval_ = copy(kval)
    if mode == "matrix":
        kval_[mode] = np.dot(kval_[mode], v)
    assert_allclose(value, kval_[mode], atol=1e-6)
