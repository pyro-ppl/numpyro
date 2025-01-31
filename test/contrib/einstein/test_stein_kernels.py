# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from copy import copy

import numpy as np
from numpy.testing import assert_allclose
import pytest

from jax import numpy as jnp, random

from numpyro import sample
from numpyro.contrib.einstein import SteinVI
from numpyro.contrib.einstein.stein_kernels import (
    GraphicalKernel,
    IMQKernel,
    LinearKernel,
    MixtureKernel,
    ProbabilityProductKernel,
    RadialGaussNewtonKernel,
    RandomFeatureKernel,
    RBFKernel,
)
from numpyro.distributions import Normal
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam

T = namedtuple(
    "TestSteinKernel", ["name", "kernel", "particle_info", "loss_fn", "kval"]
)

PARTICLES = np.array([[1.0, 2.0], [10.0, 5.0], [7.0, 3.0], [2.0, -1]])


def MOCK_MODEL():
    sample("x", Normal())


TEST_CASES = [
    T(
        "RBFKernel",
        RBFKernel,
        lambda d: {},
        lambda x: x,
        {
            # let
            #   m = 4
            #   median trick (x_i in PARTICLES)
            #   h = med( [||x_i-x_j||_2]_{i,j=(0,0)}^{(m,m)} )^2 / log(m) = 16.92703711264772
            #   x = (1, 2); y=(10,5)
            # in
            # k(x,y) = exp(-.5 * ||x-y||_2^2 / h) = 0.00490776
            "norm": 0.00490776,
            # let
            #   h = 16.92703711264772 (from norm case)
            #   x = (1, 2); y=(10,5)
            # in
            # k(x,y) = exp(-.5 * (x-y)^2 / h) = (0.00835209, 0.5876088)
            "vector": np.array([0.00835209, 0.5876088]),
            # I(n) is n by n identity matrix
            # let
            #   k_norm = 0.00490776  (from norm case)
            #   x = (1, 2); y=(10,5)
            # in
            # k(x,y) = k_norm * I
            "matrix": np.array([[0.00490776, 0.0], [0.0, 0.00490776]]),
        },
    ),
    T(
        "RandomFeatureKernel",
        RandomFeatureKernel,
        lambda d: {},
        lambda x: x,
        {"norm": 13.805723},
    ),
    T(
        "IMQKernel",
        IMQKernel,
        lambda d: {},
        lambda x: x,
        {
            # let
            #   x = (1,2); y=(10,5)
            #   b = -.5; c=1
            # in
            # k(x,y) = (c**2 + ||x-y||^2)^b = (1 + 90)^(-.5) = 0.10482848367219183
            "norm": 0.104828484,
            # let
            #   x = (1,2); y=(10,5)
            #   b = -.5; c=1
            # in
            # k(x,y) = (c**2 + (x-y)^2)^b = (1 + [81,9])^(-.5) = [0.11043153, 0.31622777]
            "vector": np.array([0.11043153, 0.31622776]),
        },
    ),
    T(
        "LinearKernel",
        LinearKernel,
        lambda d: {},
        lambda x: x,
        {
            # let
            #   x = (1,2); y=(10,5)
            # in
            # k(x,y) = (x^Ty + 1) = 20 + 1 = 21
            "norm": 21.0
        },
    ),
    T(
        "MixtureKernel",
        lambda mode: MixtureKernel(
            mode=mode,
            ws=np.array([0.2, 0.8]),
            kernel_fns=[RBFKernel(mode), RBFKernel(mode)],
        ),
        lambda d: {},
        lambda x: x,
        # simply .2rbf_matrix + .8 rbf_matrix = rbf_matrix
        {"matrix": np.array([[0.00490776, 0.0], [0.0, 0.00490776]])},
    ),
    T(
        "GraphicalKernel",
        lambda mode: GraphicalKernel(
            mode=mode, local_kernel_fns={"p1": RBFKernel("norm")}
        ),
        lambda d: {"p1": (0, d)},
        lambda x: x,
        {
            # let
            #   d = 2 => l = [0,1]
            #   x = (1,2); y=(10,5)
            #   x_0 = x_1 = x; y_0=y_1=y
            #   k_0(x_0,y_0) = k_1(x_1,y_1) = RBFKernel(norm)(x,y) = 0.00490776
            # in
            # k(x,y) = diag({k_l(x_l,y_l)}) = [[0.00490776, 0.0], [0.0, 0.00490776]]
            "matrix": np.array([[0.00490776, 0.0], [0.0, 0.00490776]])
        },
    ),
    T(
        "ProbibilityProductKernel",
        lambda mode: ProbabilityProductKernel(mode=mode, guide=AutoNormal(MOCK_MODEL)),
        lambda d: {"x_auto_loc": (0, 1), "x_auto_scale": (1, 2)},
        lambda x: x,
        # eq. 5 Probability Product Kernels
        # x := (loc_x, softplus-inv(std_x)); y =: (loc_y, softplus-inv(std_y))
        # let
        #   s+(z) = softplus(z) = log(exp(z)+1);
        #   x =(1,2); y=(10,5)
        # in
        # k(x,y) = exp(-.5((1/s+(2))^2 +
        #                  (10/s+(5))^2 -
        #                  (1/(s+(2)^2 + (10/s+(5))^2)) ** 2 / (1/s+(2)^2 + 1/s+(5)^2)))
        #        = 0.2544481
        {"norm": 0.2544481},
    ),
    T(
        "RadialGaussNewtonKernel",
        lambda mode: RadialGaussNewtonKernel(),
        lambda d: {},
        lambda key, particle, i: jnp.linalg.norm(particle),  # Mock ELBO
        # let
        #   J(z) = (2/sqrt(z.sum()))*z.T . z
        #   M = mean(map(J, particles)) = [[0.6612069 , 0.19051724],
        #                                  [0.19051724, 0.3387931 ]]
        #   diff = [1, 2] -  [10, 5] = [-9, -3]
        #   quad_form = diff.T . M . diff = 66.89482758620689
        # in
        # k(x,y) = exp(-1/(2*2) * quad_form) = 5.457407430444593e-08
        {"norm": 5.457407430444593e-08},
    ),
]


TEST_IDS = [t.name for t in TEST_CASES]


@pytest.mark.parametrize("mode", ["norm", "vector", "matrix"])
@pytest.mark.parametrize(
    "name, kernel, particle_info, loss_fn, kval", TEST_CASES, ids=TEST_IDS
)
def test_kernel_forward(name, kernel, particle_info, loss_fn, mode, kval):
    particles = PARTICLES
    if mode not in kval:
        pytest.skip()
    (d,) = particles[0].shape
    kernel = kernel(mode=mode)
    key1, key2 = random.split(random.PRNGKey(0))
    kernel.init(key1, particles.shape)
    kernel_fn = kernel.compute(key2, particles, particle_info(d), loss_fn)
    value = kernel_fn(particles[0], particles[1])
    assert_allclose(value, jnp.array(kval[mode]), atol=0.5)


@pytest.mark.parametrize(
    "name, kernel, particle_info, loss_fn, kval", TEST_CASES, ids=TEST_IDS
)
@pytest.mark.parametrize("mode", ["norm", "vector", "matrix"])
def test_apply_kernel(name, kernel, particle_info, loss_fn, mode, kval):
    particles = PARTICLES
    if mode not in kval:
        pytest.skip()
    (d,) = particles[0].shape
    kernel_fn = kernel(mode=mode)
    key1, key2 = random.split(random.PRNGKey(0))
    kernel_fn.init(key1, particles.shape)
    kernel_fn = kernel_fn.compute(key2, particles, particle_info(d), loss_fn)
    v = np.ones_like(kval[mode])
    stein = SteinVI(id, id, Adam(1.0), kernel(mode))
    value = stein._apply_kernel(kernel_fn, particles[0], particles[1], v)
    kval_ = copy(kval)
    if mode == "matrix":
        kval_[mode] = np.dot(kval_[mode], v)
    assert_allclose(value, kval_[mode], atol=0.5)
