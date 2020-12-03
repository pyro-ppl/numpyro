from collections import namedtuple

import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose
from numpyro.util import ravel_pytree

from numpyro.infer.einstein.kernels import (
    RBFKernel,
    RandomFeatureKernel,
    GraphicalKernel,
    IMQKernel,
    LinearKernel,
    MixtureKernel,
    HessianPrecondMatrix,
    PrecondMatrixKernel
)

T = namedtuple('TestSteinKernel', ['kernel', 'particle_info', 'loss_fn', 'kval'])

PARTICLES_2D = jnp.array([[1., 2.,], [-10., 10.], [0., 0.], [2., -1]])

TPARTICLES_2D = (jnp.array([1., 2.]), jnp.array([10., 5.]))  # transformed particles

TEST_CASES = [
    T(RBFKernel,
      lambda d: {},
      lambda x: x,
      {'norm': 3.8147664e-06,
       'vector': jnp.array([0., 0.2500005]),
       'matrix': jnp.array([[3.8147664e-06, 0.],
                            [0., 3.8147664e-06]])}
      ),
    T(RandomFeatureKernel,
      lambda d: {},
      lambda x: x,
      {'norm': -4.566867}),
    T(IMQKernel,
      lambda d: {},
      lambda x: x,
      {'norm': .104828484,
       'vector': jnp.array([0.11043153, 0.31622776])}
      ),
    T(LinearKernel,
      lambda d: {},
      lambda x: x,
      {'norm': 21.}
      ),
    T(lambda mode: MixtureKernel(mode=mode, ws=jnp.array([.2, .8]), kernel_fns=[RBFKernel(mode), RBFKernel(mode)]),
      lambda d: {},
      lambda x: x,
      {'matrix': jnp.array([[3.8147664e-06, 0.],
                            [0., 3.8147664e-06]])}
      ),
    T(lambda mode: GraphicalKernel(mode=mode, local_kernel_fns={'p1': RBFKernel('norm')}),
      lambda d: {'p1': (0, d)},
      lambda x: x,
      {'matrix': jnp.array([[3.8147664e-06, 0.],
                            [0., 3.8147664e-06]])}
      ),
    T(lambda mode: PrecondMatrixKernel(HessianPrecondMatrix(), RBFKernel(mode='matrix')),
      lambda d: {},
      lambda x: x**2 + x**3,
      {'matrix': jnp.array([[3.8147664e-06, 0.],
                            [0., 3.8147664e-06]])}
      )
]

PARTICLES = [(PARTICLES_2D, TPARTICLES_2D)]

TEST_IDS = [t[0].__class__.__name__ for t in TEST_CASES]


@pytest.mark.parametrize('kernel, particle_info, loss_fn, kval', TEST_CASES, ids=TEST_IDS)
@pytest.mark.parametrize('particles, tparticles', PARTICLES)
@pytest.mark.parametrize('mode', ['norm', 'vector', 'matrix'])
def test_kernel_forward(kernel, particles, particle_info, loss_fn, tparticles, mode, kval):
    if mode not in kval:
        return
    d, = tparticles[0].shape
    kernel_fn = kernel(mode=mode).compute(particles, particle_info(d), loss_fn)
    value = kernel_fn(*tparticles)

    assert_allclose(value, kval[mode])
