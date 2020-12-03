from collections import namedtuple

import jax.numpy as jnp
import pytest

from numpyro.infer.einstein.kernels import (
    RBFKernel,
    RandomFeatureKernel,
    GraphicalKernel,
    IMQKernel,
    LinearKernel,
    MixtureKernel,
)

T = namedtuple('TestSteinKernel', ['kernel', 'particle_info', 'loss_fn', 'kernel_modes'])

PARTICLES_1D = jnp.array([[1.], [10.], [-1.], [4.]])
PARTICLES_2D = jnp.array([[1., 2.], [-10., 10.], [0., 0.], [2., -1]])
PARTICLES_10D = jnp.array([jnp.arange(10) + 5., jnp.arange(10) * 2., (jnp.arange(10) - 3.) % 5, jnp.arange(10) + 4.])

TPARTICLES_1D = (jnp.array([1.]), jnp.array([10.]))  # transformed particles
TPARTICLES_2D = (jnp.array([1., 2.]), jnp.array([10., 5.]))  # transformed particles
TPARTICLES_10D = (jnp.arange(10) + .5, (jnp.arange(10) + 3.) / 7.)  # transformed particles

TEST_CASES = [
    T(RBFKernel,
      lambda d: {},
      lambda x: x,
      ('norm', 'vector', 'matrix')
      ),
    T(RandomFeatureKernel,
      lambda d: {},
      lambda x: x,
      ('norm',)),
    T(IMQKernel,
      lambda d: {},
      lambda x: x,
      ('norm', 'vector')
      ),
    T(LinearKernel,
      lambda d: {},
      lambda x: x,
      ('norm',)
      ),
    T(lambda mode: MixtureKernel(mode=mode, ws=jnp.array([.2, .8]), kernel_fns=[RBFKernel(mode), RBFKernel(mode)]),
      lambda d: {},
      lambda x: x,
      ('matrix',)
      ),
    T(lambda mode: GraphicalKernel(mode=mode, local_kernel_fns={'p1': RBFKernel('norm')}),
      lambda d: {'p1': (0, d)},
      lambda x: x,
      ('matrix',)
      )
]

PARTICLES = [(PARTICLES_1D, TPARTICLES_1D), (PARTICLES_2D, TPARTICLES_2D), (PARTICLES_10D, TPARTICLES_10D)]

TEST_IDS = [t[0].__class__.__name__ for t in TEST_CASES]


@pytest.mark.parametrize('kernel, particle_info, loss_fn, kernel_modes', TEST_CASES, ids=TEST_IDS)
@pytest.mark.parametrize('particles, tparticles', PARTICLES)
@pytest.mark.parametrize('mode', ['norm', 'vector', 'matrix'])
def test_kernel_forward(kernel, particles, particle_info, loss_fn, tparticles, mode, kernel_modes):
    if mode not in kernel_modes:
        return
    d, = tparticles[0].shape
    kernel_fn = kernel(mode=mode).compute(particles, particle_info(d), loss_fn)
    value = kernel_fn(*tparticles)

    if mode == 'norm':
        assert value.shape == ()
    elif mode == 'vector':
        assert value.shape == (d,)
    elif mode == 'matrix':
        assert value.shape == (d, d)
