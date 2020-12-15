from collections import namedtuple

import numpy as np
from numpy.testing import assert_allclose
import pytest

import jax.numpy as jnp

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
from numpyro.infer.einstein.utils import posdef, sqrth, sqrth_and_inv_sqrth

T = namedtuple('TestSteinKernel', ['kernel', 'particle_info', 'loss_fn', 'kval'])

PARTICLES_2D = jnp.array([[1., 2.], [-10., 10.], [0., 0.], [2., -1]])

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
      lambda x: x[0] ** 4 - x[1] ** 3 / 2,
      {'matrix': jnp.array([[5.608312e-09, 0.],
                            [0., 9.347186e-05]])}
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


@pytest.mark.parametrize('batch_shape', [(), (2,), (3, 1)])
def test_posdef(batch_shape):
    dim = 4
    x = np.random.normal(size=batch_shape + (dim, dim + 1))
    m = x @ np.swapaxes(x, -2, -1)
    assert_allclose(posdef(m), m, rtol=1e-5)


@pytest.mark.parametrize('batch_shape', [(), (2,), (3, 1)])
def test_sqrth(batch_shape):
    dim = 4
    x = np.random.normal(size=batch_shape + (dim, dim + 1))
    m = x @ np.swapaxes(x, -2, -1)
    s = sqrth(m)
    assert_allclose(s @ np.swapaxes(s, -2, -1), m, rtol=1e-5)


@pytest.mark.parametrize('batch_shape', [(), (2,), (3, 1)])
def test_sqrth_and_inv_sqrth(batch_shape):
    dim = 4
    x = np.random.normal(size=batch_shape + (dim, dim + 1))
    m = x @ np.swapaxes(x, -2, -1)
    s, i, si = sqrth_and_inv_sqrth(m)
    assert_allclose(s @ np.swapaxes(s, -2, -1), m, rtol=1e-5)
    assert_allclose(i, np.linalg.inv(m), rtol=1e-5)
    assert_allclose(si @ np.swapaxes(si, -2, -1), i, rtol=1e-5)
