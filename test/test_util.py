import jax.numpy as np
import pytest
import scipy.special as sp
from jax import jit, lax, partial, grad
from numpy.testing import assert_allclose

from numpyro.distributions.util import xlogy
from numpyro.util import dual_averaging

_zeros = partial(lax.full_like, fill_value=0)


@pytest.mark.parametrize('x, y', [
    (np.array([1]), np.array([1, 2, 3])),
    (np.array([0]), np.array([0, 0])),
    (np.array([[0.], [0.]]), np.array([1., 2.])),
])
@pytest.mark.parametrize('jit_fn', [False, True])
def test_xlogy(x, y, jit_fn):
    fn = xlogy if not jit_fn else jit(xlogy)
    assert np.allclose(fn(x, y), sp.xlogy(x, y))


@pytest.mark.parametrize('x, y, grad1, grad2', [
    (np.array([1., 1., 1.]), np.array([1., 2., 3.]),
     np.log(np.array([1, 2, 3])), np.array([1. / 1., 1. / 2., 1. / 3.])),
    (np.array([1.]), np.array([1., 2., 3.]),
     np.sum(np.log(np.array([1, 2, 3]))), np.array([1. / 1., 1. / 2., 1. / 3.])),
    (np.array([1., 2., 3.]), np.array([2.]),
     np.log(np.array([2., 2., 2.])), np.array([3.])),
    (np.array([0.]), np.array([0, 0]),
     np.array([-float('inf')]), np.array([0, 0])),
    (np.array([[0], [0]]), np.array([1., 2.]),
     np.array([[0], [0]]), np.array([0, 0])),
])
def test_xlogy_jac(x, y, grad1, grad2):
    assert_allclose(jit(grad(lambda x, y: np.sum(xlogy(x, y))))(x, y), grad1)
    assert_allclose(jit(grad(lambda x, y: np.sum(xlogy(x, y)), 1))(x, y), grad2)


@pytest.mark.parametrize('jitted', [True, False])
def test_dual_averaging(jitted):
    def optimize(f):
        da_init, da_update = dual_averaging(gamma=0.5)
        da_state = da_init()
        for i in range(10):
            x = da_state[0]
            g = grad(f)(x)
            da_state = da_update(g, da_state)
        x_avg = da_state[1]
        return x_avg

    f = lambda x: (x + 1) ** 2  # noqa: E731
    if jitted:
        x_opt = jit(optimize, static_argnums=(0,))(f)
    else:
        x_opt = optimize(f)

    assert np.allclose(x_opt, -1., atol=1e-3)
