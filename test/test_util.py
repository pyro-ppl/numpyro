import logging
from collections import namedtuple

import jax.numpy as np
import numpy as onp
import pytest
import scipy.special as sp
from jax import device_put, grad, jit, lax, partial, tree_map
from numpy.testing import assert_allclose

from numpyro.distributions.util import xlogy, xlog1py
from numpyro.util import dual_averaging, velocity_verlet, welford_covariance

logger = logging.getLogger(__name__)
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
     np.log(np.array([1, 2, 3])), np.array([1., 0.5, 1./3])),
    (np.array([1.]), np.array([1., 2., 3.]),
     np.sum(np.log(np.array([1, 2, 3]))), np.array([1., 0.5, 1./3])),
    (np.array([1., 2., 3.]), np.array([2.]),
     np.log(np.array([2., 2., 2.])), np.array([3.])),
    (np.array([0.]), np.array([0, 0]),
     np.array([-float('inf')]), np.array([0, 0])),
    (np.array([[0.], [0.]]), np.array([1., 2.]),
     np.array([[np.log(2.)], [np.log(2.)]]), np.array([0, 0])),
])
def test_xlogy_jac(x, y, grad1, grad2):
    assert_allclose(grad(lambda x, y: np.sum(xlogy(x, y)))(x, y), grad1)
    assert_allclose(grad(lambda x, y: np.sum(xlogy(x, y)), 1)(x, y), grad2)


@pytest.mark.parametrize('x, y', [
    (np.array([1]), np.array([0, 1, 2])),
    (np.array([0]), np.array([-1, -1])),
    (np.array([[0.], [0.]]), np.array([1., 2.])),
])
@pytest.mark.parametrize('jit_fn', [False, True])
def test_xlog1py(x, y, jit_fn):
    fn = xlog1py if not jit_fn else jit(xlog1py)
    assert_allclose(fn(x, y), sp.xlog1py(x, y))


@pytest.mark.parametrize('x, y, grad1, grad2', [
    (np.array([1., 1., 1.]), np.array([0., 1., 2.]),
     np.log(np.array([1, 2, 3])), np.array([1., 0.5, 1./3])),
    (np.array([1., 1., 1.]), np.array([-1., 0., 1.]),
     np.log(np.array([0, 1, 2])), np.array([float('inf'), 1., 0.5])),
    (np.array([1.]), np.array([0., 1., 2.]),
     np.sum(np.log(np.array([1, 2, 3]))), np.array([1., 0.5, 1./3])),
    (np.array([1., 2., 3.]), np.array([1.]),
     np.log(np.array([2., 2., 2.])), np.array([3.])),
    (np.array([0.]), np.array([-1, -1]),
     np.array([-float('inf')]), np.array([0, 0])),
    (np.array([[0.], [0.]]), np.array([1., 2.]),
     np.array([[np.log(6.)], [np.log(6.)]]), np.array([0, 0])),
])
def test_xlog1py_jac(x, y, grad1, grad2):
    assert_allclose(grad(lambda x, y: np.sum(xlog1py(x, y)))(x, y), grad1)
    assert_allclose(grad(lambda x, y: np.sum(xlog1py(x, y)), 1)(x, y), grad2)


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

    assert_allclose(x_opt, -1., atol=1e-3)


@pytest.mark.parametrize('jitted', [True, False])
@pytest.mark.parametrize('diagonal', [True, False])
@pytest.mark.parametrize('regularize', [True, False])
def test_welford_covariance(jitted, diagonal, regularize):
    onp.random.seed(0)
    loc = onp.random.randn(3)
    a = onp.random.randn(3, 3)
    target_cov = onp.matmul(a, a.T)
    x = onp.random.multivariate_normal(loc, target_cov, size=(2000,))
    x = device_put(x)

    def get_cov(x):
        wc_init, wc_update, wc_final = welford_covariance(diagonal=diagonal)
        wc_state = wc_init(3)
        wc_state = lax.fori_loop(0, 2000, lambda i, val: wc_update(x[i], val), wc_state)
        cov = wc_final(wc_state, regularize=regularize)
        return cov

    if jitted:
        cov = jit(get_cov)(x)
    else:
        cov = get_cov(x)

    if diagonal:
        assert_allclose(cov, np.diagonal(target_cov), rtol=0.06)
    else:
        assert_allclose(cov, target_cov, rtol=0.06)


########################################
# verlocity_verlet Test
########################################

TEST_EXAMPLES = []
EXAMPLE_IDS = []

ModelArgs = namedtuple('model_args', ['step_size', 'num_steps', 'q_i', 'p_i', 'q_f', 'p_f', 'prec'])
Example = namedtuple('test_case', ['model', 'args'])


def register_model(init_args):
    """
    Register the model along with each of the model arguments
    as test examples.
    """
    def register_fn(model):
        for args in init_args:
            test_example = Example(model, args)
            TEST_EXAMPLES.append(test_example)
            EXAMPLE_IDS.append(model.__name__)
    return register_fn


@register_model([
    ModelArgs(
        step_size=0.01,
        num_steps=100,
        q_i={'x': 0.0},
        p_i={'x': 1.0},
        q_f={'x': np.sin(1.0)},
        p_f={'x': np.cos(1.0)},
        prec=1e-4
    )
])
class HarmonicOscillator(object):
    @staticmethod
    def kinetic_fn(p):
        return 0.5 * p['x'] ** 2

    @staticmethod
    def potential_fn(q):
        return 0.5 * q['x'] ** 2


@register_model([
    ModelArgs(
        step_size=0.01,
        num_steps=628,
        q_i={'x': 1.0, 'y': 0.0},
        p_i={'x': 0.0, 'y': 1.0},
        q_f={'x': 1.0, 'y': 0.0},
        p_f={'x': 0.0, 'y': 1.0},
        prec=5.0e-3
    )
])
class CircularPlanetaryMotion(object):
    @staticmethod
    def kinetic_fn(p):
        return 0.5 * p['x'] ** 2 + 0.5 * p['y'] ** 2

    @staticmethod
    def potential_fn(q):
        return - 1.0 / np.power(q['x'] ** 2 + q['y'] ** 2, 0.5)


@register_model([
    ModelArgs(
        step_size=0.1,
        num_steps=1810,
        q_i={'x': 0.02},
        p_i={'x': 0.0},
        q_f={'x': -0.02},
        p_f={'x': 0.0},
        prec=1.0e-4
    )
])
class QuarticOscillator(object):
    @staticmethod
    def kinetic_fn(p):
        return 0.5 * p['x'] ** 2

    @staticmethod
    def potential_fn(q):
        return 0.25 * np.power(q['x'], 4.0)


@pytest.mark.parametrize('jitted', [True, False])
@pytest.mark.parametrize('example', TEST_EXAMPLES, ids=EXAMPLE_IDS)
def test_velocity_verlet(jitted, example):
    def get_final_state(model, step_size, num_steps, q_i, p_i):
        vv_init, vv_update = velocity_verlet(model.potential_fn, model.kinetic_fn)
        vv_state = vv_init(q_i, p_i)
        q_f, p_f, _, _ = lax.fori_loop(0, num_steps,
                                       lambda i, val: vv_update(step_size, val),
                                       vv_state)
        return (q_f, p_f)

    model, args = example
    if jitted:
        q_f, p_f = jit(get_final_state, static_argnums=(0,))(
            model, args.step_size, args.num_steps, args.q_i, args.p_i)
    else:
        q_f, p_f = get_final_state(model, args.step_size, args.num_steps, args.q_i, args.p_i)

    logger.info('Test trajectory:')
    logger.info('initial q: {}'.format(args.q_i))
    logger.info('final q: {}'.format(q_f))
    for node in args.q_f:
        assert_allclose(q_f[node], args.q_f[node], atol=args.prec)
        assert_allclose(p_f[node], args.p_f[node], atol=args.prec)

    logger.info('Test energy conservation:')
    energy_initial = model.kinetic_fn(args.p_i) + model.potential_fn(args.q_i)
    energy_final = model.kinetic_fn(p_f) + model.potential_fn(q_f)
    logger.info('initial energy: {}'.format(energy_initial))
    logger.info('final energy: {}'.format(energy_final))
    assert_allclose(energy_initial, energy_final, atol=1e-5)

    logger.info('Test time reversibility:')
    p_reverse = tree_map(lambda x: -x, p_f)
    q_i, p_i = get_final_state(model, args.step_size, args.num_steps, q_f, p_reverse)
    for node in args.q_i:
        assert_allclose(q_i[node], args.q_i[node], atol=1e-5)
