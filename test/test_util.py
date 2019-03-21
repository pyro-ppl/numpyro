import logging
from collections import namedtuple

import jax.numpy as np
import numpy as onp
import pytest
import scipy.special as sp
from jax import device_put, grad, jit, lax, partial, random, tree_map
from numpy.testing import assert_allclose

from numpyro.distributions.util import xlogy, xlog1py
from numpyro.util import (adapt_window, build_adaptation_schedule, build_tree,
                          dual_averaging, find_reasonable_step_size,
                          velocity_verlet, warmup_adapter, welford_covariance)

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
    fn = jit(optimize, static_argnums=(0,)) if jitted else optimize
    x_opt = fn(f)

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

    fn = jit(get_cov) if jitted else get_cov
    cov = fn(x)

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
    fn = jit(get_final_state, static_argnums=(0,)) if jitted else get_final_state
    q_f, p_f = fn(model, args.step_size, args.num_steps, args.q_i, args.p_i)

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
        assert_allclose(q_i[node], args.q_i[node], atol=1e-4)


@pytest.mark.parametrize('jitted', [True, False])
@pytest.mark.parametrize('init_step_size', [0.1, 10.0])
def test_find_reasonable_step_size(jitted, init_step_size):
    def kinetic_fn(p):
        return 0.5 * p ** 2

    def potential_fn(q):
        return 0.5 * q ** 2

    p_generator = lambda: 1.0  # noqa: E731
    q = 0.0

    fn = (jit(find_reasonable_step_size, static_argnums=(0, 1, 2))
          if jitted else find_reasonable_step_size)
    step_size = fn(potential_fn, kinetic_fn, p_generator, q, init_step_size)

    # Apply 1 velocity verlet step with step_size=eps, we have
    # z_new = eps, r_new = 1 - eps^2 / 2, hence energy_new = 0.5 + eps^4 / 8,
    # hence delta_energy = energy_new - energy_init = eps^4 / 8.
    # We want to find a reasonable step_size such that delta_energy ~ -log(0.8),
    # hence that step_size ~ the following threshold
    threshold = np.power(-np.log(0.8) * 8, 0.25)

    # Confirm that given init_step_size, we will doubly increase/decrease it
    # until it passes threshold.
    if init_step_size < threshold:
        assert step_size / 2 < threshold
        assert step_size > threshold
    else:
        assert step_size * 2 > threshold
        assert step_size < threshold


@pytest.mark.parametrize("num_steps, expected", [
    (18, [(0, 17)]),
    (50, [(0, 6), (7, 44), (45, 49)]),
    (100, [(0, 14), (15, 89), (90, 99)]),
    (150, [(0, 74), (75, 99), (100, 149)]),
    (200, [(0, 74), (75, 99), (100, 149), (150, 199)]),
    (280, [(0, 74), (75, 99), (100, 229), (230, 279)]),
])
def test_build_adaptation_schedule(num_steps, expected):
    adaptation_schedule = build_adaptation_schedule(num_steps)
    expected_schedule = [adapt_window(i, j) for i, j in expected]
    assert adaptation_schedule == expected_schedule


@pytest.mark.parametrize('jitted', [
    pytest.param(True, marks=pytest.mark.xfail(
        reason="lax.cond issue: https://github.com/google/jax/issues/514")),
    False])
def test_warmup_adapter(jitted):
    num_steps = 150
    adaptation_schedule = build_adaptation_schedule(num_steps)
    find_reasonable_step_size = lambda x: np.where(x < 1, x * 4, x / 4)  # noqa: E731
    init_step_size = 1.
    mass_matrix_size = 3

    wa_init, wa_update = warmup_adapter(num_steps, find_reasonable_step_size)
    wa_update = jit(wa_update) if jitted else wa_update

    wa_state = wa_init(init_step_size, mass_matrix_size=mass_matrix_size)
    step_size, inverse_mass_matrix, _, _, window_idx = wa_state
    assert step_size == find_reasonable_step_size(init_step_size)
    assert_allclose(inverse_mass_matrix, np.ones(mass_matrix_size))
    assert window_idx == 0

    window = adaptation_schedule[0]
    for t in range(window.start, window.end + 1):
        wa_state = wa_update(t, 0.7 + 0.1 * t / (window.end - window.start),
                             np.ones(3), wa_state)
    last_step_size = step_size
    step_size, inverse_mass_matrix, _, _, window_idx = wa_state
    assert window_idx == 1
    # step_size is decreased because accept_prob < target_accept_prob
    assert step_size < last_step_size
    # inverse_mass_matrix does not change at the end of the first window
    assert_allclose(inverse_mass_matrix, np.ones(mass_matrix_size))

    window = adaptation_schedule[1]
    window_len = window.end - window.start
    for t in range(window.start, window.end + 1):
        wa_state = wa_update(t, 0.8 + 0.1 * (t - window.start) / window_len,
                             2 * np.ones(3), wa_state)
    last_step_size = step_size
    step_size, inverse_mass_matrix, _, _, window_idx = wa_state
    assert window_idx == 2
    # step_size is increased because accept_prob > target_accept_prob
    assert step_size > last_step_size
    # Verifies that inverse_mass_matrix changes at the end of the second window.
    # Because z_flat is constant during the second window, covariance will be 0
    # and only regularize_term of welford scheme is involved.
    # This also verifies that z_flat terms in the first window does not affect
    # the second window.
    welford_regularize_term = 1e-3 * (5 / (window.end + 1 - window.start + 5))
    assert_allclose(inverse_mass_matrix,
                    np.full((mass_matrix_size,), welford_regularize_term))

    window = adaptation_schedule[2]
    for t in range(window.start, window.end + 1):
        wa_state = wa_update(t, 0.8, t * np.ones(3), wa_state)
    last_step_size = step_size
    step_size, final_inverse_mass_matrix, _, _, window_idx = wa_state
    assert window_idx == 3
    # during the last window, because target_accept_prob=0.8,
    # log_step_size will be equal to the constant prox_center=log(10*last_step_size)
    assert_allclose(step_size, last_step_size * 10)
    # Verifies that inverse_mass_matrix does not change during the last window
    # despite z_flat changes w.r.t time t,
    assert_allclose(final_inverse_mass_matrix, inverse_mass_matrix)


@pytest.mark.parametrize('step_size', [0.01, 1., 100.])
@pytest.mark.parametrize('use_multinomial_sampling', [True, False])
def test_build_tree(step_size, use_multinomial_sampling):
    def kinetic_fn(p):
        return 0.5 * p ** 2

    def potential_fn(q):
        return 0.5 * q ** 2

    vv_init, vv_update = velocity_verlet(potential_fn, kinetic_fn)
    vv_state = vv_init(0.0, 1.0)
    inverse_mass_matrix = np.array([1.])
    rng = random.PRNGKey(0)

    tree = build_tree(vv_update, kinetic_fn, vv_state, inverse_mass_matrix,
                      step_size, rng, use_multinomial_sampling=use_multinomial_sampling)

    assert tree.num_proposals >= 2 ** (tree.depth - 1)

    assert tree.sum_accept_probs <= tree.num_proposals

    if tree.depth < 10:
        assert tree.turning | tree.diverging

    # for large step_size, assert that diverging will happen in 1 step
    if step_size > 10:
        assert tree.diverging
        assert tree.num_proposals == 1

    # for small step_size, assert that it should take a while to meet the terminate condition
    if step_size < 0.1:
        assert tree.num_proposals > 10
