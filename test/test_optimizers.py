import pytest

from jax import grad, jit, partial
import jax.numpy as np

from numpyro.optim import *


def loss(params):
    return np.sum(params['x'] ** 2 + params['y'] ** 2)


@partial(jit, static_argnums=(1,))
def step(opt_state, optim):
    params = optim.get_params(opt_state)
    g = grad(loss)(params)
    return optim.update(g, opt_state)


@pytest.mark.parametrize('optim_class, args', [
    (Adam, (1e-2,)),
    (Adagrad, (1e-1,)),
    (Momentum, (1e-2, 0.5,)),
    (RMSProp, (1e-2, 0.95)),
    (RMSPropMomentum, (1e-4,)),
    (SGD, (1e-2,))
])
def test_optim_multi_params(optim_class, args):
    params = {'x': np.array([1., 1., 1.]), 'y': np.array([-1, -1., -1.])}
    optim = optim_class(*args)
    opt_state = optim.init(params)
    for i in range(2000):
        opt_state = step(opt_state, optim)
    for _, param in optim.get_params(opt_state).items():
        assert np.allclose(param, np.zeros(3))
