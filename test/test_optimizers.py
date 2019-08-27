import pytest

from jax import grad, jit, partial
import jax.numpy as np

from numpyro import optim


def loss(params):
    return np.sum(params['x'] ** 2 + params['y'] ** 2)


@partial(jit, static_argnums=(1,))
def step(opt_state, optim):
    params = optim.get_params(opt_state)
    g = grad(loss)(params)
    return optim.update(g, opt_state)


@pytest.mark.parametrize('optim_class, args', [
    (optim.Adam, (1e-2,)),
    (optim.Adagrad, (1e-1,)),
    (optim.Momentum, (1e-2, 0.5,)),
    (optim.RMSProp, (1e-2, 0.95)),
    (optim.RMSPropMomentum, (1e-4,)),
    (optim.SGD, (1e-2,))
])
def test_optim_multi_params(optim_class, args):
    params = {'x': np.array([1., 1., 1.]), 'y': np.array([-1, -1., -1.])}
    opt = optim_class(*args)
    opt_state = opt.init(params)
    for i in range(2000):
        opt_state = step(opt_state, opt)
    for _, param in opt.get_params(opt_state).items():
        assert np.allclose(param, np.zeros(3))
