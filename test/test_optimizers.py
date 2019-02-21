import jax.numpy as np
from jax import grad, jit
from jax.experimental import optimizers


@jit
def loss(params):
    return np.sum(params['x'] ** 2 + params['y'] ** 2)


def step(i, opt_state, opt_update):
    params = optimizers.get_params(opt_state)
    g = grad(loss)(params)
    return opt_update(i, g, opt_state)


def test_optim_multi_params():
    params = {'x': np.array([1., 1., 1.]), 'y': np.array([-1, -1., -1.])}
    opt_init, opt_update = optimizers.adam(step_size=1e-2)
    opt_state = opt_init(params)
    for i in range(1000):
        opt_state = step(i, opt_state, opt_update)
    for _, param in optimizers.get_params(opt_state).items():
        assert np.allclose(param, np.zeros(3))
