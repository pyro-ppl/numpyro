import jax.numpy as np
import pytest
from jax import grad, jit

from numpyro.util import dual_averaging


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
