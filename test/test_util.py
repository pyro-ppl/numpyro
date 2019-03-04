import jax.numpy as np
import numpy as onp
import pytest
from jax import grad, jit

from numpyro.util import dual_averaging, welford_covariance


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


@pytest.mark.parametrize('jitted', [True, False])
@pytest.mark.parametrize('diagonal', [True, False])
@pytest.mark.parametrize('regularize', [True, False])
def test_welford_covariance(jitted, diagonal, regularize):
    onp.random.seed(0)
    loc = onp.random.randn(3)
    a = onp.random.randn(3, 3)
    target_cov = onp.matmul(a, a.T)
    x = onp.random.multivariate_normal(loc, target_cov, size=(2000,))

    wc_init, wc_update, wc_final = welford_covariance(diagonal=diagonal)
    if jitted:
        wc_update = jit(wc_update)
    state = wc_init()
    for sample in x:
        state = wc_update(sample, state)
    cov = wc_final(state, regularize=regularize)

    if diagonal:
        assert np.allclose(cov, np.diagonal(target_cov), rtol=0.06)
    else:
        assert np.allclose(cov, target_cov, rtol=0.06)
