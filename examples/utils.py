import jax
import jax.numpy as np
from jax.lax import stop_gradient
import numpyro


class CustomAdam(numpyro.optim.Adam):
    def init(self, params, num_stats=2):
        return super().init(params), np.zeros(num_stats)

    def update(self, g, state):
        return super().update(g, state[0]), g['stats']

    def get_params(self, state):
        return super().get_params(state[0])


def record_stats(stat_value, num_stats=2):
    stat = numpyro.param('stats', np.zeros(num_stats)) * stop_gradient(stat_value)
    numpyro.factor('stats_dummy_factor', -stat + stop_gradient(stat))


def sigmoid(x):
      return 1.0 / (1.0 + np.exp(-x))


def kdot(X, Z):
    return np.matmul(X, np.transpose(Z))

def dotdot(x):
    return np.dot(x, x)

def cho_tri_solve(A, b):
    L = cho_factor(A, lower=True)[0]
    Linv_b = solve_triangular(L, b, lower=True)
    return L, Linv_b

def sample_aux_noise(shape):
    key = numpyro.sample('rng_key', numpyro.distributions.PRNGIdentity())
    with numpyro.handlers.block():
        return jax.random.normal(key, shape=shape)


def _fori_loop(lower, upper, body_fun, init_val):
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val
