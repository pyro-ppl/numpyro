import jax
import numpyro


class CustomAdam(numpyro.optim.Adam):
    def init(self, params, num_stats=2):
        return super().init(params), np.zeros(num_stats)

    def update(self, g, state):
        return super().update(g, state[0]), g['stats']

    def get_params(self, state):
        return super().get_params(state[0])


def record_stats(stat_value, num_stats=2):
    stat = numpyro.param('stats', np.zeros(num_stats)) * jax.lax.stop_gradient(stat_value)
    numpyro.factor('stats_dummy_factor', -stat + jax.lax.stop_gradient(stat))


def sigmoid(x):
      return 1.0 / (1.0 + np.exp(-x))


def kdot(X, Z):
    return np.dot(X, Z[..., None])[..., 0]


def cho_tri_solve(A, b):
    L = cho_factor(A, lower=True)[0]
    Linv_b = solve_triangular(L, b, lower=True)
    return L, Linv_b
