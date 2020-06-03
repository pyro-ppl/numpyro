from functools import namedtuple
import numpy as onp
import jax
from jax import vmap, jit
from jax.lax import while_loop
import jax.numpy as np
from scipy.linalg import cho_solve
import time
from jax.util import partial


onp.random.seed(0)
D = 50
A = onp.random.rand(D * D // 2).reshape((D, D // 2))
A = onp.matmul(A, onp.transpose(A)) + 0.05 * onp.eye(D)
b = onp.random.rand(D)
L = onp.linalg.cholesky(A)
truth = cho_solve((L, True), b)
#print("b", b)
#print("truth", truth)



CGState = namedtuple('CGState', ['u', 'r', 'd', 'rnorm', 'iter'])

def cg_body_fun(state, mvm):
    u, r, d, rnorm, iteration = state
    v = mvm(d)
    alpha = np.dot(r, r) / np.dot(d, v)
    u = u + alpha * d
    beta_denom = np.dot(r, r)
    r = r - alpha * v
    rnorm = np.linalg.norm(r)
    beta = np.dot(r, r) / beta_denom
    d = r + beta * d
    return CGState(u, r, d, rnorm, iteration + 1)

def cg_cond_fun(state, mvm, epsilon=1.0e-14):
    return state.rnorm > epsilon# and state.iter < 50

def cg(b, A, epsilon=1.0e-14, max_iters=100):
    mvm = lambda rhs: np.matmul(A, rhs)
    _cond_fun = lambda state: cg_cond_fun(state, mvm=mvm, epsilon=epsilon)
    _body_fun = lambda state: cg_body_fun(state, mvm=mvm)
    init_state = CGState(np.zeros(D), b, b, 1.0e10, 0)  # (u, r, d, rnorm, iter)
    final_state = while_loop(_cond_fun, _body_fun, init_state)
    return final_state.u, final_state.rnorm, final_state.iter

#def batch_cg(b, A, epsilon=1.0e-14, max_iters=100):
#    return vmap(lambda _b, _A: cg(_b, _A, epsilon=epsilon, max_iters=max_iters))(b, A)

t0 = time.time()
u, rnorm, num_iters = cg(b, A, max_iters=50)
t1 = time.time()
print("took {:.5f} seconds".format(t1-t0))

delta = u - truth
print("rnorm", rnorm, " num_iters", num_iters)
print("delta norm", onp.linalg.norm(delta), onp.max(onp.abs(delta)))

