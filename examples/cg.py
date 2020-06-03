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
D = 4
A = onp.random.rand(D * D // 2).reshape((D, D // 2))
A = onp.matmul(A, onp.transpose(A)) + 0.05 * onp.eye(D)
b = onp.random.rand(D)
L = onp.linalg.cholesky(A)
truth = cho_solve((L, True), b)
#print("b", b)
#print("truth", truth)



CGState = namedtuple('CGState', ['u', 'r', 'd', 'rnorm', 'iter', 'alpha', 'beta'])

def cg_body_fun(state, mvm):
    u, r, d, rnorm, iteration, alpha, beta = state
    v = mvm(d[..., None])[..., 0]
    _alpha = np.dot(r, r) / np.dot(d, v)
    u = u + _alpha * d
    beta_denom = np.dot(r, r)
    r = r - _alpha * v
    rnorm = np.linalg.norm(r)
    _beta = np.dot(r, r) / beta_denom
    d = r + _beta * d
    alpha = jax.ops.index_update(alpha, jax.ops.index[iteration], _alpha)
    beta = jax.ops.index_update(beta, jax.ops.index[iteration], _beta)
    return CGState(u, r, d, rnorm, iteration + 1, alpha, beta)

def cg_cond_fun(state, mvm, epsilon=1.0e-14, max_iters=100):
    return (state.rnorm > epsilon) & (state.iter < max_iters)

def cg(b, A, epsilon=1.0e-14, max_iters=100):
    mvm = lambda rhs: np.matmul(A, rhs)
    _cond_fun = lambda state: cg_cond_fun(state, mvm=mvm, epsilon=epsilon, max_iters=max_iters)
    _body_fun = lambda state: cg_body_fun(state, mvm=mvm)
    zero = np.zeros(b.shape[-1])
    init_state = CGState(zero, b, b, 1.0e10, 0, zero, zero)
    final_state = while_loop(_cond_fun, _body_fun, init_state)
    alpha = final_state.alpha
    beta = final_state.beta
    P = final_state.iter
    print("alpha", alpha)
    print("beta", beta)
    diag = 1.0 / alpha[:P]
    diag = jax.ops.index_add(diag, np.arange(1, P), beta[:P-1] / alpha[:P-1])
    T = onp.zeros(P * P)
    offdiag = np.sqrt(beta[:P-1]) / alpha[:P-1]
    T = jax.ops.index_add(T, np.arange(1, P * P, P + 1), offdiag)
    T = jax.ops.index_add(T, np.arange(P, P * P, P + 1), offdiag)
    T = jax.ops.index_add(T, np.arange(0, P * P, P + 1), diag)
    T = T.reshape((P, P))
    print("T\n",T, onp.linalg.slogdet(T))
    print("diag0", 1/ alpha[0])
    print("diag1", 1/ alpha[1] + beta[0] / alpha[0])
    print("diag2", 1/ alpha[2] + beta[1] / alpha[1])
    print("offdiag0", np.sqrt(beta[0]) / alpha[0])
    print("offdiag1", np.sqrt(beta[1]) / alpha[1])
    return final_state.u, final_state.rnorm, final_state.iter

#@jit
def batch_cg(b, A, epsilon=1.0e-14, max_iters=100):
    return vmap(lambda _b, _A: cg(_b, _A, epsilon=epsilon, max_iters=max_iters))(b, A)

if 0:
    N = 5
    A = onp.random.rand(N * D * D // 2).reshape((N, D, D // 2))
    A = onp.matmul(A, onp.transpose(A, axes=(0,2,1))) + 0.05 * onp.eye(D)
    b = onp.random.rand(N * D).reshape((N, D))
    t0 = time.time()
    u, rnorm, num_iters = batch_cg(b, A, epsilon=1.0e-10, max_iters=150)
    t1 = time.time()
    print("batch took {:.5f} seconds".format(t1-t0))
    print("rnorm", rnorm)
    print("num_iters", num_iters)
else:
    t0 = time.time()
    print("Aslogdet", onp.linalg.slogdet(A))
    u, rnorm, num_iters = cg(b, A, epsilon=1.0e-14, max_iters=4)
    t1 = time.time()
    print("took {:.5f} seconds".format(t1-t0))

    delta = u - truth
    print("rnorm", rnorm, " num_iters", num_iters)
    print("delta norm", onp.linalg.norm(delta), onp.max(onp.abs(delta)))

