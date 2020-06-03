import numpy as onp
import jax
from jax import vmap, jit
from jax.lax import while_loop
import jax.numpy as np
from scipy.linalg import cho_solve
import time

onp.random.seed(0)
D = 800
A = onp.random.rand(D * D // 2).reshape((D, D // 2))
A = onp.matmul(A, onp.transpose(A)) + 0.05 * onp.eye(D)
b = onp.random.rand(D)
L = onp.linalg.cholesky(A)
truth = cho_solve((L, True), b)
#print("b", b)
#print("truth", truth)


def mvm(rhs):
    return np.matmul(A, rhs)

init_state = (np.zeros(D), b, b, 1.0e10, 0) #(u, r, d, rnorm, step)

def cg_body_fun(state):
    u, r, d, rnorm, step = state
    v = mvm(d)
    alpha = np.dot(r, r) / np.dot(d, v)
    u = u + alpha * d
    beta_denom = np.dot(r, r)
    r = r - alpha * v
    rnorm = np.linalg.norm(r)
    beta = np.dot(r, r) / beta_denom
    d = r + beta * d
    return (u, r, d, rnorm, step + 1)

def cg_cond_fun(state, epsilon=1.0e-14):
    u, r, d, rnorm, step = state
    return rnorm > epsilon

t0 = time.time()
u, r, d, rnorm, num_iters = while_loop(cg_cond_fun, cg_body_fun, init_state)
t1 = time.time()
print("took {:.5f} seconds".format(t1-t0))


delta = u - truth
print("rnorm", rnorm, " num_iters", num_iters)
print("delta norm", onp.linalg.norm(delta), onp.max(onp.abs(delta)))

