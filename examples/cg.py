from functools import namedtuple
import numpy as onp
import jax
from jax import vmap, jit, custom_jvp, grad, jvp, vjp
from jax.lax import while_loop
import jax.numpy as np
from scipy.linalg import cho_solve
import time
from jax.util import partial
from jax.scipy.linalg import cho_factor, solve_triangular, cho_solve, eigh#, eigh_tridiagonal
from numpy.testing import assert_allclose


CGState = namedtuple('CGState', ['u', 'r', 'd', 'r_dot_r', 'iter', 'alpha', 'beta'])



def cg_body_fun(state, mvm):
    u, r, d, r_dot_r, iteration, alpha, beta = state
    v = mvm(d)
    _alpha = r_dot_r / np.dot(d, v)
    u = u + _alpha * d
    r = r - _alpha * v
    _beta_denom = r_dot_r
    r_dot_r = np.dot(r, r)
    _beta = r_dot_r / _beta_denom
    d = r + _beta * d
    alpha = jax.ops.index_update(alpha, iteration, _alpha)
    beta = jax.ops.index_update(beta, iteration, _beta)
    return CGState(u, r, d, r_dot_r, iteration + 1, alpha, beta)

def cg_cond_fun(state, mvm, epsilon=1.0e-14, max_iters=100):
    return (np.sqrt(state.r_dot_r) > epsilon) & (state.iter < max_iters)

def cg(b, A, epsilon=1.0e-14, max_iters=100):
    mvm = lambda rhs: np.matmul(A, rhs)
    _cond_fun = lambda state: cg_cond_fun(state, mvm=mvm, epsilon=epsilon, max_iters=max_iters)
    _body_fun = lambda state: cg_body_fun(state, mvm=mvm)
    zero = np.zeros(b.shape[-1])
    init_state = CGState(zero, b, b, np.dot(b, b), 0, zero, zero)
    final_state = while_loop(_cond_fun, _body_fun, init_state)
    #return final_state.u, np.sqrt(final_state.r_dot_r), final_state.iter
    P, alpha, beta = final_state.iter, final_state.alpha, final_state.beta
    #print("alpha", alpha)
    #print("beta", beta)
    #alphaP = jax.lax.dynamic_slice_in_dim(alpha, 0, P)
    blah = alpha[0:P:1]
    diag = jax.ops.index_add(1.0 / alpha[0:P:1], np.arange(1, P), beta[:P-1] / alpha[:P-1])
    T = onp.zeros(P * P)
    offdiag = np.sqrt(beta[:P-1]) / alpha[:P-1]
    T = jax.ops.index_add(T, np.arange(1, P * P, P + 1), offdiag)
    T = jax.ops.index_add(T, np.arange(P, P * P, P + 1), offdiag)
    T = jax.ops.index_add(T, np.arange(0, P * P, P + 1), diag)
    T = T.reshape((P, P))
    logdetT = 0.0
    #eig, V = np.linalg.eigh(T)
    #logdetT = np.dot(V[0, :], V[0, :] * np.log(eig))
    #print("VeigV", logT)
    #print("T\n",T, onp.linalg.slogdet(T))
    #print("diag0", 1/ alpha[0])
    #print("diag1", 1/ alpha[1] + beta[0] / alpha[0])
    #print("diag2", 1/ alpha[2] + beta[1] / alpha[1])
    #print("diag3", 1/ alpha[3] + beta[2] / alpha[2])
    #print("offdiag0", np.sqrt(beta[0]) / alpha[0])
    #print("offdiag1", np.sqrt(beta[1]) / alpha[1])
    #print("offdiag2", np.sqrt(beta[2]) / alpha[2])
    return final_state.u, np.sqrt(final_state.r_dot_r), final_state.iter, logdetT

#@jit
def batch_cg(b, A, epsilon=1.0e-14, max_iters=100):
    return vmap(lambda _b: cg(_b, A, epsilon=epsilon, max_iters=max_iters))(b)

def batch_cg2(b, A, epsilon=1.0e-14, max_iters=100):
    return vmap(lambda _b, _A: cg(_b, _A, epsilon=epsilon, max_iters=max_iters))(b, A)


# compute logdet A - b A^{-1} b
@custom_jvp
def quad_form_log_det(A, b):
    L = cho_factor(A, lower=True)[0]
    Linv_b = solve_triangular(L, b, lower=True)
    quad_form = np.dot(Linv_b, Linv_b)
    log_det = 2.0 * np.sum(np.log(np.diagonal(L)))
    return 0.0 #log_det + quad_form

def vanilla_quad_form_log_det(A, b):
    L = cho_factor(A, lower=True)[0]
    Linv_b = solve_triangular(L, b, lower=True)
    quad_form = np.dot(Linv_b, Linv_b)
    log_det = 2.0 * np.sum(np.log(np.diagonal(L)))
    return log_det + quad_form

@quad_form_log_det.defjvp
def quad_form_log_det_jvp(primals, tangents, num_probes=5, max_iters=4):
    A, b = primals
    D = b.shape[-1]
    Ainv_b, _, _, _ = cg(b, A, epsilon=1.0e-14, max_iters=max_iters)
    A_dot, b_dot = tangents
    primal_out = 0.0#quad_form_log_det(A, b)

    #probes = onp.random.randn(D * num_probes).reshape((num_probes, D))
    probes = np.ones((num_probes, D))
    Ainv_probes, _, _, _ = batch_cg(probes, A, max_iters=max_iters)

    quad_form_dA = -np.dot(Ainv_b, np.matmul(A_dot, Ainv_b))
    quad_form_db = 2.0 * np.dot(Ainv_b, b_dot)
    log_det_dA = np.mean(np.einsum('...i,...i->...', np.matmul(probes, A_dot), Ainv_probes))
    tangent_out = log_det_dA + quad_form_dA + quad_form_db

    return primal_out, tangent_out


if __name__ == "__main__":
    #onp.random.seed(0)
    D = 4
    A = onp.random.rand(D * D // 2).reshape((D, D // 2))
    A = onp.matmul(A, onp.transpose(A)) + 0.05 * onp.eye(D)
    A = A + onp.diag(np.array([0.1,0.2,0.3,0.4]))
    b = onp.random.rand(D)
    #L = onp.linalg.cholesky(A)
    #truth = cho_solve((L, True), b)
    #print("b", b)
    #print("truth", truth)
    print(quad_form_log_det(A, b))
    grad(quad_form_log_det, 0)(A, b)

    import sys; sys.exit()

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
    elif 0:
        t0 = time.time()
        #print("A",A)
        #print("A symeig", np.linalg.eigh(A)[0])
        #print("Aslogdet", onp.linalg.slogdet(A))
        u, rnorm, num_iters = cg(b, A, epsilon=1.0e-18, max_iters=D)
        t1 = time.time()
        print("took {:.5f} seconds".format(t1-t0))

        delta = u - truth
        print("rnorm", rnorm, " num_iters", num_iters)
        print("delta norm", onp.linalg.norm(delta), onp.max(onp.abs(delta)))

    #import sys; sys.exit()

    def symmetrize(x):
        return 0.5 * (x + np.transpose(x))

    quad_form_log_det(A, b)

    #u, rnorm, iters, logdetT = batch_cg(b, A)
    #print("rnorm", rnorm)
    #print("iters", iters)
    #print("logdetT", logdetT)



    import sys; sys.exit()

    #print(grad(vanilla_quad_form_log_det, 0)(A, b))
    print(symmetrize(grad(vanilla_quad_form_log_det, 0)(A, b)))
    print(symmetrize(grad(quad_form_log_det, 0)(A, b)))
    assert_allclose(symmetrize(grad(vanilla_quad_form_log_det, 0)(A, b)),
                    symmetrize(grad(quad_form_log_det, 0)(A, b)),
                    rtol=1.0e-3, atol=1.0e-3)
    print()
    print(grad(vanilla_quad_form_log_det, 1)(A, b))
    print(grad(quad_form_log_det, 1)(A, b))

    #b = onp.random.rand(D)
    #g1 = symmetrize(grad(vanilla_quad_form_log_det)(A, b))
    #g2 = grad(quad_form_log_det)(A, b)
    #print("g1-g2\n",g1-g2)
    #assert_allclose(g1, g2, rtol=1.0e-5, atol=1.0e-5)
