from functools import namedtuple
import numpy as onp
import jax
from jax import vmap, jit, custom_jvp, grad, jvp, vjp, value_and_grad
from jax.lax import while_loop, fori_loop
import jax.numpy as np
from scipy.linalg import cho_solve
import time
from jax.util import partial
from jax.scipy.linalg import cho_factor, solve_triangular, cho_solve, eigh#, eigh_tridiagonal
from numpy.testing import assert_allclose

CGState = namedtuple('CGState', ['u', 'r', 'd', 'r_dot_r', 'iter', 'diag', 'offdiag', 'alpha', 'beta'])



def cg_body_fun(state, mvm):
    u, r, d, r_dot_r, iteration, diag, offdiag, prev_alpha, prev_beta = state
    v = mvm(d)
    alpha = r_dot_r / np.dot(d, v)
    u = u + alpha * d
    r = r - alpha * v
    beta_denom = r_dot_r
    r_dot_r = np.dot(r, r)
    beta = r_dot_r / beta_denom
    d = r + beta * d
    diag = jax.ops.index_update(diag, iteration, 1.0 / alpha + prev_beta / prev_alpha)
    offdiag = jax.ops.index_update(offdiag, iteration, np.sqrt(beta) / alpha)
    return CGState(u, r, d, r_dot_r, iteration + 1, diag, offdiag, alpha, beta)

#def cg_cond_fun(state, mvm, epsilon=1.0e-14, max_iters=100):
#    return (np.sqrt(state.r_dot_r) > epsilon) & (state.iter < max_iters)

def cg(b, A, num_iters=100):
    mvm = lambda rhs: np.matmul(A, rhs)
    #_cond_fun = lambda state: cg_cond_fun(state, mvm=mvm, epsilon=epsilon, max_iters=max_iters)
    _body_fun = lambda null, state: cg_body_fun(state, mvm=mvm)
    zero = np.zeros(b.shape[-1])
    init_state = CGState(zero, b, b, np.dot(b, b), 0, zero, zero, 1.0, 0.0)
    final_state = fori_loop(0, num_iters, _body_fun, init_state) #final_state = while_loop(_cond_fun, _body_fun, init_state)
    P = num_iters
    diag, offdiag = final_state.diag[:P], final_state.offdiag[:P-1]
    T = onp.zeros(P * P)
    T = jax.ops.index_add(T, np.arange(1, P * P, P + 1), offdiag)
    T = jax.ops.index_add(T, np.arange(P, P * P, P + 1), offdiag)
    T = jax.ops.index_add(T, np.arange(0, P * P, P + 1), diag)
    T = T.reshape((P, P))
    eig, V = np.linalg.eigh(T)
    logdetT = np.dot(V[0, :], V[0, :] * np.log(eig))
    #logdetT = b.shape[-1] * np.dot(V[0, :], V[0, :] * np.log(eig))
    return final_state.u, np.sqrt(final_state.r_dot_r), logdetT

def batch_cg(b, A, num_iters=100):
    return vmap(lambda _b: cg(_b, A, num_iters=num_iters))(b)

def batch_cg2(b, A, num_iters=100):
    return vmap(lambda _b, _A: cg(_b, _A, num_iters=num_iters))(b, A)


# compute logdet A + b A^{-1} b
#@partial(custom_jvp, nondiff_argnums=(2,))
@custom_jvp
def quad_form_log_det(A, b, probes):
    return np.nan


def direct_quad_form_log_det(A, b):
    L = cho_factor(A, lower=True)[0]
    Linv_b = solve_triangular(L, b, lower=True)
    quad_form = np.dot(Linv_b, Linv_b)
    log_det = 2.0 * np.sum(np.log(np.diagonal(L)))
    return log_det #+ quad_form


@quad_form_log_det.defjvp
def quad_form_log_det_jvp(primals, tangents, num_iters=4):
    A, b, probes = primals
    D = b.shape[-1]
    A_dot, b_dot, _ = tangents

    b_probes = np.concatenate([b[None, :], probes])
    Ainv_b_probes, _, log_det = batch_cg(b_probes, A, num_iters=num_iters)
    Ainv_b, Ainv_probes = Ainv_b_probes[0], Ainv_b_probes[1:]

    quad_form_dA = -np.dot(Ainv_b, np.matmul(A_dot, Ainv_b))
    quad_form_db = 2.0 * np.dot(Ainv_b, b_dot)
    log_det_dA = np.mean(np.einsum('...i,...i->...', np.matmul(probes, A_dot), Ainv_probes))
    tangent_out = log_det_dA
    #tangent_out = log_det_dA + quad_form_dA + quad_form_db

    quad_form = np.dot(Ainv_b, b)
    primal_out = np.mean(log_det[1:])
    #primal_out = quad_form + np.mean(log_det[1:])

    return primal_out, tangent_out


def symmetrize(x):
    return 0.5 * (x + np.transpose(x))


if __name__ == "__main__":
    #onp.random.seed(1)
    trials = 1
    D = 4
    N = 1000 * 1000

    for trial in range(trials):
        if trial != trials - 1:
            probes = onp.random.randn(N * D).reshape((N, D))
            factor = 1
        else:
            probes = onp.eye(D)
            factor = 10.0

        A = onp.random.rand(D * D // 2).reshape((D, D // 2))
        A = onp.matmul(A, onp.transpose(A)) + 0.05 * onp.eye(D)
        b = onp.random.rand(D)

        v1, g1 = value_and_grad(quad_form_log_det, 1)(A, b, probes)
        v2, g2 = value_and_grad(direct_quad_form_log_det, 1)(A, b)
        assert_allclose(v1, v2, atol=1.0e-3, rtol=1.0e-3)
        assert_allclose(g1, g2, atol=1.0e-3, rtol=1.0e-3)

        v1, g1 = value_and_grad(quad_form_log_det, 0)(A, b, probes)
        v2, g2 = value_and_grad(direct_quad_form_log_det, 0)(A, b)
        assert_allclose(v1, v2, atol=1.0e-3, rtol=1.0e-3)
        assert_allclose(symmetrize(g1), symmetrize(g2), atol=5.0e-2 / factor, rtol=5.0e-2 / factor)
        print("passed trial {}...".format(trial + 1))

    import sys; sys.exit()

    if 0:
        N = 5
        A = onp.random.rand(N * D * D // 2).reshape((N, D, D // 2))
        A = onp.matmul(A, onp.transpose(A, axes=(0,2,1))) + 0.05 * onp.eye(D)
        b = onp.random.rand(N * D).reshape((N, D))
        t0 = time.time()
        u, rnorm, num_iters = batch_cg(b, A, num_iters=150)
        t1 = time.time()
        print("batch took {:.5f} seconds".format(t1-t0))
        print("rnorm", rnorm)
        print("num_iters", num_iters)
    elif 0:
        t0 = time.time()
        #print("A",A)
        #print("A symeig", np.linalg.eigh(A)[0])
        #print("Aslogdet", onp.linalg.slogdet(A))
        u, rnorm, num_iters = cg(b, A, num_iters=D)
        t1 = time.time()
        print("took {:.5f} seconds".format(t1-t0))

        delta = u - truth
        print("rnorm", rnorm, " num_iters", num_iters)
        print("delta norm", onp.linalg.norm(delta), onp.max(onp.abs(delta)))

    #import sys; sys.exit()

    def symmetrize(x):
        return 0.5 * (x + np.transpose(x))

    quad_form_log_det(A, b)

    import sys; sys.exit()

    print(symmetrize(grad(quad_form_log_det, 0)(A, b)))
    assert_allclose(symmetrize(grad(vanilla_quad_form_log_det, 0)(A, b)),
                    symmetrize(grad(quad_form_log_det, 0)(A, b)),
                    rtol=1.0e-3, atol=1.0e-3)
    print()
    print(grad(vanilla_quad_form_log_det, 1)(A, b))
    print(grad(quad_form_log_det, 1)(A, b))
