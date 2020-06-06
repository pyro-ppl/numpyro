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


CGState = namedtuple('CGState', ['u', 'r', 'd', 'r_dot_r', 'iter'])


# X X^t b
def quad_mvm(b, X):
    return np.einsum('np,p->n', X, np.einsum('np,n->p', X, b))

def kernel_mvm(b, X, Xsq, eta1sq, eta2sq, c, jitter=1.0e-6):
    b_one = np.sum(b) * np.ones(b.shape)
    XXb = quad_mvm(b, X)
    expensive = np.einsum('ip,jp,iq,jq,j->i',X,X,X,X,b)
    k1b = 0.5 * eta2sq * (b_one + 2.0 * XXb + expensive)
    k2b = -0.5 * eta2sq * quad_mvm(b, Xsq)
    k3b = (eta1sq - eta2sq) * XXb
    k4b = (np.square(c) - 0.5 * eta2sq) * b_one + jitter * b
    return k1b + k2b + k3b + k4b


def structured_cg_body_fun(state, X, Xsq, eta1sq, eta2sq, c, jitter):
    u, r, d, r_dot_r, iteration = state
    v = kernel_mvm(d, X, Xsq, eta1sq, eta2sq, c, jitter)
    alpha = r_dot_r / np.dot(d, v)
    u = u + alpha * d
    r = r - alpha * v
    beta_denom = r_dot_r
    r_dot_r = np.dot(r, r)
    beta = r_dot_r / beta_denom
    d = r + beta * d
    return CGState(u, r, d, r_dot_r, iteration + 1)


def cg_body_fun(state, mvm):
    u, r, d, r_dot_r, iteration = state
    v = mvm(d)
    alpha = r_dot_r / np.dot(d, v)
    u = u + alpha * d
    r = r - alpha * v
    beta_denom = r_dot_r
    r_dot_r = np.dot(r, r)
    beta = r_dot_r / beta_denom
    d = r + beta * d
    return CGState(u, r, d, r_dot_r, iteration + 1)


def cg_cond_fun(state, epsilon=1.0e-14, max_iters=100):
    return (np.sqrt(state.r_dot_r) > epsilon) & (state.iter < max_iters)


def cg(b, A, epsilon=1.0e-14, max_iters=200):
    mvm = lambda rhs: np.matmul(A, rhs)
    cond_fun = lambda state: cg_cond_fun(state, epsilon=epsilon, max_iters=max_iters)
    body_fun = lambda state: cg_body_fun(state, mvm=mvm)
    init_state = CGState(np.zeros(b.shape[-1]), b, b, np.dot(b, b), 0)
    final_state = while_loop(cond_fun, body_fun, init_state)
    return final_state.u, np.sqrt(final_state.r_dot_r), final_state.iter


def structured_cg(b, A, X, Xsq, eta1sq, eta2sq, c, jitter, epsilon=1.0e-14, max_iters=200):
    cond_fun = lambda state: cg_cond_fun(state, epsilon=epsilon, max_iters=max_iters)
    body_fun = lambda state: structured_cg_body_fun(state, X, Xsq, eta1sq, eta2sq, c, jitter)
    init_state = CGState(np.zeros(b.shape[-1]), b, b, np.dot(b, b), 0)
    final_state = while_loop(cond_fun, body_fun, init_state)
    return final_state.u, np.sqrt(final_state.r_dot_r), final_state.iter



def cg_batch_b(b, A, epsilon=1.0e-14, max_iters=100):
    return vmap(lambda _b: cg(_b, A, epsilon=epsilon, max_iters=max_iters))(b)

def cg_batch_bA(b, A, epsilon=1.0e-14, max_iters=100):
    return vmap(lambda _b, _A: cg(_b, _A, epsilon=epsilon, max_iters=max_iters))(b, A)

def structured_cg_batch_b(b, A, X, Xsq, eta1sq, eta2sq, c, jitter, epsilon=1.0e-14, max_iters=100):
    return vmap(lambda _b: structured_cg(_b, A, X, Xsq, eta1sq, eta2sq, c, jitter, epsilon=epsilon, max_iters=max_iters))(b)


# compute logdet A + b A^{-1} b
def direct_quad_form_log_det(A, b, include_log_det=True):
    L = cho_factor(A, lower=True)[0]
    Linv_b = solve_triangular(L, b, lower=True)
    quad_form = np.dot(Linv_b, Linv_b)
    if include_log_det:
        log_det = 2.0 * np.sum(np.log(np.diagonal(L)))
        return log_det + quad_form
    else:
        return quad_form


# compute logdet A + b A^{-1} b
@custom_jvp
def cg_quad_form_log_det(A, b, probes, max_iters=100):
    return np.nan


@cg_quad_form_log_det.defjvp
def cg_quad_form_log_det_jvp(primals, tangents):
    A, b, probes, max_iters = primals
    A_dot, b_dot, _, _ = tangents
    D = b.shape[-1]

    b_probes = np.concatenate([b[None, :], probes])
    Ainv_b_probes = cg_batch_b(b_probes, A, max_iters=max_iters)[0]
    Ainv_b, Ainv_probes = Ainv_b_probes[0], Ainv_b_probes[1:]

    quad_form_dA = -np.dot(Ainv_b, np.matmul(A_dot, Ainv_b))
    quad_form_db = 2.0 * np.dot(Ainv_b, b_dot)
    log_det_dA = np.mean(np.einsum('...i,...i->...', np.matmul(probes, A_dot), Ainv_probes))
    tangent_out = log_det_dA + quad_form_dA + quad_form_db

    quad_form = np.dot(b, Ainv_b)
    primal_out = quad_form

    return primal_out, tangent_out


# compute logdet A + b A^{-1} b
@custom_jvp
def structured_cg_quad_form_log_det(A, b, probes, X, Xsq, eta1sq, eta2sq, c, jitter, max_iters=100):
    return np.nan


@structured_cg_quad_form_log_det.defjvp
def structured_cg_quad_form_log_det_jvp(primals, tangents):
    A, b, probes, X, Xsq, eta1sq, eta2sq, c, jitter, max_iters = primals
    A_dot, b_dot, _, _, _, _, _, _, _, _  = tangents
    D = b.shape[-1]

    b_probes = np.concatenate([b[None, :], probes])
    Ainv_b_probes = structured_cg_batch_b(b_probes, A, X, Xsq, eta1sq, eta2sq, c, jitter, max_iters=max_iters)[0]
    Ainv_b, Ainv_probes = Ainv_b_probes[0], Ainv_b_probes[1:]

    quad_form_dA = -np.dot(Ainv_b, np.matmul(A_dot, Ainv_b))
    quad_form_db = 2.0 * np.dot(Ainv_b, b_dot)
    log_det_dA = np.mean(np.einsum('...i,...i->...', np.matmul(probes, A_dot), Ainv_probes))
    tangent_out = log_det_dA + quad_form_dA + quad_form_db

    quad_form = np.dot(b, Ainv_b)
    primal_out = quad_form

    return primal_out, tangent_out


def symmetrize(x):
    return 0.5 * (x + np.transpose(x))


if __name__ == "__main__":
    #onp.random.seed(1)
    trials = 10
    D = 4
    N = 500 * 1000
    atol = 5.0e-2
    rtol = 1.0e-3

    def dot(X, Z):
        return np.dot(X, Z[..., None])[..., 0]

    def kernel(X, Z, eta1, eta2, c, jitter=1.0e-6):
        eta1sq = np.square(eta1)
        eta2sq = np.square(eta2)
        k1 = 0.5 * eta2sq * np.square(1.0 + dot(X, Z))
        k2 = -0.5 * eta2sq * dot(np.square(X), np.square(Z))
        k3 = (eta1sq - eta2sq) * dot(X, Z)
        k4 = np.square(c) - 0.5 * eta2sq
        if X.shape == Z.shape:
            k4 += jitter * np.eye(X.shape[0])
        return k1 + k2 + k3 + k4

    N = 50
    P = 10
    X = onp.random.randn(N * P).reshape((N, P))
    b = onp.random.randn(N)
    eta1, eta2, c = 0.4, 0.007, 0.9
    k = kernel(X, X, eta1, eta2, c, jitter=1.0e-3)
    kb = np.matmul(k, b)
    kb2 = kernel_mvm(b, X, np.square(X), np.square(eta1), np.square(eta2), c, jitter=1.0e-3)
    assert_allclose(kb, kb2, atol=1.0e-5, rtol=1.0e-5)

    import sys; sys.exit()

    for trial in range(trials):
        probes = onp.random.randn(N * D).reshape((N, D))

        A = onp.random.rand(D * D // 2).reshape((D, D // 2))
        A = onp.matmul(A, onp.transpose(A)) + 0.35 * onp.eye(D)
        b = onp.random.randn(D)

        direct_include = lambda A, b: direct_quad_form_log_det(A, b, include_log_det=True)
        direct_exclude = lambda A, b: direct_quad_form_log_det(A, b, include_log_det=False)

        v1, g1 = value_and_grad(cg_quad_form_log_det, 1)(A, b, probes)
        v2, _ = value_and_grad(direct_exclude, 1)(A, b)
        _, g2 = value_and_grad(direct_include, 1)(A, b)
        assert_allclose(v1, v2, atol=atol, rtol=rtol)
        assert_allclose(g1, g2, atol=atol, rtol=rtol)

        v1, g1 = value_and_grad(cg_quad_form_log_det, 0)(A, b, probes)
        v2, _ = value_and_grad(direct_exclude, 0)(A, b)
        _, g2 = value_and_grad(direct_include, 0)(A, b)
        assert_allclose(v1, v2, atol=atol, rtol=rtol)
        assert_allclose(symmetrize(g1), symmetrize(g2), atol=atol, rtol=rtol)

        print("passed trial {}...".format(trial + 1))
