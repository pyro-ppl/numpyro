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


CGState = namedtuple('CGState', ['x', 'r', 'p', 'r_dot_r', 'iter'])
PCGState = namedtuple('CGState', ['x', 'r', 'p', 'z', 'r_dot_z', 'iter'])


def cg_body_fun(state, mvm):
    x, r, p, r_dot_r, iteration = state
    Ap = mvm(p)
    alpha = r_dot_r / np.dot(p, Ap)
    x = x + alpha * p
    r = r - alpha * Ap
    beta_denom = r_dot_r
    r_dot_r = np.dot(r, r)
    beta = r_dot_r / beta_denom
    p = r + beta * p
    return CGState(x, r, p, r_dot_r, iteration + 1)


def pcg_body_fun(state, mvm, presolve):
    x, r, p, z, r_dot_z, iteration = state
    Ap = mvm(p)
    alpha = r_dot_z / np.dot(p, Ap)
    x = x + alpha * p
    r = r - alpha * Ap
    z = presolve(r)
    beta_denom = r_dot_z
    r_dot_z = np.dot(r, z)
    beta = r_dot_z / beta_denom
    p = z + beta * p
    return PCGState(x, r, p, z, r_dot_z, iteration + 1)


def cg_cond_fun(state, epsilon=1.0e-14, max_iters=100):
    return (np.sqrt(state.r_dot_r) > epsilon) & (state.iter < max_iters)

def pcg_cond_fun(state, epsilon=1.0e-14, max_iters=100):
    return (np.linalg.norm(state.r) > epsilon) & (state.iter < max_iters)


def cg(b, A, epsilon=1.0e-14, max_iters=4):
    mvm = lambda rhs: np.matmul(A, rhs)
    cond_fun = lambda state: cg_cond_fun(state, epsilon=epsilon, max_iters=max_iters)
    body_fun = lambda state: cg_body_fun(state, mvm=mvm)
    init_state = CGState(np.zeros(b.shape[-1]), b, b, np.dot(b, b), 0)
    final_state = while_loop(cond_fun, body_fun, init_state)
    return final_state.x, np.sqrt(final_state.r_dot_r), final_state.iter


def pcg(b, A, epsilon=1.0e-14, max_iters=4):
    presolve = lambda rhs: rhs
    mvm = lambda rhs: np.matmul(A, rhs)
    cond_fun = lambda state: pcg_cond_fun(state, epsilon=epsilon, max_iters=max_iters)
    body_fun = lambda state: pcg_body_fun(state, mvm=mvm, presolve=presolve)
    z = presolve(b)
    init_state = PCGState(np.zeros(b.shape[-1]), b, z, z, np.dot(b, z), 0)
    final_state = while_loop(cond_fun, body_fun, init_state)
    return final_state.x, np.linalg.norm(final_state.r), final_state.iter


def cg_batch_b(b, A, epsilon=1.0e-14, max_iters=100):
    return vmap(lambda _b: cg(_b, A, epsilon=epsilon, max_iters=max_iters))(b)

def cg_batch_bA(b, A, epsilon=1.0e-14, max_iters=100):
    return vmap(lambda _b, _A: cg(_b, _A, epsilon=epsilon, max_iters=max_iters))(b, A)

def pcg_batch_b(b, A, epsilon=1.0e-14, max_iters=4):
    ret = vmap(lambda _b: pcg(_b, A, epsilon=epsilon, max_iters=max_iters))(b)
    print("res", np.mean(ret[1]))
    return ret
    #return vmap(lambda _b: pcg(_b, A, epsilon=epsilon, max_iters=max_iters))(b)


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
def pcg_quad_form_log_det(A, b, probes, max_iters=5):
    return np.nan


@pcg_quad_form_log_det.defjvp
def pcg_quad_form_log_det_jvp(primals, tangents):
    A, b, probes, max_iters = primals
    A_dot, b_dot, _, _ = tangents
    D = b.shape[-1]

    b_probes = np.concatenate([b[None, :], probes])
    Ainv_b_probes = pcg_batch_b(b_probes, A, max_iters=max_iters)[0]
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
    onp.random.seed(1)
    trials = 3
    D = 10
    N = 500 * 1000
    atol = 5.0e-1
    rtol = 1.0e-1

    for trial in range(trials):
        probes = onp.random.randn(N * D).reshape((N, D))

        A = onp.random.rand(D * D // 2).reshape((D, D // 2))
        A = onp.matmul(A, onp.transpose(A)) + 0.35 * onp.eye(D)
        b = onp.random.randn(D)

        direct_include = lambda A, b: direct_quad_form_log_det(A, b, include_log_det=True)
        direct_exclude = lambda A, b: direct_quad_form_log_det(A, b, include_log_det=False)

        #v1, g1 = value_and_grad(cg_quad_form_log_det, 1)(A, b, probes)
        v2, _ = value_and_grad(direct_exclude, 1)(A, b)
        _, g2 = value_and_grad(direct_include, 1)(A, b)
        v3, g3 = value_and_grad(pcg_quad_form_log_det, 1)(A, b, probes)
        #assert_allclose(v1, v2, atol=atol, rtol=rtol)
        #assert_allclose(g1, g2, atol=atol, rtol=rtol)
        assert_allclose(v2, v3, atol=atol, rtol=rtol)
        assert_allclose(g2, g3, atol=atol, rtol=rtol)

        #v1, g1 = value_and_grad(cg_quad_form_log_det, 0)(A, b, probes)
        v2, _ = value_and_grad(direct_exclude, 0)(A, b)
        _, g2 = value_and_grad(direct_include, 0)(A, b)
        v3, g3 = value_and_grad(pcg_quad_form_log_det, 0)(A, b, probes)
        #assert_allclose(v1, v2, atol=atol, rtol=rtol)
        #assert_allclose(symmetrize(g1), symmetrize(g2), atol=atol, rtol=rtol)
        assert_allclose(v3, v2, atol=atol, rtol=rtol)
        assert_allclose(symmetrize(g3), symmetrize(g2), atol=atol, rtol=rtol)

        print("passed trial {}...".format(trial + 1))
