import jax.numpy as np
from numpy.testing import assert_allclose
from functools import partial
from jax import vmap, jit, custom_jvp, value_and_grad, jvp, grad, custom_vjp
from jax.scipy.linalg import cho_factor, solve_triangular, cho_solve, inv
import numpyro
import time
import numpy as onp

from mvm import kernel_mvm_diag
from cg import lowrank_presolve, pcg_batch_b, kernel

numpyro.set_platform('cpu')


def _get_chunks(L, chunk_size):
    num_chunks = L // chunk_size
    chunks = [np.arange(i * chunk_size, (i + 1) * chunk_size) for i in range(num_chunks)]
    if L % chunk_size != 0:
        chunks.append(np.arange(L - L % chunk_size, L))
    return chunks

def _chunk_vmap(fun, array, chunk_size=10):
    L = array.shape[0]
    if chunk_size >= L:
        return vmap(fun)(array)
    chunks = _get_chunks(L, chunk_size)
    results = [vmap(fun)(array[chunk]) for chunk in chunks]
    return np.concatenate(results)

# do a matrix vector after first materializing the matrix M
def vanilla_mvm(row):
    def do_mvm(rhs):
        M = vmap(row)(np.arange(N))
        return np.matmul(M, rhs)
    return do_mvm

# do a matrix vector multiply chunk-by-chunk
def partitioned_mvm(row, dilation):
    def do_mvm(rhs):
        @jit
        def compute_element(i):
            return np.dot(rhs, row(i))
        return _chunk_vmap(compute_element, np.arange(rhs.shape[-1]), rhs.shape[-1] // dilation)
    return do_mvm

def vanilla_f(b, N):
    def rowop(i):
        return np.sin(i / N + np.arange(N) / N)
    return np.sum(vanilla_mvm(rowop)(b))

def part_f(b, N, dilation):
    def rowop(i):
        return np.sin(i / N + np.arange(N) / N)
    return np.sum(partitioned_mvm(rowop, dilation)(b))

# compute logdet A + lhs A^{-1} rhs
def direct_quad_form_log_det(A, lhs, rhs, include_log_det=True):
    L = cho_factor(A, lower=True)[0]
    Linv_rhs = solve_triangular(L, rhs, lower=True)
    Linv_lhs = solve_triangular(L, lhs, lower=True)
    quad_form = 0.125 * np.dot(Linv_lhs, Linv_rhs)
    if include_log_det:
        log_det = - np.sum(np.log(np.diagonal(L)))
        return log_det + quad_form
    else:
        return quad_form

#@partial(custom_vjp, nondiff_argnums=(2, 3))
#def g(b, b2, N, N2):
#    return (np.nan, np.nan)
#def g_fwd(b, b2, N, N2):
#    return (part_f(b, N, 16), 0.0), (b, b2)
#def g_bwd(N, N2, residual, _g):
#    b, b2 = residual
#    def colop(i):
#        return np.sum(np.sin(i / N + np.arange(N) / N))
#    colsum = _chunk_vmap(colop, np.arange(N), N // 16)
#    return (colsum * _g[0], colsum *_g[1])
#g.defvjp(g_fwd, g_bwd)


@partial(custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9, 10, 11, 12))
def pcpcg_quad_form_log_det(kappa, b, eta1, eta2, diag, c, X, probes, rank1, rank2, cg_tol, max_iters, dilation):
    return (np.nan, np.nan, np.nan)

def pcpcg_quad_form_log_det_fwd(kappa, b, eta1, eta2, diag, c, X, probes, rank1, rank2, cg_tol, max_iters, dilation):
    kX = kappa * X
    omega_b = b * diag

    mvm = lambda _b: kernel_mvm_diag(_b, kX, eta1, eta2, c, diag, dilation=dilation)
    presolve = lowrank_presolve(kX, diag, eta1, eta2, c, kappa, rank1, rank2)

    om_b_probes = np.concatenate([omega_b[None, :], probes])
    Ainv_om_b_probes, res_norm, iters = pcg_batch_b(om_b_probes, mvm, presolve=presolve,
                                                    cg_tol=cg_tol, max_iters=max_iters)
    Ainv_om_b, Ainv_probes = Ainv_om_b_probes[0], Ainv_om_b_probes[1:]
    K_Ainv_om_b = kernel_mvm_diag(Ainv_om_b, kX, eta1, eta2, c, 0.0, dilation=dilation)
    quad_form = 0.125 * np.dot(b, K_Ainv_om_b)

    residuals = (kX, K_Ainv_om_b, Ainv_om_b, diag, Ainv_probes, probes)

    return (quad_form, np.mean(res_norm), np.mean(iters)), residuals

def pcpcg_quad_form_log_det_bwd(c, X, probes, rank1, rank2, cg_tol, max_iters, dilation, residuals, g):
    kX, K_Ainv_om_b, Ainv_om_b, diag, Ainv_probes, probes = residuals

    quad_form_bar = g[0]

    diag_bar_qf = np.square(K_Ainv_om_b / diag)
    diag_bar_ld = np.mean(Ainv_probes * probes, axis=0)

    kappa_bar = kappa
    b_bar = (0.25 * quad_form_bar) * K_Ainv_om_b
    eta1_bar = eta1
    eta2_bar = eta2
    diag_bar = (0.125 * quad_form_bar) * diag_bar_qf - (0.5 * quad_form_bar) * diag_bar_ld

    return (kappa_bar, b_bar, eta1_bar, eta2_bar, diag_bar)

pcpcg_quad_form_log_det.defvjp(pcpcg_quad_form_log_det_fwd, pcpcg_quad_form_log_det_bwd)


N = 7
P = 4
b = np.array(onp.random.randn(N))
X = np.array(onp.random.randn(N * P).reshape((N, P)))
kappa = np.array(onp.random.rand(P))
eta1 = 0.3
eta2 = 0.2
diag = np.array(onp.random.rand(N))
c = 1.0
num_probes = 20000
probes = np.array(onp.random.randn(N * num_probes).reshape((num_probes, N)))

def direct(_kappa, _b, _eta1, _eta2, _diag, include_log_det):
    kX = _kappa * X
    k = kernel(kX, kX, _eta1, _eta2, c)
    k_diag = k + np.diag(_diag)
    return direct_quad_form_log_det(k_diag, np.matmul(k, _b), _b * _diag, include_log_det=include_log_det)

v1, _ = value_and_grad(direct, 4)(kappa, b, eta1, eta2, diag, False)
_, g1 = value_and_grad(direct, 4)(kappa, b, eta1, eta2, diag, True)
v2, g2 = value_and_grad(lambda x: pcpcg_quad_form_log_det(kappa, b, eta1, eta2, x, c, X,
                                                          probes, 2, 2, 1.0e-7, 400, 4)[0], 0)(diag)
print("v1", v1, "v2", v2)
print("g1", g1.shape, "g2", g2.shape)
print("g1", g1[0:3])
print("g2", g2[0:3])

assert_allclose(g1, g2, atol=1.0e-5, rtol=1.0e-5)
