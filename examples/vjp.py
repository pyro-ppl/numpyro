import jax.numpy as np
from numpy.testing import assert_allclose
from functools import partial
from jax import vmap, jit, custom_jvp, value_and_grad, jvp, grad, custom_vjp
from jax.scipy.linalg import cho_factor, solve_triangular, cho_solve, inv
import numpyro
import time
import numpy as onp
import math


from mvm import kernel_mvm_diag, kXkXsq_mvm, quad_mvm_dil, quad_mvm_dil3, kX_mvm, kXdkXsq_mvm, kXkXkXX_qf, kXkXkXX_qf2
from cg import lowrank_presolve, pcg_batch_b, kernel

numpyro.set_platform('gpu')


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


@partial(custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9, 10, 11, 12, 13))
def pcpcg_quad_form_log_det(kappa, b, eta1, eta2, diag, c, X, probes, rank1, rank2, cg_tol, max_iters, dilation, dilation2):
    return (np.nan, np.nan, np.nan)

def pcpcg_quad_form_log_det_fwd(kappa, b, eta1, eta2, diag, c, X, probes, rank1, rank2, cg_tol, max_iters, dilation, dilation2):
    kX = kappa * X
    omega_b = b * diag

    mvm = lambda _b: kernel_mvm_diag(_b, kX, eta1, eta2, c, diag, dilation=dilation,dilation2=dilation2)
    presolve = lowrank_presolve(kX, diag, eta1, eta2, c, kappa, rank1, rank2)

    om_b_probes = np.concatenate([omega_b[None, :], probes])
    Ainv_om_b_probes, res_norm, iters = pcg_batch_b(om_b_probes, mvm, presolve=presolve,
                                                    cg_tol=cg_tol, max_iters=max_iters)
    Ainv_om_b, Ainv_probes = Ainv_om_b_probes[0], Ainv_om_b_probes[1:]
    K_Ainv_om_b = kernel_mvm_diag(Ainv_om_b, kX, eta1, eta2, c, 0.0, dilation=dilation, dilation2=dilation2)
    quad_form = 0.125 * np.dot(b, K_Ainv_om_b)

    residuals = (kX, kappa, eta1, eta2, K_Ainv_om_b, Ainv_om_b, diag, Ainv_probes, probes)

    return (quad_form, np.mean(res_norm), np.mean(iters)), residuals

def meansum(x):
    return np.sum(x) / x.shape[0]

def pcpcg_quad_form_log_det_bwd(c, X, probes, rank1, rank2, cg_tol, max_iters, dilation, dilation2,
                                residuals, g):
    kX, kappa, eta1, eta2, K_Ainv_om_b, Ainv_om_b, diag, Ainv_probes, probes = residuals

    qfld_bar = g[0]

    diag_bar_qf = np.square(K_Ainv_om_b / diag)
    diag_bar_ld = np.mean(Ainv_probes * probes, axis=0)

    zk1z = np.dot(Ainv_om_b, kXkXsq_mvm(Ainv_om_b, kX, dilation=dilation2))
    zk2z = np.dot(Ainv_om_b, quad_mvm_dil(Ainv_om_b, np.square(kX), dilation=dilation))
    zk3z = np.dot(Ainv_om_b, quad_mvm_dil(Ainv_om_b, kX, dilation=dilation))
    zk4z = np.square(np.sum(Ainv_om_b))

    probes_kX = kX_mvm(probes, kX, dilation=dilation)
    probes_sqkX = kX_mvm(probes, np.square(kX), dilation=dilation)
    Ainv_probes_kX = kX_mvm(Ainv_probes, kX, dilation=dilation)
    Ainv_probes_sqkX = kX_mvm(Ainv_probes, np.square(kX), dilation=dilation)
    kXkXsq_Ainv_probes = np.transpose(kXkXsq_mvm(Ainv_probes, kX, dilation=dilation2))

    prbk1prb = meansum(probes * kXkXsq_Ainv_probes)
    prbk2prb = meansum(probes_sqkX * Ainv_probes_sqkX)
    prbk3prb = meansum(probes_kX * Ainv_probes_kX)
    prbk4prb = np.mean(np.sum(probes, axis=-1) * np.sum(Ainv_probes, axis=-1))

    Xsq = np.square(X)
    k3Xsq = kappa ** 3 * Xsq

    zXzkX = np.square(np.sum(Ainv_om_b[:, None] * X, axis=0)) * kappa
    _kXkXkXX_qf = kXkXkXX_qf(Ainv_om_b, X, kX, dilation=dilation2)
    k3X2X2_qf = np.sum(k3Xsq * Ainv_om_b[:, None], axis=0) * np.sum(Xsq * Ainv_om_b[:, None], axis=0)
    _kXkXkXX_qf2 = kXkXkXX_qf2(probes, Ainv_probes, X, kX, dilation=dilation2)
    k3X2X2_qf2 = np.mean(np.sum(k3Xsq * probes[:, :, None], axis=-2) * np.sum(Xsq * Ainv_probes[:, :, None], axis=-2), axis=0)

    eta1sq, eta2sq = np.square(eta1), np.square(eta2)

    kappa_qf_bar = (2.0 * eta1sq) * zXzkX + (2.0 * eta2sq) * _kXkXkXX_qf - (2.0 * eta2sq) * k3X2X2_qf
    kappa_ld_bar = (2.0 * eta1sq) * np.mean(np.sum(probes[:, :, None] * kX, axis=-2) * np.sum(Ainv_probes[:, :, None] * X, axis=-2), axis=0) + (2.0 * eta2sq) * _kXkXkXX_qf2 - (2.0 * eta2sq)* k3X2X2_qf2

    kappa_bar = (0.125 * qfld_bar) * kappa_qf_bar - (0.5 * qfld_bar) * kappa_ld_bar
    b_bar = (0.25 * qfld_bar) * K_Ainv_om_b
    eta1_bar = (0.25 * eta1 * qfld_bar) * zk3z - (eta1 * qfld_bar) * prbk3prb
    eta2_qf_bar = 0.5 * (zk1z - zk2z - zk4z) - zk3z
    eta2_ld_bar = 0.5 * (prbk1prb - prbk2prb - prbk4prb) - prbk3prb
    eta2_bar = (0.25 * eta2 * qfld_bar) * eta2_qf_bar - (eta2 * qfld_bar) * eta2_ld_bar
    diag_bar = (0.125 * qfld_bar) * diag_bar_qf - (0.5 * qfld_bar) * diag_bar_ld

    return (kappa_bar, b_bar, eta1_bar, eta2_bar, diag_bar)

pcpcg_quad_form_log_det.defvjp(pcpcg_quad_form_log_det_fwd, pcpcg_quad_form_log_det_bwd)


if __name__ == '__main__':
    N = 10 * 1000
    P = 500
    b = np.array(onp.random.randn(N))
    X = np.array(onp.random.randn(N * P).reshape((N, P)))
    kappa = 0.3 + 2.0 * np.array(onp.random.rand(P))
    eta1 = 0.8
    eta2 = 0.5
    diag = np.array(onp.random.rand(N))
    c = 1.0
    num_probes = 1
    probes = np.array(onp.random.randn(N * num_probes).reshape((num_probes, N)))
    #probes = math.sqrt(N) * np.eye(N)

    def direct(_kappa, _b, _eta1, _eta2, _diag, include_log_det):
        kX = _kappa * X
        k = kernel(kX, kX, _eta1, _eta2, c)
        k_diag = k + np.diag(_diag)
        return direct_quad_form_log_det(k_diag, np.matmul(k, _b), _b * _diag, include_log_det=include_log_det)

    @jit
    def pcpcg(_kappa, _b, _eta1, _eta2, _diag):
        return pcpcg_quad_form_log_det(_kappa, _b, _eta1, _eta2, _diag, c, X,
                                       probes, 64, 16, 1.0e-2, 3, 2, 8)[0]

    #v1, _ = value_and_grad(direct, 4)(kappa, b, eta1, eta2, diag, False)
    #_, g1 = value_and_grad(direct, 4)(kappa, b, eta1, eta2, diag, True)
    t0 = time.time()
    v2, g2 = value_and_grad(pcpcg, 4)(kappa, b, eta1, eta2, diag)
    v2, g2 = value_and_grad(pcpcg, 0)(kappa, b, eta1, eta2, diag)
    #v2, g2 = value_and_grad(lambda x: pcpcg_quad_form_log_det(kappa, b, eta1, x, diag, c, X,
    #                                                          probes, 2, 2, 1.0e-3, 2, 1, 4)[0], 0)(eta2)
    #g2.block_until_ready()
    #v2, g2 = value_and_grad(lambda x: pcpcg_quad_form_log_det(kappa, b, eta1, x, diag, c, X,
    #                                                          probes, 2, 2, 1.0e-3, 2, 1, 4)[0], 0)(eta2)
    #t0 = time.time()
    #v2, g2 = value_and_grad(lambda x: pcpcg_quad_form_log_det(kappa, b, eta1, x, diag, c, X,
    #                                                          probes, 2, 2, 1.0e-3, 10, 2, 8)[0], 0)(eta2)
    t1 = time.time()
    print("[ELAPSED TIME]", t1-t0)
    print("[ELAPSED TIME]", t1-t0)
    print("[ELAPSED TIME]", t1-t0)
    #import sys;sys.exit()
    #print("v1", v1, "v2", v2)
    #print("g1", g1.shape, "g2", g2.shape)
    #print("g1", g1)#[0:3])
    #print("g2", g2)#[0:3])

    #assert_allclose(g1, g2, atol=1.0e-5, rtol=1.0e-5)
