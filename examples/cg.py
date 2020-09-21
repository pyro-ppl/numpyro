from functools import namedtuple, partial
import numpyro
import numpy as onp
import jax
from jax import vmap, jit, custom_jvp, value_and_grad, grad
from jax.lax import while_loop, dynamic_slice_in_dim
import jax.numpy as np
import time
from jax.scipy.linalg import cho_factor, solve_triangular, cho_solve
from numpy.testing import assert_allclose
from tensor_sketch import create_sketch_transform, sketch_transform
from utils import dotdot
from mvm import kXkXsq_mvm, kXdkXsq_mvm, kernel_mvm_diag, kX_mvm


CGState = namedtuple('CGState', ['x', 'r', 'p', 'r_dot_r', 'iter'])
PCGState = namedtuple('CGState', ['x', 'r', 'p', 'z', 'r_dot_z', 'iter'])


def kdot(X, Z):
    return np.dot(X, Z[..., None])[..., 0]


def kernel(X, Z, eta1, eta2, c, jitter=1.0e-6):
    eta1sq = np.square(eta1)
    eta2sq = np.square(eta2)
    k1 = 0.5 * eta2sq * np.square(1.0 + kdot(X, Z))
    k2 = -0.5 * eta2sq * kdot(np.square(X), np.square(Z))
    k3 = (eta1sq - eta2sq) * kdot(X, Z)
    k4 = np.square(c) - 0.5 * eta2sq
    if X.shape == Z.shape:
        k4 += jitter * np.eye(X.shape[0])
    return k1 + k2 + k3 + k4

def kernel_approx(X, Z, eta1, eta2, c, jitter=1.0e-6, rank=0):
    eta1sq = onp.square(eta1)
    eta2sq = onp.square(eta2)
    k3 = (eta1sq - eta2sq) * kdot(X[:, :rank], Z[:, :rank])
    k4 = onp.square(c) - 0.5 * eta2sq + jitter * onp.eye(X.shape[0])
    return k3 + k4


#@partial(jit, static_argnums=(5,7,8))
def lowrank_presolve(kX, D, eta1, eta2, c, kappa, rank1, rank2):
    N, P = kX.shape
    all_ones = np.ones((N, 1))
    kappa_indices = np.argsort(kappa)

    top_features = dynamic_slice_in_dim(kappa_indices, P - rank1, rank1)
    kX_top = np.take(kX, top_features, -1)

    if rank2 > 0:
        top_features2 = dynamic_slice_in_dim(kappa_indices, P - rank2, rank2)
        kX_top2 = np.take(kX, top_features2, -1)  # N rank2
        kX_top2 = kX_top2[:, None, :] * kX_top2[:, :, None] # N rank2 rank2
        lower_diag = np.ravel(np.arange(rank2) < np.arange(rank2)[:, None])
        kX_top2 = np.compress(lower_diag, kX_top2.reshape((N, -1)), axis=-1)

        Z = np.concatenate([eta2 * kX_top2, eta1 * kX_top, c * all_ones], axis=1)
    else:
        Z = np.concatenate([eta1 * kX_top, c * all_ones], axis=1)

    ZD = Z / D[:, None]
    ZDZ = np.eye(ZD.shape[-1]) + np.matmul(np.transpose(Z), ZD)
    L = cho_factor(ZDZ, lower=True)[0]
    return lambda b: b / D - np.matmul(ZD, cho_solve((L, True), np.matmul(np.transpose(ZD), b)))


def lowrank_presolve_linear(kX, D, eta1, c, kappa, rank):
    N, P = kX.shape
    all_ones = np.ones((N, 1))
    kappa_indices = np.argsort(kappa)

    top_features = dynamic_slice_in_dim(kappa_indices, P - rank, rank)
    kX_top = np.take(kX, top_features, -1)
    Z = np.concatenate([eta1 * kX_top, c * all_ones], axis=1)

    ZD = Z / D[:, None]
    ZDZ = np.eye(ZD.shape[-1]) + np.matmul(np.transpose(Z), ZD)
    L = cho_factor(ZDZ, lower=True)[0]
    return lambda b: b / D - np.matmul(ZD, cho_solve((L, True), np.matmul(np.transpose(ZD), b)))


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


def cg_cond_fun(state, cg_tol=1.0e-14, max_iters=100):
    return (np.sqrt(state.r_dot_r) > cg_tol) & (state.iter < max_iters)

def pcg_cond_fun(state, cg_tol=1.0e-14, max_iters=100):
    return (np.linalg.norm(state.r) > cg_tol) & (state.iter < max_iters)


def cg(b, A, cg_tol=1.0e-4, max_iters=50):
    mvm = lambda rhs: np.matmul(A, rhs)
    cond_fun = lambda state: cg_cond_fun(state, cg_tol=cg_tol, max_iters=max_iters)
    body_fun = lambda state: cg_body_fun(state, mvm=mvm)
    init_state = CGState(np.zeros(b.shape[-1]), b, b, np.dot(b, b), 0)
    final_state = while_loop(cond_fun, body_fun, init_state)
    return final_state.x, np.sqrt(final_state.r_dot_r), final_state.iter


def pcg(b, mvm, presolve, cg_tol=1.0e-4, max_iters=4):
    #mvm = lambda rhs: np.matmul(A, rhs)
    cond_fun = lambda state: pcg_cond_fun(state, cg_tol=cg_tol, max_iters=max_iters)
    body_fun = lambda state: pcg_body_fun(state, mvm=mvm, presolve=presolve)
    z = presolve(b)
    init_state = PCGState(np.zeros(b.shape[-1]), b, z, z, np.dot(b, z), 0)
    final_state = while_loop(cond_fun, body_fun, init_state)
    return final_state.x, np.linalg.norm(final_state.r), final_state.iter


def cg_batch_b(b, A, cg_tol=1.0e-14, max_iters=100):
    return vmap(lambda _b: cg(_b, A, cg_tol=cg_tol, max_iters=max_iters))(b)

def cg_batch_bA(b, A, cg_tol=1.0e-14, max_iters=100):
    return vmap(lambda _b, _A: cg(_b, _A, cg_tol=cg_tol, max_iters=max_iters))(b, A)

def pcg_batch_b(b, mvm, presolve=lambda rhs: rhs, cg_tol=1.0e-14, max_iters=8):
    return vmap(lambda _b: pcg(_b, mvm, presolve=presolve, cg_tol=cg_tol, max_iters=max_iters))(b)


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
def cg_quad_form_log_det(A, b, probes, cg_tol=1.0e-5, max_iters=100):
    return (np.nan, np.nan, np.nan)

@cg_quad_form_log_det.defjvp
def cg_quad_form_log_det_jvp(primals, tangents):
    A, b, probes, cg_tol, max_iters = primals
    A_dot, b_dot, _, _, _ = tangents
    D = b.shape[-1]

    b_probes = np.concatenate([b[None, :], probes])
    Ainv_b_probes, res_norm, iters = cg_batch_b(b_probes, A, cg_tol=cg_tol, max_iters=max_iters)
    Ainv_b, Ainv_probes = Ainv_b_probes[0], Ainv_b_probes[1:]

    quad_form_dA = -np.dot(Ainv_b, np.matmul(A_dot, Ainv_b))
    quad_form_db = 2.0 * np.dot(Ainv_b, b_dot)
    log_det_dA = np.mean(np.einsum('...i,...i->...', np.matmul(probes, A_dot), Ainv_probes))
    tangent_out = log_det_dA + quad_form_dA + quad_form_db
    quad_form = np.dot(b, Ainv_b)

    return (quad_form, np.mean(res_norm), np.mean(iters)), (tangent_out, 0.0, 0.0)


# compute logdet A + b A^{-1} b
@custom_jvp
def pcg_quad_form_log_det(A, b, probes, cg_tol=1.0e-5, max_iters=20):
    return np.nan

@pcg_quad_form_log_det.defjvp
def pcg_quad_form_log_det_jvp(primals, tangents):
    A, b, probes, cg_tol, max_iters = primals
    A_dot, b_dot, _, _, _, _ = tangents
    D = b.shape[-1]

    b_probes = np.concatenate([b[None, :], probes])
    Ainv_b_probes, res_norm, iters = pcg_batch_b(b_probes, A, max_iters=max_iters)
    Ainv_b, Ainv_probes = Ainv_b_probes[0], Ainv_b_probes[1:]

    quad_form_dA = -np.dot(Ainv_b, np.matmul(A_dot, Ainv_b))
    quad_form_db = 2.0 * np.dot(Ainv_b, b_dot)
    log_det_dA = np.mean(np.einsum('...i,...i->...', np.matmul(probes, A_dot), Ainv_probes))
    tangent_out = log_det_dA + quad_form_dA + quad_form_db
    quad_form = np.dot(b, Ainv_b)

    return quad_form, tangent_out

# compute logdet A + b A^{-1} b
@partial(custom_jvp, nondiff_argnums=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
def cpcg_quad_form_log_det(A, b, eta1, eta2, diag, c, kX, kappa, probes, rank1, rank2, cg_tol, max_iters):
    return (np.nan, np.nan, np.nan)

@cpcg_quad_form_log_det.defjvp
def cpcg_quad_form_log_det_jvp(eta1, eta2, diag, c, kX, kappa, probes, rank1, rank2, cg_tol, max_iters, primals, tangents):
    A, b = primals
    A_dot, b_dot = tangents
    D = b.shape[-1]

    presolve = lowrank_presolve(kX, diag, eta1, eta2, c, kappa, rank1, rank2)
    mvm = lambda b: np.matmul(A, b)

    b_probes = np.concatenate([b[None, :], probes])
    Ainv_b_probes, res_norm, iters = pcg_batch_b(b_probes, mvm, presolve=presolve, cg_tol=cg_tol, max_iters=max_iters)
    Ainv_b, Ainv_probes = Ainv_b_probes[0], Ainv_b_probes[1:]

    quad_form_dA = -np.dot(Ainv_b, np.matmul(A_dot, Ainv_b))
    quad_form_db = 2.0 * np.dot(Ainv_b, b_dot)
    log_det_dA = np.mean(np.einsum('...i,...i->...', np.matmul(probes, A_dot), Ainv_probes))
    tangent_out = log_det_dA + quad_form_dA + quad_form_db
    quad_form = np.dot(b, Ainv_b)

    return (quad_form, np.mean(res_norm), np.mean(iters)), (tangent_out, 0.0, 0.0)


def symmetrize(x):
    return 0.5 * (x + np.transpose(x))


if __name__ == "__main__":
    from numpyro.util import enable_x64
    #enable_x64()
    numpyro.set_platform("gpu")

    onp.random.seed(0)
    N = 40 * 1000
    P = 500
    b = onp.random.randn(N)
    X = onp.random.randn(N * P).reshape((N, P))
    kappa = 2.0 * np.exp(0.2 * onp.random.randn(P))

    eta1 = np.array(0.5)
    eta2 = np.array(0.33)
    c = 1.0
    diag = np.exp(0.2 * onp.random.randn(N))
    num_probes = 1#10 ** 5
    probes = onp.random.randn(num_probes * N).reshape((num_probes, N))

    def f1(_kappa, _b, _eta1, _eta2, _diag):
        return pcpcg_quad_form_log_det(_kappa, _b, _eta1, _eta2, _diag, c, X, probes, 3, 2,
                                       1.0e-8, 10, 16)[0]

    def f2(_kappa, _b, _eta1, _eta2, _diag):
        kX = _kappa * X
        k = kernel(kX, kX, _eta1, _eta2, c) + np.diag(_diag)
        return direct_quad_form_log_det(k, _b)

    def f3(_kappa, _b, _eta1, _eta2, _diag):
        kX = _kappa * X
        k = kernel(kX, kX, _eta1, _eta2, c) + np.diag(_diag)
        return cg_quad_form_log_det(k, _b, probes, cg_tol=1.0e-8, max_iters=300)[0]

    which = 0

    t0 = time.time()
    g1 = grad(f1, which)(kappa, b, eta1, eta2, diag)
    print("ELAPSED", time.time()-t0)
    import sys;sys.exit()


    g2 = grad(f2, which)(kappa, b, eta1, eta2, diag)
    g3 = grad(f3, which)(kappa, b, eta1, eta2, diag)

    assert_allclose(g3, g2, atol=1.0e-2, rtol=1.0e-2)
    print("passed cg = direct")
    assert_allclose(g1, g2, atol=1.0e-2, rtol=1.0e-2)
    print("passed pcpcg = direct")
    assert_allclose(g1, g3, atol=0.0, rtol=1.0e-30)
    print("passed pcpcg = cg")

    import sys; sys.exit()

    presolve = lambda b: lowrank_presolve(b, X, diag * onp.ones(N), np.square(eta1), np.square(eta2), c, rank=rank)
    k2 = kernel_approx(X, X, eta1, eta2, c, jitter=diag, rank=rank)
    presolve_b = presolve(b)
    L = cho_factor(k2, lower=True)[0]
    k2_b = cho_solve((L, True), b)
    assert_allclose(k2_b, presolve_b, atol=3.0e-3, rtol=1.0e-3)



    import sys; sys.exit()


    t0 = time.time()
    num_trials = 10
    for trial in range(num_trials):
        rng_key = jax.random.PRNGKey(trial)
        onp.random.seed(trial)

        N = 8000
        D = 500
        K = 1000
        b = onp.random.randn(N)
        X = onp.random.randn(N * D).reshape((N, D)) / onp.sqrt(D)
        X[:, 10:] *= 0.01
        transform = create_sketch_transform(rng_key, D, K)
        lowrank = sketch_transform(X, transform)

        sigmasq = 0.1
        kernel = onp.square(np.matmul(X, np.transpose(X))) + sigmasq * onp.eye(N)
        approx_kernel = onp.matmul(lowrank, onp.transpose(lowrank)) + sigmasq * onp.eye(N)

        probes = onp.random.randn(8 * N).reshape((8, N))
        value_and_grad(pcg_quad_form_log_det, 1)(kernel, b, probes)
    t1= time.time()
    print("time per comp", (t1-t0)/num_trials)



    import sys; sys.exit()


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
