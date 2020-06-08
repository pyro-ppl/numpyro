from functools import namedtuple
import numpy as onp
import jax
from jax import vmap, jit, custom_jvp, value_and_grad
from jax.lax import while_loop, dynamic_slice_in_dim
import jax.numpy as np
import time
from jax.scipy.linalg import cho_factor, solve_triangular, cho_solve
from numpy.testing import assert_allclose
from tensor_sketch import create_sketch_transform, sketch_transform


CGState = namedtuple('CGState', ['x', 'r', 'p', 'r_dot_r', 'iter'])
PCGState = namedtuple('CGState', ['x', 'r', 'p', 'z', 'r_dot_z', 'iter'])


def kdot(X, Z):
    return onp.dot(X, Z[..., None])[..., 0]


def kernel(X, Z, eta1, eta2, c, jitter=1.0e-6):
    eta1sq = onp.square(eta1)
    eta2sq = onp.square(eta2)
    k1 = 0.5 * eta2sq * onp.square(1.0 + kdot(X, Z))
    k2 = -0.5 * eta2sq * kdot(onp.square(X), onp.square(Z))
    k3 = (eta1sq - eta2sq) * kdot(X, Z)
    k4 = onp.square(c) - 0.5 * eta2sq
    if X.shape == Z.shape:
        k4 += jitter * onp.eye(X.shape[0])
    return k1 + k2 + k3 + k4

def kernel_approx(X, Z, eta1, eta2, c, jitter=1.0e-6, rank=0):
    eta1sq = onp.square(eta1)
    eta2sq = onp.square(eta2)
    k3 = (eta1sq - eta2sq) * kdot(X[:, :rank], Z[:, :rank])
    k4 = onp.square(c) - 0.5 * eta2sq + jitter * onp.eye(X.shape[0])
    return k3 + k4

#def quad_mvm(b, X):
#    return np.einsum('np,p->n', X, np.einsum('np,n->p', X, b))


def lowrank_presolve(b, X, D, eta1, eta2, c, kappa, rank=0):
    P = X.shape[-1]
    all_ones = np.ones((b.shape[-1], 1))
    top_features = dynamic_slice_in_dim(np.argsort(kappa), P - rank, P)
    X_top = np.take(X, top_features, -1)
    eta12 = np.sqrt(np.square(eta1) - np.square(eta2))
    Z = np.concatenate([eta12 * X_top, c * all_ones], axis=1)
    ZD = Z / D[:, None]
    ZDZ = np.eye(top_features.shape[-1] + 1) + np.matmul(np.transpose(Z), ZD)
    L = cho_factor(ZDZ, lower=True)[0]
    return b / D - np.matmul(ZD, cho_solve((L, True), np.matmul(np.transpose(ZD), b)))


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


def cg(b, A, epsilon=1.0e-4, max_iters=50):
    mvm = lambda rhs: np.matmul(A, rhs)
    cond_fun = lambda state: cg_cond_fun(state, epsilon=epsilon, max_iters=max_iters)
    body_fun = lambda state: cg_body_fun(state, mvm=mvm)
    init_state = CGState(np.zeros(b.shape[-1]), b, b, np.dot(b, b), 0)
    final_state = while_loop(cond_fun, body_fun, init_state)
    return final_state.x, np.sqrt(final_state.r_dot_r), final_state.iter


def pcg(b, A, presolve, epsilon=1.0e-4, max_iters=4):
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

def pcg_batch_b(b, A, presolve=lambda rhs: rhs, epsilon=1.0e-14, max_iters=8):
    return vmap(lambda _b: pcg(_b, A, presolve=presolve, epsilon=epsilon, max_iters=max_iters))(b)


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
def cg_quad_form_log_det(A, b, probes, epsilon=1.0e-5, max_iters=100):
    return (np.nan, np.nan, np.nan)

@cg_quad_form_log_det.defjvp
def cg_quad_form_log_det_jvp(primals, tangents):
    A, b, probes, epsilon, max_iters = primals
    A_dot, b_dot, _, _, _ = tangents
    D = b.shape[-1]

    b_probes = np.concatenate([b[None, :], probes])
    Ainv_b_probes, res_norm, iters = cg_batch_b(b_probes, A, epsilon=epsilon, max_iters=max_iters)
    Ainv_b, Ainv_probes = Ainv_b_probes[0], Ainv_b_probes[1:]

    quad_form_dA = -np.dot(Ainv_b, np.matmul(A_dot, Ainv_b))
    quad_form_db = 2.0 * np.dot(Ainv_b, b_dot)
    log_det_dA = np.mean(np.einsum('...i,...i->...', np.matmul(probes, A_dot), Ainv_probes))
    tangent_out = log_det_dA + quad_form_dA + quad_form_db
    quad_form = np.dot(b, Ainv_b)

    return (quad_form, np.mean(res_norm), np.mean(iters)), (tangent_out, 0.0, 0.0)


# compute logdet A + b A^{-1} b
@custom_jvp
def pcg_quad_form_log_det(A, b, probes, epsilon=1.0e-5, max_iters=20):
    return np.nan

@pcg_quad_form_log_det.defjvp
def pcg_quad_form_log_det_jvp(primals, tangents):
    A, b, probes, epsilon, max_iters = primals
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
    #print("np.mean(res_norm), np.mean(iters)",np.mean(res_norm), np.mean(iters))
    #return (quad_form, np.mean(res_norm), np.mean(iters)), (tangent_out, 0.0, 0.0)

# compute logdet A + b A^{-1} b
@custom_jvp
def cpcg_quad_form_log_det(A, b, eta1, eta2, c, X, D, kappa, probes, rank=0, epsilon=1.0e-5, max_iters=20):
    return (np.nan, np.nan, np.nan)

@cpcg_quad_form_log_det.defjvp
def cpcg_quad_form_log_det_jvp(primals, tangents):
    A, b, eta1, eta2, c, X, diag, kappa, probes, rank, epsilon, max_iters = primals
    A_dot, b_dot, _, _, _, _, _, _, _, _, _, _ = tangents
    D = b.shape[-1]

    presolve = lambda b: lowrank_presolve(b, X, diag, eta1, eta2, c, kappa, rank=rank)

    b_probes = np.concatenate([b[None, :], probes])
    Ainv_b_probes, res_norm, iters = pcg_batch_b(b_probes, A, presolve=presolve, epsilon=epsilon, max_iters=max_iters)
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
    onp.random.seed(0)
    N = 1000
    D = 200
    b = onp.random.randn(N)
    X = onp.random.randn(N * D).reshape((N, D))
    X[:, 20:] *= 0.01
    X[:, 10:20] *= 0.3

    eta1 = 0.5
    eta2 = 0.05
    c = 1.0
    diag = 10.0
    k = kernel(X, X, eta1, eta2, c, jitter=diag)
    probes = onp.random.randn(8 * N).reshape((8, N))

    rank = 30
    max_iters = 5

    def f(_k, _b):
        return cpcg_quad_form_log_det(_k, _b, eta1, eta2, c, X, diag * onp.ones(N), probes, rank=rank,
                                      epsilon=1.0e-5, max_iters=max_iters)
    value_and_grad(f, 1)(k, b)

    def f(_k, _b):
        return cg_quad_form_log_det(_k, _b, probes, epsilon=1.0e-5, max_iters=max_iters)[0]
    value_and_grad(f, 1)(k, b)

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
