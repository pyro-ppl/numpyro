from functools import partial
from jax import vmap, jit, custom_jvp, value_and_grad, jvp, grad
import jax.numpy as np
import numpyro
from numpy.testing import assert_allclose
import time
import numpy as onp
from utils import kdot


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

# do a matrix vector multiply chunk-by-chunk
def partitioned_mvm_size(row, size, dilation):
    def do_mvm(rhs):
        @jit
        def compute_element(i):
            return np.dot(rhs, row(i))
        return _chunk_vmap(compute_element, np.arange(size), size // dilation)
    return do_mvm

# np.square(1.0 + kdot(kX, kX))
def kXkXsq_row_sub(i, kX_sub, kX):
    return np.square(1.0 + np.matmul(kX, kX_sub[i]))
def kXkXsq_mvm_sub(b, kX_sub, kX, dilation=2):
    return partitioned_mvm_size(lambda i: kXkXsq_row_sub(i, kX_sub, kX), kX_sub.shape[0], dilation)(b)

# np.square(1.0 + kdot(kX, kX))
def kXkXsq_row(i, kX):
    return np.square(1.0 + np.matmul(kX, kX[i]))
def kXkXsq_mvm(b, kX, dilation=2):
    return partitioned_mvm(lambda i: kXkXsq_row(i, kX), dilation)(b)

# kdot(kX, kX) * kdot(kX, dkX)
def kXdkXsq_row(i, kX, dkX):
    return np.matmul(kX, kX[i]) * np.matmul(kX, dkX[i])
def kXdkXsq_mvm(b, kX, dkX, dilation=2):
    return partitioned_mvm(lambda i: kXdkXsq_row(i, kX, dkX), dilation)(b)

def kXdkXsq_row_sub(i, kX_sub, dkX_sub, kX, dkX):
    return np.matmul(kX, kX_sub[i]) * np.matmul(kX, dkX_sub[i])
def kXdkXsq_mvm_sub(b, kX_sub, dkX_sub, kX, dkX, dilation=2):
    return partitioned_mvm_size(lambda i: kXdkXsq_row_sub(i, kX_sub, dkX_sub, kX, dkX), kX_sub.shape[0], dilation)(b)

def partitioned_mvm2(row, P, dilation):
    def do_mvm(rhs):
        @jit
        def compute_element(i):
            return np.dot(rhs, row(i))
        return _chunk_vmap(compute_element, np.arange(P), P // dilation)
    return do_mvm

def kX_mvm(b, kX, dilation=2):
    return np.transpose(partitioned_mvm2(lambda i: kX[:, i], kX.shape[-1], dilation)(b))

def quad_mvm(b, X):
    return np.einsum('np,p->n', X, np.einsum('np,n->p', X, b))

def quad_mvm_dil(b, X, dilation=2):
    N, P = X.shape
    @jit
    def compute_element1(i):
        return np.dot(X[:, i], b)
    partial = _chunk_vmap(compute_element1, np.arange(P), P // dilation)
    @jit
    def compute_element2(i):
        return np.dot(X[i], partial)
    return _chunk_vmap(compute_element2, np.arange(N), N // dilation)

def kernel(X, Z, eta1, eta2, c):
    eta1sq = np.square(eta1)
    eta2sq = np.square(eta2)
    k1 = 0.5 * eta2sq * np.square(1.0 + kdot(X, Z))
    k2 = -0.5 * eta2sq * kdot(np.square(X), np.square(Z))
    k3 = (eta1sq - eta2sq) * kdot(X, Z)
    k4 = np.square(c) - 0.5 * eta2sq
    return k1 + k2 + k3 + k4

def kernel_mvm_diag(b, kX, eta1, eta2, c, diag, dilation=2,dilation2=2):
    eta1sq = np.square(eta1)
    eta2sq = np.square(eta2)
    k1b = 0.5 * eta2sq * kXkXsq_mvm(b, kX, dilation=dilation)
    k2b = -0.5 * eta2sq * quad_mvm_dil(b, np.square(kX), dilation=dilation)
    k3b = (eta1sq - eta2sq) * quad_mvm_dil(b, kX, dilation=dilation)
    k4b = (np.square(c) - 0.5 * eta2sq) * np.sum(b) * np.ones(b.shape)
    return k1b + k2b + k3b + k4b + diag * b

def kernel_mvm(b, kX, eta1, eta2, c, dilation=2,dilation2=2):
    eta1sq = np.square(eta1)
    eta2sq = np.square(eta2)
    k1b = 0.5 * eta2sq * kXkXsq_mvm(b, kX, dilation=dilation)
    k2b = -0.5 * eta2sq * quad_mvm_dil(b, np.square(kX), dilation=dilation)
    k3b = (eta1sq - eta2sq) * quad_mvm_dil(b, kX, dilation=dilation)
    k4b = (np.square(c) - 0.5 * eta2sq) * np.sum(b) * np.ones(b.shape)
    return k1b + k2b + k3b + k4b



if __name__ == "__main__":
    numpyro.set_platform("gpu")

    onp.random.seed(0)

    N = 7
    P = 4
    b = np.sin(np.ones(N)) / N
    a = np.cos(np.ones(N)) / N

    kX = np.array(onp.random.randn(N * P).reshape((N, P)))
    X = np.array(onp.random.randn(N * P).reshape((N, P)))

    #p1 = np.array(onp.random.rand(N*N+N).reshape((N+1,N)))
    #p2 = np.array(onp.random.rand(N*N+N).reshape((N+1,N)))
    #res1 = kXkXkXX_qf2(p1, p2, X, kX, dilation=2)
    #res2 = kXkXkXX_qf3(p1, p2, X, kX, dilation=2)
    #print("res1",res1)
    #print("res2",res1)
    #assert_allclose(res1, res2, atol=1.0e-5, rtol=1.0e-5)
    #import sys; sys.exit()

    eta1 = np.array(0.55)
    eta2 = np.array(0.22)
    c = 0.9

    kappa = np.array(onp.random.rand(P))

    b = np.sin(np.ones(N) / N)

    kXt = np.transpose(kX)
    Xt = np.transpose(X)
    kXkX = np.matmul(kX, kXt)
    kXkX_j = kXkX * kXt[:, None, :] * Xt[:, :, None]
    res1 = np.sum(np.sum(kXkX_j * b * b[:, None], axis=-1), axis=-1)
    print("res1", res1.shape, res1)
    res2 = kXkXkXX_qf(b, X, kX, dilation=2)
    print("res2", res2.shape, res2)
    import sys;sys.exit()

    res1 = np.matmul(np.transpose(X), np.transpose(b))  # N
    res2 = quad_mvm_dil(b, kX, dilation=2)
    assert_allclose(res1, res2, atol=1.0e-5, rtol=1.0e-8)

    import sys;sys.exit()

    def f(kappa, eta1, eta2):
        kX = kappa * X
        return np.matmul(kernel(kX, kX, eta1, eta2, c), b)

    def g(kappa, eta1, eta2):
        return kernel_mvm(b, kappa, X, np.broadcast_to(eta1, b.shape), np.broadcast_to(eta2, b.shape), c, 2)

    _, t1 = jvp(f, (kappa, eta1, eta2), (1.4 * kappa, 0.1, .2))
    _, t2 = jvp(g, (kappa, eta1, eta2), (1.4 * kappa, 0.1, .2))
    assert_allclose(t1, t2, atol=1.0e-5, rtol=1.0e-5)
    _, t1 = jvp(f, (kappa, eta1, eta2), (1.4 / kappa, 0.3, .4))
    _, t2 = jvp(g, (kappa, eta1, eta2), (1.4 / kappa, 0.3, .4))
    assert_allclose(t1, t2, atol=1.0e-5, rtol=1.0e-5)

    def f(_kappa, _eta1, _eta2):
        kX = _kappa * X
        return np.dot(a, np.matmul(kernel(kX, kX, _eta1, _eta2, c), b))

    def g(_kappa, _eta1, _eta2):
        kb = kernel_mvm(b, _kappa, X, np.broadcast_to(_eta1, b.shape), np.broadcast_to(_eta2, b.shape), c, 2)
        return np.dot(a, kb)

    _, t1 = jvp(f, (kappa, eta1, eta2), (1.4 * kappa, 0.1, .2))
    _, t2 = jvp(g, (kappa, eta1, eta2), (1.4 * kappa, 0.1, .2))
    assert_allclose(t1, t2, atol=1.0e-5, rtol=1.0e-5)

    g1 = grad(g, 0)(kappa, eta1, eta2)
    g2 = grad(f, 0)(kappa, eta1, eta2)
    assert_allclose(g1, g2, atol=1.0e-5, rtol=1.0e-5)

    g1 = grad(g, 1)(kappa, eta1, eta2)
    g2 = grad(f, 1)(kappa, eta1, eta2)
    assert_allclose(g1, g2, atol=1.0e-5, rtol=1.0e-5)

    g1 = grad(g, 2)(kappa, eta1, eta2)
    g2 = grad(f, 2)(kappa, eta1, eta2)
    assert_allclose(g1, g2, atol=1.0e-30, rtol=1.0e-30)

    import sys; sys.exit()

    v1, g1 = value_and_grad(g, 1)(kappa, eta1, eta2)
    v2, g2 = value_and_grad(f, 1)(kappa, eta1, eta2)
    assert_allclose(v1, v2, atol=1.0e-5, rtol=1.0e-5)
    assert_allclose(g1, g2, atol=1.0e-30, rtol=1.0e-30)

    v1, g1 = value_and_grad(g, 2)(kappa, eta1, eta2)
    v2, g2 = value_and_grad(f, 2)(kappa, eta1, eta2)
    assert_allclose(g1, g2, atol=1.0e-5, rtol=1.0e-5)

    import sys; sys.exit()

    v1, g1 = value_and_grad(kernel_quad, 1)(b, kappa, X, eta1, eta2, c, 2)
    v2, g2 = value_and_grad(f, 0)(kappa, eta1, eta2)
    assert_allclose(g1, g2, atol=1.0e-5, rtol=1.0e-5)

    import sys; sys.exit()

    res1 = quad_mvm(b, kX)
    res2 = quad_mvm_dil(b, kX, dilation=2)
    assert_allclose(res1, res2, atol=1.0e-5, rtol=1.0e-8)

    import sys; sys.exit()

    b2 = np.array(onp.random.randn(3,N))  # 3 N
    res1 = kX_mvm(b2, kX)   # 3 N
    res2 = np.matmul(b2, kX)              # 3 P
    assert_allclose(res1, res2, atol=1.0e-5, rtol=1.0e-5)

    import sys; sys.exit()

    kb1 = np.matmul(kernel(kX, kX, eta1, eta2, c), b)
    kb2 = kernel_mvm_diag(b, kX, eta1, eta2, c, dilation=2)
    kb3 = kernel_mvm_diag(b, kX, eta1, eta2, c, dilation=3)
    assert_allclose(kb1, kb2, atol=1.0e-5, rtol=1.0e-5)
    print("kb1 == kb2")
    assert_allclose(kb1, kb3, atol=1.0e-5, rtol=1.0e-5)
    print("kb1 == kb3")

    import sys; sys.exit()

    res1 = partitioned_mvm(lambda i: kXkXsq_row(i, kX), 8)(b)
    res2 = partitioned_mvm(lambda i: kXdkXsq_row(i, kX, dkX), 8)(b)

    if N < 10**4:
        res1v = vanilla_mvm(lambda i: kXkXsq_row(i, kX))(b)
        res1vv = np.matmul(np.square(1.0 + kdot(kX, kX)), b)
        assert_allclose(res1, res1v, atol=1.0e-3, rtol=1.0e-3)
        print("res1 == res1v")
        assert_allclose(res1, res1vv, atol=1.0e-3, rtol=1.0e-3)
        print("res1 == res1vv")

        res2v = vanilla_mvm(lambda i: kXdkXsq_row(i, kX, dkX))(b)
        res2vv = np.matmul(kdot(kX, kX) * kdot(dkX, kX), b)
        assert_allclose(res2, res2v, atol=1.0e-3, rtol=1.0e-3)
        print("res2 == res2v")
        assert_allclose(res2, res2vv, atol=1.0e-3, rtol=1.0e-3)
        print("res2 == res2vv")
