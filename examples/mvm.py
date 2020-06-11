from jax import vmap, jit
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


# np.square(1.0 + kdot(kX, kX))
def kXkXsq_row(i, kX):
    return np.square(1.0 + np.matmul(kX, kX[i]))

# kdot(kX, kX) * kdot(kX, dkX)
def kXdkXsq_row(i, kX, dkX):
    return np.matmul(kX, kX[i]) * np.matmul(kX, dkX[i])

def kXkXsq_mvm(b, kX, dilation=2):
    return partitioned_mvm(lambda i: kXkXsq_row(i, kX), dilation)(b)

def kXdkXsq_mvm(b, kX, dkX, dilation=2):
    return partitioned_mvm(lambda i: kXdkXsq_row(i, kX, dkX), dilation)(b)

def quad_mvm(b, X):
    return np.einsum('np,p->n', X, np.einsum('np,n->p', X, b))

def kernel(X, Z, eta1, eta2, c):
    eta1sq = np.square(eta1)
    eta2sq = np.square(eta2)
    k1 = 0.5 * eta2sq * np.square(1.0 + kdot(X, Z))
    k2 = -0.5 * eta2sq * kdot(np.square(X), np.square(Z))
    k3 = (eta1sq - eta2sq) * kdot(X, Z)
    k4 = np.square(c) - 0.5 * eta2sq
    return k1 + k2 + k3 + k4

def kernel_mvm(b, kX, eta1, eta2, c, diag, dilation=2):
    eta1sq = np.square(eta1)
    eta2sq = np.square(eta2)
    k1b = 0.5 * eta2sq * kXkXsq_mvm(b, kX, dilation=dilation)
    k2b = -0.5 * eta2sq * quad_mvm(b, np.square(kX))
    k3b = (eta1sq - eta2sq) * quad_mvm(b, kX)
    k4b = (np.square(c) - 0.5 * eta2sq) * np.sum(b) * np.ones(b.shape)
    return k1b + k2b + k3b + k4b + diag * b


if __name__ == "__main__":
    numpyro.set_platform("gpu")

    N = 9 * 10 ** 3
    P = 1000
    b = np.sin(np.ones(N)) / N

    dkX = np.array(onp.random.randn(N * P).reshape((N, P)))
    kX = np.array(onp.random.randn(N * P).reshape((N, P)))

    eta1 = 0.55
    eta2 = 0.22
    c = 0.9

    kb1 = np.matmul(kernel(kX, kX, eta1, eta2, c), b)
    kb2 = kernel_mvm(b, kX, eta1, eta2, c, dilation=2)
    kb3 = kernel_mvm(b, kX, eta1, eta2, c, dilation=3)
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
