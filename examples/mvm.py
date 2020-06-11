from jax import vmap, jit
import jax.numpy as np
import numpyro
from numpy.testing import assert_allclose
import time
import numpy as onp
from utils import kdot


numpyro.set_platform("gpu")

N = 9 * 10 ** 3
P = 1000
b = np.sin(np.ones(N)) / N

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

if __name__ == "__main__":
    dkX = np.array(onp.random.randn(N * P).reshape((N, P)))
    kX = np.array(onp.random.randn(N * P).reshape((N, P)))

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

if __name__ == "__main__":
    dkX = np.array(onp.random.randn(N * P).reshape((N, P)))
    kX = np.array(onp.random.randn(N * P).reshape((N, P)))

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
