import jax
from jax import vmap
import jax.numpy as np
import numpy as onp


def get_chunks(L, chunk_size):
    num_chunks = L // chunk_size
    chunks = [np.arange(i * chunk_size, (i + 1) * chunk_size) for i in range(num_chunks)]
    if L % chunk_size != 0:
        chunks.append(np.arange(L - L % chunk_size, L))
    return chunks, len(chunks)


def chunk_vmap(fun, array, chunk_size=10):
    L = array.shape[0]
    if chunk_size >= L:
        return vmap(fun)(array)
    chunks, num_chunks = get_chunks(L, chunk_size)
    results = [vmap(fun)(array[chunk]) for chunk in chunks]
    num_return = len(results[0])
    return tuple([np.concatenate([res[i] for res in results]) for i in range(num_return)])


### simple test ###
if __name__ == '__main__':
    from numpy.testing import assert_allclose

    def f(x):
        return x * x, x * x * x

    expected = vmap(f)(np.arange(5))

    assert_allclose(expected, chunk_vmap(f, np.arange(5), 1))
    assert_allclose(expected, chunk_vmap(f, np.arange(5), 2))
    assert_allclose(expected, chunk_vmap(f, np.arange(5), 3))
    assert_allclose(expected, chunk_vmap(f, np.arange(5), 5))
