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


def slice_tuple(array, index):
    return tuple([a[index] for a in array])


def chunk_vmap(fun, array, chunk_size=10):
    if isinstance(array, tuple):
        L = array[0].shape[0]
    else:
        L = array.shape[0]
    if chunk_size >= L and isinstance(array, tuple):
        return vmap(fun)(*array)
    elif chunk_size >= L:
        return vmap(fun)(array)
    chunks, num_chunks = get_chunks(L, chunk_size)
    if isinstance(array, tuple):
        results = [vmap(fun)(*slice_tuple(array, chunk)) for chunk in chunks]
    else:
        results = [vmap(fun)(array[chunk]) for chunk in chunks]
    num_return = len(results[0])
    #if results[0].ndim == 1:
    #    return np.concatenate(results)
    return tuple([np.concatenate([res[i] for res in results]) for i in range(num_return)])


def safe_chunk_vmap(fun, array, chunk_size=10):
    try:
        return chunk_vmap(fun, array, chunk_size)
    except:
        return safe_chunk_vmap(fun, array, chunk_size // 2)


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

    def f(x, y):
        return x * y, x * x * y

    expected = vmap(f)(np.arange(5), np.arange(5) + 1)

    assert_allclose(expected, chunk_vmap(f, (np.arange(5), np.arange(5) + 1), 1))
    assert_allclose(expected, chunk_vmap(f, (np.arange(5), np.arange(5) + 1), 2))
    assert_allclose(expected, chunk_vmap(f, (np.arange(5), np.arange(5) + 1), 3))
    assert_allclose(expected, chunk_vmap(f, (np.arange(5), np.arange(5) + 1), 5))
