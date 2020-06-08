import numpy as onp
import jax
import jax.random as random
import jax.numpy as np
from jax.numpy.fft import fft, ifft
from numpy.testing import assert_allclose


#def woodruff_presolve(lowrank, rhs):
#    sigmasq = 0.1
#    M = sigmasq * np.eye(lowrank.shape[-1]) + np.matmul(np.transpose(lowrank), lowrank)
#    L = cho_factor(M, lower=True)[0]
#    def presolve(rhs):
#        lowrank_rhs = np.matmul(np.transpose(lowrank), rhs)
#        lowrank_rhs = cho_solve((L, True), lowrank_rhs)
#        return (1.0 / sigmasq) * (rhs - np.matmul(lowrank, lowrank_rhs))



def create_sketch_transform(rng_key, D, K):
    hash_indices = random.categorical(rng_key, np.zeros(K), shape=(2, D))
    hash_signs = 2 * random.bernoulli(rng_key, shape=(2, D)) - 1

    transform = np.zeros((2, D, K))
    transform = jax.ops.index_update(transform, (0, np.arange(D), hash_indices[0]), hash_signs[0])
    transform = jax.ops.index_update(transform, (1, np.arange(D), hash_indices[1]), hash_signs[1])

    return transform


def sketch_transform(X, transform):
    X = np.matmul(X, transform)
    X = np.prod(fft(X, axis=-1), axis=0)
    return np.real(ifft(X))


if __name__ == '__main__':
    for trial in range(3):
        rng_key = random.PRNGKey(trial)
        N = 2500
        D = 3
        K = 250
        X = onp.random.randn(N * D).reshape((N, D)) / np.sqrt(D)

        transform = create_sketch_transform(rng_key, D, K)
        sketch = sketch_transform(X, transform)

        XX = onp.square(onp.matmul(X, onp.transpose(X)))
        DD = onp.matmul(sketch, onp.transpose(sketch))
        assert_allclose(XX, DD, rtol=0.0001, atol=0.0001)
        print("passed trial {}...".format(trial))
