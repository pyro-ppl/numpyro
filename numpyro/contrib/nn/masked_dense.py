from jax import random
from jax.experimental.stax import glorot, randn
import jax.numpy as np

from numpyro.distributions.util import vec_to_tril_matrix


def MaskedDense(mask, bias=True, W_init=glorot(), b_init=randn()):
    """
    As in jax.experimental.stax, each layer constructor function returns
    an (init_fun, apply_fun) pair, where `init_fun` takes an rng key and
    an input shape and returns an (output_shape, params) pair, and
    `apply_fun` takes params, inputs, and an rng key and applies the layer.

    :param array mask: Mask of shape (input_dim, out_dim) applied to the weights of the layer.
    :param bool bias: whether to include bias term.
    :param array W_init: initialization method for the weights.
    :param array b_init: initialization method for the bias terms.
    :return: a (`init_fn`, `update_fn`) pair.
    """
    def init_fun(rng, input_shape):
        k1, k2 = random.split(rng)
        W = W_init(k1, mask.shape)
        if bias:
            b = b_init(k2, mask.shape[-1:])
            params = (W, b)
        else:
            params = W
        return input_shape[:-1] + mask.shape[-1:], params

    def apply_fun(params, inputs, **kwargs):
        if bias:
            W, b = params
            return np.dot(inputs, W * mask) + b
        else:
            W = params
            return np.dot(inputs, W * mask)

    return init_fun, apply_fun
