import jax.numpy as np
from jax.experimental.stax import glorot, randn
from jax import random


def MaskedDense(out_dim, mask, W_init=glorot(), b_init=randn()):
    """
    As in jax.experimental.stax, each layer constructor function returns
    an (init_fun, apply_fun) pair, where `init_fun` takes an rng key and
    an input shape and returns an (output_shape, params) pair, and
    `apply_fun` takes params, inputs, and an rng key and applies the layer.

    :param int out_dim: Number of output dimensions.
    :param array mask: Mask applied to the weights of the layer.
    :param array W_init: initialization method for the weights.
    :param array b_init: initialization method for the bias terms.
    :return: a (`init_fn`, `update_fn`) pair.
    """
    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2 = random.split(rng)
        W, b = W_init(k1, (input_shape[-1], out_dim)), b_init(k2, (out_dim,))
        return output_shape, (W, b)

    def apply_fun(params, inputs, **kwargs):
        W, b = params
        return np.dot(inputs, W * mask) + b

    return init_fun, apply_fun
