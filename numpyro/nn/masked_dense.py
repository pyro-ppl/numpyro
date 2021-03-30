# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from jax import random
from jax.nn.initializers import glorot_normal, normal
import jax.numpy as jnp


def MaskedDense(mask, bias=True, W_init=glorot_normal(), b_init=normal()):
    """
    As in jax.experimental.stax, each layer constructor function returns
    an (init_fun, apply_fun) pair, where `init_fun` takes an rng_key key and
    an input shape and returns an (output_shape, params) pair, and
    `apply_fun` takes params, inputs, and an rng_key key and applies the layer.

    :param array mask: Mask of shape (input_dim, out_dim) applied to the weights of the layer.
    :param bool bias: whether to include bias term.
    :param array W_init: initialization method for the weights.
    :param array b_init: initialization method for the bias terms.
    :return: a (`init_fn`, `update_fn`) pair.
    """

    def init_fun(rng_key, input_shape):
        k1, k2 = random.split(rng_key)
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
            return jnp.dot(inputs, W * mask) + b
        else:
            W = params
            return jnp.dot(inputs, W * mask)

    return init_fun, apply_fun
