# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from jax import random
from jax.example_libraries import stax
from jax.nn import sigmoid, softplus
from jax.nn.initializers import glorot_uniform, normal, uniform
import jax.numpy as jnp

from numpyro.distributions.util import logmatmulexp, vec_to_tril_matrix


def BlockMaskedDense(
    num_blocks, in_factor, out_factor, bias=True, W_init=glorot_uniform()
):
    """
    Module that implements a linear layer with block matrices with positive diagonal blocks.
    Moreover, it uses Weight Normalization (https://arxiv.org/abs/1602.07868) for stability.

    :param int num_blocks: Number of block matrices.
    :param int in_factor: number of rows in each block.
    :param int out_factor: number of columns in each block.
    :param W_init: initialization method for the weights.
    :return: an (`init_fn`, `update_fn`) pair.
    """
    input_dim, out_dim = num_blocks * in_factor, num_blocks * out_factor
    # construct mask_d, mask_o for formula (8) of Ref [1]
    # Diagonal block mask
    mask_d = np.identity(num_blocks)[..., None]
    mask_d = np.tile(mask_d, (1, in_factor, out_factor)).reshape(input_dim, out_dim)
    # Off-diagonal block mask for upper triangular weight matrix
    mask_o = vec_to_tril_matrix(
        jnp.ones(num_blocks * (num_blocks - 1) // 2), diagonal=-1
    ).T[..., None]
    mask_o = jnp.tile(mask_o, (1, in_factor, out_factor)).reshape(input_dim, out_dim)

    def init_fun(rng, input_shape):
        assert input_dim == input_shape[-1]
        *k1, k2, k3 = random.split(rng, num_blocks + 2)

        # Initialize each column block using W_init
        W = jnp.zeros((input_dim, out_dim))
        for i in range(num_blocks):
            W = W.at[: (i + 1) * in_factor, i * out_factor : (i + 1) * out_factor].set(
                W_init(k1[i], ((i + 1) * in_factor, out_factor))
            )

        # initialize weight scale
        ws = jnp.log(uniform(1.0)(k2, (out_dim,)))

        if bias:
            b = (uniform(1.0)(k3, (out_dim,)) - 0.5) * (2 / jnp.sqrt(out_dim))
            params = (W, ws, b)
        else:
            params = (W, ws)
        return input_shape[:-1] + (out_dim,), params

    def apply_fun(params, inputs, **kwargs):
        x, logdet = inputs
        if bias:
            W, ws, b = params
        else:
            W, ws = params

        # Form block weight matrix, making sure it's positive on diagonal!
        w = jnp.exp(W) * mask_d + W * mask_o

        # Compute norm of each column (i.e. each output features)
        w_norm = jnp.linalg.norm(w, axis=-2, keepdims=True)

        # Normalize weight and rescale
        w = jnp.exp(ws) * w / w_norm

        out = jnp.dot(x, w)
        if bias:
            out = out + b

        dense_logdet = ws + W - jnp.log(w_norm)
        # logdet of block diagonal
        dense_logdet = dense_logdet[mask_d.astype(bool)].reshape(
            num_blocks, in_factor, out_factor
        )
        if logdet is None:
            logdet = jnp.broadcast_to(dense_logdet, x.shape[:-1] + dense_logdet.shape)
        else:
            logdet = logmatmulexp(logdet, dense_logdet)
        return out, logdet

    return init_fun, apply_fun


def Tanh():
    """
    Tanh nonlinearity with its log jacobian.

    :return: an (`init_fn`, `update_fn`) pair.
    """

    def init_fun(rng, input_shape):
        return input_shape, ()

    def apply_fun(params, inputs, **kwargs):
        x, logdet = inputs
        out = jnp.tanh(x)
        tanh_logdet = -2 * (x + softplus(-2 * x) - jnp.log(2.0))
        # logdet.shape = batch_shape + (num_blocks, in_factor, out_factor)
        # tanh_logdet.shape = batch_shape + (num_blocks x out_factor,)
        # so we need to reshape tanh_logdet to: batch_shape + (num_blocks, 1, out_factor)
        tanh_logdet = tanh_logdet.reshape(logdet.shape[:-2] + (1, logdet.shape[-1]))
        return out, logdet + tanh_logdet

    return init_fun, apply_fun


def FanInResidualNormal():
    """
    Similar to stax.FanInSum but also keeps track of log determinant of Jacobian.
    It is required that the second fan-in branch is identity.

    :return: an (`init_fn`, `update_fn`) pair.
    """

    def init_fun(rng, input_shape):
        return input_shape[0], ()

    def apply_fun(params, inputs, **kwargs):
        # f(x) + x
        (fx, logdet), (x, _) = inputs
        return fx + x, softplus(logdet)

    return init_fun, apply_fun


def FanInResidualGated(gate_init=normal(1.0)):
    """
    Similar to FanInNormal uses a learnable parameter `gate` to interpolate two fan-in branches.
    It is required that the second fan-in branch is identity.

    :param gate_init: initialization method for the gate.
    :return: an (`init_fn`, `update_fn`) pair.
    """

    def init_fun(rng, input_shape):
        return input_shape[0], gate_init(rng, ())

    def apply_fun(params, inputs, **kwargs):
        # a * f(x) + (1 - a) * x
        (fx, logdet), (x, _) = inputs
        gate = sigmoid(params)
        out = gate * fx + (1 - gate) * x
        logdet = softplus(logdet + params) - softplus(params)
        return out, logdet

    return init_fun, apply_fun


def BlockNeuralAutoregressiveNN(input_dim, hidden_factors=[8, 8], residual=None):
    """
    An implementation of Block Neural Autoregressive neural network.

    **References**

    1. *Block Neural Autoregressive Flow*,
       Nicola De Cao, Ivan Titov, Wilker Aziz

    :param int input_dim: The dimensionality of the input.
    :param list hidden_factors: Hidden layer i has ``hidden_factors[i]`` hidden units per
        input dimension. This corresponds to both :math:`a` and :math:`b` in reference [1].
        The elements of hidden_factors must be integers.
    :param str residual: Type of residual connections to use. One of `None`, `"normal"`, `"gated"`.
    :return: an (`init_fn`, `update_fn`) pair.
    """
    layers = []
    in_factor = 1
    for hidden_factor in hidden_factors:
        layers.append(BlockMaskedDense(input_dim, in_factor, hidden_factor))
        layers.append(Tanh())
        in_factor = hidden_factor
    layers.append(BlockMaskedDense(input_dim, in_factor, 1))
    arn = stax.serial(*layers)
    if residual is not None:
        FanInResidual = (
            FanInResidualGated if residual == "gated" else FanInResidualNormal
        )
        arn = stax.serial(
            stax.FanOut(2), stax.parallel(arn, stax.Identity), FanInResidual()
        )

    def init_fun(rng, input_shape):
        return arn[0](rng, input_shape)

    def apply_fun(params, inputs, **kwargs):
        out, logdet = arn[1](params, (inputs, None), **kwargs)
        return out, logdet.reshape(inputs.shape)

    return init_fun, apply_fun
