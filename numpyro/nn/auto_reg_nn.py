# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

# lightly adapted from https://github.com/pyro-ppl/pyro/blob/dev/pyro/nn/auto_reg_nn.py

import numpy as np

from jax import ops
from jax.experimental import stax
import jax.numpy as jnp

from numpyro.nn.masked_dense import MaskedDense


def sample_mask_indices(input_dim, hidden_dim):
    """
    Samples the indices assigned to hidden units during the construction of MADE masks

    :param input_dim: the dimensionality of the input variable
    :type input_dim: int
    :param hidden_dim: the dimensionality of the hidden layer
    :type hidden_dim: int
    """
    indices = jnp.linspace(1, input_dim, num=hidden_dim)
    # Simple procedure tries to space fractional indices evenly by rounding to nearest int
    return jnp.round(indices)


def create_mask(input_dim, hidden_dims, permutation, output_dim_multiplier):
    """
    Creates (non-conditional) MADE masks

    :param input_dim: the dimensionality of the input variable
    :type input_dim: int
    :param hidden_dims: the dimensionality of the hidden layers(s)
    :type hidden_dims: list[int]
    :param permutation: the order of the input variables
    :type permutation: numpy array of integers of length `input_dim`
    :param output_dim_multiplier: tiles the output (e.g. for when a separate mean and scale parameter are desired)
    :type output_dim_multiplier: int
    """
    # Create mask indices for input, hidden layers, and final layer
    var_index = jnp.zeros(permutation.shape[0])
    var_index = ops.index_update(var_index, permutation, jnp.arange(input_dim))

    # Create the indices that are assigned to the neurons
    input_indices = 1 + var_index
    hidden_indices = [sample_mask_indices(input_dim - 1, h) for h in hidden_dims]
    output_indices = jnp.tile(var_index + 1, output_dim_multiplier)

    # Create mask from input to output for the skips connections
    mask_skip = output_indices[None, :] > input_indices[:, None]

    # Create mask from input to first hidden layer, and between subsequent hidden layers
    # NB: these masks are transposed version of Pyro ones
    masks = [hidden_indices[0][None, :] >= input_indices[:, None]]
    for i in range(1, len(hidden_dims)):
        masks.append(hidden_indices[i][None, :] >= hidden_indices[i - 1][:, None])

    # Create mask from last hidden layer to output layer
    masks.append(output_indices[None, :] > hidden_indices[-1][:, None])

    return masks, mask_skip


def AutoregressiveNN(input_dim, hidden_dims, param_dims=[1, 1], permutation=None,
                     skip_connections=False, nonlinearity=stax.Relu):
    """
    An implementation of a MADE-like auto-regressive neural network.

    Similar to the purely functional layer implemented in jax.experimental.stax,
    the `AutoregressiveNN` class has `init_fun` and `apply_fun` methods,
    where `init_fun` takes an rng_key key and an input shape and returns an
    (output_shape, params) pair, and `apply_fun` takes params and inputs
    and applies the layer.

    :param input_dim: the dimensionality of the input
    :type input_dim: int
    :param hidden_dims: the dimensionality of the hidden units per layer
    :type hidden_dims: list[int]
    :param param_dims: shape the output into parameters of dimension (p_n, input_dim) for p_n in param_dims
        when p_n > 1 and dimension (input_dim) when p_n == 1. The default is [1, 1], i.e. output two parameters
        of dimension (input_dim), which is useful for inverse autoregressive flow.
    :type param_dims: list[int]
    :param permutation: an optional permutation that is applied to the inputs and controls the order of the
        autoregressive factorization. in particular for the identity permutation the autoregressive structure
        is such that the Jacobian is triangular. Defaults to identity permutation.
    :type permutation: array of ints
    :param bool skip_connection: whether to add skip connections from the input to the output.
    :type skip_connections: bool
    :param nonlinearity: The nonlinearity to use in the feedforward network such as ReLU. Note that no
        nonlinearity is applied to the final network output, so the output is an unbounded real number.
        defaults to ReLU.
    :type nonlinearity: callable.
    :return: a tuple (init_fun, apply_fun)

    Reference:

    MADE: Masked Autoencoder for Distribution Estimation [arXiv:1502.03509]
    Mathieu Germain, Karol Gregor, Iain Murray, Hugo Larochelle
    """
    output_multiplier = sum(param_dims)
    all_ones = (np.array(param_dims) == 1).all()

    # Calculate the indices on the output corresponding to each parameter
    ends = np.cumsum(np.array(param_dims), axis=0)
    starts = np.concatenate((np.zeros(1), ends[:-1]))
    param_slices = [slice(int(s), int(e)) for s, e in zip(starts, ends)]

    # Hidden dimension must be not less than the input otherwise it isn't
    # possible to connect to the outputs correctly
    for h in hidden_dims:
        if h < input_dim:
            raise ValueError('Hidden dimension must not be less than input dimension.')

    if permutation is None:
        permutation = jnp.arange(input_dim)

    # Create masks
    masks, mask_skip = create_mask(input_dim=input_dim, hidden_dims=hidden_dims,
                                   permutation=permutation,
                                   output_dim_multiplier=output_multiplier)

    main_layers = []
    # Create masked layers
    for i, mask in enumerate(masks):
        main_layers.append(MaskedDense(mask))
        if i < len(masks) - 1:
            main_layers.append(nonlinearity)

    if skip_connections:
        net_init, net = stax.serial(stax.FanOut(2),
                                    stax.parallel(stax.serial(*main_layers),
                                                  MaskedDense(mask_skip, bias=False)),
                                    stax.FanInSum)
    else:
        net_init, net = stax.serial(*main_layers)

    def init_fun(rng_key, input_shape):
        """
        :param rng_key: rng_key used to initialize parameters
        :param input_shape: input shape
        """
        assert input_dim == input_shape[-1]
        return net_init(rng_key, input_shape)

    def apply_fun(params, inputs, **kwargs):
        """
        :param params: layer parameters
        :param inputs: layer inputs
        """
        out = net(params, inputs, **kwargs)

        # reshape output as necessary
        out = jnp.reshape(out, inputs.shape[:-1] + (output_multiplier, input_dim))
        # move param dims to the first dimension
        out = jnp.moveaxis(out, -2, 0)

        if all_ones:
            # Squeeze dimension if all parameters are one dimensional
            out = tuple([out[i] for i in range(output_multiplier)])
        else:
            # If not all ones, then probably don't want to squeeze a single dimension parameter
            out = tuple([out[s] for s in param_slices])

        # if len(param_dims) == 1, we return the array instead of a tuple of arrays
        return out[0] if len(param_dims) == 1 else out

    return init_fun, apply_fun
