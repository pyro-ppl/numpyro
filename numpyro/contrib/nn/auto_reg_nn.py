# lightly adapted from https://github.com/pyro-ppl/pyro/blob/dev/pyro/nn/auto_reg_nn.py

from __future__ import absolute_import, division, print_function

import numpy as onp

from jax import random
import jax.numpy as np

from numpyro.contrib.nn import MaskedDense


def sample_mask_indices(input_dim, hidden_dim):
    """
    Samples the indices assigned to hidden units during the construction of MADE masks

    :param input_dim: the dimensionality of the input variable
    :type input_dim: int
    :param hidden_dim: the dimensionality of the hidden layer
    :type hidden_dim: int
    """
    indices = onp.linspace(1, input_dim, num=hidden_dim)
    # Simple procedure tries to space fractional indices evenly by rounding to nearest int
    return onp.round(indices)


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
    var_index = onp.zeros(permutation.shape[0])
    var_index[permutation] = onp.arange(input_dim)
    dtype = var_index.dtype

    # Create the indices that are assigned to the neurons
    input_indices = 1 + var_index
    hidden_indices = [sample_mask_indices(input_dim, h) for h in hidden_dims]
    output_indices = onp.tile(var_index + 1, output_dim_multiplier)

    # Create mask from input to output for the skips connections
    mask_skip = onp.transpose(output_indices[:, None] > input_indices[None, :]).astype(dtype)

    # Create mask from input to first hidden layer, and between subsequent hidden layers
    # NOTE: The masks created follow a slightly different pattern than that given in Germain et al. Figure 1
    # The output first in the order (e.g. x_2 in the figure) is connected to hidden units rather than being unattached
    # Tracing a path back through the network, however, this variable will still be unconnected to any input variables
    masks = [onp.transpose(hidden_indices[0][:, None] > input_indices[None, :]).astype(dtype)]
    for i in range(1, len(hidden_dims)):
        masks.append(onp.transpose(hidden_indices[i][:, None] >= hidden_indices[i - 1][None, :]).astype(dtype))

    # Create mask from last hidden layer to output layer
    masks.append(onp.transpose(output_indices[:, None] >= hidden_indices[-1][None, :]).astype(dtype))

    return masks, mask_skip


def relu(x): return np.maximum(x, 0.)


class AutoRegressiveNN(object):
    """
    An implementation of a MADE-like auto-regressive neural network.

    Similar to the purely functional layer implemented in jax.experimental.stax,
    the `AutoRegressiveNN` class has `init_fun` and `apply_fun` methods,
    where `init_fun` takes an rng key and an input shape and returns an
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
    :param nonlinearity: The nonlinearity to use in the feedforward network such as ReLU. Note that no
        nonlinearity is applied to the final network output, so the output is an unbounded real number.
        defaults to ReLU.
    :type nonlinearity: callable.

    Reference:

    MADE: Masked Autoencoder for Distribution Estimation [arXiv:1502.03509]
    Mathieu Germain, Karol Gregor, Iain Murray, Hugo Larochelle
    """
    def __init__(self, input_dim, hidden_dims, param_dims=[1, 1], nonlinearity=relu):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.masked_layers = None
        self.param_dims = param_dims
        self.nonlinearity = nonlinearity
        self.all_ones = (np.array(param_dims) == 1).all()

        # Calculate the indices on the output corresponding to each parameter
        ends = onp.cumsum(self.param_dims, axis=0)
        starts = onp.concatenate((np.zeros(1), ends[:-1]))
        self.param_slices = [slice(int(s), int(e)) for s, e in zip(starts, ends)]

        self.count_params = len(self.param_dims)
        self.output_multiplier = sum(self.param_dims)

        # Hidden dimension must be not less than the input otherwise it isn't
        # possible to connect to the outputs correctly
        for h in self.hidden_dims:
            if h < self.input_dim:
                raise ValueError('Hidden dimension must not be less than input dimension.')

    def init_fun(self, rng, input_shape, permutation=None):
        """
        :param rng: rng used to initialize parameters
        :param input_shape: input shape
        :param permutation: an optional permutation that is applied to the inputs and controls the order of the
            autoregressive factorization. in particular for the identity permutation the autoregressive structure
            is such that the Jacobian is triangular. By default this is chosen at random.
        :type permutation: array of ints
        """
        if permutation is None:
            # By default set a random permutation of variables, which is important for performance with multiple steps
            rng, rng_perm = random.split(rng)
            self.permutation = onp.array(random.shuffle(rng_perm, np.arange(self.input_dim)))
        else:
            self.permutation = permutation

        # Create masks (no skip connections allowed; TODO add support)
        masks, _ = create_mask(input_dim=self.input_dim, hidden_dims=self.hidden_dims,
                               permutation=self.permutation,
                               output_dim_multiplier=self.output_multiplier)

        # Create masked layers
        self.masked_layers = [MaskedDense(mask) for mask in masks]
        init_params = []
        for mask in self.masked_layers:
            mask_init = mask[0]
            input_shape, param = mask_init(rng, input_shape)
            init_params.append(param)

        return input_shape, init_params

    def apply_fun(self, params, inputs, **kwargs):
        """
        :param params: layer parameters
        :param inputs: layer inputs
        """
        out = inputs

        for k, (mask, param) in enumerate(zip(self.masked_layers, params)):
            mask_apply = mask[1]
            out = mask_apply(param, out)
            if k < len(self.masked_layers) - 1:
                out = self.nonlinearity(out)

        # reshape output as necessary
        if self.output_multiplier == 1:
            return out
        else:
            out = np.reshape(out, list(inputs.shape[:-1]) + [self.output_multiplier, self.input_dim])

            # Squeeze dimension if all parameters are one dimensional
            if self.count_params == 1:
                return out

            elif self.all_ones:
                return np.squeeze(out, axis=-2)

            # If not all ones, then probably don't want to squeeze a single dimension parameter
            else:
                return tuple([out[..., s, :] for s in self.param_slices])
