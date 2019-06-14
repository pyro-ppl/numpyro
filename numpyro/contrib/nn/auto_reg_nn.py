# lightly adapted from https://github.com/pyro-ppl/pyro/blob/dev/pyro/nn/auto_reg_nn.py

from __future__ import absolute_import, division, print_function

import numpy as onp


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

    # Create the indices that are assigned to the neurons
    input_indices = 1 + var_index
    hidden_indices = [sample_mask_indices(input_dim, h) for h in hidden_dims]
    output_indices = onp.tile(var_index + 1, output_dim_multiplier)

    # Create mask from input to output for the skips connections
    mask_skip = output_indices[:, None] > input_indices[None, :]

    # Create mask from input to first hidden layer, and between subsequent hidden layers
    # NOTE: The masks created follow a slightly different pattern than that given in Germain et al. Figure 1
    # The output first in the order (e.g. x_2 in the figure) is connected to hidden units rather than being unattached
    # Tracing a path back through the network, however, this variable will still be unconnected to any input variables
    masks = [hidden_indices[0][:, None] > input_indices[None, :]]
    for i in range(1, len(hidden_dims)):
        masks.append(hidden_indices[i][:, None] >= hidden_indices[i - 1][None, :])

    # Create mask from last hidden layer to output layer
    masks.append(output_indices[:, None] >= hidden_indices[-1][None, :])

    return masks, mask_skip
