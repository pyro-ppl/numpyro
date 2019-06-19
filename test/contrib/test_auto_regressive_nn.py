# lightly adapted from https://github.com/pyro-ppl/pyro/blob/dev/tests/nn/test_autoregressive.py

import numpy as onp
from numpy.testing import assert_array_equal
import pytest

from jax import jacfwd, random

from numpyro.contrib.nn.auto_reg_nn import AutoRegressiveNN, create_mask


@pytest.mark.parametrize('input_dim', [5])
@pytest.mark.parametrize('param_dims', [[1], [1, 1], [2, 3]])
@pytest.mark.parametrize('hidden_dims', [[8], [6, 7]])
def test_auto_reg_nn(input_dim, hidden_dims, param_dims):
    arn = AutoRegressiveNN(input_dim, hidden_dims, param_dims=param_dims)

    rng = random.PRNGKey(0)
    batch_size = 4
    input_shape = (batch_size, input_dim)
    _, init_params = arn.init_fun(rng, input_shape)

    output = arn.apply_fun(init_params, onp.random.rand(*input_shape))

    if param_dims == [1]:
        assert output.shape == (batch_size, input_dim)
        jac = jacfwd(lambda x: arn.apply_fun(init_params, x))(onp.random.rand(input_dim))
    elif param_dims == [1, 1]:
        assert output.shape == (batch_size, 2, input_dim)
        jac = jacfwd(lambda x: arn.apply_fun(init_params, x))(onp.random.rand(input_dim))
    elif param_dims == [2, 3]:
        assert output[0].shape == (batch_size, 2, input_dim)
        assert output[1].shape == (batch_size, 3, input_dim)
        jac = jacfwd(lambda x: arn.apply_fun(init_params, x)[0])(onp.random.rand(input_dim))

    # permute jacobian as necessary
    permuted_jac = onp.zeros(jac.shape)
    perm = arn.permutation

    for j in range(input_dim):
        for k in range(input_dim):
            permuted_jac[..., j, k] = jac[..., perm[j], perm[k]]

    # make sure jacobians are triangular
    assert onp.sum(onp.abs(onp.triu(permuted_jac))) == 0.0


@pytest.mark.parametrize('input_dim', [5, 8])
@pytest.mark.parametrize('n_layers', [1, 3])
@pytest.mark.parametrize('output_dim_multiplier', [1, 4])
def test_masks(input_dim, n_layers, output_dim_multiplier):
    hidden_dim = input_dim * 3
    hidden_dims = [hidden_dim] * n_layers
    permutation = onp.random.permutation(input_dim)
    masks, mask_skip = create_mask(input_dim, hidden_dims, permutation, output_dim_multiplier)
    masks = [onp.transpose(m) for m in masks]
    mask_skip = onp.transpose(mask_skip)

    # First test that hidden layer masks are adequately connected
    # Tracing backwards, works out what inputs each output is connected to
    # It's a dictionary of sets indexed by a tuple (input_dim, param_dim)
    _permutation = list(permutation)

    # Loop over variables
    for idx in range(input_dim):
        # Calculate correct answer
        correct = onp.array(sorted(_permutation[0:onp.where(permutation == idx)[0][0]]))

        # Loop over parameters for each variable
        for jdx in range(output_dim_multiplier):
            prev_connections = set()
            # Do output-to-penultimate hidden layer mask
            for kdx in range(masks[-1].shape[1]):
                if masks[-1][idx + jdx * input_dim, kdx]:
                    prev_connections.add(kdx)

            # Do hidden-to-hidden, and hidden-to-input layer masks
            for m in reversed(masks[:-1]):
                this_connections = set()
                for kdx in prev_connections:
                    for ldx in range(m.shape[1]):
                        if m[kdx, ldx]:
                            this_connections.add(ldx)
                prev_connections = this_connections

            assert_array_equal(list(sorted(prev_connections)), correct)

            # Test the skip-connections mask
            skip_connections = set()
            for kdx in range(mask_skip.shape[1]):
                if mask_skip[idx + jdx * input_dim, kdx]:
                    skip_connections.add(kdx)
            assert_array_equal(list(sorted(skip_connections)), correct)
