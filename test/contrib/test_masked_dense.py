import numpy as onp
import pytest

from jax import random
from jax.experimental.stax import serial

from numpyro.contrib.nn import MaskedDense
from numpyro.contrib.nn.auto_reg_nn import create_mask


@pytest.mark.parametrize('input_dim', [5, 7])
def test_masked_dense(input_dim):
    hidden_dim = input_dim * 3
    output_dim_multiplier = input_dim - 4
    mask, _ = create_mask(input_dim, [hidden_dim], onp.random.permutation(input_dim), output_dim_multiplier)
    init_random_params, masked_dense = serial(MaskedDense(mask[0]))

    rng = random.PRNGKey(0)
    batch_size = 4
    input_shape = (batch_size, input_dim)
    _, init_params = init_random_params(rng, input_shape)
    output = masked_dense(init_params, onp.random.rand(*input_shape))
    assert output.shape == (batch_size, hidden_dim)
