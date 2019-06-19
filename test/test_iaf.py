import pytest
import numpy as onp
from numpy.testing import assert_allclose
from numpyro.contrib.nn import AutoRegressiveNN
from numpyro.distributions import InverseAutoRegressiveTransform
from jax import random


@pytest.mark.parametrize('input_dim', [5, 7])
@pytest.mark.parametrize('hidden_dims', [[8, 9], [10]])
def test_auto_reg_nn(input_dim, hidden_dims):
    arn = AutoRegressiveNN(input_dim, hidden_dims, param_dims=[1, 1])

    rng = random.PRNGKey(0)
    batch_size = 4
    input_shape = (batch_size, input_dim)
    _, init_params = arn.init_fun(rng, input_shape)

    iaf = InverseAutoRegressiveTransform(arn, init_params)

    # test inverse is correct
    x = onp.random.rand(*input_shape)
    y = iaf(x)
    inv = iaf.inv(y)

    assert_allclose(x, inv, atol=1e-5)

    # smoketest
    iaf.log_abs_det_jacobian(x, y)
