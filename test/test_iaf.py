import pytest
import numpy as onp
from numpy.testing import assert_allclose
from numpyro.contrib.nn import AutoRegressiveNN
from numpyro.distributions import InverseAutoRegressiveTransform
from jax import random, jacfwd


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

    # test jacobians
    x = onp.random.rand(*input_shape[-1:])
    jac = jacfwd(iaf)(x)

    # permute jacobian as necessary
    permuted_jac = onp.zeros(jac.shape)
    perm = arn.permutation

    for j in range(input_dim):
        for k in range(input_dim):
            permuted_jac[..., j, k] = jac[..., perm[j], perm[k]]

    # make sure jacobians are triangular
    assert onp.sum(onp.abs(onp.triu(permuted_jac, 1))) == 0.00

    # make sure iaf.log_abs_det_jacobian is correct
    ldj = iaf.log_abs_det_jacobian(x, y)
    assert_allclose(ldj, onp.sum(onp.log(onp.diag(permuted_jac))), atol=1e-5)

