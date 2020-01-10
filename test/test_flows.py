# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import numpy as onp
from numpy.testing import assert_allclose
import pytest

from jax import jacfwd, random

from numpyro.contrib.nn import AutoregressiveNN
from numpyro.distributions.flows import InverseAutoregressiveTransform


def _make_iaf_args(input_dim, hidden_dims):
    _, rng_key_perm = random.split(random.PRNGKey(0))
    perm = random.shuffle(rng_key_perm, onp.arange(input_dim))
    arn_init, arn = AutoregressiveNN(input_dim, hidden_dims, param_dims=[1, 1], permutation=perm)
    _, init_params = arn_init(random.PRNGKey(0), (input_dim,))
    return partial(arn, init_params),


@pytest.mark.parametrize('flow_class, flow_args, input_dim', [
    (InverseAutoregressiveTransform, _make_iaf_args(5, hidden_dims=[10]), 5),
    (InverseAutoregressiveTransform, _make_iaf_args(7, hidden_dims=[8, 9]), 7),
])
@pytest.mark.parametrize('batch_shape', [(), (1,), (4,), (2, 3)])
def test_flows(flow_class, flow_args, input_dim, batch_shape):
    transform = flow_class(*flow_args)
    x = random.normal(random.PRNGKey(0), batch_shape + (input_dim,))

    # test inverse is correct
    y = transform(x)
    inv = transform.inv(y)
    assert_allclose(x, inv, atol=1e-5)

    # test jacobian shape
    actual = transform.log_abs_det_jacobian(x, y)
    assert onp.shape(actual) == batch_shape

    if batch_shape == ():
        # make sure transform.log_abs_det_jacobian is correct
        jac = jacfwd(transform)(x)
        expected = onp.linalg.slogdet(jac)[1]
        assert_allclose(actual, expected, atol=1e-5)

        # make sure jacobian is triangular, first permute jacobian as necessary
        if isinstance(transform, InverseAutoregressiveTransform):
            permuted_jac = onp.zeros(jac.shape)
            _, rng_key_perm = random.split(random.PRNGKey(0))
            perm = random.shuffle(rng_key_perm, onp.arange(input_dim))

            for j in range(input_dim):
                for k in range(input_dim):
                    permuted_jac[j, k] = jac[perm[j], perm[k]]

            assert onp.sum(onp.abs(onp.triu(permuted_jac, 1))) == 0.00
