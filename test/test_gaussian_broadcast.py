# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numpy.testing import assert_allclose
import pytest
import scipy.stats as osp

import jax
import jax.numpy as jnp

from numpyro import distributions as dist


@pytest.mark.parametrize("shape", [(2,), (1, 2), (4, 1, 2)])
@pytest.mark.parametrize(
    "parameter", ["covariance_matrix", "precision_matrix", "scale_tril", "student"]
)
def test_shared_gaussian_observations(shape, parameter):
    covariance = np.array(
        [[[1.0, 0.2], [0.2, 0.7]], [[0.6, -0.1], [-0.1, 1.3]], [[1.4, 0.3], [0.3, 0.8]]]
    )
    tril = np.linalg.cholesky(covariance)
    if parameter == "student":
        d = dist.MultivariateStudentT(5.0, jnp.zeros(2), jnp.array(tril))
    else:
        matrix = (
            covariance
            if parameter == "covariance_matrix"
            else np.linalg.inv(covariance)
            if parameter == "precision_matrix"
            else tril
        )
        d = dist.MultivariateNormal(**{parameter: jnp.array(matrix)})
    observation = np.broadcast_to(np.array([0.3, -0.4]), shape)
    target_shape = np.broadcast_shapes(shape[:-1], (3,))
    expected = []
    scores = []
    for cov in covariance:
        if parameter == "student":
            expected.append(
                osp.multivariate_t.logpdf(np.array([0.3, -0.4]), df=5.0, shape=cov)
            )
        else:
            expected.append(
                osp.multivariate_normal.logpdf(np.array([0.3, -0.4]), cov=cov)
            )
        precision_y = np.linalg.solve(cov, np.array([0.3, -0.4]))
        score = -precision_y
        if parameter == "student":
            score *= 7 / (5 + np.array([0.3, -0.4]) @ precision_y)
        scores.append(score)
    # Each covariance model sees the same observed vector, including sample axes.
    expected = np.broadcast_to(np.asarray(expected), target_shape)
    actual = jax.jit(d.log_prob)(jnp.array(observation))
    assert actual.shape == target_shape
    assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    derivative = jax.jit(
        jax.grad(lambda t: d.log_prob(t * jnp.array(observation)).sum())
    )(1.0)
    expected_derivative = np.sum(np.asarray(scores) * np.array([0.3, -0.4]))
    expected_derivative *= np.prod(target_shape) / 3
    assert_allclose(derivative, expected_derivative, rtol=2e-6, atol=2e-6)
