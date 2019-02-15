import jax.numpy as np
import jax.random as random

import pytest
from jax import lax

from numpyro.distributions.normal import norm


@pytest.mark.parametrize('loc, scale', [
    (1, 1),
    (1., np.array([1., 2.])),
])
def test_normal_sample(loc, scale):
    rng = random.PRNGKey(0)
    expected_shape = lax.broadcast_shapes(*[np.shape(loc), np.shape(scale)])
    print(norm.rvs(loc, scale, random_state=rng))
    assert np.shape(norm.rvs(loc, scale, random_state=rng)) == expected_shape
