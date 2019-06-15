from numpy.testing import assert_allclose

from jax import random

import numpyro.distributions as dist
from numpyro.handlers import sample, scale
from numpyro.hmc_util import log_density


def test_scale():
    def model(data):
        x = sample('x', dist.Normal(0, 1))
        with scale(10):
            sample('obs', dist.Normal(x, 1), obs=data)

    data = random.normal(random.PRNGKey(0), (3,))
    x = random.normal(random.PRNGKey(1))
    log_joint = log_density(model, (data,), {}, {'x': x})[0]
    expected = dist.Normal(0, 1).log_prob(x) + 10 * dist.Normal(x, 1).log_prob(data).sum()
    assert_allclose(log_joint, expected)
