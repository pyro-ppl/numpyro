import pytest
from numpy.testing import assert_allclose

from jax import random

import numpyro.distributions as dist
from numpyro.handlers import sample, scale, param, substitute, seed, trace, condition
from numpyro.hmc_util import log_density
from numpyro.util import optional


@pytest.mark.parametrize('use_context_manager', [True, False])
def test_scale(use_context_manager):
    def model(data):
        x = sample('x', dist.Normal(0, 1))
        with optional(use_context_manager, scale(scale_factor=10)):
            sample('obs', dist.Normal(x, 1), obs=data)

    model = model if use_context_manager else scale(model, 10.)
    data = random.normal(random.PRNGKey(0), (3,))
    x = random.normal(random.PRNGKey(1))
    log_joint = log_density(model, (data,), {}, {'x': x})[0]
    log_prob1, log_prob2 = dist.Normal(0, 1).log_prob(x), dist.Normal(x, 1).log_prob(data).sum()
    expected = log_prob1 + 10 * log_prob2 if use_context_manager else 10 * (log_prob1 + log_prob2)
    assert_allclose(log_joint, expected)


def test_substitute():
    def model():
        x = param('x', None)
        y = substitute(lambda: param('y', None) * param('x', None), {'y': x})()
        return x + y

    assert substitute(model, {'x': 3.})() == 12.


def test_condition():
    def model():
        x = sample('x', dist.Delta(0.))
        y = sample('y', dist.Normal(0., 1.))
        return x + y

    model = condition(seed(model, random.PRNGKey(1)), {'y': 2.})
    model_trace = trace(model).get_trace()
    assert model_trace['y']['value'] == 2.
    assert model_trace['y']['is_observed']
    # Raise ValueError when site is already observed.
    with pytest.raises(ValueError):
        condition(model, {'y': 3.})()
