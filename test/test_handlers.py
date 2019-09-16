from numpy.testing import assert_allclose
import pytest

from jax import random

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer_util import log_density
from numpyro.util import optional


@pytest.mark.parametrize('use_context_manager', [True, False])
def test_scale(use_context_manager):
    def model(data):
        x = numpyro.sample('x', dist.Normal(0, 1))
        with optional(use_context_manager, handlers.scale(scale_factor=10)):
            numpyro.sample('obs', dist.Normal(x, 1), obs=data)

    model = model if use_context_manager else handlers.scale(model, 10.)
    data = random.normal(random.PRNGKey(0), (3,))
    x = random.normal(random.PRNGKey(1))
    log_joint = log_density(model, (data,), {}, {'x': x})[0]
    log_prob1, log_prob2 = dist.Normal(0, 1).log_prob(x), dist.Normal(x, 1).log_prob(data).sum()
    expected = log_prob1 + 10 * log_prob2 if use_context_manager else 10 * (log_prob1 + log_prob2)
    assert_allclose(log_joint, expected)


def test_substitute():
    def model():
        x = numpyro.param('x', None)
        y = handlers.substitute(lambda: numpyro.param('y', None) * numpyro.param('x', None), {'y': x})()
        return x + y

    assert handlers.substitute(model, {'x': 3.})() == 12.


def test_condition():
    def model():
        x = numpyro.sample('x', dist.Delta(0.))
        y = numpyro.sample('y', dist.Normal(0., 1.))
        return x + y

    model = handlers.condition(handlers.seed(model, random.PRNGKey(1)), {'y': 2.})
    model_trace = handlers.trace(model).get_trace()
    assert model_trace['y']['value'] == 2.
    assert model_trace['y']['is_observed']
    # Raise ValueError when site is already observed.
    with pytest.raises(ValueError):
        handlers.condition(model, {'y': 3.})()


def test_no_split_deterministic():
    def model():
        x = numpyro.sample('x', dist.Normal(0., 1.))
        y = numpyro.sample('y', dist.Normal(0., 1.))
        return x + y

    model = handlers.condition(model, {'x': 1., 'y': 2.})
    assert model() == 3.
