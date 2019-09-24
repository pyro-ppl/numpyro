from numpy.testing import assert_allclose
import pytest

from jax import jit, random

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


def test_seed():
    with handlers.seed(rng=11):
        x = numpyro.sample('x', dist.Normal(0., 1.))

    y = handlers.seed(lambda: numpyro.sample('y', dist.Normal(0., 1.)), 11)()
    assert_allclose(x, y)


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


def model_nested_plates_0():
    with numpyro.plate('outer', 10):
        x = numpyro.sample('y', dist.Normal(0., 1.))
        assert x.shape == (10,)
        with numpyro.plate('inner', 5):
            y = numpyro.sample('x', dist.Normal(0., 1.))
            assert y.shape == (5, 10)


def model_nested_plates_1():
    with numpyro.plate('outer', 10, dim=-2):
        x = numpyro.sample('y', dist.Normal(0., 1.))
        assert x.shape == (10, 1)
        with numpyro.plate('inner', 5):
            y = numpyro.sample('x', dist.Normal(0., 1.))
            assert y.shape == (10, 5)


def model_nested_plates_2():
    outer = numpyro.plate('outer', 10)
    inner = numpyro.plate('inner', 5, dim=-3)
    with outer:
        x = numpyro.sample('x', dist.Normal(0., 1.))
        assert x.shape == (10,)
    with inner:
        y = numpyro.sample('y', dist.Normal(0., 1.))
        assert y.shape == (5, 1, 1)

    with outer, inner:
        xy = numpyro.sample('xy', dist.Normal(0., 1.), sample_shape=(10,))
        assert xy.shape == (5, 1, 10)


def model_subsample_1():
    outer = numpyro.plate('outer', 20, subsample_size=10)
    inner = numpyro.plate('inner', 10, subsample_size=5, dim=-3)
    with outer:
        x = numpyro.sample('x', dist.Normal(0., 1.))
        assert x.shape == (10,)
    with inner:
        y = numpyro.sample('y', dist.Normal(0., 1.))
        assert y.shape == (5, 1, 1)

    with outer, inner:
        xy = numpyro.sample('xy', dist.Normal(0., 1.))
        assert xy.shape == (5, 1, 10)


@pytest.mark.parametrize('model', [
    model_nested_plates_0,
    model_nested_plates_1,
    model_nested_plates_2,
    model_subsample_1,
])
def test_plate(model):
    trace = handlers.trace(handlers.seed(model, random.PRNGKey(1))).get_trace()
    jit_trace = handlers.trace(jit(handlers.seed(model, random.PRNGKey(1)))).get_trace()
    for name, site in trace.items():
        if site['type'] == 'sample':
            assert_allclose(jit_trace[name]['value'], site['value'])
