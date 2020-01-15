# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as onp
from numpy.testing import assert_allclose, assert_raises
import pytest

from jax import jit
from jax import numpy as np
from jax import random, vmap

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer.util import log_density
from numpyro.util import optional


@pytest.mark.parametrize('mask_last', [1, 5, 10])
@pytest.mark.parametrize('use_jit', [False, True])
def test_mask(mask_last, use_jit):
    N = 10
    mask = onp.ones(N, dtype=onp.bool)
    mask[-mask_last] = 0

    def model(data, mask):
        with numpyro.plate('N', N):
            x = numpyro.sample('x', dist.Normal(0, 1))
            with handlers.mask(mask_array=mask):
                numpyro.sample('y', dist.Delta(x, log_density=1.))
                with handlers.scale(scale_factor=2):
                    numpyro.sample('obs', dist.Normal(x, 1), obs=data)

    data = random.normal(random.PRNGKey(0), (N,))
    x = random.normal(random.PRNGKey(1), (N,))
    if use_jit:
        log_joint = jit(log_density, static_argnums=(0,))(model, (data, mask), {}, {'x': x, 'y': x})[0]
    else:
        log_joint = log_density(model, (data, mask), {}, {'x': x, 'y': x})[0]
    log_prob_x = dist.Normal(0, 1).log_prob(x)
    log_prob_y = mask
    log_prob_z = dist.Normal(x, 1).log_prob(data)
    expected = (log_prob_x + np.where(mask,  log_prob_y + 2 * log_prob_z, 0.)).sum()
    assert_allclose(log_joint, expected, atol=1e-4)


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
    def _sample():
        x = numpyro.sample('x', dist.Normal(0., 1.))
        y = numpyro.sample('y', dist.Normal(1., 2.))
        return np.stack([x, y])

    xs = []
    for i in range(100):
        with handlers.seed(rng_seed=i):
            xs.append(_sample())
    xs = np.stack(xs)

    ys = vmap(lambda rng_key: handlers.seed(lambda: _sample(), rng_key)())(np.arange(100))
    assert_allclose(xs, ys, atol=1e-6)


def test_nested_seeding():
    def fn(rng_key_1, rng_key_2, rng_key_3):
        xs = []
        with handlers.seed(rng_seed=rng_key_1):
            with handlers.seed(rng_seed=rng_key_2):
                xs.append(numpyro.sample('x', dist.Normal(0., 1.)))
                with handlers.seed(rng_seed=rng_key_3):
                    xs.append(numpyro.sample('y', dist.Normal(0., 1.)))
        return np.stack(xs)

    s1, s2 = fn(0, 1, 2), fn(3, 1, 2)
    assert_allclose(s1, s2)
    s1, s2 = fn(0, 1, 2), fn(3, 1, 4)
    assert_allclose(s1[0], s2[0])
    assert_raises(AssertionError, assert_allclose, s1[1], s2[1])


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
            z = numpyro.deterministic('z', x ** 2)
            assert z.shape == (10,)


def model_nested_plates_1():
    with numpyro.plate('outer', 10, dim=-2):
        x = numpyro.sample('y', dist.Normal(0., 1.))
        assert x.shape == (10, 1)
        with numpyro.plate('inner', 5):
            y = numpyro.sample('x', dist.Normal(0., 1.))
            assert y.shape == (10, 5)
            z = numpyro.deterministic('z', x ** 2)
            assert z.shape == (10, 1)


def model_nested_plates_2():
    outer = numpyro.plate('outer', 10)
    inner = numpyro.plate('inner', 5, dim=-3)
    with outer:
        x = numpyro.sample('x', dist.Normal(0., 1.))
        assert x.shape == (10,)
    with inner:
        y = numpyro.sample('y', dist.Normal(0., 1.))
        assert y.shape == (5, 1, 1)
        z = numpyro.deterministic('z', x ** 2)
        assert z.shape == (10,)

    with outer, inner:
        xy = numpyro.sample('xy', dist.Normal(0., 1.), sample_shape=(10,))
        assert xy.shape == (5, 1, 10)


def model_dist_batch_shape():
    outer = numpyro.plate('outer', 10)
    inner = numpyro.plate('inner', 5, dim=-3)
    with outer:
        x = numpyro.sample('x', dist.Normal(np.zeros(10), 1.))
        assert x.shape == (10,)
    with inner:
        y = numpyro.sample('y', dist.Normal(0., np.ones(10)))
        assert y.shape == (5, 1, 10)
        z = numpyro.deterministic('z', x ** 2)
        assert z.shape == (10,)

    with outer, inner:
        xy = numpyro.sample('xy', dist.Normal(0., np.ones(10)), sample_shape=(10,))
        assert xy.shape == (5, 10, 10)


def model_subsample_1():
    outer = numpyro.plate('outer', 20, subsample_size=10)
    inner = numpyro.plate('inner', 10, subsample_size=5, dim=-3)
    with outer:
        x = numpyro.sample('x', dist.Normal(0., 1.))
        assert x.shape == (10,)
    with inner:
        y = numpyro.sample('y', dist.Normal(0., 1.))
        assert y.shape == (5, 1, 1)
        z = numpyro.deterministic('z', x ** 2)
        assert z.shape == (10,)

    with outer, inner:
        xy = numpyro.sample('xy', dist.Normal(0., 1.))
        assert xy.shape == (5, 1, 10)


@pytest.mark.parametrize('model', [
    model_nested_plates_0,
    model_nested_plates_1,
    model_nested_plates_2,
    model_dist_batch_shape,
    model_subsample_1,
])
def test_plate(model):
    trace = handlers.trace(handlers.seed(model, random.PRNGKey(1))).get_trace()
    jit_trace = handlers.trace(jit(handlers.seed(model, random.PRNGKey(1)))).get_trace()
    assert 'z' in trace
    for name, site in trace.items():
        if site['type'] == 'sample':
            assert_allclose(jit_trace[name]['value'], site['value'])
