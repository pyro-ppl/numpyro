# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numpy.testing import assert_allclose, assert_raises
import pytest

from jax import jit, random, vmap
import jax.numpy as jnp

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer.util import log_density
from numpyro.util import optional


@pytest.mark.parametrize('mask_last', [1, 5, 10])
@pytest.mark.parametrize('use_jit', [False, True])
def test_mask(mask_last, use_jit):
    N = 10
    mask = np.ones(N, dtype=np.bool)
    mask[-mask_last] = 0

    def model(data, mask):
        with numpyro.plate('N', N):
            x = numpyro.sample('x', dist.Normal(0, 1))
            with handlers.mask(mask=mask):
                numpyro.sample('y', dist.Delta(x, log_density=1.))
                with handlers.scale(scale=2):
                    numpyro.sample('obs', dist.Normal(x, 1), obs=data)

    data = random.normal(random.PRNGKey(0), (N,))
    x = random.normal(random.PRNGKey(1), (N,))
    if use_jit:
        log_joint = jit(lambda *args: log_density(*args)[0], static_argnums=(0,))(
            model, (data, mask), {}, {'x': x, 'y': x})
    else:
        log_joint = log_density(model, (data, mask), {}, {'x': x, 'y': x})[0]
    log_prob_x = dist.Normal(0, 1).log_prob(x)
    log_prob_y = mask
    log_prob_z = dist.Normal(x, 1).log_prob(data)
    expected = (log_prob_x + jnp.where(mask,  log_prob_y + 2 * log_prob_z, 0.)).sum()
    assert_allclose(log_joint, expected, atol=1e-4)


def test_mask_inf():
    def model():
        with handlers.mask(mask=jnp.zeros(10, dtype=bool)):
            numpyro.factor('inf', -jnp.inf)

    log_joint = log_density(model, (), {}, {})[0]
    assert_allclose(log_joint, 0.)


@pytest.mark.parametrize('use_context_manager', [True, False])
def test_scale(use_context_manager):
    def model(data):
        x = numpyro.sample('x', dist.Normal(0, 1))
        with optional(use_context_manager, handlers.scale(scale=10)):
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
        return jnp.stack([x, y])

    xs = []
    for i in range(100):
        with handlers.seed(rng_seed=i):
            xs.append(_sample())
    xs = jnp.stack(xs)

    ys = vmap(lambda rng_key: handlers.seed(lambda: _sample(), rng_key)())(jnp.arange(100))
    assert_allclose(xs, ys, atol=1e-6)


def test_nested_seeding():
    def fn(rng_key_1, rng_key_2, rng_key_3):
        xs = []
        with handlers.seed(rng_seed=rng_key_1):
            with handlers.seed(rng_seed=rng_key_2):
                xs.append(numpyro.sample('x', dist.Normal(0., 1.)))
                with handlers.seed(rng_seed=rng_key_3):
                    xs.append(numpyro.sample('y', dist.Normal(0., 1.)))
        return jnp.stack(xs)

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
    assert handlers.condition(model, {'y': 3.})() == 3.


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


def model_nested_plates_3():
    outer = numpyro.plate('outer', 10, dim=-1)
    inner = numpyro.plate('inner', 5, dim=-2)
    numpyro.deterministic('z', 1.)

    with inner, outer:
        xy = numpyro.sample('xy', dist.Normal(jnp.zeros((5, 10)), 1.))
        assert xy.shape == (5, 10)


def model_dist_batch_shape():
    outer = numpyro.plate('outer', 10)
    inner = numpyro.plate('inner', 5, dim=-3)
    with outer:
        x = numpyro.sample('x', dist.Normal(jnp.zeros(10), 1.))
        assert x.shape == (10,)
    with inner:
        y = numpyro.sample('y', dist.Normal(0., jnp.ones(10)))
        assert y.shape == (5, 1, 10)
        z = numpyro.deterministic('z', x ** 2)
        assert z.shape == (10,)

    with outer, inner:
        xy = numpyro.sample('xy', dist.Normal(0., jnp.ones(10)), sample_shape=(10,))
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
    model_nested_plates_3,
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


def test_subsample_data():
    data = jnp.arange(100.)
    subsample_size = 7
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        with numpyro.plate("a", len(data), subsample_size=subsample_size) as idx:
            assert data[idx].shape == (subsample_size,)


def test_subsample_substitute():
    data = jnp.arange(100.)
    subsample_size = 7
    subsample = jnp.array([13, 3, 30, 4, 1, 68, 5])
    with handlers.trace() as tr, handlers.seed(rng_seed=0), handlers.substitute(data={"a": subsample}):
        with numpyro.plate("a", len(data), subsample_size=subsample_size) as idx:
            assert data[idx].shape == (subsample_size,)
            assert_allclose(idx, subsample)
    assert tr["a"]["kwargs"]["rng_key"] is None


def test_messenger_fn_invalid():
    with pytest.raises(ValueError, match="to be a Python callable object"):
        with numpyro.handlers.mask(False):
            pass


@pytest.mark.parametrize('shape', [(), (5,), (2, 3)])
def test_plate_stack(shape):
    def guide():
        with numpyro.plate_stack("plates", shape):
            return numpyro.sample("x", dist.Normal(0, 1))

    x = handlers.seed(guide, 0)()
    assert x.shape == shape


@pytest.mark.parametrize('intervene,observe,flip', [
    (True, False, False),
    (False, True, False),
    (True, True, False),
    (True, True, True),
])
def test_counterfactual_query(intervene, observe, flip):
    # x -> y -> z -> w

    sites = ["x", "y", "z", "w"]
    observations = {"x": 1., "y": None, "z": 1., "w": 1.}
    interventions = {"x": None, "y": 0., "z": 2., "w": 1.}

    def model():
        with handlers.seed(rng_seed=0):
            x = numpyro.sample("x", dist.Normal(0, 1))
            y = numpyro.sample("y", dist.Normal(x, 1))
            z = numpyro.sample("z", dist.Normal(y, 1))
            w = numpyro.sample("w", dist.Normal(z, 1))
            return dict(x=x, y=y, z=z, w=w)

    if not flip:
        if intervene:
            model = handlers.do(model, data=interventions)
        if observe:
            model = handlers.condition(model, data=observations)
    elif flip and intervene and observe:
        model = handlers.do(
            handlers.condition(model, data=observations),
            data=interventions)

    with handlers.trace() as tr:
        actual_values = model()
    for name in sites:
        # case 1: purely observational query like handlers.condition
        if not intervene and observe:
            if observations[name] is not None:
                assert tr[name]['is_observed']
                assert_allclose(observations[name], actual_values[name])
                assert_allclose(observations[name], tr[name]['value'])
            if interventions[name] != observations[name]:
                if interventions[name] is not None:
                    assert_raises(AssertionError, assert_allclose, interventions[name], actual_values[name])
        # case 2: purely interventional query like old handlers.do
        elif intervene and not observe:
            assert not tr[name]['is_observed']
            if interventions[name] is not None:
                assert_allclose(interventions[name], actual_values[name])
            if observations[name] is not None:
                assert_raises(AssertionError, assert_allclose, observations[name], tr[name]['value'])
            if interventions[name] is not None:
                assert_raises(AssertionError, assert_allclose, interventions[name], tr[name]['value'])
        # case 3: counterfactual query mixing intervention and observation
        elif intervene and observe:
            if observations[name] is not None:
                assert tr[name]['is_observed']
                assert_allclose(observations[name], tr[name]['value'])
            if interventions[name] is not None:
                assert_allclose(interventions[name], actual_values[name])
            if interventions[name] != observations[name]:
                if interventions[name] is not None:
                    assert_raises(AssertionError, assert_allclose, interventions[name], tr[name]['value'])


def test_block():
    with handlers.trace() as trace:
        with handlers.block(hide=['x']):
            with handlers.seed(rng_seed=0):
                numpyro.sample('x', dist.Normal())
    assert 'x' not in trace


def test_scope():
    def fn():
        return numpyro.sample('x', dist.Normal())

    with handlers.trace() as trace:
        with handlers.seed(rng_seed=1):
            with handlers.scope(prefix='a'):
                fn()
            with handlers.scope(prefix='b'):
                with handlers.scope(prefix='a'):
                    fn()

    assert 'a/x' in trace
    assert 'b/a/x' in trace


def test_lift():
    def model():
        loc1 = numpyro.param("loc1", 0.)
        scale1 = numpyro.param("scale1", 1., constraint=constraints.positive)
        numpyro.sample("latent1", dist.Normal(loc1, scale1))

        loc2 = numpyro.param("loc2", 1.)
        scale2 = numpyro.param("scale2", 2., constraint=constraints.positive)
        latent2 = numpyro.sample("latent2", dist.Normal(loc2, scale2))
        return latent2

    loc1_prior = dist.Normal()
    scale1_prior = dist.LogNormal()
    prior = {"loc1": loc1_prior, "scale1": scale1_prior}

    with handlers.trace() as tr:
        with handlers.seed(rng_seed=1):
            model()

    with handlers.trace() as lifted_tr:
        with handlers.seed(rng_seed=2):
            with handlers.lift(prior=prior):
                model()

    for name in tr.keys():
        assert name in lifted_tr
        if name in prior:
            assert lifted_tr[name]['fn'] is prior[name]
            assert lifted_tr[name]['type'] == 'sample'
            assert lifted_tr[name]['value'] not in (0., 1.)
        elif name in ('loc2', 'scale2'):
            assert lifted_tr[name]['type'] == 'param'


def test_lift_memoize():
    def model():
        a = numpyro.param("loc")
        b = numpyro.param("loc")
        assert a == b

    with handlers.seed(rng_seed=1):
        with handlers.lift(prior=dist.Normal(0, 1)):
            model()
