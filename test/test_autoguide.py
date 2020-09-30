# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

from numpy.testing import assert_allclose
import pytest

from jax import jit, lax, random
from jax.experimental.stax import Dense
import jax.numpy as jnp
from jax.test_util import check_eq

import numpyro
from numpyro import handlers, optim
from numpyro.contrib.control_flow import scan
import numpyro.distributions as dist
from numpyro.distributions import constraints, transforms
from numpyro.distributions.flows import InverseAutoregressiveTransform
from numpyro.handlers import substitute
from numpyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from numpyro.infer.autoguide import (
    AutoBNAFNormal,
    AutoDiagonalNormal,
    AutoIAFNormal,
    AutoLaplaceApproximation,
    AutoLowRankMultivariateNormal,
    AutoMultivariateNormal,
    AutoNormal
)
from numpyro.infer.initialization import init_to_median
from numpyro.infer.reparam import TransformReparam
from numpyro.nn.auto_reg_nn import AutoregressiveNN
from numpyro.util import fori_loop

init_strategy = init_to_median(num_samples=2)


@pytest.mark.parametrize('auto_class', [
    AutoDiagonalNormal,
    AutoIAFNormal,
    AutoBNAFNormal,
    AutoMultivariateNormal,
    AutoLaplaceApproximation,
    AutoLowRankMultivariateNormal,
    AutoNormal,
])
def test_beta_bernoulli(auto_class):
    data = jnp.array([[1.0] * 8 + [0.0] * 2,
                     [1.0] * 4 + [0.0] * 6]).T

    def model(data):
        f = numpyro.sample('beta', dist.Beta(jnp.ones(2), jnp.ones(2)))
        numpyro.sample('obs', dist.Bernoulli(f), obs=data)

    adam = optim.Adam(0.01)
    guide = auto_class(model, init_loc_fn=init_strategy)
    svi = SVI(model, guide, adam, Trace_ELBO())
    svi_state = svi.init(random.PRNGKey(1), data)

    def body_fn(i, val):
        svi_state, loss = svi.update(val, data)
        return svi_state

    svi_state = fori_loop(0, 3000, body_fn, svi_state)
    params = svi.get_params(svi_state)
    true_coefs = (jnp.sum(data, axis=0) + 1) / (data.shape[0] + 2)
    # test .sample_posterior method
    posterior_samples = guide.sample_posterior(random.PRNGKey(1), params, sample_shape=(1000,))
    assert_allclose(jnp.mean(posterior_samples['beta'], 0), true_coefs, atol=0.05)


@pytest.mark.parametrize('auto_class', [
    AutoDiagonalNormal,
    AutoIAFNormal,
    AutoBNAFNormal,
    AutoMultivariateNormal,
    AutoLaplaceApproximation,
    AutoLowRankMultivariateNormal,
    AutoNormal,
])
@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceMeanField_ELBO])
def test_logistic_regression(auto_class, Elbo):
    N, dim = 3000, 3
    data = random.normal(random.PRNGKey(0), (N, dim))
    true_coefs = jnp.arange(1., dim + 1.)
    logits = jnp.sum(true_coefs * data, axis=-1)
    labels = dist.Bernoulli(logits=logits).sample(random.PRNGKey(1))

    def model(data, labels):
        coefs = numpyro.sample('coefs', dist.Normal(jnp.zeros(dim), jnp.ones(dim)))
        logits = jnp.sum(coefs * data, axis=-1)
        return numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=labels)

    adam = optim.Adam(0.01)
    rng_key_init = random.PRNGKey(1)
    guide = auto_class(model, init_loc_fn=init_strategy)
    svi = SVI(model, guide, adam, Elbo())
    svi_state = svi.init(rng_key_init, data, labels)

    # smoke test if analytic KL is used
    if auto_class is AutoNormal and Elbo is TraceMeanField_ELBO:
        _, mean_field_loss = svi.update(svi_state, data, labels)
        svi.loss = Trace_ELBO()
        _, elbo_loss = svi.update(svi_state, data, labels)
        svi.loss = TraceMeanField_ELBO()
        assert abs(mean_field_loss - elbo_loss) > 0.5

    def body_fn(i, val):
        svi_state, loss = svi.update(val, data, labels)
        return svi_state

    svi_state = fori_loop(0, 2000, body_fn, svi_state)
    params = svi.get_params(svi_state)
    if auto_class not in (AutoIAFNormal, AutoBNAFNormal):
        median = guide.median(params)
        assert_allclose(median['coefs'], true_coefs, rtol=0.1)
        # test .quantile method
        median = guide.quantiles(params, [0.2, 0.5])
        assert_allclose(median['coefs'][1], true_coefs, rtol=0.1)
    # test .sample_posterior method
    posterior_samples = guide.sample_posterior(random.PRNGKey(1), params, sample_shape=(1000,))
    assert_allclose(jnp.mean(posterior_samples['coefs'], 0), true_coefs, rtol=0.1)


def test_iaf():
    # test for substitute logic for exposed methods `sample_posterior` and `get_transforms`
    N, dim = 3000, 3
    data = random.normal(random.PRNGKey(0), (N, dim))
    true_coefs = jnp.arange(1., dim + 1.)
    logits = jnp.sum(true_coefs * data, axis=-1)
    labels = dist.Bernoulli(logits=logits).sample(random.PRNGKey(1))

    def model(data, labels):
        coefs = numpyro.sample('coefs', dist.Normal(jnp.zeros(dim), jnp.ones(dim)))
        offset = numpyro.sample('offset', dist.Uniform(-1, 1))
        logits = offset + jnp.sum(coefs * data, axis=-1)
        return numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=labels)

    adam = optim.Adam(0.01)
    rng_key_init = random.PRNGKey(1)
    guide = AutoIAFNormal(model)
    svi = SVI(model, guide, adam, Trace_ELBO())
    svi_state = svi.init(rng_key_init, data, labels)
    params = svi.get_params(svi_state)

    x = random.normal(random.PRNGKey(0), (dim + 1,))
    rng_key = random.PRNGKey(1)
    actual_sample = guide.sample_posterior(rng_key, params)
    actual_output = guide._unpack_latent(guide.get_transform(params)(x))

    flows = []
    for i in range(guide.num_flows):
        if i > 0:
            flows.append(transforms.PermuteTransform(jnp.arange(dim + 1)[::-1]))
        arn_init, arn_apply = AutoregressiveNN(dim + 1, [dim + 1, dim + 1],
                                               permutation=jnp.arange(dim + 1),
                                               skip_connections=guide._skip_connections,
                                               nonlinearity=guide._nonlinearity)
        arn = partial(arn_apply, params['auto_arn__{}$params'.format(i)])
        flows.append(InverseAutoregressiveTransform(arn))
    flows.append(guide._unpack_latent)

    transform = transforms.ComposeTransform(flows)
    _, rng_key_sample = random.split(rng_key)
    expected_sample = transform(dist.Normal(jnp.zeros(dim + 1), 1).sample(rng_key_sample))
    expected_output = transform(x)
    assert_allclose(actual_sample['coefs'], expected_sample['coefs'])
    assert_allclose(actual_sample['offset'],
                    transforms.biject_to(constraints.interval(-1, 1))(expected_sample['offset']))
    check_eq(actual_output, expected_output)


def test_uniform_normal():
    true_coef = 0.9
    data = true_coef + random.normal(random.PRNGKey(0), (1000,))

    def model(data):
        alpha = numpyro.sample('alpha', dist.Uniform(0, 1))
        with numpyro.handlers.reparam(config={'loc': TransformReparam()}):
            loc = numpyro.sample('loc', dist.Uniform(0, alpha))
        numpyro.sample('obs', dist.Normal(loc, 0.1), obs=data)

    adam = optim.Adam(0.01)
    rng_key_init = random.PRNGKey(1)
    guide = AutoDiagonalNormal(model)
    svi = SVI(model, guide, adam, Trace_ELBO())
    svi_state = svi.init(rng_key_init, data)

    def body_fn(i, val):
        svi_state, loss = svi.update(val, data)
        return svi_state

    svi_state = fori_loop(0, 1000, body_fn, svi_state)
    params = svi.get_params(svi_state)
    median = guide.median(params)
    assert_allclose(median['loc'], true_coef, rtol=0.05)
    # test .quantile method
    median = guide.quantiles(params, [0.2, 0.5])
    assert_allclose(median['loc'][1], true_coef, rtol=0.1)


def test_param():
    # this test the validity of model having
    # param sites contain composed transformed constraints
    rng_keys = random.split(random.PRNGKey(0), 3)
    a_minval = 1
    a_init = jnp.exp(random.normal(rng_keys[0])) + a_minval
    b_init = jnp.exp(random.normal(rng_keys[1]))
    x_init = random.normal(rng_keys[2])

    def model():
        a = numpyro.param('a', a_init, constraint=constraints.greater_than(a_minval))
        b = numpyro.param('b', b_init, constraint=constraints.positive)
        numpyro.sample('x', dist.Normal(a, b))

    # this class is used to force init value of `x` to x_init
    class _AutoGuide(AutoDiagonalNormal):
        def __call__(self, *args, **kwargs):
            return substitute(super(_AutoGuide, self).__call__,
                              {'_auto_latent': x_init})(*args, **kwargs)

    adam = optim.Adam(0.01)
    rng_key_init = random.PRNGKey(1)
    guide = _AutoGuide(model)
    svi = SVI(model, guide, adam, Trace_ELBO())
    svi_state = svi.init(rng_key_init)

    params = svi.get_params(svi_state)
    assert_allclose(params['a'], a_init)
    assert_allclose(params['b'], b_init)
    assert_allclose(params['auto_loc'], guide._init_latent)
    assert_allclose(params['auto_scale'], jnp.ones(1) * guide._init_scale)

    actual_loss = svi.evaluate(svi_state)
    assert jnp.isfinite(actual_loss)
    expected_loss = dist.Normal(guide._init_latent, guide._init_scale).log_prob(x_init) \
        - dist.Normal(a_init, b_init).log_prob(x_init)
    assert_allclose(actual_loss, expected_loss, rtol=1e-6)


def test_dynamic_supports():
    true_coef = 0.9
    data = true_coef + random.normal(random.PRNGKey(0), (1000,))

    def actual_model(data):
        alpha = numpyro.sample('alpha', dist.Uniform(0, 1))
        with numpyro.handlers.reparam(config={'loc': TransformReparam()}):
            loc = numpyro.sample('loc', dist.Uniform(0, alpha))
        numpyro.sample('obs', dist.Normal(loc, 0.1), obs=data)

    def expected_model(data):
        alpha = numpyro.sample('alpha', dist.Uniform(0, 1))
        loc = numpyro.sample('loc', dist.Uniform(0, 1)) * alpha
        numpyro.sample('obs', dist.Normal(loc, 0.1), obs=data)

    adam = optim.Adam(0.01)
    rng_key_init = random.PRNGKey(1)

    guide = AutoDiagonalNormal(actual_model)
    svi = SVI(actual_model, guide, adam, Trace_ELBO())
    svi_state = svi.init(rng_key_init, data)
    actual_opt_params = adam.get_params(svi_state.optim_state)
    actual_params = svi.get_params(svi_state)
    actual_values = guide.median(actual_params)
    actual_loss = svi.evaluate(svi_state, data)

    guide = AutoDiagonalNormal(expected_model)
    svi = SVI(expected_model, guide, adam, Trace_ELBO())
    svi_state = svi.init(rng_key_init, data)
    expected_opt_params = adam.get_params(svi_state.optim_state)
    expected_params = svi.get_params(svi_state)
    expected_values = guide.median(expected_params)
    expected_loss = svi.evaluate(svi_state, data)

    # test auto_loc, auto_scale
    check_eq(actual_opt_params, expected_opt_params)
    check_eq(actual_params, expected_params)
    # test latent values
    assert_allclose(actual_values['alpha'], expected_values['alpha'])
    assert_allclose(actual_values['loc_base'], expected_values['loc'])
    assert_allclose(actual_loss, expected_loss)


def test_laplace_approximation_warning():
    def model(x, y):
        a = numpyro.sample("a", dist.Normal(0, 10))
        b = numpyro.sample("b", dist.Normal(0, 10), sample_shape=(3,))
        mu = a + b[0] * x + b[1] * x ** 2 + b[2] * x ** 3
        numpyro.sample("y", dist.Normal(mu, 0.001), obs=y)

    x = random.normal(random.PRNGKey(0), (3,))
    y = 1 + 2 * x + 3 * x ** 2 + 4 * x ** 3
    guide = AutoLaplaceApproximation(model)
    svi = SVI(model, guide, optim.Adam(0.1), Trace_ELBO(), x=x, y=y)
    init_state = svi.init(random.PRNGKey(0))
    svi_state = fori_loop(0, 10000, lambda i, val: svi.update(val)[0], init_state)
    params = svi.get_params(svi_state)
    with pytest.warns(UserWarning, match="Hessian of log posterior"):
        guide.sample_posterior(random.PRNGKey(1), params)


def test_improper():
    y = random.normal(random.PRNGKey(0), (100,))

    def model(y):
        lambda1 = numpyro.sample('lambda1', dist.ImproperUniform(dist.constraints.real, (), ()))
        lambda2 = numpyro.sample('lambda2', dist.ImproperUniform(dist.constraints.real, (), ()))
        sigma = numpyro.sample('sigma', dist.ImproperUniform(dist.constraints.positive, (), ()))
        mu = numpyro.deterministic('mu', lambda1 + lambda2)
        numpyro.sample('y', dist.Normal(mu, sigma), obs=y)

    guide = AutoDiagonalNormal(model)
    svi = SVI(model, guide, optim.Adam(0.003), Trace_ELBO(), y=y)
    svi_state = svi.init(random.PRNGKey(2))
    lax.scan(lambda state, i: svi.update(state), svi_state, jnp.zeros(10000))


def test_module():
    x = random.normal(random.PRNGKey(0), (100, 10))
    y = random.normal(random.PRNGKey(1), (100,))

    def model(x, y):
        nn = numpyro.module("nn", Dense(1), (10,))
        mu = nn(x).squeeze(-1)
        sigma = numpyro.sample("sigma", dist.HalfNormal(1))
        numpyro.sample('y', dist.Normal(mu, sigma), obs=y)

    guide = AutoDiagonalNormal(model)
    svi = SVI(model, guide, optim.Adam(0.003), Trace_ELBO(), x=x, y=y)
    svi_state = svi.init(random.PRNGKey(2))
    lax.scan(lambda state, i: svi.update(state), svi_state, jnp.zeros(1000))


@pytest.mark.parametrize("auto_class", [AutoNormal])
def test_subsample_guide(auto_class):

    # The model adapted from tutorial/source/easyguide.ipynb
    def model(batch, subsample, full_size):
        drift = numpyro.sample("drift", dist.LogNormal(-1, 0.5))
        with handlers.substitute(data={"data": subsample}):
            plate = numpyro.plate("data", full_size, subsample_size=len(subsample))
        assert plate.size == 50

        def transition_fn(z_prev, y_curr):
            with plate:
                z_curr = numpyro.sample("state", dist.Normal(z_prev, drift))
                y_curr = numpyro.sample("obs", dist.Bernoulli(logits=z_curr), obs=y_curr)
            return z_curr, y_curr

        _, result = scan(transition_fn, jnp.zeros(len(subsample)), batch, length=num_time_steps)
        return result

    def create_plates(batch, subsample, full_size):
        with handlers.substitute(data={"data": subsample}):
            return numpyro.plate("data", full_size, subsample_size=subsample.shape[0])

    guide = auto_class(model, create_plates=create_plates)

    full_size = 50
    batch_size = 20
    num_time_steps = 8
    with handlers.seed(rng_seed=0):
        data = model(None, jnp.arange(full_size), full_size)
    assert data.shape == (num_time_steps, full_size)

    svi = SVI(model, guide, optim.Adam(0.02), Trace_ELBO())
    svi_state = svi.init(random.PRNGKey(0), data[:, :batch_size],
                         jnp.arange(batch_size), full_size=full_size)
    update_fn = jit(svi.update, static_argnums=(3,))
    for epoch in range(2):
        beg = 0
        while beg < full_size:
            end = min(full_size, beg + batch_size)
            subsample = jnp.arange(beg, end)
            batch = data[:, beg:end]
            beg = end
            svi_state, loss = update_fn(svi_state, batch, subsample, full_size)
