from functools import partial

from numpy.testing import assert_allclose
import pytest

from jax import random
import jax.numpy as np
from jax.test_util import check_eq

import numpyro
from numpyro import optim
from numpyro.contrib.autoguide import (
    AutoContinuousELBO,
    AutoDiagonalNormal,
    AutoIAFNormal,
    AutoLaplaceApproximation,
    AutoMultivariateNormal
)
from numpyro.contrib.nn.auto_reg_nn import AutoregressiveNN
import numpyro.distributions as dist
from numpyro.distributions import constraints, transforms
from numpyro.distributions.flows import InverseAutoregressiveTransform
from numpyro.handlers import substitute
from numpyro.infer import SVI
from numpyro.infer.util import init_to_median
from numpyro.util import fori_loop

init_strategy = init_to_median(num_samples=2)


@pytest.mark.parametrize('auto_class', [
    AutoDiagonalNormal,
    AutoIAFNormal,
    AutoMultivariateNormal,
    AutoLaplaceApproximation,
])
def test_beta_bernoulli(auto_class):
    data = np.array([[1.0] * 8 + [0.0] * 2,
                     [1.0] * 4 + [0.0] * 6]).T

    def model(data):
        f = numpyro.sample('beta', dist.Beta(np.ones(2), np.ones(2)))
        numpyro.sample('obs', dist.Bernoulli(f), obs=data)

    adam = optim.Adam(0.01)
    guide = auto_class(model, init_strategy=init_strategy)
    svi = SVI(model, guide, adam, AutoContinuousELBO())
    svi_state = svi.init(random.PRNGKey(1), data)

    def body_fn(i, val):
        svi_state, loss = svi.update(val, data)
        return svi_state

    svi_state = fori_loop(0, 2000, body_fn, svi_state)
    params = svi.get_params(svi_state)
    true_coefs = (np.sum(data, axis=0) + 1) / (data.shape[0] + 2)
    # test .sample_posterior method
    posterior_samples = guide.sample_posterior(random.PRNGKey(1), params, sample_shape=(1000,))
    assert_allclose(np.mean(posterior_samples['beta'], 0), true_coefs, atol=0.04)


@pytest.mark.parametrize('auto_class', [
    AutoDiagonalNormal,
    AutoIAFNormal,
    AutoMultivariateNormal,
    AutoLaplaceApproximation,
])
def test_logistic_regression(auto_class):
    N, dim = 3000, 3
    data = random.normal(random.PRNGKey(0), (N, dim))
    true_coefs = np.arange(1., dim + 1.)
    logits = np.sum(true_coefs * data, axis=-1)
    labels = dist.Bernoulli(logits=logits).sample(random.PRNGKey(1))

    def model(data, labels):
        coefs = numpyro.sample('coefs', dist.Normal(np.zeros(dim), np.ones(dim)))
        logits = np.sum(coefs * data, axis=-1)
        return numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=labels)

    adam = optim.Adam(0.01)
    rng_init = random.PRNGKey(1)
    guide = auto_class(model, init_strategy=init_strategy)
    svi = SVI(model, guide, adam, AutoContinuousELBO())
    svi_state = svi.init(rng_init, data, labels)

    def body_fn(i, val):
        svi_state, loss = svi.update(val, data, labels)
        return svi_state

    svi_state = fori_loop(0, 2000, body_fn, svi_state)
    params = svi.get_params(svi_state)
    if auto_class is not AutoIAFNormal:
        median = guide.median(params)
        assert_allclose(median['coefs'], true_coefs, rtol=0.1)
        # test .quantile method
        median = guide.quantiles(params, [0.2, 0.5])
        assert_allclose(median['coefs'][1], true_coefs, rtol=0.1)
    # test .sample_posterior method
    posterior_samples = guide.sample_posterior(random.PRNGKey(1), params, sample_shape=(1000,))
    assert_allclose(np.mean(posterior_samples['coefs'], 0), true_coefs, rtol=0.1)


def test_iaf():
    # test for substitute logic for exposed methods `sample_posterior` and `get_transforms`
    N, dim = 3000, 3
    data = random.normal(random.PRNGKey(0), (N, dim))
    true_coefs = np.arange(1., dim + 1.)
    logits = np.sum(true_coefs * data, axis=-1)
    labels = dist.Bernoulli(logits=logits).sample(random.PRNGKey(1))

    def model(data, labels):
        coefs = numpyro.sample('coefs', dist.Normal(np.zeros(dim), np.ones(dim)))
        offset = numpyro.sample('offset', dist.Uniform(-1, 1))
        logits = offset + np.sum(coefs * data, axis=-1)
        return numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=labels)

    adam = optim.Adam(0.01)
    rng_init = random.PRNGKey(1)
    guide = AutoIAFNormal(model)
    svi = SVI(model, guide, adam, AutoContinuousELBO())
    svi_state = svi.init(rng_init, data, labels)
    params = svi.get_params(svi_state)

    x = random.normal(random.PRNGKey(0), (dim + 1,))
    rng = random.PRNGKey(1)
    actual_sample = guide.sample_posterior(rng, params)
    actual_output = guide.get_transform(params)(x)

    flows = []
    for i in range(guide.num_flows):
        if i > 0:
            flows.append(transforms.PermuteTransform(np.arange(dim + 1)[::-1]))
        arn_init, arn_apply = AutoregressiveNN(dim + 1, [dim + 1, dim + 1],
                                               permutation=np.arange(dim + 1),
                                               skip_connections=guide._skip_connections,
                                               nonlinearity=guide._nonlinearity)
        arn = partial(arn_apply, params['auto_arn__{}$params'.format(i)])
        flows.append(InverseAutoregressiveTransform(arn))
    flows.append(transforms.UnpackTransform(guide._unpack_latent))

    transform = transforms.ComposeTransform(flows)
    rng_seed, rng_sample = random.split(rng)
    expected_sample = transform(dist.Normal(np.zeros(dim + 1), 1).sample(rng_sample))
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
        loc = numpyro.sample('loc', dist.Uniform(0, alpha))
        numpyro.sample('obs', dist.Normal(loc, 0.1), obs=data)

    adam = optim.Adam(0.01)
    rng_init = random.PRNGKey(1)
    guide = AutoDiagonalNormal(model)
    svi = SVI(model, guide, adam, AutoContinuousELBO())
    svi_state = svi.init(rng_init, data)

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
    rngs = random.split(random.PRNGKey(0), 3)
    a_minval = 1
    a_init = np.exp(random.normal(rngs[0])) + a_minval
    b_init = np.exp(random.normal(rngs[1]))
    x_init = random.normal(rngs[2])

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
    rng_init = random.PRNGKey(1)
    guide = _AutoGuide(model)
    svi = SVI(model, guide, adam, AutoContinuousELBO())
    svi_state = svi.init(rng_init)

    params = svi.get_params(svi_state)
    assert_allclose(params['a'], a_init)
    assert_allclose(params['b'], b_init)
    assert_allclose(params['auto_loc'], guide._init_latent)
    assert_allclose(params['auto_scale'], np.ones(1))

    actual_loss = svi.evaluate(svi_state)
    assert np.isfinite(actual_loss)
    expected_loss = dist.Normal(guide._init_latent, 1).log_prob(x_init) - dist.Normal(a_init, b_init).log_prob(x_init)
    assert_allclose(actual_loss, expected_loss)


def test_dynamic_supports():
    true_coef = 0.9
    data = true_coef + random.normal(random.PRNGKey(0), (1000,))

    def actual_model(data):
        alpha = numpyro.sample('alpha', dist.Uniform(0, 1))
        loc = numpyro.sample('loc', dist.Uniform(0, alpha))
        numpyro.sample('obs', dist.Normal(loc, 0.1), obs=data)

    def expected_model(data):
        alpha = numpyro.sample('alpha', dist.Uniform(0, 1))
        loc = numpyro.sample('loc', dist.Uniform(0, 1)) * alpha
        numpyro.sample('obs', dist.Normal(loc, 0.1), obs=data)

    adam = optim.Adam(0.01)
    rng_init = random.PRNGKey(1)

    guide = AutoDiagonalNormal(actual_model)
    svi = SVI(actual_model, guide, adam, AutoContinuousELBO())
    svi_state = svi.init(rng_init, data)
    actual_opt_params = adam.get_params(svi_state.optim_state)
    actual_params = svi.get_params(svi_state)
    actual_values = guide.median(actual_params)
    actual_loss = svi.evaluate(svi_state, data)

    guide = AutoDiagonalNormal(expected_model)
    svi = SVI(expected_model, guide, adam, AutoContinuousELBO())
    svi_state = svi.init(rng_init, data)
    expected_opt_params = adam.get_params(svi_state.optim_state)
    expected_params = svi.get_params(svi_state)
    expected_values = guide.median(expected_params)
    expected_loss = svi.evaluate(svi_state, data)

    # test auto_loc, auto_scale
    check_eq(actual_opt_params, expected_opt_params)
    check_eq(actual_params, expected_params)
    # test latent values
    assert_allclose(actual_values['alpha'], expected_values['alpha'])
    assert_allclose(actual_values['loc'], expected_values['alpha'] * expected_values['loc'])
    assert_allclose(actual_loss, expected_loss)


def test_elbo_dynamic_support():
    x_prior = dist.Uniform(0, 5)
    x_unconstrained = 2.

    def model():
        numpyro.sample('x', x_prior)

    class _AutoGuide(AutoDiagonalNormal):
        def __call__(self, *args, **kwargs):
            return substitute(super(_AutoGuide, self).__call__,
                              {'_auto_latent': x_unconstrained})(*args, **kwargs)

    adam = optim.Adam(0.01)
    guide = _AutoGuide(model)
    svi = SVI(model, guide, adam, AutoContinuousELBO())
    svi_state = svi.init(random.PRNGKey(0))
    actual_loss = svi.evaluate(svi_state)
    assert np.isfinite(actual_loss)

    guide_log_prob = dist.Normal(guide._init_latent).log_prob(x_unconstrained).sum()
    transfrom = transforms.biject_to(constraints.interval(0, 5))
    x = transfrom(x_unconstrained)
    logdet = transfrom.log_abs_det_jacobian(x_unconstrained, x)
    model_log_prob = x_prior.log_prob(x) + logdet
    expected_loss = guide_log_prob - model_log_prob
    assert_allclose(actual_loss, expected_loss)
