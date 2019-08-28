from numpy.testing import assert_allclose
import pytest

from jax import random
import jax.numpy as np
from jax.test_util import check_eq

import numpyro
from numpyro import optim
from numpyro.contrib.autoguide import AutoDiagonalNormal, AutoIAFNormal
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.handlers import substitute
from numpyro.svi import elbo, svi
from numpyro.util import fori_loop


@pytest.mark.parametrize('auto_class', [
    AutoDiagonalNormal,
    AutoIAFNormal,
])
def test_beta_bernoulli(auto_class):
    data = np.array([[1.0] * 8 + [0.0] * 2,
                     [1.0] * 4 + [0.0] * 6]).T

    def model(data):
        f = numpyro.sample('beta', dist.Beta(np.ones(2), np.ones(2)))
        numpyro.sample('obs', dist.Bernoulli(f), obs=data)

    adam = optim.Adam(0.01)
    rng_guide, rng_init, rng_train = random.split(random.PRNGKey(1), 3)
    guide = auto_class(rng_guide, model)
    svi_init, svi_update, _ = svi(model, guide, elbo, adam)
    opt_state, get_params = svi_init(rng_init, model_args=(data,), guide_args=(data,))

    def body_fn(i, val):
        opt_state_, rng_ = val
        loss, opt_state_, rng_ = svi_update(rng_, opt_state_, model_args=(data,), guide_args=(data,))
        return opt_state_, rng_

    opt_state, _ = fori_loop(0, 1000, body_fn, (opt_state, rng_train))
    params = get_params(opt_state)
    true_coefs = (np.sum(data, axis=0) + 1) / (data.shape[0] + 2)
    # test .sample_posterior method
    posterior_samples = guide.sample_posterior(random.PRNGKey(1), params, sample_shape=(1000,))
    assert_allclose(np.mean(posterior_samples['beta'], 0), true_coefs, atol=0.04)


@pytest.mark.parametrize('auto_class', [
    AutoDiagonalNormal,
    AutoIAFNormal,
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
    rng_guide, rng_init, rng_train = random.split(random.PRNGKey(1), 3)
    guide = auto_class(rng_guide, model)
    svi_init, svi_update, _ = svi(model, guide, elbo, adam)
    opt_state, get_params = svi_init(rng_init, model_args=(data, labels), guide_args=(data, labels))

    def body_fn(i, val):
        opt_state_, rng_ = val
        loss, opt_state_, rng_ = svi_update(rng_, opt_state_,
                                            model_args=(data, labels),
                                            guide_args=(data, labels))
        return opt_state_, rng_

    opt_state, _ = fori_loop(0, 1000, body_fn, (opt_state, rng_train))
    params = get_params(opt_state)
    if auto_class is not AutoIAFNormal:
        median = guide.median(params)
        assert_allclose(median['coefs'], true_coefs, rtol=0.1)
        # test .quantile method
        median = guide.quantiles(params, [0.2, 0.5])
        assert_allclose(median['coefs'][1], true_coefs, rtol=0.1)
    # test .sample_posterior method
    posterior_samples = guide.sample_posterior(random.PRNGKey(1), params, sample_shape=(1000,))
    assert_allclose(np.mean(posterior_samples['coefs'], 0), true_coefs, rtol=0.1)


def test_uniform_normal():
    true_coef = 0.9
    data = true_coef + random.normal(random.PRNGKey(0), (1000,))

    def model(data):
        alpha = numpyro.sample('alpha', dist.Uniform(0, 1))
        loc = numpyro.sample('loc', dist.Uniform(0, alpha))
        numpyro.sample('obs', dist.Normal(loc, 0.1), obs=data)

    adam = optim.Adam(0.01)
    rng_guide, rng_init, rng_train = random.split(random.PRNGKey(1), 3)
    guide = AutoDiagonalNormal(rng_guide, model)
    svi_init, svi_update, _ = svi(model, guide, elbo, adam)
    opt_state, get_params = svi_init(rng_init, model_args=(data,), guide_args=(data,))

    def body_fn(i, val):
        opt_state_, rng_ = val
        loss, opt_state_, rng_ = svi_update(rng_, opt_state_, model_args=(data,), guide_args=(data,))
        return opt_state_, rng_

    opt_state, _ = fori_loop(0, 1000, body_fn, (opt_state, rng_train))
    params = get_params(opt_state)
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
    rng_guide, rng_init, rng_train = random.split(random.PRNGKey(1), 3)
    guide = _AutoGuide(rng_guide, model)
    svi_init, _, svi_eval = svi(model, guide, elbo, adam)
    opt_state, get_params = svi_init(rng_init)

    params = get_params(opt_state)
    assert_allclose(params['a'], a_init)
    assert_allclose(params['b'], b_init)
    assert_allclose(params['auto_loc'], guide._init_latent)
    assert_allclose(params['auto_scale'], np.ones(1))

    actual_loss = svi_eval(random.PRNGKey(1), opt_state)
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
    rng_guide, rng_init, rng_train = random.split(random.PRNGKey(1), 3)

    guide = AutoDiagonalNormal(rng_guide, actual_model)
    svi_init, _, svi_eval = svi(actual_model, guide, elbo, adam)
    opt_state, get_params = svi_init(rng_init, (data,), (data,))
    actual_opt_params = adam.get_params(opt_state)
    actual_params = get_params(opt_state)
    actual_values = guide.median(actual_params)
    actual_loss = svi_eval(random.PRNGKey(1), opt_state, (data,), (data,))

    guide = AutoDiagonalNormal(rng_guide, expected_model)
    svi_init, _, svi_eval = svi(expected_model, guide, elbo, adam)
    opt_state, get_params = svi_init(rng_init, (data,), (data,))
    expected_opt_params = adam.get_params(opt_state)
    expected_params = get_params(opt_state)
    expected_values = guide.median(expected_params)
    expected_loss = svi_eval(random.PRNGKey(1), opt_state, (data,), (data,))

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
    guide = _AutoGuide(random.PRNGKey(0), model)
    svi_init, _, svi_eval = svi(model, guide, elbo, adam)
    opt_state, get_params = svi_init(random.PRNGKey(0), (), ())
    actual_loss = svi_eval(random.PRNGKey(1), opt_state)
    assert np.isfinite(actual_loss)

    guide_log_prob = dist.Normal(guide._init_latent).log_prob(x_unconstrained).sum()
    transfrom = constraints.biject_to(constraints.interval(0, 5))
    x = transfrom(x_unconstrained)
    logdet = transfrom.log_abs_det_jacobian(x_unconstrained, x)
    model_log_prob = x_prior.log_prob(x) + logdet
    expected_loss = guide_log_prob - model_log_prob
    assert_allclose(actual_loss, expected_loss)
