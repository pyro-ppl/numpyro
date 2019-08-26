from numpy.testing import assert_allclose
import pytest

from jax import random
from jax.experimental import optimizers
import jax.numpy as np
from jax.test_util import check_eq

import numpyro
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

    opt_init, opt_update, opt_params = optimizers.adam(0.01)
    rng_guide, rng_init, rng_train = random.split(random.PRNGKey(1), 3)
    guide = auto_class(rng_guide, model, opt_params)
    svi_init, svi_update, _ = svi(model, guide, elbo, opt_init, opt_update, opt_params)
    opt_state, get_params = svi_init(rng_init, model_args=(data,), guide_args=(data,))

    def body_fn(i, val):
        opt_state_, rng_ = val
        loss, opt_state_, rng_ = svi_update(i, rng_, opt_state_, model_args=(data,), guide_args=(data,))
        return opt_state_, rng_

    opt_state, _ = fori_loop(0, 1000, body_fn, (opt_state, rng_train))
    true_coefs = (np.sum(data, axis=0) + 1) / (data.shape[0] + 2)
    # test .sample_posterior method
    posterior_samples = guide.sample_posterior(random.PRNGKey(1), opt_state, sample_shape=(1000,))
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

    opt_init, opt_update, get_opt_params = optimizers.adam(0.01)
    rng_guide, rng_init, rng_train = random.split(random.PRNGKey(1), 3)
    guide = auto_class(rng_guide, model, get_opt_params)
    svi_init, svi_update, _ = svi(model, guide, elbo, opt_init, opt_update, get_opt_params)
    opt_state, constrain_fn = svi_init(rng_init,
                                       model_args=(data, labels),
                                       guide_args=(data, labels))

    def body_fn(i, val):
        opt_state_, rng_ = val
        loss, opt_state_, rng_ = svi_update(i, rng_, opt_state_,
                                            model_args=(data, labels),
                                            guide_args=(data, labels))
        return opt_state_, rng_

    opt_state, _ = fori_loop(0, 1000, body_fn, (opt_state, rng_train))
    if auto_class is not AutoIAFNormal:
        median = guide.median(opt_state)
        assert_allclose(median['coefs'], true_coefs, rtol=0.1)
        # test .quantile method
        median = guide.quantiles(opt_state, [0.2, 0.5])
        assert_allclose(median['coefs'][1], true_coefs, rtol=0.1)
    # test .sample_posterior method
    posterior_samples = guide.sample_posterior(random.PRNGKey(1), opt_state, sample_shape=(1000,))
    # TODO: reduce rtol to 0.1 when issues in autoguide is fixed
    assert_allclose(np.mean(posterior_samples['coefs'], 0), true_coefs, rtol=0.2)


def test_uniform_normal():
    true_coef = 0.9
    data = true_coef + random.normal(random.PRNGKey(0), (1000,))

    def model(data):
        alpha = numpyro.sample('alpha', dist.Uniform(0, 1))
        loc = numpyro.sample('loc', dist.Uniform(0, alpha))
        numpyro.sample('obs', dist.Normal(loc, 0.1), obs=data)

    opt_init, opt_update, get_opt_params = optimizers.adam(0.01)
    rng_guide, rng_init, rng_train = random.split(random.PRNGKey(1), 3)
    guide = AutoDiagonalNormal(rng_guide, model, get_opt_params)
    svi_init, svi_update, _ = svi(model, guide, elbo, opt_init, opt_update, get_opt_params)
    opt_state, get_params = svi_init(rng_init, model_args=(data,), guide_args=(data,))

    def body_fn(i, val):
        opt_state_, rng_ = val
        loss, opt_state_, rng_ = svi_update(i, rng_, opt_state_, model_args=(data,), guide_args=(data,))
        return opt_state_, rng_

    opt_state, _ = fori_loop(0, 1000, body_fn, (opt_state, rng_train))
    median = guide.median(opt_state)
    assert_allclose(median['loc'], true_coef, rtol=0.05)
    # test .quantile method
    median = guide.quantiles(opt_state, [0.2, 0.5])
    assert_allclose(median['loc'][1], true_coef, rtol=0.1)


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

    opt_init, opt_update, get_opt_params = optimizers.adam(0.01)
    rng_guide, rng_init, rng_train = random.split(random.PRNGKey(1), 3)

    guide = AutoDiagonalNormal(rng_guide, actual_model, get_opt_params)
    svi_init, _, svi_eval = svi(actual_model, guide, elbo, opt_init, opt_update, get_opt_params)
    opt_state, get_params = svi_init(rng_init, (data,), (data,))
    actual_params = get_opt_params(opt_state)
    actual_base_params = get_params(opt_state)
    actual_values = guide.median(opt_state)
    actual_loss = svi_eval(random.PRNGKey(1), opt_state, (data,), (data,))

    guide = AutoDiagonalNormal(rng_guide, expected_model, get_params)
    svi_init, _, svi_eval = svi(expected_model, guide, elbo, opt_init, opt_update, get_opt_params)
    opt_state, get_params = svi_init(rng_init, (data,), (data,))
    expected_params = get_opt_params(opt_state)
    expected_base_params = get_params(opt_state)
    expected_values = guide.median(opt_state)
    expected_loss = svi_eval(random.PRNGKey(1), opt_state, (data,), (data,))

    check_eq(actual_params, expected_params)
    check_eq(actual_base_params, expected_base_params)
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

    opt_init, opt_update, get_opt_params = optimizers.adam(0.01)
    guide = _AutoGuide(random.PRNGKey(0), model, get_opt_params)
    svi_init, _, svi_eval = svi(model, guide, elbo, opt_init, opt_update, get_opt_params)
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
