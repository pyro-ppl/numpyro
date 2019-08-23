from numpy.testing import assert_allclose

from jax import random
from jax.experimental import optimizers
import jax.numpy as np

import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.constraints import SigmoidTransform, AffineTransform
from numpyro.handlers import param, sample, substitute
from numpyro.svi import elbo, svi
from numpyro.util import fori_loop


def test_beta_bernoulli():
    data = np.array([1.0] * 8 + [0.0] * 2)

    def model(data):
        f = sample("beta", dist.Beta(1., 1.))
        sample("obs", dist.Bernoulli(f), obs=data)

    def guide():
        alpha_q = param("alpha_q", 1.0,
                        constraint=constraints.positive)
        beta_q = param("beta_q", 1.0,
                       constraint=constraints.positive)
        sample("beta", dist.Beta(alpha_q, beta_q))

    opt_init, opt_update, get_opt_params = optimizers.adam(0.05)
    svi_init, svi_update, _ = svi(model, guide, elbo, opt_init, opt_update, get_opt_params)
    rng_init, rng_train = random.split(random.PRNGKey(1))
    opt_state, get_params = svi_init(rng_init, model_args=(data,))

    def body_fn(i, val):
        opt_state_, rng_ = val
        loss, opt_state_, rng_ = svi_update(i, rng_, opt_state_, model_args=(data,))
        return opt_state_, rng_

    opt_state, _ = fori_loop(0, 300, body_fn, (opt_state, rng_train))

    params = get_params(opt_state)
    assert_allclose(params['alpha_q'] / (params['alpha_q'] + params['beta_q']), 0.8, atol=0.05, rtol=0.05)


def test_dynamic_constraints():
    true_coef = 0.9
    data = true_coef + random.normal(random.PRNGKey(0), (1000,))

    def model(data):
        # NB: model's constraints will play no effect
        loc = param('loc', 0., constraint=constraints.interval(0, 0.5))
        sample('obs', dist.Normal(loc, 0.1), obs=data)

    def guide():
        alpha = param('alpha', 0.5, constraint=constraints.unit_interval)
        param('loc', 0, constraint=constraints.interval(0, alpha))

    opt_init, opt_update, opt_get_state = optimizers.adam(0.05)
    svi_init, svi_update, _ = svi(model, guide, elbo, opt_init, opt_update, opt_get_state)
    rng_init, rng_train = random.split(random.PRNGKey(1))
    opt_state, get_params = svi_init(rng_init, model_args=(data,))

    def body_fn(i, val):
        opt_state_, rng_ = val
        loss, opt_state_, rng_ = svi_update(i, rng_, opt_state_, model_args=(data,))
        return opt_state_, rng_

    opt_state, rng = fori_loop(0, 300, body_fn, (opt_state, rng_train))
    params = get_params(opt_state, rng, guide_args=())
    assert_allclose(params['loc'], true_coef, atol=0.05)


def test_dynamic_supports():
    x_prior = dist.TransformedDistribution(
        dist.Normal(), [AffineTransform(0, 2), SigmoidTransform(), AffineTransform(0, 3)])
    x_guide = dist.Uniform(0, 3)

    def model():
        sample('x', x_prior)

    def guide():
        sample('x', x_guide)

    opt_init, opt_update, get_opt_params = optimizers.adam(0.01)
    # set base value of x_guide is 0.9
    x_base = 10
    guide = substitute(guide, base_param_map={'x': x_base})
    svi_init, _, svi_eval = svi(model, guide, elbo, opt_init, opt_update, get_opt_params)
    opt_state, get_params = svi_init(random.PRNGKey(0), (), ())
    actual_loss = svi_eval(random.PRNGKey(1), opt_state)
    x, _ = x_guide.transform_with_intermediates(x_base)
    expected_loss = x_guide.log_prob(x) - x_prior.log_prob(x)
    assert_allclose(actual_loss, expected_loss)
