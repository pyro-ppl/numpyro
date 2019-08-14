from numpy.testing import assert_allclose

from jax import random
from jax.experimental import optimizers
import jax.numpy as np

import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.handlers import param, sample
from numpyro.svi import elbo, get_param, svi
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

    _, _, get_params = optim = optimizers.adam(0.05)
    svi_init, svi_update, _ = svi(model, guide, elbo, optim)
    rng_init, rng_train = random.split(random.PRNGKey(1))
    opt_state, constrain_fn = svi_init(rng_init, model_args=(data,))

    def body_fn(i, val):
        opt_state_, rng_ = val
        loss, opt_state_, rng_ = svi_update(i, rng_, opt_state_, model_args=(data,))
        return opt_state_, rng_

    opt_state, _ = fori_loop(0, 300, body_fn, (opt_state, rng_train))

    params = constrain_fn(get_params(opt_state))
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

    _, _, get_params = optim = optimizers.adam(0.05)
    svi_init, svi_update, _ = svi(model, guide, elbo, optim)
    rng_init, rng_train = random.split(random.PRNGKey(1))
    opt_state, constrain_fn = svi_init(rng_init, model_args=(data,))

    def body_fn(i, val):
        opt_state_, rng_ = val
        loss, opt_state_, rng_ = svi_update(i, rng_, opt_state_, model_args=(data,))
        return opt_state_, rng_

    opt_state, rng = fori_loop(0, 300, body_fn, (opt_state, rng_train))
    params = get_param(opt_state, model, guide, get_params, constrain_fn, rng,
                       guide_args=())
    assert_allclose(params['loc'], true_coef, atol=0.05)
