from numpy.testing import assert_allclose

from jax import random
import jax.numpy as np

import numpyro
from numpyro import optim
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.constraints import AffineTransform, SigmoidTransform
from numpyro.handlers import substitute
from numpyro.svi import elbo, svi
from numpyro.util import fori_loop


def test_beta_bernoulli():
    data = np.array([1.0] * 8 + [0.0] * 2)

    def model(data):
        f = numpyro.sample("beta", dist.Beta(1., 1.))
        numpyro.sample("obs", dist.Bernoulli(f), obs=data)

    def guide():
        alpha_q = numpyro.param("alpha_q", 1.0,
                                constraint=constraints.positive)
        beta_q = numpyro.param("beta_q", 1.0,
                               constraint=constraints.positive)
        numpyro.sample("beta", dist.Beta(alpha_q, beta_q))

    adam = optim.Adam(0.05)
    svi_init, svi_update, _ = svi(model, guide, elbo, adam)
    rng_init, rng_train = random.split(random.PRNGKey(1))
    opt_state, get_params = svi_init(rng_init, model_args=(data,))
    assert_allclose(adam.get_params(opt_state)['alpha_q'], 0.)

    def body_fn(i, val):
        opt_state_, rng_ = val
        loss, opt_state_, rng_ = svi_update(rng_, opt_state_, model_args=(data,))
        return opt_state_, rng_

    opt_state, _ = fori_loop(0, 300, body_fn, (opt_state, rng_train))

    params = get_params(opt_state)
    assert_allclose(params['alpha_q'] / (params['alpha_q'] + params['beta_q']), 0.8, atol=0.05, rtol=0.05)


def test_elbo_dynamic_support():
    x_prior = dist.TransformedDistribution(
        dist.Normal(), [AffineTransform(0, 2), SigmoidTransform(), AffineTransform(0, 3)])
    x_guide = dist.Uniform(0, 3)

    def model():
        numpyro.sample('x', x_prior)

    def guide():
        numpyro.sample('x', x_guide)

    adam = optim.Adam(0.01)
    # set base value of x_guide is 0.9
    x_base = 0.9
    guide = substitute(guide, base_param_map={'x': x_base})
    svi_init, _, svi_eval = svi(model, guide, elbo, adam)
    opt_state, get_params = svi_init(random.PRNGKey(0), (), ())
    actual_loss = svi_eval(random.PRNGKey(1), opt_state)
    assert np.isfinite(actual_loss)
    x, _ = x_guide.transform_with_intermediates(x_base)
    expected_loss = x_guide.log_prob(x) - x_prior.log_prob(x)
    assert_allclose(actual_loss, expected_loss)
