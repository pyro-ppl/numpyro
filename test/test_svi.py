from numpy.testing import assert_allclose

from jax import lax, random
from jax.experimental import optimizers
import jax.numpy as np

import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.handlers import param, sample
from numpyro.svi import elbo, svi


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

    opt_init, opt_update, get_params = optimizers.adam(0.05)
    svi_init, svi_update, _ = svi(model, guide, elbo, opt_init, opt_update, get_params)
    rng_init, rng_train = random.split(random.PRNGKey(1))
    opt_state, constrain_fn = svi_init(rng_init, model_args=(data,))

    def body_fn(i, val):
        opt_state_, rng_ = val
        loss, opt_state_, rng_ = svi_update(i, rng_, opt_state_, model_args=(data,))
        return opt_state_, rng_

    opt_state, _ = lax.fori_loop(0, 300, body_fn, (opt_state, rng_train))

    params = constrain_fn(get_params(opt_state))
    assert_allclose(params['alpha_q'] / (params['alpha_q'] + params['beta_q']), 0.8, rtol=0.05)
