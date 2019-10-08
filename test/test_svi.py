from numpy.testing import assert_allclose

from jax import jit, random
import jax.numpy as np
from jax.test_util import check_eq

import numpyro
from numpyro import optim
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.transforms import AffineTransform, SigmoidTransform
from numpyro.handlers import substitute
from numpyro.infer import SVI, Trace_ELBO
from numpyro.util import fori_loop


def test_beta_bernoulli():
    data = np.array([1.0] * 8 + [0.0] * 2)

    def model(data):
        f = numpyro.sample("beta", dist.Beta(1., 1.))
        numpyro.sample("obs", dist.Bernoulli(f), obs=data)

    def guide(data):
        alpha_q = numpyro.param("alpha_q", 1.0,
                                constraint=constraints.positive)
        beta_q = numpyro.param("beta_q", 1.0,
                               constraint=constraints.positive)
        numpyro.sample("beta", dist.Beta(alpha_q, beta_q))

    adam = optim.Adam(0.05)
    svi = SVI(model, guide, Trace_ELBO(), adam)
    svi_state = svi.init(random.PRNGKey(1), data)
    assert_allclose(adam.get_params(svi_state.optim_state)['alpha_q'], 0.)

    def body_fn(i, val):
        svi_state, _ = svi.update(val, data)
        return svi_state

    svi_state = fori_loop(0, 300, body_fn, svi_state)
    params = svi.get_params(svi_state)
    assert_allclose(params['alpha_q'] / (params['alpha_q'] + params['beta_q']), 0.8, atol=0.05, rtol=0.05)


def test_jitted_update_fn():
    data = np.array([1.0] * 8 + [0.0] * 2)

    def model(data):
        f = numpyro.sample("beta", dist.Beta(1., 1.))
        numpyro.sample("obs", dist.Bernoulli(f), obs=data)

    def guide(data):
        alpha_q = numpyro.param("alpha_q", 1.0,
                                constraint=constraints.positive)
        beta_q = numpyro.param("beta_q", 1.0,
                               constraint=constraints.positive)
        numpyro.sample("beta", dist.Beta(alpha_q, beta_q))

    adam = optim.Adam(0.05)
    svi = SVI(model, guide, Trace_ELBO(), adam)
    svi_state = svi.init(random.PRNGKey(1), data)
    expected = svi.get_params(svi.update(svi_state, data)[0])

    actual = svi.get_params(jit(svi.update)(svi_state, data=data)[0])
    check_eq(actual, expected)


def test_param():
    # this test the validity of model/guide sites having
    # param constraints contain composed transformed
    rngs = random.split(random.PRNGKey(0), 5)
    a_minval = 1
    c_minval = -2
    c_maxval = -1
    a_init = np.exp(random.normal(rngs[0])) + a_minval
    b_init = np.exp(random.normal(rngs[1]))
    c_init = random.uniform(rngs[2], minval=c_minval, maxval=c_maxval)
    d_init = random.uniform(rngs[3])
    obs = random.normal(rngs[4])

    def model():
        a = numpyro.param('a', a_init, constraint=constraints.greater_than(a_minval))
        b = numpyro.param('b', b_init, constraint=constraints.positive)
        numpyro.sample('x', dist.Normal(a, b), obs=obs)

    def guide():
        c = numpyro.param('c', c_init, constraint=constraints.interval(c_minval, c_maxval))
        d = numpyro.param('d', d_init, constraint=constraints.unit_interval)
        numpyro.sample('y', dist.Normal(c, d), obs=obs)

    adam = optim.Adam(0.01)
    svi = SVI(model, guide, Trace_ELBO(), adam)
    svi_state = svi.init(random.PRNGKey(0))

    params = svi.get_params(svi_state)
    assert_allclose(params['a'], a_init)
    assert_allclose(params['b'], b_init)
    assert_allclose(params['c'], c_init)
    assert_allclose(params['d'], d_init)

    actual_loss = svi.evaluate(svi_state)
    assert np.isfinite(actual_loss)
    expected_loss = dist.Normal(c_init, d_init).log_prob(obs) - dist.Normal(a_init, b_init).log_prob(obs)
    # not so precisely because we do transform / inverse transform stuffs
    assert_allclose(actual_loss, expected_loss, rtol=1e-6)


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
    svi = SVI(model, guide, Trace_ELBO(), adam)
    svi_state = svi.init(random.PRNGKey(0))
    actual_loss = svi.evaluate(svi_state)
    assert np.isfinite(actual_loss)
    x, _ = x_guide.transform_with_intermediates(x_base)
    expected_loss = x_guide.log_prob(x) - x_prior.log_prob(x)
    assert_allclose(actual_loss, expected_loss)
