# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from numpy.testing import assert_allclose

import jax
import jax.numpy as jnp

import numpyro
from numpyro.contrib.control_flow.scan import scan
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive


def test_scan():
    def model(T=10, q=1, r=1, phi=0., beta=0.):

        def transition(state, i):
            x0, mu0 = state
            x1 = numpyro.sample('x', dist.Normal(phi * x0, q))
            mu1 = beta * mu0 + x1
            y1 = numpyro.sample('y', dist.Normal(mu1, r))
            numpyro.deterministic('y2', y1 * 2)
            return (x1, mu1), (x1, y1)

        mu0 = x0 = numpyro.sample('x_0', dist.Normal(0, q))
        y0 = numpyro.sample('y_0', dist.Normal(mu0, r))

        _, xy = scan(transition, (x0, mu0), jnp.arange(T))
        x, y = xy

        return jnp.append(x0, x), jnp.append(y0, y)

    T = 10
    num_samples = 100
    kernel = NUTS(model)
    mcmc = MCMC(kernel, 100, num_samples)
    mcmc.run(jax.random.PRNGKey(0), T=T)
    assert set(mcmc.get_samples()) == {'x', 'y', 'y2', 'x_0', 'y_0'}
    mcmc.print_summary()

    samples = mcmc.get_samples()
    x = samples.pop('x')[0]  # take 1 sample of x
    # this tests for the composition of condition and substitute
    # this also tests if we can use `vmap` for predictive.
    future = 5
    predictive = Predictive(numpyro.handlers.condition(model, {'x': x}),
                            samples, return_sites=['x', 'y', 'y2'], parallel=True)
    result = predictive(jax.random.PRNGKey(1), T=T + future)
    expected_shape = (num_samples, T + future)
    assert result['x'].shape == expected_shape
    assert result['y'].shape == expected_shape
    assert result['y2'].shape == expected_shape
    assert_allclose(result['x'][:, :T], jnp.broadcast_to(x, (num_samples, T)))
    assert_allclose(result['y'][:, :T], samples['y'])
