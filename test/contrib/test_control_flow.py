import jax
import numpyro
import jax.numpy as np
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.contrib.control_flow.scan import scan


def test_scan():
    def target(T=10, q=1, r=1, phi=0., beta=0.):

        def transition(state, i):
            x0, mu0 = state
            x1 = numpyro.sample('x', dist.Normal(phi * x0, q))
            mu1 = beta * mu0 + x1
            y1 = numpyro.sample('y', dist.Normal(mu1, r))
            return (x1, mu1), (x1, y1)

        mu0 = x0 = numpyro.sample('x_0', dist.Normal(0, q))
        y0 = numpyro.sample('y_0', dist.Normal(mu0, r))

        _, xy = scan("scan", transition, (x0, mu0), np.arange(1, T))
        x, y = xy

        return np.append(x0, x), np.append(y0, y)

    kernel = NUTS(target)
    mcmc = MCMC(kernel, 100, 100)
    mcmc.run(jax.random.PRNGKey(0))
    assert set(mcmc.get_samples().keys()) == {'x', 'y', 'x_0', 'y_0'}
    mcmc.print_summary()
