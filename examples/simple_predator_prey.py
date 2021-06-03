import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy.random as npr
from jax.config import config
from jax.lax import scan

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.optim import Adam
from jaxinterp.interpreter import _make_jaxpr_with_consts, interpret


def predator_prey(prey0, predator0, *, r=0.6, k=100,
                  s=1.2, a=25, u=0.5, v=0.3,
                  step_size=0.01, num_iterations=100):
    def pp_upd(state, _):
        prey, predator = state
        sh = prey * predator / (a + prey + 1e-3)
        prey_upd = r * prey * (1 - prey / (k + 1e-3)) - s * sh
        predator_upd = u * sh - v * predator
        prey = prey + step_size * prey_upd
        predator = predator + step_size * predator_upd
        return (prey, predator), (prey, predator)

    _, (prey, predator) = scan(pp_upd, (prey0, predator0),
                               jnp.arange(step_size, num_iterations + step_size, step_size))
    prey = jnp.reshape(prey, (num_iterations, -1))[:, 0]
    predator = jnp.reshape(predator, (num_iterations, -1))[:, 0]
    return prey, predator


def model(prey, predator, idxs):
    prior = dist.HalfNormal(10.)
    prey0 = numpyro.sample('prey0', prior)
    predator0 = numpyro.sample('predator0', prior)
    r = numpyro.sample('r', prior)
    k = numpyro.sample('k', prior)
    s = numpyro.sample('s', prior)
    a = numpyro.sample('a', prior)
    u = numpyro.sample('u', prior)
    v = numpyro.sample('v', prior)
    sprey, spredator = predator_prey(prey0, predator0,
                                     r=r, k=k, s=s,
                                     a=a, u=u, v=v)
    with numpyro.plate('data', prey.shape[0], dim=-2):
        numpyro.sample('prey_obs', dist.Normal(sprey[idxs], 100.), obs=prey)
        numpyro.sample('predator_obs', dist.Normal(spredator[idxs], 100.), obs=predator)


if __name__ == '__main__':
    config.update('jax_debug_nans', True)
    prey, predator = predator_prey(50., 5.)
    scale = 3
    idxs = [2, 13, 29, 31, 47]
    obs_prey = prey[idxs] + scale * npr.randn(1000, len(idxs))
    obs_predator = predator[idxs] + scale * npr.randn(1000, len(idxs))
    plt.plot(obs_prey.transpose(), color='b')
    plt.plot(obs_predator.transpose(), color='r')
    plt.show()
    guide = AutoDelta(model)
    svi = SVI(model, guide, Adam(0.25), ELBO())
    rng_key = jax.random.PRNGKey(1377)
    state = svi.init(rng_key, obs_prey, obs_predator, idxs)
    prev_state = state
    upd_fn = jax.jit(svi.update)
    try:
        for i in range(10_000):
            prev_state = state
            state, loss = upd_fn(state, obs_prey, obs_predator, idxs)
            if i % 100 == 0:
                print(f"{i}: {loss}")
    except FloatingPointError:
        config.update('jax_debug_nans', False)
        _, rng_key_eval = jax.random.split(prev_state.rng_key)
        params = svi.get_params(prev_state)
        print(svi.evaluate(prev_state, obs_prey, obs_predator, idxs))
        grady = jax.grad(lambda p: svi.loss.loss(rng_key_eval, p, svi.model, svi.guide,
                                                 obs_prey, obs_predator, idxs))
        # For getting the Jaxpr
        grad_jaxpr, consts, trees = _make_jaxpr_with_consts(grady, stage_out=True)(params)
        print(grad_jaxpr)
        # For debugging using interpreter
        interpret(grady, stage_out=True)(params)
