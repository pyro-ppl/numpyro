import argparse

from jax import lax, random
from jax.config import config as jax_config
from jax.experimental import optimizers
import jax.numpy as np
from jax.scipy.special import logsumexp

from numpyro.contrib.autoguide import AutoIAFNormal
import numpyro.distributions as dist
from numpyro.handlers import sample
from numpyro.mcmc import mcmc
from numpyro.svi import elbo, svi

"""
This example illustrates how to use a trained AutoIAFNormal autoguide to transform a posterior to a
Gaussian-like one. The transform will be used to get better mixing rate for NUTS sampler.

[1] Hoffman, M. et al. (2019), ["NeuTra-lizing Bad Geometry in Hamiltonian Monte Carlo Using Neural Transport"]
    (https://arxiv.org/abs/1903.03704).
"""


def dual_moon_pe(x):
    term1 = 0.5 * ((np.linalg.norm(x, axis=-1) - 2) / 0.4) ** 2
    term2 = -0.5 * ((x[..., :1] + np.array([-2., 2.])) / 0.6) ** 2
    return term1 - logsumexp(term2, axis=-1)


def dual_moon_model():
    x = sample('x', dist.Uniform(-10 * np.ones(2), 10 * np.ones(2)))
    pe = dual_moon_pe(x)
    sample('log_density', dist.Delta(log_density=-pe), obs=0.)


def main(args):
    jax_config.update('jax_platform_name', args.device)

    opt_init, opt_update, get_params = optimizers.adam(0.001)
    rng_guide, rng_init, rng_train = random.split(random.PRNGKey(1), 3)
    guide = AutoIAFNormal(rng_guide, dual_moon_model, get_params, hidden_dims=[20])
    svi_init, svi_update, _ = svi(dual_moon_model, guide, elbo, opt_init, opt_update, get_params)
    opt_state, _ = svi_init(rng_init)

    def body_fn(i, state):
        opt_state_, rng_ = state
        loss, opt_state_, rng_ = svi_update(i, rng_, opt_state_)
        return opt_state_, rng_

    last_state, _ = lax.fori_loop(0, 100000, body_fn, (opt_state, rng_train))
    samples = guide.sample_posterior(random.PRNGKey(0), last_state, sample_shape=(1000,))
    # only need to get flows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuTra HMC")
    parser.add_argument('--device', default='cpu', type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()
    main(args)
