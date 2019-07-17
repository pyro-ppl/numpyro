import argparse

import matplotlib.pyplot as plt
import seaborn as sns

from jax import lax, random
from jax.config import config as jax_config
from jax.experimental import optimizers
import jax.numpy as np
from jax.scipy.special import logsumexp

from numpyro.contrib.autoguide import AutoIAFNormal
import numpyro.distributions as dist
from numpyro.handlers import sample
from numpyro.hmc_util import initialize_model
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


def make_transformed_pe(potential_fn, transform, unpack_fn):
    def transformed_potential_fn(z):
        # NB: currently, intermediates for ComposeTransform is None, so this has no effect
        # see https://github.com/pyro-ppl/numpyro/issues/242
        u, intermediates = transform.call_with_intermediates(z)
        logdet = transform.log_abs_det_jacobian(z, u, intermediates=intermediates)
        return potential_fn(unpack_fn(u)) + logdet

    return transformed_potential_fn


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

    print("Start training guide...")
    last_state, _ = lax.fori_loop(0, 100000, body_fn, (opt_state, rng_train))
    print("Finish training guide. Start sampling...")

    transform = guide.get_transform(last_state)
    unpack_fn = lambda u: guide.unpack_latent(u, transform={})  # noqa: E731

    _, potential_fn, constrain_fn = initialize_model(random.PRNGKey(0), dual_moon_model)
    transformed_potential_fn = make_transformed_pe(potential_fn, transform, unpack_fn)
    transformed_constrain_fn = lambda x: constrain_fn(unpack_fn(transform(x)))  # noqa: E731

    # TODO: expose latent_size in autoguide
    init_params = np.zeros(np.size(guide._init_latent))
    samples = mcmc(4000, 4000, init_params, potential_fn=transformed_potential_fn,
                   constrain_fn=transformed_constrain_fn)

    # make plots
    x1 = np.linspace(-3, 3, 100)
    x2 = np.linspace(-3, 3, 100)
    X1, X2 = np.meshgrid(x1, x2)
    P = np.clip(np.exp(-dual_moon_pe(np.stack([X1, X2], axis=-1))), a_min=0.)

    plt.figure(figsize=(8, 8))
    plt.contourf(X1, X2, P, cmap='OrRd')
    sns.kdeplot(samples['x'][:, 0], samples['x'][:, 1])
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.gca().set_aspect('equal')

    plt.savefig("neutra.pdf")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuTra HMC")
    parser.add_argument('--device', default='cpu', type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()
    main(args)
