# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0
"""
Example: Hamiltonian Monte Carlo with Energy Conserving Subsampling
=========================

This example illustrates the use of subsampling of HMC using Energy Conserving Subsampling when the likelihood is
additive.

**References:**

    1. *Hamiltonian Monte Carlo with energy conserving subsampling*,
       Dang, K. D., Quiroz, M., Kohn, R., Minh-Ngoc, T., & Villani, M. (2019)

    :align: center
"""

import argparse

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random
from sklearn.datasets import load_breast_cancer

import numpyro
import numpyro.distributions as dist
from numpyro.infer import autoguide, SVI, Trace_ELBO, HMC, NUTS, HMCECS, MCMC
from numpyro.infer.hmc_gibbs import taylor_proxy


def model(data, obs, subsample_size):
    n, m = data.shape
    theta = numpyro.sample('theta', dist.continuous.Normal(jnp.zeros(m), .5 * jnp.ones(m)))
    with numpyro.plate('N', n, subsample_size=subsample_size):
        batch_feats = numpyro.subsample(data, event_dim=1)
        batch_obs = numpyro.subsample(obs, event_dim=0)
        numpyro.sample('obs', dist.Bernoulli(logits=theta @ batch_feats.T), obs=batch_obs)


def breast_cancer_data():
    dataset = load_breast_cancer()
    feats = dataset.data
    feats = (feats - feats.mean(0)) / feats.std(0)
    feats = jnp.hstack((feats, jnp.ones((feats.shape[0], 1))))
    return feats, dataset.target


def run_hmcecs(hmcecs_key, args, data, obs, inner_kernel):
    svi_key, mcmc_key = random.split(hmcecs_key)

    optimizer = numpyro.optim.Adam(step_size=1e-3)
    guide = autoguide.AutoDelta(model)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    params, losses = svi.run(svi_key, 5000, data, obs, args.subsample_size)
    kernel = HMCECS(inner_kernel(model), num_blocks=args.num_blocks,
                    proxy=taylor_proxy({'theta': params['theta_auto_loc']}))
    mcmc = MCMC(kernel, num_warmup=args.num_warmup, num_samples=args.num_samples)
    mcmc.run(mcmc_key, data, obs, args.subsample_size)
    mcmc.print_summary()
    return mcmc.get_samples()


def run_hmc(mcmc_key, args, data, obs, kernel):
    mcmc = MCMC(kernel(model), num_warmup=args.num_warmup, num_samples=args.num_samples)
    mcmc.run(mcmc_key, data, obs, None)
    mcmc.print_summary()
    return mcmc.get_samples()


def main(args):
    # _, fetch = load_dataset(HIGGS, shuffle=False)
    data, obs = breast_cancer_data()
    hmcecs_key, hmc_key = random.split(random.PRNGKey(args.rng_seed))
    if args.inner_kernel.lower() == 'hmc':
        inner_kernel = HMC
    elif args.inner_kernel.lower() == 'nuts':
        inner_kernel = NUTS
    else:
        inner_kernel = lambda _: None

    hmcecs_samples = run_hmcecs(hmcecs_key, args, data, obs, inner_kernel)
    if inner_kernel:
        hmc_samples = run_hmc(hmc_key, args, data, obs, inner_kernel)

    plot_mean_variance(hmc_samples, hmcecs_samples)


def plot_mean_variance(hmc_samples, hmcecs_samples):
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(jnp.sort(hmc_samples['theta'].mean(0)), 'or')
    ax[0].plot(jnp.sort(hmcecs_samples['theta'].mean(0)), 'b')
    ax[0].set_title(r'$\mathrm{\mathbb{E}}[\theta]$')

    ax[1].plot(jnp.sort(hmc_samples['theta'].var(0)), 'or')
    ax[1].plot(jnp.sort(hmcecs_samples['theta'].var(0)), 'b')
    ax[1].set_title(r'Var$[\theta]$')

    for a in ax:
        a.set_xticks([])

    fig.tight_layout()
    fig.savefig('expected_variance.pdf', bbox_inches='tight', transparent=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hamiltonian Monte Carlo with Energy Conserving Subsampling")
    parser.add_argument('--subsample_size', type=int, default=75)
    parser.add_argument('--num_blocks', type=int, default=25)
    parser.add_argument('--num_warmup', type=int, default=1000)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--inner_kernel', type=str, default='nuts')
    parser.add_argument('--device', default='cpu', type=str, help='use "cpu" or "gpu".')
    parser.add_argument('--rng_seed', default=21, type=int, help='random number generator seed')

    args = parser.parse_args()

    numpyro.set_platform('gpu')

    main(args)
