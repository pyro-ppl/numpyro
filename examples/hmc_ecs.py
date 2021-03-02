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
from jax import random

import numpyro
import numpyro.distributions as dist
from numpyro.examples.datasets import HIGGS, load_dataset
from numpyro.infer import autoguide, SVI, Trace_ELBO, HMC, NUTS, HMCECS, MCMC
from numpyro.infer.hmc_gibbs import taylor_proxy


def model(data, obs, subsample_size):
    n, m = data.shape
    theta = numpyro.sample('theta', dist.continuous.Normal(jnp.zeros(m), .5 * jnp.ones(m)))
    with numpyro.plate('N', n, subsample_size=subsample_size):
        batch_feats = numpyro.subsample(data, event_dim=1)
        batch_obs = numpyro.subsample(obs, event_dim=0)
        numpyro.sample('obs', dist.Bernoulli(logits=theta @ batch_feats.T), obs=batch_obs)


def main(args):
    _, fetch = load_dataset(HIGGS, shuffle=False)
    data, obs = fetch()

    svi_key, mcmc_key = random.split(random.PRNGKey(args.rng_seed))

    optimizer = numpyro.optim.Adam(step_size=1e-3)
    guide = autoguide.AutoDelta(model)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    params, losses = svi.run(svi_key, 5000, data, obs, args.subsample_size)

    if args.inner_kernel.lower() == 'hmc':
        inner_kernel = HMC(model)
    elif args.inner_kernel.lower() == 'nuts':
        inner_kernel = NUTS(model)
    else:
        inner_kernel = None

    kernel = HMCECS(inner_kernel, num_blocks=args.num_blocks, proxy=taylor_proxy({'theta': params['theta_auto_loc']}))
    mcmc = MCMC(kernel, num_warmup=args.num_warmup, num_samples=args.num_samples)
    mcmc.run(mcmc_key, data, obs, args.subsample_size)
    mcmc.print_summary()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hamiltonian Monte Carlo with Energy Conserving Subsampling")
    parser.add_argument('--subsample_size', type=int, default=1300)
    parser.add_argument('--num_blocks', type=int, default=100)
    parser.add_argument('--num_warmup', type=int, default=1000)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--inner_kernel', type=str, default='nuts')
    parser.add_argument('--device', default='cpu', type=str, help='use "cpu" or "gpu".')
    parser.add_argument('--rng_seed', default=21, type=int, help='random number generator seed')

    args = parser.parse_args()

    numpyro.set_platform(args.device)

    main(args)
