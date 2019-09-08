import argparse

import numpy as onp

from jax.config import config as jax_config
import jax.numpy as np
import jax.random as random

import numpyro
import numpyro.distributions as dist
from numpyro.examples.datasets import SP500, load_dataset
from numpyro.hmc_util import initialize_model
from numpyro.mcmc import hmc
from numpyro.util import fori_collect


"""
Generative model:

sigma ~ Exponential(50)
nu ~ Exponential(.1)
s_i ~ Normal(s_{i-1}, sigma - 2)
r_i ~ StudentT(nu, 0, exp(-2 s_i))

This example is from PyMC3 [1], which itself is adapted from the original experiment
from [2]. A discussion about translating this in Pyro appears in [3].

For more details, refer to:
 1. *Stochastic Volatility Model*, https://docs.pymc.io/notebooks/stochastic_volatility.html
 2. *The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo*,
    https://arxiv.org/pdf/1111.4246.pdf
 3. Forum discussion, https://forum.pyro.ai/t/problems-transforming-a-pymc3-model-to-pyro-mcmc/208/14

"""


def model(returns):
    step_size = numpyro.sample('sigma', dist.Exponential(50.))
    s = numpyro.sample('s', dist.GaussianRandomWalk(scale=step_size, num_steps=np.shape(returns)[0]))
    nu = numpyro.sample('nu', dist.Exponential(.1))
    return numpyro.sample('r', dist.StudentT(df=nu, loc=0., scale=np.exp(-2*s)),
                          obs=returns)


def print_results(posterior, dates):
    def _print_row(values, row_name=''):
        quantiles = [0.2, 0.4, 0.5, 0.6, 0.8]
        row_name_fmt = '{:>' + str(len(row_name)) + '}'
        header_format = row_name_fmt + '{:>12}' * 5
        row_format = row_name_fmt + '{:>12.3f}' * 5
        columns = ['(p{})'.format(q * 100) for q in quantiles]
        q_values = onp.quantile(values, quantiles, axis=0)
        print(header_format.format('', *columns))
        print(row_format.format(row_name, *q_values))
        print('\n')

    print('=' * 5, 'sigma', '=' * 5)
    _print_row(posterior['sigma'])
    print('=' * 5, 'nu', '=' * 5)
    _print_row(posterior['nu'])
    print('=' * 5, 'volatility', '=' * 5)
    for i in range(0, len(dates), 180):
        _print_row(np.exp(-2 * posterior['s'][:, i]), dates[i])


def main(args):
    jax_config.update('jax_platform_name', args.device)
    _, fetch = load_dataset(SP500, shuffle=False)
    dates, returns = fetch()
    init_rng, sample_rng = random.split(random.PRNGKey(args.rng))
    init_params, potential_fn, constrain_fn = initialize_model(init_rng, model, returns)
    init_kernel, sample_kernel = hmc(potential_fn, algo='NUTS')
    hmc_state = init_kernel(init_params, args.num_warmup, rng=sample_rng)
    hmc_states = fori_collect(0, args.num_samples, sample_kernel, hmc_state,
                              transform=lambda hmc_state: constrain_fn(hmc_state.z))
    print_results(hmc_states, dates)


if __name__ == "__main__":
    assert numpyro.__version__.startswith('0.2.0')
    parser = argparse.ArgumentParser(description="Stochastic Volatility Model")
    parser.add_argument('-n', '--num-samples', nargs='?', default=3000, type=int)
    parser.add_argument('--num-warmup', nargs='?', default=1500, type=int)
    parser.add_argument('--device', default='cpu', type=str, help='use "cpu" or "gpu".')
    parser.add_argument('--rng', default=21, type=int, help='random number generator seed')
    args = parser.parse_args()
    main(args)
