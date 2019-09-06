import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns

from jax import random
from jax.config import config as jax_config
import jax.numpy as np

import numpyro
import numpyro.distributions as dist
from numpyro.distributions.constraints import AffineTransform
from numpyro.mcmc import MCMC, NUTS

sns.set(context='talk')


"""
This example, which is adapted from [1], illustrates how to leverage non-centered
parameterization using the class `~numpyro.distributions.TransformedDistribution`.
We will examine the difference between two types of parameterizations on the
10-dimensional Neal's funnel distribution. As we will see, HMC gets trouble at
the neck of the funnel if centered parameterization is used. On the contrary,
the problem can be solved by using non-centered parameterization.

Using non-centered parameterization through TransformedDistribution in NumPyro
has the same effect as the automatic reparameterisation technique introduced in
[2]. However, in [2], users need to implement a (non-trivial) reparameterization
rule for each type of transform. Instead, in NumPyro the only requirement to let
inference algorithms know to do reparameterization automatically is to declare
the random variable as a transformed distribution.

[1] *Stan User's Guide*, https://mc-stan.org/docs/2_19/stan-users-guide/reparameterization-section.html
[2] Maria I. Gorinova, Dave Moore, Matthew D. Hoffman (2019), "Automatic
    Reparameterisation of Probabilistic Programs", (https://arxiv.org/abs/1906.03028)
"""


def model(dim=10):
    y = numpyro.sample('y', dist.Normal(0, 3))
    numpyro.sample('x', dist.Normal(np.zeros(dim - 1), np.exp(y / 2)))


def reparam_model(dim=10):
    y = numpyro.sample('y', dist.Normal(0, 3))
    numpyro.sample('x', dist.TransformedDistribution(
        dist.Normal(np.zeros(dim - 1), 1), AffineTransform(0, np.exp(y / 2))))


def run_inference(model, args, rng):
    kernel = NUTS(model)
    mcmc = MCMC(kernel, args.num_warmup, args.num_samples, num_chains=args.num_chains)
    mcmc.run(rng)
    return mcmc.get_samples()


def main(args):
    jax_config.update('jax_platform_name', args.device)
    rng = random.PRNGKey(0)

    # do inference with centered parameterization
    samples = run_inference(model, args, rng)

    # do inference with non-centered parameterization
    reparam_samples = run_inference(reparam_model, args, rng)

    # make plots
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 16))

    sns.scatterplot(samples['x'][:, 0], samples['y'], color='g', alpha=0.3, ax=ax1)
    ax1.set(xlim=(-20, 20), ylim=(-9, 9), xlabel='x[0]', ylabel='y',
            title='Funnel samples with centered parameterization')

    sns.scatterplot(reparam_samples['x'][:, 0], reparam_samples['y'], color='g', alpha=0.3, ax=ax2)
    ax2.set(xlim=(-20, 20), ylim=(-9, 9), xlabel='x[0]', ylabel='y',
            title='Funnel samples with non-centered parameterization')

    plt.savefig('funnel_plot.pdf')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Non-centered reparameterization example")
    parser.add_argument("-n", "--num-samples", nargs="?", default=1000, type=int)
    parser.add_argument("--num-warmup", nargs='?', default=1000, type=int)
    parser.add_argument("--num-chains", nargs='?', default=4, type=int)
    parser.add_argument("--device", default='cpu', type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    if args.device == 'cpu' and args.num_chains <= os.cpu_count():
        os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count={}'.format(
            args.num_chains)

    main(args)
