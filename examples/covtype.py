# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import time

from jax import random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.examples.datasets import COVTYPE, load_dataset
from numpyro.infer import MCMC, NUTS


def _load_dataset():
    _, fetch = load_dataset(COVTYPE, shuffle=False)
    features, labels = fetch()

    # normalize features and add intercept
    features = (features - features.mean(0)) / features.std(0)
    features = jnp.hstack([features, jnp.ones((features.shape[0], 1))])

    # make binary feature
    _, counts = jnp.unique(labels, return_counts=True)
    specific_category = jnp.argmax(counts)
    labels = (labels == specific_category)

    N, dim = features.shape
    print("Data shape:", features.shape)
    print("Label distribution: {} has label 1, {} has label 0"
          .format(labels.sum(), N - labels.sum()))
    return features, labels


def model(data, labels):
    dim = data.shape[1]
    coefs = numpyro.sample('coefs', dist.Normal(jnp.zeros(dim), jnp.ones(dim)))
    logits = jnp.dot(data, coefs)
    return numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=labels)


def benchmark_hmc(args, features, labels):
    step_size = jnp.sqrt(0.5 / features.shape[0])
    trajectory_length = step_size * args.num_steps
    rng_key = random.PRNGKey(1)
    start = time.time()
    kernel = NUTS(model, trajectory_length=trajectory_length)
    mcmc = MCMC(kernel, 0, args.num_samples)
    mcmc.run(rng_key, features, labels)
    mcmc.print_summary()
    print('\nMCMC elapsed time:', time.time() - start)


def main(args):
    features, labels = _load_dataset()
    benchmark_hmc(args, features, labels)


if __name__ == '__main__':
    assert numpyro.__version__.startswith('0.4.0')
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-samples', default=100, type=int, help='number of samples')
    parser.add_argument('--num-steps', default=10, type=int, help='number of steps (for "HMC")')
    parser.add_argument('--num-chains', nargs='?', default=1, type=int)
    parser.add_argument('--algo', default='NUTS', type=str, help='whether to run "HMC" or "NUTS"')
    parser.add_argument('--device', default='cpu', type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
