import argparse
import time

import numpy as onp
from sklearn.datasets import fetch_covtype

import jax.numpy as np
from jax import random

import numpyro.distributions as dist
from numpyro.handlers import sample
from numpyro.hmc_util import initialize_model
from numpyro.mcmc import hmc
from numpyro.util import tscan


# TODO: add to datasets.py so as to avoid dependency on scikit-learn
def load_dataset():
    data = fetch_covtype()
    features = data.data
    labels = data.target

    # normalize features and add intercept
    features = (features - features.mean(0)) / features.std(0)
    features = np.hstack([features, np.ones((features.shape[0], 1))])

    # make binary feature
    _, counts = onp.unique(labels, return_counts=True)
    specific_category = np.argmax(counts)
    labels = (labels == specific_category)

    N, dim = features.shape
    print("Data shape:", features.shape)
    print("Label distribution: {} has label 1, {} has label 0"
          .format(labels.sum(), N - labels.sum()))
    return features, labels


def model(data, labels):
    N, dim = data.shape
    coefs = sample('coefs', dist.norm(np.zeros(dim), np.ones(dim)))
    logits = np.dot(data, coefs)
    return sample('obs', dist.bernoulli(logits, is_logits=True), obs=labels)


def benchmark_hmc(args, features, labels):
    N, dim = features.shape
    step_size = np.sqrt(0.5 / N)
    trajectory_length = step_size * args.num_steps
    init_params = {'coefs': random.normal(key=random.PRNGKey(0), shape=(dim,))}

    _, potential_fn, _ = initialize_model(random.PRNGKey(1), model, (features, labels,), {})
    init_kernel, sample_kernel = hmc(potential_fn, algo=args.algo)
    t0 = time.time()
    # Do not run warmup for consistent benchmark
    hmc_state, _, _ = init_kernel(init_params, num_warmup_steps=0, step_size=step_size,
                                  trajectory_length=trajectory_length, run_warmup=False)
    t1 = time.time()
    print("time for hmc_init: ", t1 - t0)

    def transform(state): return {'coefs': state.z['coefs'],
                                  'num_steps': state.num_steps}

    def body_fn(state, i):
        return sample_kernel(state)

    hmc_state = hmc_state.update(step_size=step_size)
    hmc_states = tscan(body_fn, hmc_state, np.arange(args.num_samples), transform=transform)
    num_leapfrogs = np.sum(hmc_states['num_steps'])
    print('number of leapfrog steps: ', num_leapfrogs)
    print('avg. time for each step: ', (time.time() - t1) / num_leapfrogs)
    print(hmc_states['coefs'])


def main(args):
    features, labels = load_dataset()
    benchmark_hmc(args, features, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-samples', default=100, type=int, help='number of samples')
    parser.add_argument('--num-steps', default=10, type=int, help='number of steps (for "HMC")')
    parser.add_argument('--algo', default='NUTS', type=str, help='whether to run "HMC" or "NUTS"')
    args = parser.parse_args()
    main(args)
