import argparse
import time

import numpy as onp
from sklearn.datasets import fetch_covtype

import jax
import jax.numpy as np
from jax import random

import numpyro.distributions as dist
from numpyro.handlers import sample
from numpyro.hmc_util import initialize_model
from numpyro.mcmc import hmc
from numpyro.util import tscan


step_size = 0.00167132
init_params = {"coefs": np.array(
    [+2.03420663e+00, -3.53567265e-02, -1.49223924e-01, -3.07049364e-01,
     -1.00028366e-01, -1.46827862e-01, -1.64167881e-01, -4.20344204e-01,
     +9.47479829e-02, -1.12681836e-02, +2.64442056e-01, -1.22087866e-01,
     -6.00568838e-02, -3.79419506e-01, -1.06668741e-01, -2.97053963e-01,
     -2.05253899e-01, -4.69537191e-02, -2.78072730e-02, -1.43250525e-01,
     -6.77954629e-02, -4.34899796e-03, +5.90927452e-02, +7.23133609e-02,
     +1.38526391e-02, -1.24497898e-01, -1.50733739e-02, -2.68872194e-02,
     -1.80925727e-02, +3.47936489e-02, +4.03552800e-02, -9.98773426e-03,
     +6.20188080e-02, +1.15002751e-01, +1.32145107e-01, +2.69109547e-01,
     +2.45785132e-01, +1.19035013e-01, -2.59744357e-02, +9.94279515e-04,
     +3.39266285e-02, -1.44057125e-02, -6.95222765e-02, -7.52013028e-02,
     +1.21171586e-01, +2.29205526e-02, +1.47308692e-01, -8.34354162e-02,
     -9.34122875e-02, -2.97472421e-02, -3.03937674e-01, -1.70958012e-01,
     -1.59496680e-01, -1.88516974e-01, -1.20889175e+00])}


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
    trajectory_length = step_size * args.num_steps
    init_params = {'coefs': random.normal(key=random.PRNGKey(0), shape=(dim,))}

    _, potential_fn, _ = initialize_model(random.PRNGKey(1), model, (features, labels,), {})
    init_kernel, sample_kernel = hmc(potential_fn, algo=args.algo)
    t0 = time.time()
    # TODO: Use init_params from `initialize_model` instead of fixed params.
    hmc_state, _, _ = init_kernel(init_params, num_warmup_steps=0, step_size=step_size,
                                  trajectory_length=trajectory_length, run_warmup=False,
                                  adapt_step_size=False)
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


def main(args):
    jax.config.update("jax_platform_name", args.device)
    features, labels = load_dataset()
    benchmark_hmc(args, features, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-samples', default=100, type=int, help='number of samples')
    parser.add_argument('--num-steps', default=10, type=int, help='number of steps (for "HMC")')
    parser.add_argument('--algo', default='NUTS', type=str, help='whether to run "HMC" or "NUTS"')
    parser.add_argument('--device', default='cpu', type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()
    main(args)
