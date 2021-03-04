# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: MCMC Methods for Tall Data
===================================

This example illustrates the usages of various MCMC methods which are suitable for tall data:

    - `algo="SA"` uses the sample adaptive MCMC method in [1]
    - `algo="HMCECS"` uses the energy conserving subsampling method in [2]
    - `algo="FlowHMCECS"` utilizes a normalizing flow to neutralize the posterior
      geometry into a Gaussian-like one. Then HMCECS is used to draw the posterior
      samples. Currently, this method gives the best mixing rate among those methods.

**References:**

    1. *Sample Adaptive MCMC*,
       Michael Zhu (2019)
    2. *Hamiltonian Monte Carlo with energy conserving subsampling*,
       Dang, K. D., Quiroz, M., Kohn, R., Minh-Ngoc, T., & Villani, M. (2019)
    3. *NeuTra-lizing Bad Geometry in Hamiltonian Monte Carlo Using Neural Transport*,
       Hoffman, M. et al. (2019)

"""

import argparse
import time

import matplotlib.pyplot as plt

from jax import random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.examples.datasets import COVTYPE, load_dataset
from numpyro.infer import HMC, HMCECS, MCMC, NUTS, SA, SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoBNAFNormal
from numpyro.infer.hmc_gibbs import taylor_proxy
from numpyro.infer.reparam import NeuTraReparam


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


def model(data, labels, subsample_size=None):
    dim = data.shape[1]
    coefs = numpyro.sample('coefs', dist.Normal(jnp.zeros(dim), jnp.ones(dim)))
    with numpyro.plate("N", data.shape[0], subsample_size=subsample_size) as idx:
        logits = jnp.dot(data[idx], coefs)
        return numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=labels[idx])


def benchmark_hmc(args, features, labels):
    rng_key = random.PRNGKey(1)
    start = time.time()
    # a MAP estimate at the following source
    # https://github.com/google/edward2/blob/master/examples/no_u_turn_sampler/logistic_regression.py#L117
    ref_params = {"coefs": jnp.array([
        +2.03420663e+00, -3.53567265e-02, -1.49223924e-01, -3.07049364e-01,
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
    if args.algo == "HMC":
        step_size = jnp.sqrt(0.5 / features.shape[0])
        trajectory_length = step_size * args.num_steps
        kernel = HMC(model, step_size=step_size, trajectory_length=trajectory_length, adapt_step_size=False,
                     dense_mass=args.dense_mass)
        subsample_size = None
    elif args.algo == "NUTS":
        kernel = NUTS(model, dense_mass=args.dense_mass)
        subsample_size = None
    elif args.algo == "HMCECS":
        subsample_size = 1000
        inner_kernel = NUTS(model, init_strategy=init_to_value(values=ref_params),
                            dense_mass=args.dense_mass)
        # note: if num_blocks=100, we'll update 10 index at each MCMC step
        # so it took 50000 MCMC steps to iterative the whole dataset
        kernel = HMCECS(inner_kernel, num_blocks=100, proxy=taylor_proxy(ref_params))
    elif args.algo == "SA":
        # NB: this kernel requires large num_warmup and num_samples
        # and running on GPU is much faster than on CPU
        kernel = SA(model, adapt_state_size=1000, init_strategy=init_to_value(values=ref_params))
        subsample_size = None
    elif args.algo == "FlowHMCECS":
        subsample_size = 1000
        guide = AutoBNAFNormal(model, num_flows=1, hidden_factors=[8])
        svi = SVI(model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
        params, losses = svi.run(random.PRNGKey(2), 2000, features, labels)
        plt.plot(losses)
        plt.show()

        neutra = NeuTraReparam(guide, params)
        neutra_model = neutra.reparam(model)
        neutra_ref_params = {"auto_shared_latent": jnp.zeros(55)}
        # no need to adapt mass matrix if the flow does a good job
        inner_kernel = NUTS(neutra_model, init_strategy=init_to_value(values=neutra_ref_params),
                            adapt_mass_matrix=False)
        kernel = HMCECS(inner_kernel, num_blocks=100, proxy=taylor_proxy(neutra_ref_params))
    else:
        raise ValueError("Invalid algorithm, either 'HMC', 'NUTS', or 'HMCECS'.")
    mcmc = MCMC(kernel, args.num_warmup, args.num_samples)
    mcmc.run(rng_key, features, labels, subsample_size, extra_fields=("accept_prob",))
    print("Mean accept prob:", jnp.mean(mcmc.get_extra_fields()["accept_prob"]))
    mcmc.print_summary(exclude_deterministic=False)
    print('\nMCMC elapsed time:', time.time() - start)


def main(args):
    features, labels = _load_dataset()
    benchmark_hmc(args, features, labels)


if __name__ == '__main__':
    assert numpyro.__version__.startswith('0.5.0')
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-samples', default=1000, type=int, help='number of samples')
    parser.add_argument('--num-warmup', default=1000, type=int, help='number of warmup steps')
    parser.add_argument('--num-steps', default=10, type=int, help='number of steps (for "HMC")')
    parser.add_argument('--num-chains', nargs='?', default=1, type=int)
    parser.add_argument('--algo', default='HMCECS', type=str,
                        help='whether to run "HMC", "NUTS", "HMCECS", "SA" or "FlowHMCECS"')
    parser.add_argument('--dense-mass', action="store_true")
    parser.add_argument('--x64', action="store_true")
    parser.add_argument('--device', default='cpu', type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)
    if args.x64:
        numpyro.enable_x64()

    main(args)
