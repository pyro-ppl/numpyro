import argparse

from jax.random import PRNGKey

import numpyro
from numpyro.infer import MCMC, NUTS

from age_model import model
from data import get_data, transform_data


def main(args):
    data = get_data()
    transformed_data = transform_data(data)

    print("M = {}  N0 = {}  N2 = {}  A = {}  SI_CUT = {}".format(data['M'], data['N0'], data['N2'], data['A'], data['SI_CUT']))

    kernel = NUTS(model, step_size=1.0e-4, max_tree_depth=1, target_accept_prob=0.8)
    mcmc = MCMC(kernel, num_warmup=args.num_warmup, num_samples=args.num_samples, num_chains=args.num_chains, progress_bar=True)

    rng_key = PRNGKey(0)
    mcmc.run(rng_key, transformed_data)
    mcmc.print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Covid Age Model")
    parser.add_argument("-n", "--num-samples", nargs="?", default=800, type=int)
    parser.add_argument("--num-warmup", default=0, type=int)
    parser.add_argument("--device", default='cpu', type=str, choices=['cpu', 'gpu'])
    parser.add_argument("--num-chains", default=1, type=int)
    args = parser.parse_args()

    numpyro.enable_x64()
    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
