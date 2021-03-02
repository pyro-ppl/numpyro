import argparse

from jax.random import PRNGKey

import numpyro
from numpyro.infer import MCMC, NUTS

from age_model import model
from data import get_data, transform_data


def main(args):
    data = get_data()
    transformed_data = transform_data(data)

    kernel = NUTS(model, step_size=1.0e-3, max_tree_depth=15, target_accept_prob=0.6)
    mcmc = MCMC(kernel, num_warmup=args.num_warmup, num_samples=args.num_samples)

    rng_key = PRNGKey(0)
    mcmc.run(rng_key, transformed_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Covid Age Model")
    parser.add_argument("-n", "--num-samples", nargs="?", default=800, type=int)
    parser.add_argument("--num-warmup", nargs='?', default=400, type=int)
    parser.add_argument("--device", default='cpu', type=str, choices=['cpu', 'gpu'])
    args = parser.parse_args()

    numpyro.enable_x64()
    numpyro.set_platform(args.device)

    main(args)
