import argparse
import numpy as np
from jax.random import PRNGKey

import numpyro
from numpyro.infer import MCMC, NUTS, init_to_value

from age_model import model
from data import get_data, transform_data, generate_init_values

import pickle


def main(args):
    print(args)

    data = get_data(M=args.M)
    transformed_data = transform_data(data)

    s = "M = {}  N0 = {}  N2 = {}  A = {}  SI_CUT = {}  N_IMP = {}"
    print(s.format(data['M'], data['N0'], data['N2'],
                   data['A'], data['SI_CUT'], data['N_IMP']))

    rng_key = PRNGKey(0)

    init_strategy=init_to_value(values=generate_init_values(data, seed=0))

    if args.dense_mass == 'true':
        dense_mass = [ ("R0_{}".format(m), "dip_rnde_{}".format(m), "e_cases_N0_{}".format(m))
                      for m in range(args.M) ]
        dense_mass.extend([ ("sd_upswing_timeeff_reduced", "beta1") ])
        dense_mass.extend([ ("hyper_log_ifr_age_rnde_mid1", "log_ifr_age_rnde_mid1") ])
        dense_mass.extend([ ("hyper_log_ifr_age_rnde_mid2", "log_ifr_age_rnde_mid2") ])
        dense_mass.extend([ ("hyper_log_ifr_age_rnde_old", "log_ifr_age_rnde_old")])
        dense_mass.extend([ ("timeeff_shift_mid1", "hyper_timeeff_shift_mid1") ])
        dense_mass.extend([('upswing_timeeff_reduced_{}'.format(m),) for m in range(args.M)])
        dense_mass.extend([ ("log_ifr_age_base",) ])
        print(dense_mass)
    else:
        dense_mass = []

    kernel = NUTS(model, step_size=args.step_size,
                  max_tree_depth=(args.mtd_warmup, args.mtd),
                  target_accept_prob=args.tap,
                  init_strategy=init_strategy,
                  dense_mass=dense_mass)
    mcmc = MCMC(kernel, num_warmup=args.num_warmup, num_samples=args.num_samples,
                num_chains=args.num_chains, progress_bar=True)
    mcmc.run(rng_key, transformed_data, reparam=(args.reparam=='true'))
    samples = mcmc.get_samples()
    mcmc.print_summary()

    f = 'samples.default.M{}.repa_{}.ns_nw_{}_{}.ss_{:.5f}.mtd_{}_{}.tap_{:.2f}.dm_{}.pkl'
    f = f.format(args.M, args.reparam, args.num_samples, args.num_samples,
                 args.step_size, args.mtd_warmup, args.mtd, args.tap, args.dense_mass)

    with open(f, "wb") as f:
        pickle.dump(samples, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Covid Age Model")
    parser.add_argument("-n", "--num-samples", nargs="?", default=500, type=int)
    parser.add_argument("--num-warmup", default=500, type=int)
    parser.add_argument("--device", default='cpu', type=str, choices=['cpu', 'gpu'])
    parser.add_argument("--num-chains", default=1, type=int)
    parser.add_argument("--step-size", default=0.01, type=float)
    parser.add_argument("--tap", default=0.90, type=float)
    parser.add_argument("--mtd", default=11, type=int)
    parser.add_argument("--mtd-warmup", default=10, type=int)
    parser.add_argument("--M", default=4, type=int)
    parser.add_argument("--reparam", default='true', type=str, choices=['true', 'false'])
    parser.add_argument("--dense-mass", default='true', type=str, choices=['true', 'false'])
    args = parser.parse_args()

    numpyro.enable_x64()
    numpyro.set_platform(args.device)
    #numpyro.set_host_device_count(args.num_chains)

    main(args)
