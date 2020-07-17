# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import itertools
import time

import numpy as onp

import jax.numpy as np

import numpyro
from numpyro.util import enable_x64

import pickle
from analysis import process_singleton_svi, process_quad_svi


def main(args):
    analysis_results = {'args': args}

    with open(args.results_file, "rb") as f:
        results = pickle.load(f)

    X, Y, samples, hypers = results['X'], results['Y'], results['samples'], results['hypers']
    X, Y = np.array(X), np.array(Y)
    samples = {k: np.array(v) for k, v in samples.items()}
    P = X.shape[-1]

    # compute the mean and square root variance of each coefficient theta_i
    #means, stds = chunk_vmap(lambda dim: analyze_dimension(samples, X, Y, dim, hypers),
    #                         np.arange(P), chunk_size=999)
    #print("analyze_dimension time", time.time()-t0)

    t0 = time.time()
    means, stds = process_singleton_svi(X, Y, samples, hypers['c'], omega_chunk_size=1, probe_chunk_size=200,
                                        method='direct', rank1=16, rank2=16)
                                        #method=args['inference'][4:])
    print("analyze_dimension time", time.time()-t0)

    analysis_results['singleton_coeff_means'] = onp.array(means).tolist()
    analysis_results['singleton_coeff_stds'] = onp.array(stds).tolist()

    print("Coefficients theta_1 to theta_%d used to generate the data:" % results['args'].active_dimensions,
          results['expected_thetas'])
    active_dims = []
    expected_active_dims = onp.arange(results['args'].active_dimensions).tolist()

    strictness = 3.0

    for dim, (mean, std) in enumerate(zip(means, stds)):
        # we mark the dimension as inactive if the interval [mean - 2 * std, mean + 2 * std] contains zero
        lower, upper = mean - strictness * std, mean + strictness * std
        inactive = "inactive" if lower < 0.0 and upper > 0.0 else "active"
        if inactive == "active":
            active_dims.append(dim)
        if dim < results['args'].active_dimensions or inactive == "active":
            print("[dimension %02d/%02d]  %s:\t%.2e +- %.2e" % (dim + 1, P, inactive, mean, std))

    correct_singletons = len(set(active_dims) & set(expected_active_dims))
    false_singletons = len(set(active_dims) - set(expected_active_dims))
    missed_singletons = len(set(expected_active_dims) - set(active_dims))

    analysis_results['correct_singletons'] = correct_singletons
    analysis_results['false_singletons'] = false_singletons
    analysis_results['missed_singletons'] = missed_singletons

    print("correct_singletons: ", correct_singletons, "  false_singletons: ", false_singletons,
          "  missed_singletons: ", missed_singletons)

    print("Identified a total of %d active dimensions; expected %d." % (len(active_dims),
                                                                        results['args'].active_dimensions))

    #import sys; sys.exit()
    strictness = 5.0

    # Compute the mean and square root variance of coefficients theta_ij for i,j active dimensions.
    # Note that the resulting numbers are only meaningful for i != j.
    active_quad_dims = []
    t0 = time.time()
    if len(active_dims) > 0:
        dim_pairs = np.array(list(itertools.product(active_dims, active_dims)))
        #fun = lambda dim_pair: analyze_pair_of_dimensions(samples, X, Y, dim_pair[0], dim_pair[1], hypers)
        #means, stds = chunk_vmap(fun, dim_pairs, chunk_size=32)
        means, stds = process_quad_svi(X, Y, samples, dim_pairs, hypers['c'], omega_chunk_size=1, probe_chunk_size=200,
                                       method='ppcg', rank1=16, rank2=16)
        analysis_results['pairwise_coeff_means'] = onp.array(means).tolist()
        analysis_results['pairwise_coeff_stds'] = onp.array(stds).tolist()
        for dim_pair, mean, std in zip(dim_pairs, means, stds):
            dim1, dim2 = dim_pair
            if dim1 >= dim2:
                continue
            lower, upper = mean - strictness * std, mean + strictness * std
            if not (lower < 0.0 and upper > 0.0):
                format_str = "Identified pairwise interaction between dimensions %d and %d: %.2e +- %.2e"
                print(format_str % (dim1 + 1, dim2 + 1, mean, std))
                active_quad_dims.append((dim1, dim2))
            #elif dim1 < args['active_dimensions'] and dim2 < args['active_dimensions']:
            #    format_str = "No pairwise interaction between dimensions %d and %d: %.2e +- %.2e"
            #    print(format_str % (dim1 + 1, dim2 + 1, mean, std))

    print("analyze_pair_dimension time", time.time()-t0)

    expected_quad_dims = results['expected_quad_dims']

    correct_quads = len(set(active_quad_dims) & set(expected_quad_dims))
    false_quads = len(set(active_quad_dims) - set(expected_quad_dims))
    missed_quads = len(set(expected_quad_dims) - set(active_quad_dims))

    analysis_results['correct_quads'] = correct_quads
    analysis_results['false_quads'] = false_quads
    analysis_results['missed_quads'] = missed_quads

    print("correct_quads: ", correct_quads, "  false_quads: ", false_quads,
          "  missed_quads: ", missed_quads)

    #print("RESULTS\n", results)
    #log_file = 'slog.{}.P_{}.S_{}.seed_{}.ns_{}_{}.mtd_{}'
    #log_file = log_file.format(args.inference, P, args.active_dimensions, args.seed,
    #                           args.num_warmup, args.num_samples, args.mtd)

    #with open(args.log_dir + log_file + '.pkl', 'wb') as f:
    #    pickle.dump(results, f, protocol=2)
    #print("saved results to {}".format(args.log_dir + log_file + '.pkl'))


if __name__ == "__main__":
    assert numpyro.__version__.startswith('0.2.4')
    parser = argparse.ArgumentParser(description="Analyze results of pairwise.py inference")
    parser.add_argument("--seed", nargs='?', default=0, type=int)
    parser.add_argument("--cg-tol", nargs='?', default=0.001, type=float)
    parser.add_argument("--thinning", nargs='?', default=10, type=int)
    parser.add_argument("--device", default='gpu', type=str, help='use "cpu" or "gpu".')
    parser.add_argument("--log-dir", default='./', type=str)
    parser.add_argument("--results-file", default='results.out', type=str)
    parser.add_argument("--double", action="store_true")
    args = parser.parse_args()

    numpyro.set_platform(args.device)

    if args.double:
        enable_x64()

    main(args)
