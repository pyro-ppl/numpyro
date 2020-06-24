# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import itertools
import os
import time

import numpy as onp

from jax import jit
import jax.numpy as np
import jax.random as random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, ELBO
from numpyro.distributions import constraints
from numpyro.distributions.transforms import AffineTransform, SigmoidTransform
from numpyro.infer.util import Predictive
from numpyro.diagnostics import print_summary
from numpyro.util import enable_x64

import pickle
from cg import cg_quad_form_log_det, direct_quad_form_log_det, cpcg_quad_form_log_det, pcpcg_quad_form_log_det
from utils import CustomAdam, record_stats, kdot, sample_aux_noise, _fori_loop
from mvm import kernel_mvm

from data import get_data
from analysis import process_singleton_svi, process_quad_svi


# The kernel that corresponds to our quadratic logit function
def kernel(X, Z, eta1, eta2, c, jitter=1.0e-6):
    eta1sq = np.square(eta1)
    eta2sq = np.square(eta2)
    k1 = 0.5 * eta2sq * np.square(1.0 + kdot(X, Z))
    k2 = -0.5 * eta2sq * kdot(np.square(X), np.square(Z))
    k3 = (eta1sq - eta2sq) * kdot(X, Z)
    k4 = np.square(c) - 0.5 * eta2sq
    if X.shape == Z.shape:
        k4 += jitter * np.eye(X.shape[0])
    return k1 + k2 + k3 + k4


# Most of the model code is concerned with constructing the sparsity inducing prior.
def model(X, Y, hypers, method="direct", num_probes=1, cg_tol=0.001):
    S, sigma, P, N = hypers['expected_sparsity'], hypers['sigma'], X.shape[1], X.shape[0]

    phi = sigma * (S / np.sqrt(N)) / (P - S)
    eta1 = numpyro.sample("eta1", dist.HalfCauchy(phi))

    msq = numpyro.sample("msq", dist.InverseGamma(hypers['alpha1'], hypers['beta1']))
    xisq = numpyro.sample("xisq", dist.InverseGamma(hypers['alpha2'], hypers['beta2']))

    eta2 = numpyro.deterministic('eta2', np.square(eta1) * np.sqrt(xisq) / msq)

    lam = numpyro.sample("lambda", dist.HalfCauchy(np.ones(P)))
    kappa = numpyro.deterministic('kappa', np.sqrt(msq) * lam / np.sqrt(msq + np.square(eta1 * lam)))

    omega = numpyro.sample("omega", dist.TruncatedPolyaGamma(batch_shape=(N,)))

    kX = kappa * X

    dilation = 8

    if method != 'ppcg':
        k = kernel(kX, kX, eta1, eta2, hypers['c'])
        k_omega = k + np.eye(N) * (1.0 / omega)
        kY = np.matmul(k, Y)
        log_factor = 0.125 * np.dot(Y, kY) - 0.5 * np.sum(np.log(omega))
    else:
        #kY = kernel_mvm_diag(Y, kX, eta1, eta2, hypers['c'], 0.0, dilation=dilation)
        kY = kernel_mvm(Y, kappa, X, np.broadcast_to(eta1, Y.shape), np.broadcast_to(eta2, Y.shape), hypers['c'], dilation)
        log_factor = 0.125 * np.dot(Y, kY) - 0.5 * np.sum(np.log(omega))

    max_iters = 200
    rank1, rank2 = 32, 12
    res_norm, cg_iters, qfld = 0.0, 0.0, 0.0

    if method == "direct":
        qfld = direct_quad_form_log_det(k_omega, 0.5 * kY)
    elif method == "cg":
        probe = sample_aux_noise(shape=(num_probes, N))
        qfld, res_norm, cg_iters = cg_quad_form_log_det(k_omega, 0.5 * kY, probe, epsilon=cg_tol, max_iters=max_iters)
    elif method == "pcg":
        probe = sample_aux_noise(shape=(num_probes, N))
        qfld, res_norm, cg_iters = jit(cpcg_quad_form_log_det, static_argnums=(5, 9, 10, 11, 12))(k_omega,
            0.5 * kY, eta1, eta2, 1.0 / omega, hypers['c'], kX, kappa, probe, rank1, rank2, cg_tol, max_iters)
    elif method == "ppcg":
        probe = sample_aux_noise(shape=(num_probes, N))
        qfld, res_norm, cg_iters = jit(pcpcg_quad_form_log_det, static_argnums=(5, 6, 7, 8, 9, 10, 11, 12))(kappa,
            0.5 * kY, eta1, eta2, 1.0 / omega, hypers['c'], X, probe, rank1, rank2, cg_tol, max_iters, dilation)

    record_stats(np.array([res_norm, cg_iters]))

    numpyro.factor("obs", log_factor - 0.5 * qfld)


def guide(X, Y, hypers, method="direct", num_probes=4, cg_tol=0.001):
    S, sigma, P, N = hypers['expected_sparsity'], hypers['sigma'], X.shape[1], X.shape[0]

    phi = sigma * (S / np.sqrt(N)) / (P - S)

    eta1_loc = numpyro.param("eta1_loc", 0.25, constraint=constraints.positive)
    eta1 = numpyro.sample("eta1", dist.Delta(eta1_loc))

    msq_loc = numpyro.param("msq_loc", 1.0, constraint=constraints.positive)
    msq = numpyro.sample("msq", dist.Delta(msq_loc))

    xisq_loc = numpyro.param("xisq_loc", 1.0, constraint=constraints.positive)
    xisq = numpyro.sample("xisq", dist.Delta(xisq_loc))

    lam_loc = numpyro.param("lam_loc", 0.5 * np.ones(P), constraint=constraints.positive)
    lam = numpyro.sample("lambda", dist.Delta(lam_loc))

    omega_loc = numpyro.param('omega_loc', -2.0 * np.ones(N))
    omega_scale = numpyro.param('omega_scale', 0.8 * np.ones(N), constraint=constraints.positive)
    base_dist = dist.Normal(omega_loc, omega_scale)
    omega_dist = dist.TransformedDistribution(base_dist, [SigmoidTransform(), AffineTransform(0, 2.5)])
    omega = numpyro.sample("omega", omega_dist)

# Helper function for doing HMC inference
def run_hmc(model, args, rng_key, X, Y, hypers):
    start = time.time()
    kernel = NUTS(model, max_tree_depth=args['mtd'])
    mcmc = MCMC(kernel, args['num_warmup'], args['num_samples'], num_chains=args['num_chains'],
                progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True)
    mcmc.run(rng_key, X, Y, hypers)
    mcmc.print_summary()
    elapsed_time = time.time() - start

    samples = mcmc.get_samples()
    # thin samples
    for k, v in samples.items():
        samples[k] = v[::args['thinning']]

    return samples, elapsed_time

def do_svi(model, guide, args, rng_key, X, Y, hypers, num_samples=4):
    rng_key_init, rng_key_post = random.split(rng_key, 2)
    adam = CustomAdam(args['lr'])
    svi = SVI(model, guide, adam, ELBO())
    svi_state = svi.init(rng_key_init, X, Y, hypers, method=args['inference'][4:], cg_tol=args['cg_tol'])

    num_steps = args['num_samples']
    report_frequency = 50
    beta = 0.95
    bias_correction = 1.0 / (1.0 - beta ** report_frequency)

    @jit
    def body_fn(i, init_val):
        svi_state, old_loss, old_stats = init_val
        svi_state, loss = svi.update(svi_state, X, Y, hypers, method=args['inference'][4:], cg_tol=args['cg_tol'])
        loss = (1.0 - beta) * loss + beta * old_loss
        stats = (1.0 - beta) * svi_state.optim_state[1] + beta * old_stats
        return (svi_state, loss, stats)

    def do_chunk(svi_state):
        return _fori_loop(0, report_frequency, body_fn, (svi_state, 0.0, np.zeros(2)))

    ts = [time.time()]
    res_norm_history = []
    cg_iters_history = []

    for step_chunk in range(1, 1 + num_steps // report_frequency):
        svi_state, loss, (res_norm, cg_iters) = do_chunk(svi_state)
        loss *= bias_correction
        res_norm *= bias_correction
        cg_iters *= bias_correction
        ts.append(time.time())
        dt = (ts[-1] - ts[-2]) / float(report_frequency)
        if "direct" not in args['inference']:
            print("[iter %03d]  %.3f \t\t  res_norm: %.2e  cg_iters: %.1f \t\t [dt: %.3f]" % (step_chunk * report_frequency,
                  loss, res_norm, cg_iters, dt))
            res_norm_history.append(res_norm)
            cg_iters_history.append(cg_iters)
        else:
            print("[iter %03d]  %.3f \t\t [dt: %.3f]" % (step_chunk * report_frequency, loss, dt))

    print("mean res_norm {:.5f}   mean cg_iters {:.2f}".format(onp.mean(res_norm_history), onp.mean(cg_iters_history)))
    print("res_norm_history", res_norm_history)
    print("cg_iters_history", cg_iters_history)
    elapsed_time = time.time() - ts[0]

    params = svi.get_params(svi_state)
    return_sites = ['eta1', 'eta2', 'kappa', 'omega', 'lambda']
    ## TODO drop obs in model?
    samples = Predictive(model, guide=guide, num_samples=num_samples, params=params,
                         return_sites=return_sites)(rng_key_post, X, Y, hypers, 0, 0.0)

    for k, v in samples.items():
        if v.ndim == 1:
            print("{}  {:.4f}".format(k, v[0]))

    _report = {k: v for k, v in samples.items() if v.ndim == 2}
    print_summary(_report)

    return samples, elapsed_time


def main(**args):
    results = {'args': args}
    P = args['num_dimensions']
    print(args)

    # setup hyperparameters
    hypers = {'expected_sparsity': args['active_dimensions'],
              'alpha1': 2.0, 'beta1': 1.0, 'sigma': 2.0,
              'alpha2': 2.0, 'beta2': 1.0, 'c': 1.0}

    for N in [2000]:
    #for N in [500]: #800, 1600, 2400, 3600]:
        results[N] = {}

        X, Y, expected_thetas, _, expected_quad_dims = get_data(N=N, P=P, Q=12,
                                                                S=args['active_dimensions'], seed=args['seed'])
        print("X, Y", X.shape, Y.shape)

        rng_key = random.PRNGKey(args['seed'])

        print("starting {} inference...".format(args['inference']))
        if 'svi' in args['inference']:
            samples, inf_time = do_svi(model, guide, args, rng_key, X, Y, hypers, num_samples=48)
        elif args['inference'] == 'hmc':
            samples, inf_time = run_hmc(model, args, rng_key, X, Y, hypers)
        print("done with inference! [took {:.2f} seconds]".format(inf_time))

        print("leading lambda", onp.mean(samples['lambda'], axis=0)[:40])
        print("leading kappa", onp.mean(samples['kappa'], axis=0)[:40])

        # compute the mean and square root variance of each coefficient theta_i
        #means, stds = chunk_vmap(lambda dim: analyze_dimension(samples, X, Y, dim, hypers),
        #                         np.arange(P), chunk_size=999)
        #print("analyze_dimension time", time.time()-t0)
        t0 = time.time()
        means, stds = process_singleton_svi(X, Y, samples, hypers['c'], omega_chunk_size=1, probe_chunk_size=256,
                                            method='ppcg', rank1=64, rank2=16)
                                            #method=args['inference'][4:])
        print("analyze_dimension time", time.time()-t0)

        results[N]['inf_time'] = inf_time
        results[N]['expected_thetas'] = onp.array(expected_thetas).tolist()
        results[N]['singleton_coeff_means'] = onp.array(means).tolist()
        results[N]['singleton_coeff_stds'] = onp.array(stds).tolist()

        print("Coefficients theta_1 to theta_%d used to generate the data:" % args['active_dimensions'], expected_thetas)
        active_dims = []
        expected_active_dims = onp.arange(args['active_dimensions']).tolist()

        strictness = 3.0

        for dim, (mean, std) in enumerate(zip(means, stds)):
            # we mark the dimension as inactive if the interval [mean - 2 * std, mean + 2 * std] contains zero
            lower, upper = mean - strictness * std, mean + strictness * std
            inactive = "inactive" if lower < 0.0 and upper > 0.0 else "active"
            if inactive == "active":
                active_dims.append(dim)
            if dim < args['active_dimensions'] or inactive == "active":
                print("[dimension %02d/%02d]  %s:\t%.2e +- %.2e" % (dim + 1, P, inactive, mean, std))

        correct_singletons = len(set(active_dims) & set(expected_active_dims))
        false_singletons = len(set(active_dims) - set(expected_active_dims))
        missed_singletons = len(set(expected_active_dims) - set(active_dims))

        results[N]['correct_singletons'] = correct_singletons
        results[N]['false_singletons'] = false_singletons
        results[N]['missed_singletons'] = missed_singletons

        print("correct_singletons: ", correct_singletons, "  false_singletons: ", false_singletons,
              "  missed_singletons: ", missed_singletons)

        print("Identified a total of %d active dimensions; expected %d." % (len(active_dims),
                                                                            args['active_dimensions']))

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
            means, stds = process_quad_svi(X, Y, samples, dim_pairs, hypers['c'], omega_chunk_size=1, probe_chunk_size=256,
                                           method='ppcg', rank1=64, rank2=16)
            results[N]['pairwise_coeff_means'] = onp.array(means).tolist()
            results[N]['pairwise_coeff_stds'] = onp.array(stds).tolist()
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

        correct_quads = len(set(active_quad_dims) & set(expected_quad_dims))
        false_quads = len(set(active_quad_dims) - set(expected_quad_dims))
        missed_quads = len(set(expected_quad_dims) - set(active_quad_dims))

        results[N]['correct_quads'] = correct_quads
        results[N]['false_quads'] = false_quads
        results[N]['missed_quads'] = missed_quads

        print("correct_quads: ", correct_quads, "  false_quads: ", false_quads,
              "  missed_quads: ", missed_quads)

    #print("RESULTS\n", results)
    log_file = 'slog.{}.P_{}.S_{}.seed_{}.ns_{}_{}.mtd_{}'
    log_file = log_file.format(args['inference'], P, args['active_dimensions'], args['seed'],
                               args['num_warmup'], args['num_samples'], args['mtd'])

    #with open(args['log_dir'] + log_file + '.pkl', 'wb') as f:
    #    pickle.dump(results, f, protocol=2)
    #print("saved results to {}".format(args['log_dir'] + log_file + '.pkl'))


if __name__ == "__main__":
    assert numpyro.__version__.startswith('0.2.4')
    parser = argparse.ArgumentParser(description="Sparse Logistic Regression example")
    parser.add_argument("--inference", nargs="?", default='svi-direct', type=str,
                        choices=['hmc','svi-direct','svi-cg','svi-pcg', 'svi-ppcg'])
    parser.add_argument("-n", "--num-samples", nargs="?", default=800, type=int)
    parser.add_argument("--num-warmup", nargs='?', default=0, type=int)
    parser.add_argument("--num-chains", nargs='?', default=1, type=int)
    parser.add_argument("--mtd", nargs='?', default=5, type=int)
    parser.add_argument("--num-data", nargs='?', default=0, type=int)
    parser.add_argument("--num-dimensions", nargs='?', default=70, type=int)
    parser.add_argument("--seed", nargs='?', default=0, type=int)
    parser.add_argument("--lr", nargs='?', default=0.005, type=float)
    parser.add_argument("--cg-tol", nargs='?', default=0.001, type=float)
    parser.add_argument("--active-dimensions", nargs='?', default=14, type=int)
    parser.add_argument("--thinning", nargs='?', default=10, type=int)
    parser.add_argument("--device", default='cpu', type=str, help='use "cpu" or "gpu".')
    parser.add_argument("--log-dir", default='./very_large/', type=str)
    parser.add_argument("--double", action="store_true")
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    if args.double:
        enable_x64()

    main(**vars(args))
