# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
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
from cg import cg_quad_form_log_det, direct_quad_form_log_det, cpcg_quad_form_log_det
from utils import CustomAdam, record_stats, kdot, sample_aux_noise, _fori_loop, sample_permutation

from data import get_data
from vjp import pcpcg_quad_form_log_det2


def kernel(X, Z, eta1, eta2, c):
    eta1sq = np.square(eta1)
    eta2sq = np.square(eta2)
    k1 = 0.5 * eta2sq * np.square(1.0 + kdot(X, Z))
    k2 = -0.5 * eta2sq * kdot(np.square(X), np.square(Z))
    k3 = (eta1sq - eta2sq) * kdot(X, Z)
    k4 = np.square(c) - 0.5 * eta2sq
    return k1 + k2 + k3 + k4

def sample_hypers(sigma, S, N, P, hypers):
    phi = sigma * (S / np.sqrt(N)) / (P - S)
    eta1 = numpyro.sample("eta1", dist.HalfCauchy(phi))

    msq = numpyro.sample("msq", dist.InverseGamma(hypers['alpha1'], hypers['beta1']))
    xisq = numpyro.sample("xisq", dist.InverseGamma(hypers['alpha2'], hypers['beta2']))
    eta2 = numpyro.deterministic('eta2', np.square(eta1) * np.sqrt(xisq) / msq)

    lam = numpyro.sample("lambda", dist.HalfCauchy(np.ones(P)))
    kappa = numpyro.deterministic('kappa', np.sqrt(msq) * lam / np.sqrt(msq + np.square(eta1 * lam)))
    return eta1, eta2, kappa

def bernoulli_model(X, Y, hypers, method="direct", num_probes=1, cg_tol=0.001):
    S, sigma, P, N = hypers['expected_sparsity'], hypers['sigma'], X.shape[1], X.shape[0]

    eta1, eta2, kappa = sample_hypers(sigma, S, N, P, hypers)

    omega = numpyro.sample("omega", dist.TruncatedPolyaGamma(batch_shape=(N,)))

    kX = kappa * X

    dilation = 1

    if method != 'ppcg':
        k = kernel(kX, kX, eta1, eta2, hypers['c'])
        k_omega = k + np.eye(N) * (1.0 / omega)
        kY = np.matmul(k, Y)
        log_factor = 0.125 * np.dot(Y, kY) - 0.5 * np.sum(np.log(omega))
    else:
        log_factor = - 0.5 * np.sum(np.log(omega))

    max_iters = 200
    rank1, rank2 = 16, 12
    res_norm, cg_iters, qfld = 0.0, 0.0, 0.0

    if method == "direct":
        qfld = -0.5 * jit(direct_quad_form_log_det)(k_omega, 0.5 * kY)
    elif method == "cg":
        probe = sample_aux_noise(shape=(num_probes, N))
        qfld, res_norm, cg_iters = cg_quad_form_log_det(k_omega, 0.5 * kY, probe, epsilon=cg_tol, max_iters=max_iters)
    elif method == "pcg":
        probe = sample_aux_noise(shape=(num_probes, N))
        qfld, res_norm, cg_iters = jit(cpcg_quad_form_log_det, static_argnums=(5, 9, 10, 11, 12))(k_omega,
            0.5 * kY, eta1, eta2, 1.0 / omega, hypers['c'], kX, kappa, probe, rank1, rank2, cg_tol, max_iters)
    elif method == "ppcg":
        probe = sample_aux_noise(shape=(num_probes, N))
        subsample = sample_permutation(N)[:N // 10]
        #qfld, res_norm, cg_iters = jit(pcpcg_quad_form_log_det, static_argnums=(5, 6, 7, 8, 9, 10, 11, 12))(kappa,
        #    0.5 * kY, eta1, eta2, 1.0 / omega, hypers['c'], X, probe, rank1, rank2, cg_tol, max_iters, dilation)
        #qfld, res_norm, cg_iters = pcpcg_quad_form_log_det2(kappa,
        #     Y, eta1, eta2, 1.0 / omega, hypers['c'], X, probe, rank1, rank2, cg_tol, max_iters, dilation, subsample)
        qfld, res_norm, cg_iters = jit(pcpcg_quad_form_log_det2, static_argnums=(5, 6, 7, 8, 9, 10, 11, 12))(kappa,
             Y, eta1, eta2, 1.0 / omega, hypers['c'], X, probe, rank1, rank2, cg_tol, max_iters, dilation, subsample)

    record_stats(np.array([res_norm, cg_iters]))

    numpyro.factor("obs", log_factor + qfld)

def gaussian_model(X, Y, hypers, method="direct", num_probes=1, cg_tol=0.001):
    S, P, N = hypers['expected_sparsity'], X.shape[1], X.shape[0]

    sigma = numpyro.sample("sigma", dist.HalfNormal(hypers['alpha3']))
    eta1, eta2, kappa = sample_hypers(sigma, S, N, P, hypers)

    kX = kappa * X

    dilation = 1

    if method != 'ppcg':
        k_sigma = kernel(kX, kX, eta1, eta2, hypers['c']) + sigma ** 2 * np.eye(N)

    max_iters = 200
    rank1, rank2 = 16, 12
    res_norm, cg_iters, qfld = 0.0, 0.0, 0.0

    if method == "direct":
        numpyro.sample("Y", dist.MultivariateNormal(loc=np.zeros(X.shape[0]), covariance_matrix=k_sigma),
                       obs=Y)
    elif method == "cg":
        probe = sample_aux_noise(shape=(num_probes, N))
        qfld, res_norm, cg_iters = cg_quad_form_log_det(k_omega, 0.5 * kY, probe, epsilon=cg_tol, max_iters=max_iters)
    elif method == "pcg":
        probe = sample_aux_noise(shape=(num_probes, N))
        qfld, res_norm, cg_iters = jit(cpcg_quad_form_log_det, static_argnums=(5, 9, 10, 11, 12))(k_omega,
            0.5 * kY, eta1, eta2, 1.0 / omega, hypers['c'], kX, kappa, probe, rank1, rank2, cg_tol, max_iters)
    elif method == "ppcg":
        probe = sample_aux_noise(shape=(num_probes, N))
        subsample = sample_permutation(N)[:N // 10]
        qfld, res_norm, cg_iters = pcpcg_quad_form_log_det2(kappa,
             Y, eta1, eta2, 1.0 / omega, hypers['c'], X, probe, rank1, rank2, cg_tol, max_iters, dilation, subsample)

    record_stats(np.array([res_norm, cg_iters]))

    if method != "direct":
        numpyro.factor("obs", - 0.5 * qfld)


def bernoulli_guide(X, Y, hypers, method="direct", num_probes=4, cg_tol=0.001):
    S, sigma, P, N = hypers['expected_sparsity'], hypers['sigma'], X.shape[1], X.shape[0]

    phi = sigma * (S / np.sqrt(N)) / (P - S)

    eta1_loc = numpyro.param("eta1_loc", 0.25, constraint=constraints.positive)
    numpyro.sample("eta1", dist.Delta(eta1_loc))

    msq_loc = numpyro.param("msq_loc", 1.0, constraint=constraints.positive)
    numpyro.sample("msq", dist.Delta(msq_loc))

    xisq_loc = numpyro.param("xisq_loc", 1.0, constraint=constraints.positive)
    numpyro.sample("xisq", dist.Delta(xisq_loc))

    lam_loc = numpyro.param("lam_loc", 0.5 * np.ones(P), constraint=constraints.positive)
    numpyro.sample("lambda", dist.Delta(lam_loc))

    omega_loc = numpyro.param('omega_loc', -2.0 * np.ones(N))
    omega_scale = numpyro.param('omega_scale', 0.8 * np.ones(N), constraint=constraints.positive)
    base_dist = dist.Normal(omega_loc, omega_scale)
    omega_dist = dist.TransformedDistribution(base_dist, [SigmoidTransform(), AffineTransform(0, 2.5)])
    omega = numpyro.sample("omega", omega_dist)


def gaussian_guide(X, Y, hypers, method="direct", num_probes=4, cg_tol=0.001):
    S, P, N = hypers['expected_sparsity'], X.shape[1], X.shape[0]

    sigma_loc = numpyro.param("sigma_loc", 0.25, constraint=constraints.positive)
    sigma = numpyro.sample("sigma", dist.Delta(sigma_loc))
    phi = sigma * (S / np.sqrt(N)) / (P - S)

    eta1_loc = numpyro.param("eta1_loc", 0.25, constraint=constraints.positive)
    eta1 = numpyro.sample("eta1", dist.Delta(eta1_loc))

    msq_loc = numpyro.param("msq_loc", 1.0, constraint=constraints.positive)
    msq = numpyro.sample("msq", dist.Delta(msq_loc))

    xisq_loc = numpyro.param("xisq_loc", 1.0, constraint=constraints.positive)
    xisq = numpyro.sample("xisq", dist.Delta(xisq_loc))

    lam_loc = numpyro.param("lam_loc", 0.5 * np.ones(P), constraint=constraints.positive)
    numpyro.sample("lambda", dist.Delta(lam_loc))


def run_hmc(model, args, rng_key, X, Y, hypers):
    start = time.time()
    kernel = NUTS(model, max_tree_depth=args.mtd)
    mcmc = MCMC(kernel, args.num_warmup, args.num_samples, num_chains=args.num_chains,
                progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True)
    mcmc.run(rng_key, X, Y, hypers)
    mcmc.print_summary()
    elapsed_time = time.time() - start

    samples = mcmc.get_samples()
    # thin samples
    for k, v in samples.items():
        samples[k] = v[::args.thinning]

    return samples, elapsed_time

def do_svi(model, guide, args, rng_key, X, Y, hypers, num_samples=4):
    rng_key_init, rng_key_post = random.split(rng_key, 2)
    adam = CustomAdam(args.lr)
    svi = SVI(model, guide, adam, ELBO())
    svi_state = svi.init(rng_key_init, X, Y, hypers, method=args.inference[4:], cg_tol=args.cg_tol)

    num_steps = args.num_samples
    report_frequency = 50
    beta = 0.95
    bias_correction = 1.0 / (1.0 - beta ** report_frequency)

    @jit
    def body_fn(i, init_val):
        svi_state, old_loss, old_stats = init_val
        svi_state, loss = svi.update(svi_state, X, Y, hypers, method=args.inference[4:], cg_tol=args.cg_tol)
        loss = (1.0 - beta) * loss + beta * old_loss
        stats = (1.0 - beta) * svi_state.optim_state[1] + beta * old_stats
        return (svi_state, loss, stats)

    def do_chunk(svi_state):
        return _fori_loop(np.array(0), np.array(report_frequency), body_fn, (svi_state, np.array(0.0), np.zeros(2)))

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
        if "direct" not in args.inference:
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
    samples = Predictive(model, guide=guide, num_samples=num_samples, params=params,
                         return_sites=return_sites)(rng_key_post, X, Y, hypers, 0, 0.0)

    for k, v in samples.items():
        if v.ndim == 1:
            print("{}  {:.4f}".format(k, v[0]))

    _report = {k: v for k, v in samples.items() if v.ndim == 2}
    print_summary(_report)

    return samples, elapsed_time


def main(args):
    results = {'args': args}
    N = args.num_data
    P = args.num_dimensions
    print(args)

    # setup hyperparameters
    hypers = {'expected_sparsity': args.active_dimensions,
              'alpha1': 2.0, 'beta1': 1.0, 'sigma': 2.0, 'alpha3': 1.0,
              'alpha2': 2.0, 'beta2': 1.0, 'c': 1.0}
    results['hypers'] = hypers

    X, Y, expected_thetas, _, expected_quad_dims = get_data(N=N, P=P, Q=12,
                                                            S=args.active_dimensions, seed=args.seed,
                                                            likelihood=args.likelihood)
    print("X, Y", X.shape, Y.shape)
    results['X'] = X
    results['Y'] = Y
    results['expected_thetas'] = expected_thetas
    results['expected_quad_dims'] = expected_quad_dims

    rng_key = random.PRNGKey(args.seed)

    print("starting {} inference...".format(args.inference))
    model = bernoulli_model if args.likelihood == 'bernoulli' else gaussian_model
    if 'svi' in args.inference:
        guide = bernoulli_guide if args.likelihood == 'bernoulli' else gaussian_guide
        samples, inf_time = do_svi(model, guide, args, rng_key, X, Y, hypers, num_samples=48)
    elif args.inference == 'hmc':
        samples, inf_time = run_hmc(model, args, rng_key, X, Y, hypers)
    results['samples'] = samples
    print("done with inference! [took {:.2f} seconds]".format(inf_time))

    print("leading lambda", onp.mean(samples['lambda'], axis=0)[:40])
    print("leading kappa", onp.mean(samples['kappa'], axis=0)[:40])

    #print("RESULTS\n", results)
    #log_file = 'slog.{}.P_{}.S_{}.seed_{}.ns_{}_{}.mtd_{}'
    #log_file = log_file.format(args.inference, P, args.active_dimensions, args.seed,
    #                           args.num_warmup, args.num_samples, args.mtd)

    #with open(args.log_dir + log_file + '.pkl', 'wb') as f:
    #    pickle.dump(results, f, protocol=2)
    #print("saved results to {}".format(args.log_dir + log_file + '.pkl'))

    with open(args.log_dir + args.results_file, 'wb') as f:
        pickle.dump(results, f, protocol=2)
    print("saved results to {}".format(args.log_dir + args.results_file))


if __name__ == "__main__":
    assert numpyro.__version__.startswith('0.2.4')
    parser = argparse.ArgumentParser(description="Pairwise Interaction Discovery")
    parser.add_argument("--inference", nargs="?", default='svi-ppcg', type=str,
                        choices=['hmc','svi-direct','svi-cg','svi-pcg', 'svi-ppcg'])
    parser.add_argument("-n", "--num-samples", nargs="?", default=800, type=int)
    parser.add_argument("--num-data", default=2000, type=int)
    parser.add_argument("--num-warmup", default=0, type=int)
    parser.add_argument("--num-chains", default=1, type=int)
    parser.add_argument("--mtd", default=5, type=int)
    parser.add_argument("--num-dimensions", default=100, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--lr", default=0.005, type=float)
    parser.add_argument("--cg-tol", default=0.001, type=float)
    parser.add_argument("--active-dimensions", default=14, type=int)
    parser.add_argument("--thinning", default=10, type=int)
    parser.add_argument("--device", default='gpu', type=str, help='use "cpu" or "gpu".')
    parser.add_argument("--likelihood", default='bernoulli', type=str)
    parser.add_argument("--log-dir", default='./', type=str)
    parser.add_argument("--results-file", default='results.out', type=str)
    parser.add_argument("--double", action="store_true")
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    if args.double:
        enable_x64()

    main(args)
