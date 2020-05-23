# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import itertools
import os
import time

import numpy as onp

import jax
from jax import vmap
import jax.numpy as np
import jax.random as random

import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.infer import MCMC, NUTS
from numpyro.contrib.autoguide import AutoDiagonalNormal, AutoContinuousELBO
from numpyro.infer import SVI, ELBO
from numpyro.distributions import constraints
from numpyro.distributions.transforms import AffineTransform, SigmoidTransform

from jax.scipy.linalg import cho_factor, solve_triangular, cho_solve
from numpyro.util import enable_x64
from numpyro.util import fori_loop

import pickle


def sigmoid(x):
      return 1.0 / (1.0 + np.exp(-x))


def dot(X, Z):
    return np.dot(X, Z[..., None])[..., 0]


def cho_tri_solve(A, b):
    L = cho_factor(A, lower=True)[0]
    Linv_b = solve_triangular(L, b, lower=True)
    return L, Linv_b


# The kernel that corresponds to our quadratic logit function
def kernel(X, Z, eta1, eta2, c, jitter=1.0e-6):
    eta1sq = np.square(eta1)
    eta2sq = np.square(eta2)
    k1 = 0.5 * eta2sq * np.square(1.0 + dot(X, Z))
    k2 = -0.5 * eta2sq * dot(np.square(X), np.square(Z))
    k3 = (eta1sq - eta2sq) * dot(X, Z)
    k4 = np.square(c) - 0.5 * eta2sq
    if X.shape == Z.shape:
        k4 += jitter * np.eye(X.shape[0])
    return k1 + k2 + k3 + k4


# Most of the model code is concerned with constructing the sparsity inducing prior.
def model(X, Y, hypers):
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
    k = kernel(kX, kX, eta1, eta2, hypers['c'])

    k_omega = k + np.eye(N) * (1.0 / omega)

    kY = np.matmul(k, Y)
    L, Linv_kY = cho_tri_solve(k_omega, kY)

    log_factor1 = dot(Y, kY)
    log_factor2 = dot(Linv_kY, Linv_kY)
    log_factor3 = np.sum(np.log(np.diagonal(L))) + 0.5 * np.sum(np.log(omega))

    obs_factor = 0.125 * (log_factor1 - log_factor2) - log_factor3
    numpyro.factor("obs", obs_factor)


def guide(X, Y, hypers):
    S, sigma, P, N = hypers['expected_sparsity'], hypers['sigma'], X.shape[1], X.shape[0]

    phi = sigma * (S / np.sqrt(N)) / (P - S)

    eta1_loc = numpyro.param("eta1_loc", 0.25, constraint=constraints.positive)
    #eta1_loc = numpyro.param("eta1_loc", phi, constraint=constraints.positive)
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


# helper for computing the posterior marginal N(theta_i) or N(theta_ij)
def compute_coefficient_mean_variance(X, Y, probe, vec, eta1, eta2, c, kappa, omega):
    kprobe, kX = kappa * probe, kappa * X

    k_xx = kernel(kX, kX, eta1, eta2, c)
    k_probeX = kernel(kprobe, kX, eta1, eta2, c)
    k_prbprb = kernel(kprobe, kprobe, eta1, eta2, c)

    L = cho_factor(k_xx + np.eye(X.shape[0]) * (1.0 / omega), lower=True)[0]

    mu = 0.5 * cho_solve((L, True), Y / omega)
    mu = np.dot(vec, dot(k_probeX, mu))

    Linv_kXprobe = solve_triangular(L, np.transpose(k_probeX), lower=True)
    var = k_prbprb - np.matmul(np.transpose(Linv_kXprobe), Linv_kXprobe)
    var = np.dot(vec, np.matmul(var, vec))

    return mu, var


# compute the posterior marginal N(theta_i)
def compute_singleton_mean_variance(X, Y, dimension, eta1, eta2, c, kappa, omega):
    probe = np.zeros((2, X.shape[1]))
    probe = jax.ops.index_update(probe, jax.ops.index[:, dimension], np.array([1.0, -1.0]))
    vec = np.array([0.50, -0.50])
    return compute_coefficient_mean_variance(X, Y, probe, vec, eta1, eta2, c, kappa, omega)


# compute the posterior marginal N(theta_ij)
def compute_pairwise_mean_variance(X, Y, dim1, dim2, eta1, eta2, c, kappa, omega):
    probe = np.zeros((4, X.shape[1]))
    probe = jax.ops.index_update(probe, jax.ops.index[:, dim1], np.array([1.0, 1.0, -1.0, -1.0]))
    probe = jax.ops.index_update(probe, jax.ops.index[:, dim2], np.array([1.0, -1.0, 1.0, -1.0]))
    vec = np.array([0.25, -0.25, -0.25, 0.25])
    return compute_coefficient_mean_variance(X, Y, probe, vec, eta1, eta2, c, kappa, omega)


# Helper function for doing HMC inference
def run_inference(model, args, rng_key, X, Y, hypers):
    start = time.time()
    kernel = NUTS(model, max_tree_depth=args['mtd'])
    mcmc = MCMC(kernel, args['num_warmup'], args['num_samples'], num_chains=args['num_chains'],
                progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True)
    mcmc.run(rng_key, X, Y, hypers)
    mcmc.print_summary()
    print('\nMCMC elapsed time:', time.time() - start)

    samples = mcmc.get_samples()
    # thin samples
    for k, v in samples.items():
        samples[k] = v[::args['thinning']]

    return samples


# Get the mean and variance of a gaussian mixture
def gaussian_mixture_stats(mus, variances):
    mean_mu = np.mean(mus)
    mean_var = np.mean(variances) + np.mean(np.square(mus)) - np.square(mean_mu)
    return mean_mu, mean_var


# Create artificial regression dataset where only S out of P feature
# dimensions contain signal and where there are two pairwise interactions
def get_data(N=20, S=2, P=10, seed=0):
    assert S < P and P > 1 and S > 0
    onp.random.seed(seed)

    # generate S coefficients with non-negligible magnitude
    W = 1.0 + 1.5 * onp.random.rand(S)
    flip = 2 * onp.random.binomial(1, 0.5, W.shape) - 1
    W *= flip

    # generate covariates with non-negligible magnitude
    X = onp.random.rand(N, P) + 0.5
    flip = 2 * onp.random.binomial(1, 0.5, X.shape) - 1
    X *= flip

    # generate data using the S coefficients and two pairwise interactions
    pairwise_coefficient = 3.0
    expected_quad_dims = [(0, 1), (2, 3)]
    Y = onp.sum(X[:, 0:S] * W, axis=-1) + pairwise_coefficient * (X[:, 0] * X[:, 1] - X[:, 2] * X[:, 3])
    Y = 2 * onp.random.binomial(1, sigmoid(Y)) - 1
    print("number of 1s: {}  number of -1s: {}".format(np.sum(Y == 1.0), np.sum(Y == -1.0)))

    assert X.shape == (N, P)
    assert Y.shape == (N,)

    return X, Y, W, pairwise_coefficient, expected_quad_dims


# Helper function for analyzing the posterior statistics for coefficient theta_i
def analyze_dimension(samples, X, Y, dimension, hypers):
    vmap_args = (samples['eta1'], samples['eta2'], samples['kappa'], samples['omega'])
    mus, variances = vmap(lambda eta1, eta2, kappa, omega:
                          compute_singleton_mean_variance(X, Y, dimension, eta1, eta2, hypers['c'], kappa, omega))(*vmap_args)
    mean, variance = gaussian_mixture_stats(mus, variances)
    std = np.sqrt(variance)
    return mean, std


# Helper function for analyzing the posterior statistics for coefficient theta_ij
def analyze_pair_of_dimensions(samples, X, Y, dim1, dim2, hypers):
    vmap_args = (samples['eta1'], samples['eta2'], samples['kappa'], samples['omega'])
    mus, variances = vmap(lambda eta1, eta2, kappa, omega:
                          compute_pairwise_mean_variance(X, Y, dim1, dim2, eta1, eta2, hypers['c'], kappa, omega))(*vmap_args)
    mean, variance = gaussian_mixture_stats(mus, variances)
    std = np.sqrt(variance)
    return mean, std


def main(**args):
    results = {'args': args}
    P = args['num_dimensions']

    for N in [400]:
        results[N] = {}

        X, Y, expected_thetas, expected_pairwise, expected_quad_dims = \
            get_data(N=N, P=P, S=args['active_dimensions'], seed=args['seed'])
            #get_data(N=args['num_data, P=args['num_dimensions, S=args['active_dimensions)

        # setup hyperparameters
        hypers = {'expected_sparsity': args['active_dimensions'],
                  'alpha1': 2.0, 'beta1': 1.0, 'sigma': 2.0,
                  'alpha2': 2.0, 'beta2': 1.0, 'c': 1.0}

        rng_key = random.PRNGKey(args['seed'])

        if args['num_warmup'] == 0: # do SVI
            adam = optim.Adam(0.01)
            #guide = AutoDiagonalNormal(model)
            svi = SVI(model, guide, adam, ELBO())
            #svi = SVI(model, guide, adam, AutoContinuousELBO())
            svi_state = svi.init(rng_key, X, Y, hypers)

            def body_fn(i, val):
                svi_state, loss = svi.update(val, X, Y, hypers)
                return svi_state

            svi_state = fori_loop(0, args['num_samples'], body_fn, svi_state)
            params = svi.get_params(svi_state)
            print(params)
            import sys; sys.exit()
            #print("mean autoloc", onp.mean(params['auto_loc']))
            #print("mean autoscale", onp.mean(params['auto_scale']))
            rng_key_post = random.PRNGKey(args['seed'] + 1373483)
            samples = guide.sample_posterior(rng_key_post, params, sample_shape=(1,))
            print("posterior_samples", samples)

        else: # do HMC
            samples = run_inference(model, args, rng_key, X, Y, hypers)

        # compute the mean and square root variance of each coefficient theta_i
        means, stds = vmap(lambda dim: analyze_dimension(samples, X, Y, dim, hypers))(np.arange(P))

        results[N]['expected_thetas'] = onp.array(expected_thetas).tolist()
        results[N]['coeff_means'] = onp.array(means).tolist()
        results[N]['coeff_stds'] = onp.array(stds).tolist()

        print("Coefficients theta_1 to theta_%d used to generate the data:" % args['active_dimensions'], expected_thetas)
        active_dims = []
        expected_active_dims = onp.arange(args['active_dimensions']).tolist()

        for dim, (mean, std) in enumerate(zip(means, stds)):
            # we mark the dimension as inactive if the interval [mean - 2 * std, mean + 2 * std] contains zero
            lower, upper = mean - 2.0 * std, mean + 2.0 * std
            inactive = "inactive" if lower < 0.0 and upper > 0.0 else "active"
            if inactive == "active":
                active_dims.append(dim)
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
        print("The magnitude of the quadratic coefficients theta_{1,2} and theta_{3,4} used to generate the data:",
              expected_pairwise)

        # Compute the mean and square root variance of coefficients theta_ij for i,j active dimensions.
        # Note that the resulting numbers are only meaningful for i != j.
        active_quad_dims = []
        if len(active_dims) > 0:
            dim_pairs = np.array(list(itertools.product(active_dims, active_dims)))
            means, stds = vmap(lambda dim_pair: analyze_pair_of_dimensions(samples, X, Y,
                                                                           dim_pair[0], dim_pair[1], hypers))(dim_pairs)
            for dim_pair, mean, std in zip(dim_pairs, means, stds):
                dim1, dim2 = dim_pair
                if dim1 >= dim2:
                    continue
                lower, upper = mean - 2.0 * std, mean + 2.0 * std
                if not (lower < 0.0 and upper > 0.0):
                    format_str = "Identified pairwise interaction between dimensions %d and %d: %.2e +- %.2e"
                    print(format_str % (dim1 + 1, dim2 + 1, mean, std))
                    active_quad_dims.append((dim1, dim2))
                else:
                    format_str = "No pairwise interaction between dimensions %d and %d: %.2e +- %.2e"
                    print(format_str % (dim1 + 1, dim2 + 1, mean, std))

        correct_quads = len(set(active_quad_dims) & set(expected_quad_dims))
        false_quads = len(set(active_quad_dims) - set(expected_quad_dims))
        missed_quads = len(set(expected_quad_dims) - set(active_quad_dims))

        results[N]['correct_quads'] = correct_quads
        results[N]['false_quads'] = false_quads
        results[N]['missed_quads'] = missed_quads

        print("correct_quads: ", correct_quads, "  false_quads: ", false_quads,
              "  missed_quads: ", missed_quads)

    print("RESULTS\n", results)
    log_file = 'slog.P_{}.S_{}.seed_{}.ns_{}_{}.mtd_{}'
    log_file = log_file.format(P, args['active_dimensions'], args['seed'],
                               args['num_warmup'], args['num_samples'], args['mtd'])

    with open(args['log_dir'] + log_file + '.pkl', 'wb') as f:
        pickle.dump(results, f, protocol=2)


if __name__ == "__main__":
    assert numpyro.__version__.startswith('0.2.4')
    parser = argparse.ArgumentParser(description="Gaussian Process example")
    parser.add_argument("-n", "--num-samples", nargs="?", default=300, type=int)
    parser.add_argument("--num-warmup", nargs='?', default=0, type=int)
    parser.add_argument("--num-chains", nargs='?', default=1, type=int)
    parser.add_argument("--mtd", nargs='?', default=6, type=int)
    parser.add_argument("--num-data", nargs='?', default=0, type=int)
    parser.add_argument("--num-dimensions", nargs='?', default=250, type=int)
    parser.add_argument("--seed", nargs='?', default=0, type=int)
    parser.add_argument("--active-dimensions", nargs='?', default=10, type=int)
    parser.add_argument("--thinning", nargs='?', default=10, type=int)
    parser.add_argument("--device", default='cpu', type=str, help='use "cpu" or "gpu".')
    parser.add_argument("--log-dir", default='./', type=str)
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)
    enable_x64()

    main(**vars(args))
