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
from numpyro.infer import MCMC, NUTS

from jax.scipy.linalg import cho_factor, solve_triangular, cho_solve
from numpyro.util import enable_x64

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
    S, P, N = hypers['expected_sparsity'], X.shape[1], X.shape[0]

    #sigma = numpyro.sample("sigma", dist.HalfNormal(hypers['alpha3']))
    sigma = 4.0
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


def compute_singleton_mean_variance(X, Y, dimension, eta1, eta2, c, kappa, omega):
    probe = np.zeros((2, X.shape[1]))
    probe = jax.ops.index_update(probe, jax.ops.index[:, dimension], np.array([1.0, -1.0]))
    vec = np.array([0.50, -0.50])
    return compute_coefficient_mean_variance(X, Y, probe, vec, eta1, eta2, c, kappa, omega)


def compute_pairwise_mean_variance(X, Y, dim1, dim2, eta1, eta2, c, kappa, omega):
    probe = np.zeros((4, X.shape[1]))
    probe = jax.ops.index_update(probe, jax.ops.index[:, dim1], np.array([1.0, 1.0, -1.0, -1.0]))
    probe = jax.ops.index_update(probe, jax.ops.index[:, dim2], np.array([1.0, -1.0, 1.0, -1.0]))
    vec = np.array([0.25, -0.25, -0.25, 0.25])
    return compute_coefficient_mean_variance(X, Y, probe, vec, eta1, eta2, c, kappa, omega)


# Helper function for doing HMC inference
def run_inference(model, args, rng_key, X, Y, hypers):
    start = time.time()
    kernel = NUTS(model, max_tree_depth=args.mtd)
    mcmc = MCMC(kernel, args.num_warmup, args.num_samples, num_chains=args.num_chains,
                progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True)
    mcmc.run(rng_key, X, Y, hypers)
    mcmc.print_summary()
    print('\nMCMC elapsed time:', time.time() - start)
    return mcmc.get_samples()


# Get the mean and variance of a gaussian mixture
def gaussian_mixture_stats(mus, variances):
    mean_mu = np.mean(mus)
    mean_var = np.mean(variances) + np.mean(np.square(mus)) - np.square(mean_mu)
    return mean_mu, mean_var


# Create artificial regression dataset where only S out of P feature
# dimensions contain signal and where there is a single pairwise interaction
# between the first and second dimensions.
def get_data(N=20, S=2, P=10):
    assert S < P and P > 1 and S > 0
    onp.random.seed(0)

    X = onp.random.rand(N, P) + 0.5
    flip = 2 * onp.random.binomial(1, 0.5, X.shape) - 1
    X *= flip

    # generate S coefficients with non-negligible magnitude
    W = 2.0 + 3.0 * onp.random.rand(S)
    flip = 2 * onp.random.binomial(1, 0.5, W.shape) - 1
    W *= flip

    # generate data using the S coefficients and a single pairwise interaction
    pairwise_coefficient = 3.0
    Y = onp.sum(X[:, 0:S] * W, axis=-1) + pairwise_coefficient * (X[:, 0] * X[:, 1] - X[:, 2] * X[:, 3])
    Y = 2 * onp.random.binomial(1, sigmoid(Y)) - 1
    print("number of 1s: {}  number of -1s: {}".format(np.sum(Y == 1.0), np.sum(Y == -1.0)))

    assert X.shape == (N, P)
    assert Y.shape == (N,)

    return X, Y, W, pairwise_coefficient


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


def main(args):

    for P in [10]:

        X, Y, expected_thetas, expected_pairwise = get_data(N=args.num_data, P=args.num_dimensions,
                                                            S=args.active_dimensions)

        # setup hyperparameters
        hypers = {'expected_sparsity': max(1.0, args.num_dimensions / 10.0),
                  'alpha1': 3.0, 'beta1': 1.0,
                  'alpha2': 3.0, 'beta2': 1.0,
                  'alpha3': 1.0, 'c': 1.0}

        # do inference
        rng_key = random.PRNGKey(0)
        samples = run_inference(model, args, rng_key, X, Y, hypers)

        # compute the mean and square root variance of each coefficient theta_i
        means, stds = vmap(lambda dim: analyze_dimension(samples, X, Y, dim, hypers))(np.arange(args.num_dimensions))

        print("Coefficients theta_1 to theta_%d used to generate the data:" % args.active_dimensions, expected_thetas)
        active_dimensions = []

        for dim, (mean, std) in enumerate(zip(means, stds)):
            # we mark the dimension as inactive if the interval [mean - 2 * std, mean + 2 * std] contains zero
            lower, upper = mean - 2.0 * std, mean + 2.0 * std
            inactive = "inactive" if lower < 0.0 and upper > 0.0 else "active"
            if inactive == "active":
                active_dimensions.append(dim)
            print("[dimension %02d/%02d]  %s:\t%.2e +- %.2e" % (dim + 1, args.num_dimensions, inactive, mean, std))

        print("Identified a total of %d active dimensions; expected %d." % (len(active_dimensions),
                                                                            args.active_dimensions))
        print("The single quadratic coefficient theta_{1,2} used to generate the data:", expected_pairwise)

        # Compute the mean and square root variance of coefficients theta_ij for i,j active dimensions.
        # Note that the resulting numbers are only meaningful for i != j.
        if len(active_dimensions) > 0:
            dim_pairs = np.array(list(itertools.product(active_dimensions, active_dimensions)))
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
                else:
                    format_str = "No pairwise interaction between dimensions %d and %d: %.2e +- %.2e"
                    print(format_str % (dim1 + 1, dim2 + 1, mean, std))


if __name__ == "__main__":
    assert numpyro.__version__.startswith('0.2.4')
    parser = argparse.ArgumentParser(description="Gaussian Process example")
    parser.add_argument("-n", "--num-samples", nargs="?", default=300, type=int)
    parser.add_argument("--num-warmup", nargs='?', default=100, type=int)
    parser.add_argument("--num-chains", nargs='?', default=1, type=int)
    parser.add_argument("--mtd", nargs='?', default=6, type=int)
    parser.add_argument("--num-data", nargs='?', default=180, type=int)
    parser.add_argument("--num-dimensions", nargs='?', default=40, type=int)
    parser.add_argument("--active-dimensions", nargs='?', default=4, type=int)
    parser.add_argument("--device", default='cpu', type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)
    enable_x64()

    main(args)
