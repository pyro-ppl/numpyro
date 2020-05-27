# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import itertools
import os
import time

import numpy as onp

import jax
from jax import vmap, jit
import jax.numpy as np
import jax.random as random

import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.infer import MCMC, NUTS, SVI, ELBO
from numpyro.contrib.autoguide import AutoDiagonalNormal, AutoContinuousELBO
from numpyro.distributions import constraints, Normal
from numpyro.distributions.transforms import AffineTransform, SigmoidTransform
from numpyro.infer.util import Predictive
from numpyro.handlers import seed, trace
from numpyro.diagnostics import print_summary, summary
from jax.scipy.linalg import cho_factor, solve_triangular, cho_solve
from numpyro.util import enable_x64
from numpyro.util import fori_loop

from chunk_vmap import chunk_vmap, safe_chunk_vmap

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


def sample_posterior(X, Y, probe, eta1, eta2, c, kappa, omega, rng):
    kprobe, kX = kappa * probe, kappa * X

    k_xx = kernel(kX, kX, eta1, eta2, c)
    k_probeX = kernel(kprobe, kX, eta1, eta2, c)
    k_prbprb = kernel(kprobe, kprobe, eta1, eta2, c)

    L = cho_factor(k_xx + np.eye(X.shape[0]) * (1.0 / omega), lower=True)[0]
    mu = 0.5 * cho_solve((L, True), Y / omega)
    mu = dot(k_probeX, mu)

    Linv_kXprobe = solve_triangular(L, np.transpose(k_probeX), lower=True)
    var = k_prbprb - np.matmul(np.transpose(Linv_kXprobe), Linv_kXprobe)

    epsilon = Normal(np.zeros(mu.shape), 1.0).sample(rng)
    L = cho_factor(var, lower=True)[0]
    sample = mu + np.matmul(L, epsilon)

    return sample



# Helper function for doing HMC inference
def run_hmc(model, args, rng_key, X, Y, hypers):
    start = time.time()
    kernel = NUTS(model, max_tree_depth=args['mtd'])
    mcmc = MCMC(kernel, args['num_warmup'], args['num_samples'], num_chains=1,
                progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True)
    mcmc.run(rng_key, X, Y, hypers)
    #mcmc.print_summary()
    elapsed_time = time.time() - start

    samples = mcmc.get_samples()

    return samples, elapsed_time


# Create artificial regression dataset where only S out of P feature
# dimensions contain signal and where there are two pairwise interactions
def get_data(N=20, S=2, P=10, seed=0):
    assert S < P and P > 1 and S > 0
    onp.random.seed(seed)

    # generate S coefficients with non-negligible magnitude
    W = 0.5 + 1.0 * onp.random.rand(S)
    #W = 1.0 + 1.5 * onp.random.rand(S)
    flip = 2 * onp.random.binomial(1, 0.5, W.shape) - 1
    W *= flip

    # generate covariates with non-negligible magnitude
    X = onp.random.rand(N, P) + 0.5
    flip = 2 * onp.random.binomial(1, 0.5, X.shape) - 1
    X *= flip

    # generate data using the S coefficients and four pairwise interactions
    #pairwise_coefficient1 = 3.0
    #pairwise_coefficient2 = 2.0
    pairwise_coefficient1 = 2.0
    pairwise_coefficient2 = 1.0
    expected_quad_dims = [(0, 1), (2, 3), (4, 5), (6, 7)]
    Y = onp.sum(X[:, 0:S] * W, axis=-1) + \
        pairwise_coefficient1 * (X[:, 0] * X[:, 1] - X[:, 2] * X[:, 3]) - 2.0
        #pairwise_coefficient2 * (X[:, 4] * X[:, 5] - X[:, 6] * X[:, 7]) \
    Y = 2 * onp.random.binomial(1, sigmoid(Y)) - 1
    print("number of 1s: {}  number of -1s: {}".format(np.sum(Y == 1.0), np.sum(Y == -1.0)))

    assert X.shape == (N, P)
    assert Y.shape == (N,)

    return X, Y, W, pairwise_coefficient1, expected_quad_dims


def main(**args):
    results = {'args': args}
    P = args['num_dimensions']
    N = args['num_data']
    print(args)

    # setup hyperparameters
    hypers = {'expected_sparsity': args['active_dimensions'],
              'alpha1': 2.0, 'beta1': 1.0, 'sigma': 2.0,
              'alpha2': 2.0, 'beta2': 1.0, 'c': 1.0}

    _X, _Y, expected_thetas, expected_pairwise, expected_quad_dims = \
        get_data(N=N, P=P, S=args['active_dimensions'], seed=args['seed'])

    start = N // 2
    num_arms = 4
    num_rounds = (N // 2) // num_arms

    X, Y = _X[start:], _Y[start:]
    total_reward = 0

    for r in range(num_rounds):
        rng_key = random.PRNGKey(args['seed'] + r)
        samples, _ = run_hmc(model, args, rng_key, X, Y, hypers)

        X_test = _X[start + r * num_arms:start + (r+1) * num_arms]
        Y_test = _Y[start + r * num_arms:start + (r+1) * num_arms]

        eta1, eta2 = samples['eta1'][-1],samples['eta2'][-1]
        kappa, omega = samples['kappa'][-1], samples['omega'][-1]
        rng_key2 = random.PRNGKey(args['seed'] + 1738 * r)
        sample = sample_posterior(X, Y, X_test, eta1, eta2, hypers['c'], kappa, omega, rng_key2)
        argmax = np.argmax(sample)
        total_reward += Y_test[argmax]
        print("Choosing arm #{}... Got reward of {}".format(argmax+1, Y_test[argmax]))
        print("ts sample", sample, "Y_test", Y_test)

        X = np.concatenate([X, X_test[argmax:argmax+1]])
        Y = np.concatenate([Y, Y_test[argmax:argmax+1]])
        print("New X, Y", X.shape, Y.shape)

    print("total reward", total_reward)

        #log_file = 'slog.{}.P_{}.S_{}.seed_{}.ns_{}_{}.mtd_{}'
        #log_file = log_file.format(args['inference'], P, args['active_dimensions'], args['seed'],
        #                           args['num_warmup'], args['num_samples'], args['mtd'])

        #with open(args['log_dir'] + log_file + '.pkl', 'wb') as f:
        #    pickle.dump(results, f, protocol=2)


if __name__ == "__main__":
    assert numpyro.__version__.startswith('0.2.4')
    parser = argparse.ArgumentParser(description="contextual bandits")
    parser.add_argument("-n", "--num-samples", nargs="?", default=100, type=int)
    parser.add_argument("--num-warmup", nargs='?', default=50, type=int)
    parser.add_argument("--mtd", nargs='?', default=5, type=int)
    parser.add_argument("--num-data", nargs='?', default=128, type=int)
    parser.add_argument("--num-dimensions", nargs='?', default=12, type=int)
    parser.add_argument("--seed", nargs='?', default=0, type=int)
    parser.add_argument("--active-dimensions", nargs='?', default=6, type=int)
    parser.add_argument("--device", default='cpu', type=str, help='use "cpu" or "gpu".')
    parser.add_argument("--log-dir", default='./large/', type=str)
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    #numpyro.set_host_device_count(args.num_chains)
    enable_x64()

    main(**vars(args))
