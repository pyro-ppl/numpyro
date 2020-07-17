# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import jax
from jax import jit
import jax.numpy as np

from jax.scipy.linalg import cho_factor, solve_triangular, cho_solve

from chunk_vmap import chunk_vmap
from utils import kdot
from cg import kernel_mvm_diag, lowrank_presolve, pcg_batch_b
from pairwise import kernel


# helper for computing the posterior marginal N(theta_i) or N(theta_ij)
def compute_coefficient_mean_variance(X, Y, probe, vec, eta1, eta2, c, kappa, omega):
    kprobe, kX = kappa * probe, kappa * X

    k_xx = kernel(kX, kX, eta1, eta2, c)
    k_probeX = kernel(kprobe, kX, eta1, eta2, c)
    k_prbprb = kernel(kprobe, kprobe, eta1, eta2, c)

    L = cho_factor(k_xx + np.eye(X.shape[0]) * (1.0 / omega), lower=True)[0]

    mu = 0.5 * cho_solve((L, True), Y / omega)
    mu = np.dot(vec, np.dot(k_probeX, mu))

    Linv_kXprobe = solve_triangular(L, np.transpose(k_probeX), lower=True)
    var = k_prbprb - np.matmul(np.transpose(Linv_kXprobe), Linv_kXprobe)
    var = np.dot(vec, np.matmul(var, vec))

    return mu, var

def process_singleton_svi(X, Y, samples, c, omega_chunk_size=8, probe_chunk_size=8, method="direct",
                          rank1=64, rank2=16, cg_tol=1.0e-3, max_iters=200):
    kappa = samples['kappa'][-1]
    eta1, eta2 = samples['eta1'][-1], samples['eta2'][-1]
    P = X.shape[1]

    kX = kappa * X

    if method == "direct":
        k_xx = kernel(kX, kX, eta1, eta2, c)
        fun = lambda omega: process_omega_singleton(k_xx, kX, kappa, omega, Y, P, eta1, eta2, c, probe_chunk_size)
    elif method == "ppcg":
        @jit
        def fun(omega):
            _fun = lambda dim: process_singleton_pcg(dim, P, kappa, kX, omega, Y, eta1, eta2, c, rank1, rank2,
                                                     cg_tol=cg_tol, max_iters=max_iters)
            return chunk_vmap(_fun, np.arange(P), chunk_size=probe_chunk_size)

    mu, var = chunk_vmap(fun, samples['omega'], chunk_size=omega_chunk_size)
    mu, var = gaussian_mixture_stats(mu, var)

    return mu, np.sqrt(var)

def process_quad_svi(X, Y, samples, dim_pairs, c, omega_chunk_size=8, probe_chunk_size=8, method="direct",
                     rank1=64, rank2=16, cg_tol=1.0e-3, max_iters=200):
    kappa = samples['kappa'][-1]
    eta1, eta2 = samples['eta1'][-1], samples['eta2'][-1]
    P = X.shape[1]

    kX = kappa * X

    if method == "direct":
        k_xx = kernel(kX, kX, eta1, eta2, c)
        fun = lambda omega: process_omega_quad(dim_pairs, k_xx, kX, kappa, omega, Y, P, eta1, eta2, c, probe_chunk_size)
    elif method == "ppcg":
        @jit
        def fun(omega):
            _fun = lambda dim_pair: process_quad_pcg(dim_pair[0], dim_pair[1], P, kappa, kX, omega,
                                                     Y, eta1, eta2, c, rank1, rank2, cg_tol=cg_tol, max_iters=max_iters)
            return chunk_vmap(_fun, dim_pairs, chunk_size=probe_chunk_size)

    mu, var = chunk_vmap(fun, samples['omega'], chunk_size=omega_chunk_size)
    mu, var = gaussian_mixture_stats(mu, var)

    return mu, np.sqrt(var)

def process_omega_singleton(k_xx, kX, kappa, omega, Y, P, eta1, eta2, c, probe_chunk_size):
    L = cho_factor(k_xx + np.eye(k_xx.shape[0]) * (1.0 / omega), lower=True)[0]
    LL_Y = 0.5 * cho_solve((L, True), Y / omega)

    fun = lambda dim: process_singleton(dim, P, kappa, kX, L, LL_Y, eta1, eta2, c)
    mu, var = chunk_vmap(fun, np.arange(P), chunk_size=probe_chunk_size)
    return mu, var

def process_omega_quad(dim_pairs, k_xx, kX, kappa, omega, Y, P, eta1, eta2, c, probe_chunk_size):
    L = cho_factor(k_xx + np.eye(k_xx.shape[0]) * (1.0 / omega), lower=True)[0]
    LL_Y = 0.5 * cho_solve((L, True), Y / omega)

    fun = lambda dim_pair: process_quad(dim_pair[0], dim_pair[1], P, kappa, kX, L, LL_Y, eta1, eta2, c)
    mu, var = chunk_vmap(fun, dim_pairs, chunk_size=probe_chunk_size)
    return mu, var

def process_singleton(dim, P, kappa, kX, L, LL_Y, eta1, eta2, c):
    probe = np.zeros((2, P))
    probe = jax.ops.index_update(probe, jax.ops.index[:, dim], np.array([1.0, -1.0]))
    vec = np.array([0.50, -0.50])
    mu, var = process_probe(kappa * probe, kX, L, LL_Y, vec, eta1, eta2, c)
    return mu, var

def process_singleton_pcg(dim, P, kappa, kX, omega, Y, eta1, eta2, c, rank1, rank2,
                          cg_tol=1.0e-3, max_iters=200):
    probe = np.zeros((2, P))
    probe = jax.ops.index_update(probe, jax.ops.index[:, dim], np.array([1.0, -1.0]))
    vec = np.array([0.50, -0.50])
    mu, var = process_probe_pcg(kappa * probe, kX, kappa, omega, Y, vec, eta1, eta2, c, rank1, rank2,
                                cg_tol=cg_tol, max_iters=max_iters)
    return mu, var

def process_quad(dim1, dim2, P, kappa, kX, L, LL_Y, eta1, eta2, c):
    probe = np.zeros((4, P))
    probe = jax.ops.index_update(probe, jax.ops.index[:, dim1], np.array([1.0, 1.0, -1.0, -1.0]))
    probe = jax.ops.index_update(probe, jax.ops.index[:, dim2], np.array([1.0, -1.0, 1.0, -1.0]))
    vec = np.array([0.25, -0.25, -0.25, 0.25])
    mu, var = process_probe(kappa * probe, kX, L, LL_Y, vec, eta1, eta2, c)
    return mu, var

def process_quad_pcg(dim1, dim2, P, kappa, kX, omega, Y, eta1, eta2, c, rank1, rank2,
                     cg_tol=1.0e-3, max_iters=200):
    probe = np.zeros((4, P))
    probe = jax.ops.index_update(probe, jax.ops.index[:, dim1], np.array([1.0, 1.0, -1.0, -1.0]))
    probe = jax.ops.index_update(probe, jax.ops.index[:, dim2], np.array([1.0, -1.0, 1.0, -1.0]))
    vec = np.array([0.25, -0.25, -0.25, 0.25])
    mu, var = process_probe_pcg(kappa * probe, kX, kappa, omega, Y, vec, eta1, eta2, c, rank1, rank2,
                                cg_tol=cg_tol, max_iters=max_iters)
    return mu, var

def process_probe_pcg(kprobe, kX, kappa, omega, Y, vec, eta1, eta2, c, rank1, rank2,
                      dilation=4, max_iters=200, cg_tol=1.0e-3):
    k_probeX = kernel(kprobe, kX, eta1, eta2, c)
    k_prbprb = kernel(kprobe, kprobe, eta1, eta2, c)
    diag = 1.0 / omega

    mvm = lambda b: kernel_mvm_diag(b, kX, eta1, eta2, c, diag, dilation=dilation)
    presolve = lowrank_presolve(kX, diag, eta1, eta2, c, kappa, rank1, rank2)

    Y_omega = 0.5 * Y / omega
    Y_kprb = np.concatenate([Y_omega[None, :], k_probeX])
    Kinv_Y_kprb = pcg_batch_b(Y_kprb, mvm, presolve=presolve, cg_tol=cg_tol, max_iters=max_iters)[0]

    mu = np.dot(vec, np.dot(k_probeX, Kinv_Y_kprb[0]))

    var = k_prbprb - np.matmul(Kinv_Y_kprb[1:], np.transpose(k_probeX))
    var = np.dot(vec, np.matmul(var, vec))
    return mu, var

def process_probe(kprobe, kX, L, LL_Y, vec, eta1, eta2, c):
    k_probeX = kernel(kprobe, kX, eta1, eta2, c)
    k_prbprb = kernel(kprobe, kprobe, eta1, eta2, c)

    mu = np.dot(vec, np.dot(k_probeX, LL_Y))

    Linv_kXprobe = solve_triangular(L, np.transpose(k_probeX), lower=True)
    var = k_prbprb - np.matmul(np.transpose(Linv_kXprobe), Linv_kXprobe)
    var = np.dot(vec, np.matmul(var, vec))
    return mu, var

# compute the posterior marginal N(theta_i)
@jit
def compute_singleton_mean_variance(X, Y, dimension, eta1, eta2, c, kappa, omega):
    probe = np.zeros((2, X.shape[1]))
    probe = jax.ops.index_update(probe, jax.ops.index[:, dimension], np.array([1.0, -1.0]))
    vec = np.array([0.50, -0.50])
    return compute_coefficient_mean_variance(X, Y, probe, vec, eta1, eta2, c, kappa, omega)


# compute the posterior marginal N(theta_ij)
@jit
def compute_pairwise_mean_variance(X, Y, dim1, dim2, eta1, eta2, c, kappa, omega):
    probe = np.zeros((4, X.shape[1]))
    probe = jax.ops.index_update(probe, jax.ops.index[:, dim1], np.array([1.0, 1.0, -1.0, -1.0]))
    probe = jax.ops.index_update(probe, jax.ops.index[:, dim2], np.array([1.0, -1.0, 1.0, -1.0]))
    vec = np.array([0.25, -0.25, -0.25, 0.25])
    return compute_coefficient_mean_variance(X, Y, probe, vec, eta1, eta2, c, kappa, omega)

# Get the mean and variance of a gaussian mixture
def gaussian_mixture_stats(mus, variances):
    mean_mu = np.mean(mus, axis=0)
    mean_var = np.mean(variances, axis=0) + np.mean(np.square(mus), axis=0) - np.square(mean_mu)
    return mean_mu, mean_var

# Helper function for analyzing the posterior statistics for coefficient theta_i
@jit
def analyze_dimension(samples, X, Y, dimension, hypers, chunk_size=1):
    vmap_args = (samples['eta1'], samples['eta2'], samples['kappa'], samples['omega'])
    fun = lambda eta1, eta2, kappa, omega: compute_singleton_mean_variance(X, Y, dimension, eta1, eta2,
                                                                           hypers['c'], kappa, omega)
    mus, variances = chunk_vmap(fun, vmap_args, chunk_size=chunk_size)
    mean, variance = gaussian_mixture_stats(mus, variances)
    std = np.sqrt(variance)
    return mean, std

# Helper function for analyzing the posterior statistics for coefficient theta_ij
@jit
def analyze_pair_of_dimensions(samples, X, Y, dim1, dim2, hypers, chunk_size=1):
    vmap_args = (samples['eta1'], samples['eta2'], samples['kappa'], samples['omega'])
    fun = lambda eta1, eta2, kappa, omega: compute_pairwise_mean_variance(X, Y, dim1, dim2, eta1, eta2,
                                                                          hypers['c'], kappa, omega)
    mus, variances = chunk_vmap(fun, vmap_args, chunk_size=chunk_size)
    mean, variance = gaussian_mixture_stats(mus, variances)
    std = np.sqrt(variance)
    return mean, std

