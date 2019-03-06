from __future__ import division

import jax.lax as lax
import jax.numpy as np
from jax import grad, value_and_grad
from jax.tree_util import tree_multimap


def dual_averaging(t0=10, kappa=0.75, gamma=0.05):
    """
    Dual Averaging is a scheme to solve convex optimization problems. It belongs
    to a class of subgradient methods which uses subgradients to update parameters
    (in primal space) of a model. Under some conditions, the averages of generated
    parameters during the scheme are guaranteed to converge to an optimal value.
    However, a counter-intuitive aspect of traditional subgradient methods is
    "new subgradients enter the model with decreasing weights" (see :math:`[1]`).
    Dual Averaging scheme solves that phenomenon by updating parameters using
    weights equally for subgradients (which lie in a dual space), hence we have
    the name "dual averaging".
    This class implements a dual averaging scheme which is adapted for Markov chain
    Monte Carlo (MCMC) algorithms. To be more precise, we will replace subgradients
    by some statistics calculated during an MCMC trajectory. In addition,
    introducing some free parameters such as ``t0`` and ``kappa`` is helpful and
    still guarantees the convergence of the scheme.

    **References**
    [1] `Primal-dual subgradient methods for convex problems`,
    Yurii Nesterov
    [2] `The No-U-turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo`,
    Matthew D. Hoffman, Andrew Gelman
    """
    def init_fn(prox_center=0.):
        x_t = 0.
        x_avg = 0.  # average of primal sequence
        g_avg = 0.  # average of dual sequence
        t = 0
        return (x_t, x_avg, g_avg, t, prox_center)

    def update_fn(g, state):
        x_t, x_avg, g_avg, t, prox_center = state
        t = t + 1
        # g_avg = (g_1 + ... + g_t) / t
        g_avg = (1 - 1 / (t + t0)) * g_avg + g / (t + t0)
        # According to formula (3.4) of [1], we have
        #     x_t = argmin{ g_avg . x + loc_t . |x - x0|^2 },
        # where loc_t := beta_t / t, beta_t := (gamma/2) * sqrt(t)
        x_t = prox_center - (t ** 0.5) / gamma * g_avg
        # weight for the new x_t
        weight_t = t ** (-kappa)
        x_avg = (1 - weight_t) * x_avg + weight_t * x_t
        return (x_t, x_avg, g_avg, t, prox_center)

    return init_fn, update_fn


def welford_covariance(diagonal=True):
    """
    Implements Welford's online method for estimating (co)variance (see :math:`[1]`).
    Useful for adapting diagonal and dense mass structures for HMC.

    **References**
    [1] `The Art of Computer Programming`,
    Donald E. Knuth
    """
    def init_fn(size):
        # TODO: replace by a better pattern
        mean = np.zeros(size)
        if diagonal:
            m2 = np.zeros(size)
        else:
            m2 = np.zeros((size, size))
        n = 0
        return (mean, m2, n)

    def update_fn(sample, state):
        mean, m2, n = state
        n = n + 1
        delta_pre = sample - mean
        mean = mean + delta_pre / n
        delta_post = sample - mean
        if diagonal:
            m2 = m2 + delta_pre * delta_post
        else:
            m2 = m2 + np.outer(delta_post, delta_pre)
        return (mean, m2, n)

    def final_fn(state, regularize=False):
        mean, m2, n = state
        # TODO: when n=1, return 0; we temporarily do not check for that case
        # because lax.cond is not yet available
        cov = m2 / (n - 1)
        if regularize:
            # Regularization from Stan
            scaled_cov = (n / (n + 5)) * cov
            shrinkage = 1e-3 * (5 / (n + 5))
            if diagonal:
                cov = scaled_cov + shrinkage
            else:
                cov = scaled_cov + shrinkage * np.identity(mean.shape[0], dtype=mean.dtype)
        return cov

    return init_fn, update_fn, final_fn


def velocity_verlet(potential_fn, kinetic_fn):
    r"""
    Second order symplectic integrator that uses the velocity verlet algorithm
    for position `z` and momentum `r`.
    """
    def init_fn(z, r):
        # TODO: init using the cache of potential_energy and z_grad?
        potential_energy, z_grad = value_and_grad(potential_fn)(z)
        return (z, r, potential_energy, z_grad)

    def update_fn(step_size, state):
        """
        Single step velocity verlet.
        """
        z, r, _, z_grad = state
        r = tree_multimap(lambda r, z_grad: r - 0.5 * step_size * z_grad, r, z_grad)  # r(n+1/2)
        r_grad = grad(kinetic_fn)(r)
        z = tree_multimap(lambda z, r_grad: z + step_size * r_grad, z, r_grad)  # z(n+1)
        potential_energy, z_grad = value_and_grad(potential_fn)(z)
        r = tree_multimap(lambda r, z_grad: r - 0.5 * step_size * z_grad, r, z_grad)  # r(n+1)
        return (z, r, potential_energy, z_grad)

    return init_fn, update_fn


def find_reasonable_step_size(potential_fn, kinetic_fn, momentum_generator, position, init_step_size):
    # We are going to find a step_size which make accept_prob (Metropolis correction)
    # near the target_accept_prob. If accept_prob:=exp(-delta_energy) is small,
    # then we have to decrease step_size; otherwise, increase step_size.
    target_accept_prob = np.log(0.8)

    _, vv_update = velocity_verlet(potential_fn, kinetic_fn)
    z = position
    potential_energy, z_grad = value_and_grad(potential_fn)(z)

    def _body_fn(state):
        step_size, _, direction = state
        # scale step_size: increase 2x or decrease 2x depends on direction;
        # direction=1 means keep increasing step_size, otherwise decreasing step_size.
        # Note that the direction is -1 if delta_energy is `NaN`, which may be the
        # case for a diverging trajectory (e.g. in the case of evaluating log prob
        # of a value simulated using a large step size for a constrained sample site).
        step_size = (2.0 ** direction) * step_size
        r = momentum_generator()  # generate r upon calling
        _, r_new, potential_energy_new, _ = vv_update(step_size,
                                                      (z, r, potential_energy, z_grad))
        energy_current = kinetic_fn(r) + potential_energy
        energy_new = kinetic_fn(r_new) + potential_energy_new
        delta_energy = energy_new - energy_current
        direction_new = np.where(target_accept_prob < -delta_energy, 1, -1)
        return (step_size, direction, direction_new)

    step_size, _, _ = lax.while_loop(lambda sdd: (sdd[1] == 0) | (sdd[1] == sdd[2]),
                                     _body_fn, (init_step_size, 0, 0))
    return step_size
