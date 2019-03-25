import random
from collections import namedtuple

import numpy as onp

import jax.numpy as np
from jax import grad, lax, value_and_grad
from jax.tree_util import tree_multimap


def velocity_verlet(potential_fn, kinetic_fn):
    r"""
    Second order symplectic integrator that uses the velocity verlet algorithm
    for position `z` and momentum `r`.

    param: 
    """
    def init_fn(z, r):
        potential_energy, z_grad = value_and_grad(potential_fn)(z)
        return z, r, potential_energy, z_grad

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
        return z, r, potential_energy, z_grad

    return init_fn, update_fn


def dual_averaging(t0=10, kappa=0.75, gamma=0.05):
    """
    Dual Averaging is a scheme to solve convex optimization problems. It
    belongs to a class of subgradient methods which uses subgradients (which
    lie in a dual space) to update states (in primal space) of a model. Under
    some conditions, the averages of generated parameters during the scheme are
    guaranteed to converge to an optimal value. However, a counter-intuitive
    aspect of traditional subgradient methods is "new subgradients enter the
    model with decreasing weights" (see reference [1]). Dual Averaging scheme
    resolves that issue by updating parameters using weights equally for
    subgradients, hence we have the name "dual averaging".

    This class implements a dual averaging scheme which is adapted for Markov
    chain Monte Carlo (MCMC) algorithms. To be more precise, we will replace
    subgradients by some statistics calculated at the end of MCMC trajectories.
    Following [2], we introduce some free parameters such as ``t0`` and
    ``kappa``, which is helpful and still guarantees the convergence of the
    scheme.

    :param int t0: A free parameter introduced in reference [2] that stabilizes
        the initial steps of the scheme. Defaults to 10.
    :param float kappa: A free parameter introduced in reference [2] that
        controls the weights of steps of the scheme. For a small ``kappa``, the
        scheme will quickly forget states from early steps. This should be a
        number in :math:`(0.5, 1]`. Defaults to 0.75.
    :param float gamma: A free parameter introduced in reference [1] which
        controls the speed of the convergence of the scheme. Defaults to 0.05.
    :returns: a (init_fn, update_fn) pair

    **References**

    [1] `Primal-dual subgradient methods for convex problems`,
    Yurii Nesterov

    [2] `The No-U-turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo`,
    Matthew D. Hoffman, Andrew Gelman
    """
    def init_fn(prox_center=0.):
        """
        :param float prox_center: A parameter introduced in reference [1] which
            pulls the primal sequence towards it. Defaults to 0.
        """
        x_t = 0.
        x_avg = 0.  # average of primal sequence
        g_avg = 0.  # average of dual sequence
        t = 0
        return x_t, x_avg, g_avg, t, prox_center

    def update_fn(g, state):
        """
        :param float g: The current subgradient or statistics calculated during
            an MCMC trajectory.
        """
        x_t, x_avg, g_avg, t, prox_center = state
        t = t + 1
        # g_avg = (g_1 + ... + g_t) / t
        g_avg = (1 - 1 / (t + t0)) * g_avg + g / (t + t0)
        # According to formula (3.4) of [1], we have
        #     x_t = argmin{ g_avg . x + loc_t . |x - x0|^2 },
        # hence x_t = x0 - g_avg / (2 * loc_t),
        # where loc_t := beta_t / t, beta_t := (gamma/2) * sqrt(t).
        x_t = prox_center - (t ** 0.5) / gamma * g_avg
        # weight for the new x_t
        weight_t = t ** (-kappa)
        x_avg = (1 - weight_t) * x_avg + weight_t * x_t
        return x_t, x_avg, g_avg, t, prox_center

    return init_fn, update_fn


def welford_covariance(diagonal=True):
    """
    Implements Welford's online method for estimating (co)variance. Useful for
    adapting diagonal and dense mass structures for HMC. It is required that
    each sample is a 1-dimensional array.

    :param bool diagonal: If True, we estimate the variance of samples.
        Otherwise, we estimate the covariance of the samples. Defaults to True.
    :returns: a (init_fn, update_fn, finalize) triple

    **References**

    [1] `The Art of Computer Programming`,
    Donald E. Knuth
    """
    def init_fn(size):
        """
        :param int size: size of each sample
        """
        mean = np.zeros(size)
        if diagonal:
            m2 = np.zeros(size)
        else:
            m2 = np.zeros((size, size))
        n = 0
        return mean, m2, n

    def update_fn(x, state):
        """
        :param x: value of the current sample
        """
        mean, m2, n = state
        n = n + 1
        delta_pre = sample - mean
        mean = mean + delta_pre / n
        delta_post = sample - mean
        if diagonal:
            m2 = m2 + delta_pre * delta_post
        else:
            m2 = m2 + np.outer(delta_post, delta_pre)
        return mean, m2, n

    def finalize(state, regularize=False):
        """
        param bool regularize: 
        """
        mean, m2, n = state
        # XXX it is not necessary to check for the case n=1
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

    return init_fn, update_fn, finalize


def find_reasonable_step_size(potential_fn, kinetic_fn, r_generator, z, init_step_size):
    """
    We are going to find a step_size which make accept_prob (Metropolis correction)
    near the target_accept_prob. If accept_prob:=exp(-delta_energy) is small,
    then we have to decrease step_size; otherwise, increase step_size.
    """

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
        return step_size, direction, direction_new

    step_size, _, _ = lax.while_loop(lambda sdd: (sdd[1] == 0) | (sdd[1] == sdd[2]),
                                     _body_fn, (init_step_size, 0, 0))
    return step_size


adapt_window = namedtuple("adapt_window", ["start", "end"])


def build_adaptation_schedule(num_steps):
    adaptation_schedule = []
    # from Stan, for small num_steps
    if num_steps < 20:
        adaptation_schedule.append(adapt_window(0, num_steps - 1))
        return adaptation_schedule

    # We separate num_steps into windows:
    #   start_buffer + window 1 + window 2 + window 3 + ... + end_buffer
    # where the length of each window will be doubled for the next window.
    # We won't adapt mass matrix during start and end buffers; and mass
    # matrix will be updated at the end of each window. This is helpful
    # for dealing with the intense computation of sampling momentum from the
    # inverse of mass matrix.
    start_buffer_size = 75  # from Stan
    end_buffer_size = 50  # from Stan
    init_window_size = 25  # from Stan
    if (start_buffer_size + end_buffer_size + init_window_size) > num_steps:
        start_buffer_size = int(0.15 * num_steps)
        end_buffer_size = int(0.1 * num_steps)
        init_window_size = num_steps - start_buffer_size - end_buffer_size

    adaptation_schedule.append(adapt_window(start=0, end=start_buffer_size - 1))
    end_window_start = num_steps - end_buffer_size

    next_window_size = init_window_size
    next_window_start = start_buffer_size
    while next_window_start < end_window_start:
        cur_window_start, cur_window_size = next_window_start, next_window_size
        # Ensure that slow adaptation windows are monotonically increasing
        if 3 * cur_window_size <= end_window_start - cur_window_start:
            next_window_size = 2 * cur_window_size
        else:
            cur_window_size = end_window_start - cur_window_start
        next_window_start = cur_window_start + cur_window_size
        adaptation_schedule.append(adapt_window(cur_window_start, next_window_start - 1))
    adaptation_schedule.append(adapt_window(end_window_start, num_steps - 1))
    return adaptation_schedule


def warmup_adapter(num_steps, find_reasonable_step_size=None,
                   adapt_step_size=True, adapt_mass_matrix=True,
                   diag_mass=True, target_accept_prob=0.8):
    # XXX here we use the default dual_averaging, welford_covariance,
    # build_adaptation_schedule if users need to 
    ss_init, ss_update = dual_averaging()
    mm_init, mm_update, mm_final = welford_covariance(diagonal=diag_mass)
    adaptation_schedule = np.array(build_adaptation_schedule(num_steps))
    num_windows = len(adaptation_schedule)

    def init_fn(step_size=1.0, inverse_mass_matrix=None, mass_matrix_size=None):
        if find_reasonable_step_size is not None:
            step_size = find_reasonable_step_size(step_size)
        ss_state = ss_init(np.log(10 * step_size))

        if inverse_mass_matrix is None:
            assert mass_matrix_size is not None
            if diag_mass:
                inverse_mass_matrix = np.ones(mass_matrix_size)
            else:
                inverse_mass_matrix = np.identity(mass_matrix_size)
        mm_state = mm_init(inverse_mass_matrix.shape[-1])

        window_idx = 0
        return step_size, inverse_mass_matrix, ss_state, mm_state, window_idx

    def _update_at_window_end(state):
        step_size, inverse_mass_matrix, ss_state, mm_state, window_idx = state

        if adapt_step_size:
            if find_reasonable_step_size is not None:
                step_size = find_reasonable_step_size(step_size)
            ss_state = ss_init(np.log(10 * step_size))

        if adapt_mass_matrix:
            inverse_mass_matrix = mm_final(mm_state, regularize=True)
            mm_state = mm_init(inverse_mass_matrix.shape[-1])

        return step_size, inverse_mass_matrix, ss_state, mm_state, window_idx

    def update_fn(t, accept_prob, z_flat, state):
        step_size, inverse_mass_matrix, ss_state, mm_state, window_idx = state

        # update step size state
        if adapt_step_size:
            ss_state = ss_update(target_accept_prob - accept_prob, ss_state)
            # note: at the end of warmup phase, use average of log step_size
            # TODO: should we make sure that we won't update step_size if t >= num_steps?
            log_step_size, log_step_size_avg, *_ = ss_state
            step_size = np.where(t == (num_steps - 1),
                                 np.exp(log_step_size_avg),
                                 np.exp(log_step_size))

        # update mass matrix state
        is_middle_window = (0 < window_idx) & (window_idx < (num_windows - 1))
        if adapt_mass_matrix:
            mm_state = lax.cond(is_middle_window,
                                (z_flat, mm_state), lambda args: mm_update(*args),
                                mm_state, lambda x: x)

        t_at_window_end = t == adaptation_schedule[window_idx, 1]
        window_idx = np.where(t_at_window_end, window_idx + 1, window_idx)
        state = step_size, inverse_mass_matrix, ss_state, mm_state, window_idx
        # TODO: enable lax.cond when https://github.com/google/jax/issues/514 is resolved
        # state = lax.cond(t_at_window_end & is_middle_window,
        #                  state, _end_window_fn, state, lambda x: x)
        if t_at_window_end & is_middle_window:
            state = _update_at_window_end(state)
        return state

    return init_fn, update_fn
