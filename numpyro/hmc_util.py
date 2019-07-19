from collections import namedtuple

import jax
from jax import grad, jit, lax, partial, random, value_and_grad, vmap
from jax.flatten_util import ravel_pytree
import jax.numpy as np
from jax.ops import index_update
from jax.scipy.special import expit
from jax.tree_util import tree_multimap

from numpyro.distributions.constraints import biject_to, real
from numpyro.distributions.util import cholesky_inverse
from numpyro.handlers import seed, trace
from numpyro.infer_util import log_density, transform_fn
from numpyro.util import cond, while_loop

AdaptWindow = namedtuple('AdaptWindow', ['start', 'end'])
AdaptState = namedtuple('AdaptState', ['step_size', 'inverse_mass_matrix', 'mass_matrix_sqrt',
                                       'ss_state', 'mm_state', 'window_idx', 'rng'])
IntegratorState = namedtuple('IntegratorState', ['z', 'r', 'potential_energy', 'z_grad'])

TreeInfo = namedtuple('TreeInfo', ['z_left', 'r_left', 'z_left_grad',
                                   'z_right', 'r_right', 'z_right_grad',
                                   'z_proposal', 'z_proposal_pe', 'z_proposal_grad',
                                   'depth', 'weight', 'r_sum', 'turning', 'diverging',
                                   'sum_accept_probs', 'num_proposals'])


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

    **References:**

    1. *Primal-dual subgradient methods for convex problems*,
       Yurii Nesterov
    2. *The No-U-turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo*,
       Matthew D. Hoffman, Andrew Gelman

    :param int t0: A free parameter introduced in reference [2] that stabilizes
        the initial steps of the scheme. Defaults to 10.
    :param float kappa: A free parameter introduced in reference [2] that
        controls the weights of steps of the scheme. For a small ``kappa``, the
        scheme will quickly forget states from early steps. This should be a
        number in :math:`(0.5, 1]`. Defaults to 0.75.
    :param float gamma: A free parameter introduced in reference [1] which
        controls the speed of the convergence of the scheme. Defaults to 0.05.
    :return: a (`init_fn`, `update_fn`) pair.
    """
    def init_fn(prox_center=0.):
        """
        :param float prox_center: A parameter introduced in reference [1] which
            pulls the primal sequence towards it. Defaults to 0.
        :return: initial state for the scheme.
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
        :param state: Current state of the scheme.
        :return: new state for the scheme.
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

    **References:**

    1. *The Art of Computer Programming*,
       Donald E. Knuth

    :param bool diagonal: If True, we estimate the variance of samples.
        Otherwise, we estimate the covariance of the samples. Defaults to True.
    :return: a (`init_fn`, `update_fn`, `final_fn`) triple.
    """
    def init_fn(size):
        """
        :param int size: size of each sample.
        :return: initial state for the scheme.
        """
        mean = np.zeros(size)
        if diagonal:
            m2 = np.zeros(size)
        else:
            m2 = np.zeros((size, size))
        n = 0
        return mean, m2, n

    def update_fn(sample, state):
        """
        :param sample: A new sample.
        :param state: Current state of the scheme.
        :return: new state for the scheme.
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

    def final_fn(state, regularize=False):
        """
        :param state: Current state of the scheme.
        :param bool regularize: Whether to adjust diagonal for numerical stability.
        :return: a pair of estimated covariance and the square root of precision.
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
                cov = scaled_cov + shrinkage * np.identity(mean.shape[0])
        if np.ndim(cov) == 2:
            cov_inv_sqrt = cholesky_inverse(cov)
        else:
            cov_inv_sqrt = np.sqrt(np.reciprocal(cov))
        return cov, cov_inv_sqrt

    return init_fn, update_fn, final_fn


def velocity_verlet(potential_fn, kinetic_fn):
    r"""
    Second order symplectic integrator that uses the velocity verlet algorithm
    for position `z` and momentum `r`.

    :param potential_fn: Python callable that computes the potential energy
        given input parameters. The input parameters to `potential_fn` can be
        any python collection type.
    :param kinetic_fn: Python callable that returns the kinetic energy given
        inverse mass matrix and momentum.
    :return: a pair of (`init_fn`, `update_fn`).
    """
    def init_fn(z, r):
        """
        :param z: Position of the particle.
        :param r: Momentum of the particle.
        :return: initial state for the integrator.
        """
        potential_energy, z_grad = value_and_grad(potential_fn)(z)
        return IntegratorState(z, r, potential_energy, z_grad)

    def update_fn(step_size, inverse_mass_matrix, state):
        """
        :param float step_size: Size of a single step.
        :param inverse_mass_matrix: Inverse of mass matrix, which is used to
            calculate kinetic energy.
        :param state: Current state of the integrator.
        :return: new state for the integrator.
        """
        z, r, _, z_grad = state
        r = tree_multimap(lambda r, z_grad: r - 0.5 * step_size * z_grad, r, z_grad)  # r(n+1/2)
        r_grad = grad(kinetic_fn, argnums=1)(inverse_mass_matrix, r)
        z = tree_multimap(lambda z, r_grad: z + step_size * r_grad, z, r_grad)  # z(n+1)
        potential_energy, z_grad = value_and_grad(potential_fn)(z)
        r = tree_multimap(lambda r, z_grad: r - 0.5 * step_size * z_grad, r, z_grad)  # r(n+1)
        return IntegratorState(z, r, potential_energy, z_grad)

    return init_fn, update_fn


def find_reasonable_step_size(potential_fn, kinetic_fn, momentum_generator, inverse_mass_matrix,
                              position, rng, init_step_size):
    """
    Finds a reasonable step size by tuning `init_step_size`. This function is used
    to avoid working with a too large or too small step size in HMC.

    **References:**

    1. *The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo*,
       Matthew D. Hoffman, Andrew Gelman

    :param potential_fn: A callable to compute potential energy.
    :param kinetic_fn: A callable to compute kinetic energy.
    :param momentum_generator: A generator to get a random momentum variable.
    :param inverse_mass_matrix: Inverse of mass matrix.
    :param position: Current position of the particle.
    :param jax.random.PRNGKey rng: Random key to be used as the source of randomness.
    :param float init_step_size: Initial step size to be tuned.
    :return: a reasonable value for step size.
    :rtype: float
    """
    # We are going to find a step_size which make accept_prob (Metropolis correction)
    # near the target_accept_prob. If accept_prob:=exp(-delta_energy) is small,
    # then we have to decrease step_size; otherwise, increase step_size.
    target_accept_prob = np.log(0.8)

    _, vv_update = velocity_verlet(potential_fn, kinetic_fn)
    z = position
    potential_energy, z_grad = value_and_grad(potential_fn)(z)
    tiny = np.finfo(lax.dtype(init_step_size)).tiny

    def _body_fn(state):
        step_size, _, direction, rng = state
        rng, rng_momentum = random.split(rng)
        # scale step_size: increase 2x or decrease 2x depends on direction;
        # direction=1 means keep increasing step_size, otherwise decreasing step_size.
        # Note that the direction is -1 if delta_energy is `NaN`, which may be the
        # case for a diverging trajectory (e.g. in the case of evaluating log prob
        # of a value simulated using a large step size for a constrained sample site).
        step_size = (2.0 ** direction) * step_size
        r = momentum_generator(inverse_mass_matrix, rng_momentum)
        _, r_new, potential_energy_new, _ = vv_update(step_size,
                                                      inverse_mass_matrix,
                                                      (z, r, potential_energy, z_grad))
        energy_current = kinetic_fn(inverse_mass_matrix, r) + potential_energy
        energy_new = kinetic_fn(inverse_mass_matrix, r_new) + potential_energy_new
        delta_energy = energy_new - energy_current
        direction_new = np.where(target_accept_prob < -delta_energy, 1, -1)
        return step_size, direction, direction_new, rng

    def _cond_fn(state):
        step_size, last_direction, direction, _ = state
        # condition to run only if step_size is not so small or we are not decreasing step_size
        not_small_step_size_cond = (step_size > tiny) | (direction >= 0)
        return not_small_step_size_cond & ((last_direction == 0) | (direction == last_direction))

    step_size, _, _, _ = while_loop(_cond_fn, _body_fn, (init_step_size, 0, 0, rng))
    return step_size


def build_adaptation_schedule(num_steps):
    """
    Builds a window adaptation schedule to be used during warmup phase of HMC.

    :param int num_steps: Number of warmup steps.
    :return: a list of contiguous windows, each has attributes `start` and `end`,
        where `start` is the starting index and `end` is the ending index of the window.

    **References:**

    1. *Stan Reference Manual version 2.18*,
       Stan Development Team
    """
    adaptation_schedule = []
    # from Stan, for small num_steps
    if num_steps < 20:
        adaptation_schedule.append(AdaptWindow(0, num_steps - 1))
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

    adaptation_schedule.append(AdaptWindow(start=0, end=start_buffer_size - 1))
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
        adaptation_schedule.append(AdaptWindow(cur_window_start, next_window_start - 1))
    adaptation_schedule.append(AdaptWindow(end_window_start, num_steps - 1))
    return adaptation_schedule


def _identity_step_size(inverse_mass_matrix, z, rng, step_size):
    return step_size


def warmup_adapter(num_adapt_steps, find_reasonable_step_size=_identity_step_size,
                   adapt_step_size=True, adapt_mass_matrix=True,
                   dense_mass=False, target_accept_prob=0.8):
    """
    A scheme to adapt tunable parameters, namely step size and mass matrix, during
    the warmup phase of HMC.

    :param int num_adapt_steps: Number of warmup steps.
    :param find_reasonable_step_size: A callable to find a reasonable step size
        at the beginning of each adaptation window.
    :param bool adapt_step_size: A flag to decide if we want to adapt step_size
        during warm-up phase using Dual Averaging scheme (defaults to ``True``).
    :param bool adapt_mass_matrix: A flag to decide if we want to adapt mass
        matrix during warm-up phase using Welford scheme (defaults to ``True``).
    :param bool dense_mass: A flag to decide if mass matrix is dense or
        diagonal (defaults to ``False``).
    :param float target_accept_prob: Target acceptance probability for step size
        adaptation using Dual Averaging. Increasing this value will lead to a smaller
        step size, hence the sampling will be slower but more robust. Default to 0.8.
    :return: a pair of (`init_fn`, `update_fn`).
    """
    ss_init, ss_update = dual_averaging()
    mm_init, mm_update, mm_final = welford_covariance(diagonal=not dense_mass)
    adaptation_schedule = np.array(build_adaptation_schedule(num_adapt_steps))
    num_windows = len(adaptation_schedule)

    def init_fn(z, rng, step_size=1.0, inverse_mass_matrix=None, mass_matrix_size=None):
        """
        :param z: Initial position of the integrator.
        :param jax.random.PRNGKey rng: Random key to be used as the source of randomness.
        :param float step_size: Initial step size.
        :param inverse_mass_matrix: Inverse of the initial mass matrix. If ``None``,
            inverse of mass matrix will be an identity matrix with size is decided
            by the argument `mass_matrix_size`.
        :param int mass_matrix_size: Size of the mass matrix.
        :return: initial state of the adapt scheme.
        """
        rng, rng_ss = random.split(rng)
        if inverse_mass_matrix is None:
            assert mass_matrix_size is not None
            if dense_mass:
                inverse_mass_matrix = np.identity(mass_matrix_size)
            else:
                inverse_mass_matrix = np.ones(mass_matrix_size)
            mass_matrix_sqrt = inverse_mass_matrix
        else:
            if dense_mass:
                mass_matrix_sqrt = cholesky_inverse(inverse_mass_matrix)
            else:
                mass_matrix_sqrt = np.sqrt(np.reciprocal(inverse_mass_matrix))

        if adapt_step_size:
            step_size = find_reasonable_step_size(inverse_mass_matrix, z, rng_ss, step_size)
        ss_state = ss_init(np.log(10 * step_size))

        mm_state = mm_init(inverse_mass_matrix.shape[-1])

        window_idx = 0
        return AdaptState(step_size, inverse_mass_matrix, mass_matrix_sqrt,
                          ss_state, mm_state, window_idx, rng)

    def _update_at_window_end(z, rng_ss, state):
        step_size, inverse_mass_matrix, mass_matrix_sqrt, ss_state, mm_state, window_idx, rng = state

        if adapt_mass_matrix:
            inverse_mass_matrix, mass_matrix_sqrt = mm_final(mm_state, regularize=True)
            mm_state = mm_init(inverse_mass_matrix.shape[-1])

        if adapt_step_size:
            step_size = find_reasonable_step_size(inverse_mass_matrix, z, rng_ss, step_size)
            ss_state = ss_init(np.log(10 * step_size))

        return AdaptState(step_size, inverse_mass_matrix, mass_matrix_sqrt,
                          ss_state, mm_state, window_idx, rng)

    def update_fn(t, accept_prob, z, state):
        """
        :param int t: The current time step.
        :param float accept_prob: Acceptance probability of the current trajectory.
        :param z: New position drawn at the end of the current trajectory.
        :param state: Current state of the adapt scheme.
        :return: new state of the adapt scheme.
        """
        step_size, inverse_mass_matrix, mass_matrix_sqrt, ss_state, mm_state, window_idx, rng = state
        rng, rng_ss = random.split(rng)

        # update step size state
        if adapt_step_size:
            ss_state = ss_update(target_accept_prob - accept_prob, ss_state)
            # note: at the end of warmup phase, use average of log step_size
            log_step_size, log_step_size_avg, *_ = ss_state
            step_size = np.where(t == (num_adapt_steps - 1),
                                 np.exp(log_step_size_avg),
                                 np.exp(log_step_size))
            # account the the case log_step_size is a so small negative number
            step_size = np.clip(step_size, a_min=np.finfo(lax.dtype(step_size)).tiny)

        # update mass matrix state
        is_middle_window = (0 < window_idx) & (window_idx < (num_windows - 1))
        if adapt_mass_matrix:
            z_flat, _ = ravel_pytree(z)
            mm_state = cond(is_middle_window,
                            (z_flat, mm_state), lambda args: mm_update(*args),
                            mm_state, lambda x: x)

        t_at_window_end = t == adaptation_schedule[window_idx, 1]
        window_idx = np.where(t_at_window_end, window_idx + 1, window_idx)
        state = AdaptState(step_size, inverse_mass_matrix, mass_matrix_sqrt,
                           ss_state, mm_state, window_idx, rng)
        state = cond(t_at_window_end & is_middle_window,
                     (z, rng_ss, state), lambda args: _update_at_window_end(*args),
                     state, lambda x: x)
        return state

    return init_fn, update_fn


def _is_turning(inverse_mass_matrix, r_left, r_right, r_sum):
    r_left, _ = ravel_pytree(r_left)
    r_right, _ = ravel_pytree(r_right)
    r_sum, _ = ravel_pytree(r_sum)

    if inverse_mass_matrix.ndim == 2:
        v_left = np.matmul(inverse_mass_matrix, r_left)
        v_right = np.matmul(inverse_mass_matrix, r_right)
    elif inverse_mass_matrix.ndim == 1:
        v_left = np.multiply(inverse_mass_matrix, r_left)
        v_right = np.multiply(inverse_mass_matrix, r_right)

    # This implements dynamic termination criterion (ref [2], section A.4.2).
    turning_at_left = np.dot(v_left, r_sum - r_left) <= 0
    turning_at_right = np.dot(v_right, r_sum - r_right) <= 0
    return turning_at_left | turning_at_right


def _uniform_transition_kernel(current_tree, new_tree):
    # This function computes transition prob for subtrees (ref [2], section A.3.1).
    # e^new_weight / (e^new_weight + e^current_weight)
    transition_prob = expit(new_tree.weight - current_tree.weight)
    return transition_prob


def _biased_transition_kernel(current_tree, new_tree):
    # This function computes transition prob for main trees (ref [2], section A.3.2).
    transition_prob = np.exp(new_tree.weight - current_tree.weight)
    # If new tree is turning or diverging, we won't move the proposal
    # to the new tree.
    transition_prob = np.where(new_tree.turning | new_tree.diverging,
                               0.0, np.clip(transition_prob, a_max=1.0))
    return transition_prob


@partial(jit, static_argnums=(5,))
def _combine_tree(current_tree, new_tree, inverse_mass_matrix, going_right, rng, biased_transition):
    # Now we combine the current tree and the new tree. Note that outside
    # leaves of the combined tree are determined by the direction.
    z_left, r_left, z_left_grad, z_right, r_right, r_right_grad = cond(
        going_right,
        (current_tree, new_tree),
        lambda trees: (trees[0].z_left, trees[0].r_left,
                       trees[0].z_left_grad, trees[1].z_right,
                       trees[1].r_right, trees[1].z_right_grad),
        (new_tree, current_tree),
        lambda trees: (trees[0].z_left, trees[0].r_left,
                       trees[0].z_left_grad, trees[1].z_right,
                       trees[1].r_right, trees[1].z_right_grad)
    )
    r_sum = tree_multimap(np.add, current_tree.r_sum, new_tree.r_sum)

    if biased_transition:
        transition_prob = _biased_transition_kernel(current_tree, new_tree)
        turning = new_tree.turning | _is_turning(inverse_mass_matrix, r_left, r_right, r_sum)
    else:
        transition_prob = _uniform_transition_kernel(current_tree, new_tree)
        turning = current_tree.turning

    transition = random.bernoulli(rng, transition_prob)
    z_proposal, z_proposal_pe, z_proposal_grad = cond(
        transition,
        new_tree, lambda tree: (tree.z_proposal, tree.z_proposal_pe, tree.z_proposal_grad),
        current_tree, lambda tree: (tree.z_proposal, tree.z_proposal_pe, tree.z_proposal_grad)
    )

    tree_depth = current_tree.depth + 1
    tree_weight = np.logaddexp(current_tree.weight, new_tree.weight)
    diverging = new_tree.diverging

    sum_accept_probs = current_tree.sum_accept_probs + new_tree.sum_accept_probs
    num_proposals = current_tree.num_proposals + new_tree.num_proposals

    return TreeInfo(z_left, r_left, z_left_grad, z_right, r_right, r_right_grad,
                    z_proposal, z_proposal_pe, z_proposal_grad,
                    tree_depth, tree_weight, r_sum, turning, diverging,
                    sum_accept_probs, num_proposals)


def _build_basetree(vv_update, kinetic_fn, z, r, z_grad, inverse_mass_matrix, step_size, going_right,
                    energy_current, max_delta_energy):
    step_size = np.where(going_right, step_size, -step_size)
    z_new, r_new, potential_energy_new, z_new_grad = vv_update(
        step_size,
        inverse_mass_matrix,
        (z, r, energy_current, z_grad),
    )

    energy_new = potential_energy_new + kinetic_fn(inverse_mass_matrix, r_new)
    delta_energy = energy_new - energy_current
    # Handles the NaN case.
    delta_energy = np.where(np.isnan(delta_energy), np.inf, delta_energy)
    tree_weight = -delta_energy

    diverging = delta_energy > max_delta_energy
    accept_prob = np.clip(np.exp(-delta_energy), a_max=1.0)
    return TreeInfo(z_new, r_new, z_new_grad, z_new, r_new, z_new_grad,
                    z_new, potential_energy_new, z_new_grad,
                    depth=0, weight=tree_weight, r_sum=r_new, turning=False,
                    diverging=diverging, sum_accept_probs=accept_prob, num_proposals=1)


def _get_leaf(tree, going_right):
    return cond(going_right,
                tree,
                lambda tree: (tree.z_right, tree.r_right, tree.z_right_grad),
                tree,
                lambda tree: (tree.z_left, tree.r_left, tree.z_left_grad))


def _double_tree(current_tree, vv_update, kinetic_fn, inverse_mass_matrix, step_size,
                 going_right, rng, energy_current, max_delta_energy, r_ckpts, r_sum_ckpts):
    key, transition_key = random.split(rng)
    # If we are going to the right, start from the right leaf of the current tree.
    z, r, z_grad = _get_leaf(current_tree, going_right)

    new_tree = _iterative_build_subtree(current_tree.depth, vv_update, kinetic_fn,
                                        z, r, z_grad, inverse_mass_matrix, step_size,
                                        going_right, key, energy_current, max_delta_energy,
                                        r_ckpts, r_sum_ckpts)

    return _combine_tree(current_tree, new_tree, inverse_mass_matrix, going_right, transition_key,
                         True)


def _leaf_idx_to_ckpt_idxs(n):
    # computes the number of non-zero bits except the last bit
    # e.g. 6 -> 2, 7 -> 2, 13 -> 2
    _, idx_max = while_loop(lambda nc: nc[0] > 0,
                            lambda nc: (nc[0] >> 1, nc[1] + (nc[0] & 1)),
                            (n >> 1, 0))
    # computes the number of contiguous last non-zero bits
    # e.g. 6 -> 0, 7 -> 3, 13 -> 1
    _, num_subtrees = while_loop(lambda nc: (nc[0] & 1) != 0,
                                 lambda nc: (nc[0] >> 1, nc[1] + 1),
                                 (n, 0))
    idx_min = idx_max - num_subtrees + 1
    return idx_min, idx_max


def _is_iterative_turning(inverse_mass_matrix, r, r_sum, r_ckpts, r_sum_ckpts, idx_min, idx_max):
    def _body_fn(state):
        i, _ = state
        subtree_r_sum = r_sum - r_sum_ckpts[i] + r_ckpts[i]
        return i - 1, _is_turning(inverse_mass_matrix, r_ckpts[i], r, subtree_r_sum)

    _, turning = while_loop(lambda it: (it[0] >= idx_min) & ~it[1],
                            _body_fn,
                            (idx_max, False))
    return turning


def _iterative_build_subtree(depth, vv_update, kinetic_fn, z, r, z_grad,
                             inverse_mass_matrix, step_size, going_right, rng,
                             energy_current, max_delta_energy, r_ckpts, r_sum_ckpts):
    max_num_proposals = 2 ** depth

    def _cond_fn(state):
        tree, turning, _, _, _ = state
        return (tree.num_proposals < max_num_proposals) & ~turning & ~tree.diverging

    def _body_fn(state):
        current_tree, _, r_ckpts, r_sum_ckpts, rng = state
        rng, transition_rng = random.split(rng)
        z, r, z_grad = _get_leaf(current_tree, going_right)
        new_leaf = _build_basetree(vv_update, kinetic_fn, z, r, z_grad, inverse_mass_matrix, step_size,
                                   going_right, energy_current, max_delta_energy)
        new_tree = _combine_tree(current_tree, new_leaf, inverse_mass_matrix, going_right,
                                 transition_rng, False)

        leaf_idx = current_tree.num_proposals
        ckpt_idx_min, ckpt_idx_max = _leaf_idx_to_ckpt_idxs(leaf_idx)
        r, _ = ravel_pytree(new_leaf.r_right)
        r_sum, _ = ravel_pytree(new_tree.r_sum)
        # we update checkpoints when leaf_idx is even
        r_ckpts, r_sum_ckpts = cond(leaf_idx % 2 == 0,
                                    (r_ckpts, r_sum_ckpts),
                                    lambda x: (index_update(x[0], ckpt_idx_max, r),
                                               index_update(x[1], ckpt_idx_max, r_sum)),
                                    (r_ckpts, r_sum_ckpts),
                                    lambda x: x)

        turning = _is_iterative_turning(inverse_mass_matrix, r, r_sum, r_ckpts, r_sum_ckpts,
                                        ckpt_idx_min, ckpt_idx_max)
        return new_tree, turning, r_ckpts, r_sum_ckpts, rng

    basetree = _build_basetree(vv_update, kinetic_fn, z, r, z_grad, inverse_mass_matrix, step_size,
                               going_right, energy_current, max_delta_energy)
    r_init, _ = ravel_pytree(basetree.r_left)
    r_ckpts = index_update(r_ckpts, 0, r_init)
    r_sum_ckpts = index_update(r_sum_ckpts, 0, r_init)

    tree, turning, _, _, _ = while_loop(
        _cond_fn,
        _body_fn,
        (basetree, False, r_ckpts, r_sum_ckpts, rng)
    )
    # update depth and turning condition
    return TreeInfo(tree.z_left, tree.r_left, tree.z_left_grad,
                    tree.z_right, tree.r_right, tree.z_right_grad,
                    tree.z_proposal, tree.z_proposal_pe, tree.z_proposal_grad,
                    depth, tree.weight, tree.r_sum, turning, tree.diverging,
                    tree.sum_accept_probs, tree.num_proposals)


def build_tree(verlet_update, kinetic_fn, verlet_state, inverse_mass_matrix, step_size, rng,
               max_delta_energy=1000., max_tree_depth=10):
    """
    Builds a binary tree from the `verlet_state`. This is used in NUTS sampler.

    **References:**

    1. *The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo*,
       Matthew D. Hoffman, Andrew Gelman
    2. *A Conceptual Introduction to Hamiltonian Monte Carlo*,
       Michael Betancourt

    :param verlet_update: A callable to get a new integrator state given a current
        integrator state.
    :param kinetic_fn: A callable to compute kinetic energy.
    :param verlet_state: Initial integrator state.
    :param inverse_mass_matrix: Inverse of the mass matrix.
    :param float step_size: Step size for the current trajectory.
    :param jax.random.PRNGKey rng: random key to be used as the source of
        randomness.
    :param float max_delta_energy: A threshold to decide if the new state diverges
        (based on the energy difference) too much from the initial integrator state.
    :return: information of the tree.
    :rtype: :data:`TreeInfo`
    """
    z, r, potential_energy, z_grad = verlet_state
    energy_current = potential_energy + kinetic_fn(inverse_mass_matrix, r)
    r_ckpts = np.zeros((max_tree_depth, inverse_mass_matrix.shape[-1]))
    r_sum_ckpts = np.zeros((max_tree_depth, inverse_mass_matrix.shape[-1]))

    tree = TreeInfo(z, r, z_grad, z, r, z_grad, z, potential_energy, z_grad,
                    depth=0, weight=0., r_sum=r, turning=False, diverging=False,
                    sum_accept_probs=0., num_proposals=0)

    def _cond_fn(state):
        tree, _ = state
        return (tree.depth < max_tree_depth) & ~tree.turning & ~tree.diverging

    def _body_fn(state):
        tree, key = state
        key, direction_key, doubling_key = random.split(key, 3)
        going_right = random.bernoulli(direction_key)
        tree = _double_tree(tree, verlet_update, kinetic_fn, inverse_mass_matrix, step_size,
                            going_right, doubling_key, energy_current, max_delta_energy,
                            r_ckpts, r_sum_ckpts)
        return tree, key

    state = (tree, rng)
    tree, _ = while_loop(_cond_fn, _body_fn, state)
    return tree


def euclidean_kinetic_energy(inverse_mass_matrix, r):
    r, _ = ravel_pytree(r)

    if inverse_mass_matrix.ndim == 2:
        v = np.matmul(inverse_mass_matrix, r)
    elif inverse_mass_matrix.ndim == 1:
        v = np.multiply(inverse_mass_matrix, r)

    return 0.5 * np.dot(v, r)


def potential_energy(model, model_args, model_kwargs, inv_transforms):
    def _potential_energy(params):
        params_constrained = transform_fn(inv_transforms, params)
        log_joint, model_trace = log_density(model, model_args, model_kwargs, params_constrained)
        for name, t in inv_transforms.items():
            t_log_det = np.sum(t.log_abs_det_jacobian(params[name], params_constrained[name]))
            if 'scale' in model_trace[name]:
                t_log_det = model_trace[name]['scale'] * t_log_det
            log_joint = log_joint + t_log_det
        return - log_joint

    return _potential_energy


def initialize_model(rng, model, *model_args, init_strategy='uniform', **model_kwargs):
    """
    Given a model with Pyro primitives, returns a function which, given
    unconstrained parameters, evaluates the potential energy (negative
    joint density). In addition, this also returns initial parameters
    sampled from the prior to initiate MCMC sampling and functions to
    transform unconstrained values at sample sites to constrained values
    within their respective support.

    :param jax.random.PRNGKey rng: random number generator seed to
        sample from the prior. The returned `init_params` will have the
        batch shape ``rng.shape[:-1]``.
    :param model: Python callable containing Pyro primitives.
    :param `*model_args`: args provided to the model.
    :param str init_strategy: initialization strategy - `uniform`
        initializes the unconstrained parameters by drawing from
        a `Uniform(-2, 2)` distribution (as used by Stan), whereas
        `prior` initializes the parameters by sampling from the prior
        for each of the sample sites.
    :param `**model_kwargs`: kwargs provided to the model.
    :return: tuple of (`init_params`, `potential_fn`, `constrain_fn`),
        `init_params` are values from the prior used to initiate MCMC,
        `constrain_fn` is a callable that uses inverse transforms
        to convert unconstrained HMC samples to constrained values that
        lie within the site's support.
    """
    def single_chain_init(key, only_params=False):
        seeded_model = seed(model, key)
        model_trace = trace(seeded_model).get_trace(*model_args, **model_kwargs)
        constrained_values, inv_transforms = {}, {}
        for k, v in model_trace.items():
            if v['type'] == 'sample' and not v['is_observed']:
                constrained_values[k] = v['value']
                inv_transforms[k] = biject_to(v['fn'].support)
            elif v['type'] == 'param':
                constrained_values[k] = v['value']
                constraint = v['kwargs'].pop('constraint', real)
                inv_transforms[k] = biject_to(constraint)
        prior_params = transform_fn(inv_transforms,
                                    {k: v for k, v in constrained_values.items()}, invert=True)
        if init_strategy == 'uniform':
            init_params = {}
            for k, v in prior_params.items():
                key, = random.split(key, 1)
                init_params[k] = random.uniform(key, shape=np.shape(v), minval=-2, maxval=2)
        elif init_strategy == 'prior':
            init_params = prior_params
        else:
            raise ValueError('initialize={} is not a valid initialization strategy.'.format(init_strategy))

        if only_params:
            return init_params
        else:
            return (init_params,
                    potential_energy(seeded_model, model_args, model_kwargs, inv_transforms),
                    jax.partial(transform_fn, inv_transforms))

    if rng.ndim == 1:
        return single_chain_init(rng)
    else:
        _, potential_fn, constrain_fun = single_chain_init(rng[0])
        init_params = vmap(lambda rng: single_chain_init(rng, only_params=True))(rng)
        return init_params, potential_fn, constrain_fun
