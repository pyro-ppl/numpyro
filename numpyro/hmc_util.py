import jax
import jax.numpy as np
from jax import grad, jit, partial, random, value_and_grad
from jax.flatten_util import ravel_pytree
from jax.ops import index_update
from jax.scipy.special import expit
from jax.tree_util import tree_multimap

from numpyro.distributions.constraints import biject_to
from numpyro.handlers import seed, substitute, trace
from numpyro.util import cond, laxtuple, while_loop

AdaptWindow = laxtuple("AdaptWindow", ["start", "end"])
AdaptState = laxtuple("AdaptState", ["step_size", "inverse_mass_matrix", "ss_state", "mm_state",
                                     "window_idx", "rng"])
IntegratorState = laxtuple("IntegratorState", ["z", "r", "potential_energy", "z_grad"])

_TreeInfo = laxtuple('_TreeInfo', ['z_left', 'r_left', 'z_left_grad',
                                   'z_right', 'r_right', 'z_right_grad',
                                   'z_proposal', 'z_proposal_pe', 'z_proposal_grad',
                                   'depth', 'weight', 'r_sum', 'turning', 'diverging',
                                   'sum_accept_probs', 'num_proposals'])


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
        return x_t, x_avg, g_avg, t, prox_center

    def update_fn(g, state):
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
        return mean, m2, n

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
        return mean, m2, n

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
        return IntegratorState(z, r, potential_energy, z_grad)

    def update_fn(step_size, inverse_mass_matrix, state):
        """
        Single step velocity verlet.
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
    # We are going to find a step_size which make accept_prob (Metropolis correction)
    # near the target_accept_prob. If accept_prob:=exp(-delta_energy) is small,
    # then we have to decrease step_size; otherwise, increase step_size.
    target_accept_prob = np.log(0.8)

    _, vv_update = velocity_verlet(potential_fn, kinetic_fn)
    z = position
    potential_energy, z_grad = value_and_grad(potential_fn)(z)

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
        return (state[1] == 0) | (state[1] == state[2])

    step_size, _, _, _ = while_loop(_cond_fn, _body_fn, (init_step_size, 0, 0, rng))
    return step_size


def build_adaptation_schedule(num_steps):
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
                   diag_mass=True, target_accept_prob=0.8):
    ss_init, ss_update = dual_averaging()
    mm_init, mm_update, mm_final = welford_covariance(diagonal=diag_mass)
    adaptation_schedule = np.array(build_adaptation_schedule(num_adapt_steps))
    num_windows = len(adaptation_schedule)

    def init_fn(z, rng, step_size=1.0, inverse_mass_matrix=None, mass_matrix_size=None):
        rng, rng_ss = random.split(rng)
        if inverse_mass_matrix is None:
            assert mass_matrix_size is not None
            if diag_mass:
                inverse_mass_matrix = np.ones(mass_matrix_size)
            else:
                inverse_mass_matrix = np.identity(mass_matrix_size)

        if adapt_step_size:
            step_size = find_reasonable_step_size(inverse_mass_matrix, z, rng_ss, step_size)
        ss_state = ss_init(np.log(10 * step_size))

        mm_state = mm_init(inverse_mass_matrix.shape[-1])

        window_idx = 0
        return AdaptState(step_size, inverse_mass_matrix, ss_state, mm_state, window_idx, rng)

    def _update_at_window_end(z, rng_ss, state):
        step_size, inverse_mass_matrix, ss_state, mm_state, window_idx, rng = state

        if adapt_mass_matrix:
            inverse_mass_matrix = mm_final(mm_state, regularize=True)
            mm_state = mm_init(inverse_mass_matrix.shape[-1])

        if adapt_step_size:
            step_size = find_reasonable_step_size(inverse_mass_matrix, z, rng_ss, step_size)
            ss_state = ss_init(np.log(10 * step_size))

        return AdaptState(step_size, inverse_mass_matrix, ss_state, mm_state, window_idx, rng)

    def update_fn(t, accept_prob, z, state):
        step_size, inverse_mass_matrix, ss_state, mm_state, window_idx, rng = state
        rng, rng_ss = random.split(rng)

        # update step size state
        if adapt_step_size:
            ss_state = ss_update(target_accept_prob - accept_prob, ss_state)
            # note: at the end of warmup phase, use average of log step_size
            # TODO: should we make sure that we won't update step_size if t >= num_steps?
            log_step_size, log_step_size_avg, *_ = ss_state
            step_size = np.where(t == (num_adapt_steps - 1),
                                 np.exp(log_step_size_avg),
                                 np.exp(log_step_size))

        # update mass matrix state
        is_middle_window = (0 < window_idx) & (window_idx < (num_windows - 1))
        if adapt_mass_matrix:
            z_flat, _ = ravel_pytree(z)
            mm_state = cond(is_middle_window,
                            (z_flat, mm_state), lambda args: mm_update(*args),
                            mm_state, lambda x: x)

        t_at_window_end = t == adaptation_schedule[window_idx, 1]
        window_idx = np.where(t_at_window_end, window_idx + 1, window_idx)
        state = AdaptState(step_size, inverse_mass_matrix, ss_state, mm_state, window_idx, rng)
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

    return _TreeInfo(z_left, r_left, z_left_grad, z_right, r_right, r_right_grad,
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
    return _TreeInfo(z_new, r_new, z_new_grad, z_new, r_new, z_new_grad,
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
    return _TreeInfo(tree.z_left, tree.r_left, tree.z_left_grad,
                     tree.z_right, tree.r_right, tree.z_right_grad,
                     tree.z_proposal, tree.z_proposal_pe, tree.z_proposal_grad,
                     depth, tree.weight, tree.r_sum, turning, tree.diverging,
                     tree.sum_accept_probs, tree.num_proposals)


def build_tree(verlet_update, kinetic_fn, verlet_state, inverse_mass_matrix, step_size, rng,
               max_delta_energy=1000., max_tree_depth=10):
    """
    **References:**
    [1] `The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo`,
    Matthew D. Hoffman, Andrew Gelman
    [2] `A Conceptual Introduction to Hamiltonian Monte Carlo`,
    Michael Betancourt
    """
    z, r, potential_energy, z_grad = verlet_state
    energy_current = potential_energy + kinetic_fn(inverse_mass_matrix, r)
    r_ckpts = np.zeros((max_tree_depth, inverse_mass_matrix.shape[-1]),
                       dtype=inverse_mass_matrix.dtype)
    r_sum_ckpts = np.zeros((max_tree_depth, inverse_mass_matrix.shape[-1]),
                           dtype=inverse_mass_matrix.dtype)

    tree = _TreeInfo(z, r, z_grad, z, r, z_grad, z, potential_energy, z_grad,
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


def log_density(model, model_args, model_kwargs, params):
    model = substitute(model, params)
    model_trace = trace(model).get_trace(*model_args, **model_kwargs)
    log_joint = 0.
    for site in model_trace.values():
        if site['type'] == 'sample':
            log_joint = log_joint + np.sum(site['fn'].log_prob(site['value']))
    return log_joint, model_trace


def potential_energy(model, model_args, model_kwargs, transforms):
    def _potential_energy(params):
        params_constrained = {k: transforms[k](v) for k, v in params.items()}
        log_joint = jax.partial(log_density, model, model_args, model_kwargs)(params_constrained)[0]
        for name, t in transforms.items():
            log_joint = log_joint + np.sum(t.log_abs_det_jacobian(params[name], params_constrained[name]))
        return - log_joint

    return _potential_energy


def transform_fn(transforms, params, invert=False):
    return {k: transforms[k](v) if not invert else transforms[k].inv(v)
            for k, v in params.items()}


def initialize_model(rng, model, *model_args, **model_kwargs):
    """
    Given a model with Pyro primitives, returns a function which, given
    unconstrained parameters, evaluates the potential energy (negative
    joint density). In addition, this also returns initial parameters
    sampled from the prior to initiate MCMC sampling and functions to
    transform unconstrained values at sample sites to constrained values
    within their respective support.

    :param jax.random.PRNGKey rng: random number generator seed to
        sample from the prior.
    :param model: Python callable containing Pyro primitives.
    :param tuple model_args: args provided to the model.
    :param dict model_kwargs: kwargs provided to the model.
    :return: tuple of (`init_params`, `potential_fn`, `transforms`)
        `init_params` are values from the prior used to initiate MCMC.
        `transforms` are useful to convert unconstrained HMC samples
        to constrained values that lie within the site's support.
    """
    model = seed(model, rng)
    model_trace = trace(model).get_trace(*model_args, **model_kwargs)
    sample_sites = {k: v for k, v in model_trace.items() if v['type'] == 'sample' and not v['is_observed']}
    transforms = {k: biject_to(v['fn'].support) for k, v in sample_sites.items()}
    init_params = transform_fn(transforms, {k: v['value'] for k, v in sample_sites.items()}, invert=True)
    return init_params, potential_energy(model, model_args, model_kwargs, transforms), \
        jax.partial(transform_fn, transforms)
