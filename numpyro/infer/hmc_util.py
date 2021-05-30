# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict, namedtuple

from jax import grad, jacfwd, random, value_and_grad, vmap
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
from jax.ops import index_update
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import expit
from jax.tree_util import tree_flatten, tree_map, tree_multimap

import numpyro.distributions as dist
from numpyro.util import cond, identity, while_loop

AdaptWindow = namedtuple("AdaptWindow", ["start", "end"])
# XXX: we need to store rng_key here in case we use find_reasonable_step_size functionality
HMCAdaptState = namedtuple(
    "HMCAdaptState",
    [
        "step_size",
        "inverse_mass_matrix",
        "mass_matrix_sqrt",
        "mass_matrix_sqrt_inv",
        "ss_state",
        "mm_state",
        "window_idx",
        "rng_key",
    ],
)
IntegratorState = namedtuple(
    "IntegratorState", ["z", "r", "potential_energy", "z_grad"]
)
IntegratorState.__new__.__defaults__ = (None,) * len(IntegratorState._fields)

TreeInfo = namedtuple(
    "TreeInfo",
    [
        "z_left",
        "r_left",
        "z_left_grad",
        "z_right",
        "r_right",
        "z_right_grad",
        "z_proposal",
        "z_proposal_pe",
        "z_proposal_grad",
        "z_proposal_energy",
        "depth",
        "weight",
        "r_sum",
        "turning",
        "diverging",
        "sum_accept_probs",
        "num_proposals",
    ],
)


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

    def init_fn(prox_center=0.0):
        """
        :param float prox_center: A parameter introduced in reference [1] which
            pulls the primal sequence towards it. Defaults to 0.
        :return: initial state for the scheme.
        """
        x_t = jnp.zeros(())
        x_avg = jnp.zeros(())  # average of primal sequence
        g_avg = jnp.zeros(())  # average of dual sequence
        t = jnp.array(0, dtype=jnp.result_type(int))
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
        :param int size: size of each sample. For a structured mass matrix,
            this is a dict mapping from tuples of site names to the shape
            of the mass matrix.
        :return: initial state for the scheme.
        """
        if isinstance(size, dict):
            state = {}
            for site_names, size_block in size.items():
                state[site_names] = init_fn(size_block)
            return state

        if isinstance(size, int):
            shape = (size,) if diagonal else (size, size)
        else:
            shape = size

        mean = jnp.zeros(shape[-1])
        m2 = jnp.zeros(shape)
        n = 0
        return mean, m2, n

    def update_fn(sample, state):
        """
        :param sample: A new sample.
        :param state: Current state of the scheme.
        :return: new state for the scheme.
        """
        if isinstance(state, dict):
            assert isinstance(sample, dict)
            new_state = {}
            for site_names, state_block in state.items():
                sample_block = tuple(sample[k] for k in site_names)
                new_state[site_names] = update_fn(sample_block, state_block)
            return new_state

        sample, _ = ravel_pytree(sample)
        mean, m2, n = state
        n = n + 1
        delta_pre = sample - mean
        mean = mean + delta_pre / n
        delta_post = sample - mean
        if jnp.ndim(m2) == 1:
            m2 = m2 + delta_pre * delta_post
        else:
            m2 = m2 + jnp.outer(delta_post, delta_pre)
        return mean, m2, n

    def final_fn(state, regularize=False):
        """
        :param state: Current state of the scheme.
        :param bool regularize: Whether to adjust diagonal for numerical stability.
        :return: a triple of estimated covariance, the square root of precision, and
            the inverse of that square root.
        """
        if isinstance(state, dict):
            cov, cov_inv_sqrt, tril_inv = {}, {}, {}
            for site_names, state_block in state.items():
                cov_block, cov_inv_sqrt_block, tril_inv_block = final_fn(
                    state_block, regularize=regularize
                )
                cov[site_names] = cov_block
                cov_inv_sqrt[site_names] = cov_inv_sqrt_block
                tril_inv[site_names] = tril_inv_block
            return cov, cov_inv_sqrt, tril_inv

        mean, m2, n = state
        # XXX it is not necessary to check for the case n=1
        cov = m2 / (n - 1)
        if regularize:
            # Regularization from Stan
            scaled_cov = (n / (n + 5)) * cov
            shrinkage = 1e-3 * (5 / (n + 5))
            if jnp.ndim(scaled_cov) == 1:
                cov = scaled_cov + shrinkage
            else:
                cov = scaled_cov + shrinkage * jnp.identity(mean.shape[0])
        if jnp.ndim(cov) == 2:
            # copy the implementation of distributions.util.cholesky_of_inverse here
            tril_inv = jnp.swapaxes(
                jnp.linalg.cholesky(cov[..., ::-1, ::-1])[..., ::-1, ::-1], -2, -1
            )
            identity = jnp.identity(cov.shape[-1])
            cov_inv_sqrt = solve_triangular(tril_inv, identity, lower=True)
        else:
            tril_inv = jnp.sqrt(cov)
            cov_inv_sqrt = jnp.reciprocal(tril_inv)
        return cov, cov_inv_sqrt, tril_inv

    return init_fn, update_fn, final_fn


def _value_and_grad(f, x, forward_mode_differentiation=False):
    if forward_mode_differentiation:
        return f(x), jacfwd(f)(x)
    else:
        return value_and_grad(f)(x)


def _kinetic_grad(kinetic_fn, inverse_mass_matrix, r):
    if hasattr(kinetic_fn, "_kinetic_grad"):
        return kinetic_fn._kinetic_grad(inverse_mass_matrix, r)
    else:
        return grad(kinetic_fn, argnums=1)(inverse_mass_matrix, r)


def velocity_verlet(potential_fn, kinetic_fn, forward_mode_differentiation=False):
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

    def init_fn(z, r, potential_energy=None, z_grad=None):
        """
        :param z: Position of the particle.
        :param r: Momentum of the particle.
        :param potential_energy: Potential energy at `z`.
        :param z_grad: gradient of potential energy at `z`.
        :return: initial state for the integrator.
        """
        if potential_energy is None or z_grad is None:
            potential_energy, z_grad = _value_and_grad(
                potential_fn, z, forward_mode_differentiation
            )
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
        r = tree_multimap(
            lambda r, z_grad: r - 0.5 * step_size * z_grad, r, z_grad
        )  # r(n+1/2)
        r_grad = _kinetic_grad(kinetic_fn, inverse_mass_matrix, r)
        z = tree_multimap(lambda z, r_grad: z + step_size * r_grad, z, r_grad)  # z(n+1)
        potential_energy, z_grad = _value_and_grad(
            potential_fn, z, forward_mode_differentiation
        )
        r = tree_multimap(
            lambda r, z_grad: r - 0.5 * step_size * z_grad, r, z_grad
        )  # r(n+1)
        return IntegratorState(z, r, potential_energy, z_grad)

    return init_fn, update_fn


def find_reasonable_step_size(
    potential_fn,
    kinetic_fn,
    momentum_generator,
    init_step_size,
    inverse_mass_matrix,
    z_info,
    rng_key,
):
    """
    Finds a reasonable step size by tuning `init_step_size`. This function is used
    to avoid working with a too large or too small step size in HMC.

    **References:**

    1. *The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo*,
       Matthew D. Hoffman, Andrew Gelman

    :param potential_fn: A callable to compute potential energy.
    :param kinetic_fn: A callable to compute kinetic energy.
    :param momentum_generator: A generator to get a random momentum variable.
    :param float init_step_size: Initial step size to be tuned.
    :param inverse_mass_matrix: Inverse of mass matrix.
    :param IntegratorState z_info: The current integrator state.
    :param jax.random.PRNGKey rng_key: Random key to be used as the source of randomness.
    :return: a reasonable value for step size.
    :rtype: float
    """
    # We are going to find a step_size which make accept_prob (Metropolis correction)
    # near the target_accept_prob. If accept_prob:=exp(-delta_energy) is small,
    # then we have to decrease step_size; otherwise, increase step_size.
    target_accept_prob = jnp.log(0.8)

    _, vv_update = velocity_verlet(potential_fn, kinetic_fn)
    z, _, potential_energy, z_grad = z_info
    if potential_energy is None or z_grad is None:
        potential_energy, z_grad = value_and_grad(potential_fn)(z)
    finfo = jnp.finfo(jnp.result_type(init_step_size))

    def _body_fn(state):
        step_size, _, direction, rng_key = state
        rng_key, rng_key_momentum = random.split(rng_key)
        # scale step_size: increase 2x or decrease 2x depends on direction;
        # direction=1 means keep increasing step_size, otherwise decreasing step_size.
        # Note that the direction is -1 if delta_energy is `NaN`, which may be the
        # case for a diverging trajectory (e.g. in the case of evaluating log prob
        # of a value simulated using a large step size for a constrained sample site).
        step_size = (2.0 ** direction) * step_size
        r = momentum_generator(z, inverse_mass_matrix, rng_key_momentum)
        _, r_new, potential_energy_new, _ = vv_update(
            step_size, inverse_mass_matrix, (z, r, potential_energy, z_grad)
        )
        energy_current = kinetic_fn(inverse_mass_matrix, r) + potential_energy
        energy_new = kinetic_fn(inverse_mass_matrix, r_new) + potential_energy_new
        delta_energy = energy_new - energy_current
        direction_new = jnp.where(target_accept_prob < -delta_energy, 1, -1)
        return step_size, direction, direction_new, rng_key

    def _cond_fn(state):
        step_size, last_direction, direction, _ = state
        # condition to run only if step_size is not too small or we are not decreasing step_size
        not_small_step_size_cond = (step_size > finfo.tiny) | (direction >= 0)
        # condition to run only if step_size is not too large or we are not increasing step_size
        not_large_step_size_cond = (step_size < finfo.max) | (direction <= 0)
        not_extreme_cond = not_small_step_size_cond & not_large_step_size_cond
        return not_extreme_cond & (
            (last_direction == 0) | (direction == last_direction)
        )

    step_size, _, _, _ = while_loop(_cond_fn, _body_fn, (init_step_size, 0, 0, rng_key))
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


def _initialize_mass_matrix(z, inverse_mass_matrix, dense_mass):
    if isinstance(dense_mass, list):
        if inverse_mass_matrix is None:
            inverse_mass_matrix = {}
        # if user specifies an ndarray mass matrix, then we convert it to a dict
        elif not isinstance(inverse_mass_matrix, dict):
            inverse_mass_matrix = {tuple(sorted(z)): inverse_mass_matrix}
        mass_matrix_sqrt = {}
        mass_matrix_sqrt_inv = {}
        for site_names in dense_mass:
            inverse_mm = inverse_mass_matrix.get(site_names)
            z_block = tuple(z[k] for k in site_names)
            inverse_mm, mm_sqrt, mm_sqrt_inv = _initialize_mass_matrix(
                z_block, inverse_mm, True
            )
            inverse_mass_matrix[site_names] = inverse_mm
            mass_matrix_sqrt[site_names] = mm_sqrt
            mass_matrix_sqrt_inv[site_names] = mm_sqrt_inv
        # NB: this branch only happens when users want to use block diagonal
        # inverse_mass_matrix, for example, {("a",): jnp.ones(3), ("b",): jnp.ones(3)}.
        for site_names, inverse_mm in inverse_mass_matrix.items():
            if site_names in dense_mass:
                continue
            z_block = tuple(z[k] for k in site_names)
            inverse_mm, mm_sqrt, mm_sqrt_inv = _initialize_mass_matrix(
                z_block, inverse_mm, False
            )
            inverse_mass_matrix[site_names] = inverse_mm
            mass_matrix_sqrt[site_names] = mm_sqrt
            mass_matrix_sqrt_inv[site_names] = mm_sqrt_inv
        remaining_sites = tuple(sorted(set(z) - set().union(*inverse_mass_matrix)))
        if len(remaining_sites) > 0:
            z_block = tuple(z[k] for k in remaining_sites)
            inverse_mm, mm_sqrt, mm_sqrt_inv = _initialize_mass_matrix(
                z_block, None, False
            )
            inverse_mass_matrix[remaining_sites] = inverse_mm
            mass_matrix_sqrt[remaining_sites] = mm_sqrt
            mass_matrix_sqrt_inv[remaining_sites] = mm_sqrt_inv
        expected_site_names = sorted(z)
        actual_site_names = sorted(
            [k for site_names in inverse_mass_matrix for k in site_names]
        )
        assert actual_site_names == expected_site_names, (
            "There seems to be a conflict of sites names specified in the initial"
            " `inverse_mass_matrix` and in `dense_mass` argument."
        )
        return inverse_mass_matrix, mass_matrix_sqrt, mass_matrix_sqrt_inv

    mass_matrix_size = jnp.size(ravel_pytree(z)[0])
    if inverse_mass_matrix is None:
        if dense_mass:
            inverse_mass_matrix = jnp.identity(mass_matrix_size)
        else:
            inverse_mass_matrix = jnp.ones(mass_matrix_size)
        mass_matrix_sqrt = mass_matrix_sqrt_inv = inverse_mass_matrix
    else:
        if dense_mass:
            if jnp.ndim(inverse_mass_matrix) == 1:
                inverse_mass_matrix = jnp.diag(inverse_mass_matrix)
            mass_matrix_sqrt_inv = jnp.swapaxes(
                jnp.linalg.cholesky(inverse_mass_matrix[..., ::-1, ::-1])[
                    ..., ::-1, ::-1
                ],
                -2,
                -1,
            )
            identity = jnp.identity(inverse_mass_matrix.shape[-1])
            mass_matrix_sqrt = solve_triangular(
                mass_matrix_sqrt_inv, identity, lower=True
            )
        else:
            if jnp.ndim(inverse_mass_matrix) == 2:
                inverse_mass_matrix = jnp.diag(inverse_mass_matrix)
            mass_matrix_sqrt_inv = jnp.sqrt(inverse_mass_matrix)
            mass_matrix_sqrt = jnp.reciprocal(mass_matrix_sqrt_inv)
    return inverse_mass_matrix, mass_matrix_sqrt, mass_matrix_sqrt_inv


def warmup_adapter(
    num_adapt_steps,
    find_reasonable_step_size=None,
    adapt_step_size=True,
    adapt_mass_matrix=True,
    dense_mass=False,
    target_accept_prob=0.8,
    regularize_mass_matrix=True,
):
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
    if find_reasonable_step_size is None:
        find_reasonable_step_size = identity
    ss_init, ss_update = dual_averaging()
    mm_init, mm_update, mm_final = welford_covariance(diagonal=not dense_mass)
    adaptation_schedule = jnp.array(build_adaptation_schedule(num_adapt_steps))
    num_windows = len(adaptation_schedule)

    def init_fn(
        z_info, rng_key, step_size=1.0, inverse_mass_matrix=None, mass_matrix_size=None
    ):
        """
        :param IntegratorState z_info: The initial integrator state.
        :param jax.random.PRNGKey rng_key: Random key to be used as the source of randomness.
        :param float step_size: Initial step size.
        :param inverse_mass_matrix: Inverse of the initial mass matrix. If ``None``,
            inverse of mass matrix will be an identity matrix with size is decided
            by the argument `mass_matrix_size`.
        :param int mass_matrix_size: Size of the mass matrix.
        :return: initial state of the adapt scheme.
        """
        rng_key, rng_key_ss = random.split(rng_key)
        (
            inverse_mass_matrix,
            mass_matrix_sqrt,
            mass_matrix_sqrt_inv,
        ) = _initialize_mass_matrix(z_info[0], inverse_mass_matrix, dense_mass)

        if adapt_step_size:
            step_size = find_reasonable_step_size(
                step_size, inverse_mass_matrix, z_info, rng_key_ss
            )
        ss_state = ss_init(jnp.log(10 * step_size))

        if isinstance(inverse_mass_matrix, dict):
            size = {k: v.shape for k, v in inverse_mass_matrix.items()}
        else:
            size = inverse_mass_matrix.shape[-1]
        mm_state = mm_init(size)

        window_idx = jnp.array(0, dtype=jnp.result_type(int))
        return HMCAdaptState(
            step_size,
            inverse_mass_matrix,
            mass_matrix_sqrt,
            mass_matrix_sqrt_inv,
            ss_state,
            mm_state,
            window_idx,
            rng_key,
        )

    def _update_at_window_end(z_info, rng_key_ss, state):
        (
            step_size,
            inverse_mass_matrix,
            mass_matrix_sqrt,
            mass_matrix_sqrt_inv,
            ss_state,
            mm_state,
            window_idx,
            rng_key,
        ) = state

        if adapt_mass_matrix:
            inverse_mass_matrix, mass_matrix_sqrt, mass_matrix_sqrt_inv = mm_final(
                mm_state, regularize=regularize_mass_matrix
            )
            if isinstance(inverse_mass_matrix, dict):
                size = {k: v.shape for k, v in inverse_mass_matrix.items()}
            else:
                size = inverse_mass_matrix.shape[-1]
            mm_state = mm_init(size)

        if adapt_step_size:
            step_size = find_reasonable_step_size(
                step_size, inverse_mass_matrix, z_info, rng_key_ss
            )
            # NB: when step_size is large, say 1e38, jnp.log(10 * step_size) will be inf
            # and jnp.log(10) + jnp.log(step_size) will be finite
            ss_state = ss_init(jnp.log(10) + jnp.log(step_size))

        return HMCAdaptState(
            step_size,
            inverse_mass_matrix,
            mass_matrix_sqrt,
            mass_matrix_sqrt_inv,
            ss_state,
            mm_state,
            window_idx,
            rng_key,
        )

    def update_fn(t, accept_prob, z_info, state):
        """
        :param int t: The current time step.
        :param float accept_prob: Acceptance probability of the current trajectory.
        :param IntegratorState z_info: The new integrator state.
        :param state: Current state of the adapt scheme.
        :return: new state of the adapt scheme.
        """
        (
            step_size,
            inverse_mass_matrix,
            mass_matrix_sqrt,
            mass_matrix_sqrt_inv,
            ss_state,
            mm_state,
            window_idx,
            rng_key,
        ) = state
        if rng_key is not None:
            rng_key, rng_key_ss = random.split(rng_key)
        else:
            rng_key_ss = None

        # update step size state
        if adapt_step_size:
            ss_state = ss_update(target_accept_prob - accept_prob, ss_state)
            # note: at the end of warmup phase, use average of log step_size
            log_step_size, log_step_size_avg, *_ = ss_state
            step_size = jnp.where(
                t == (num_adapt_steps - 1),
                jnp.exp(log_step_size_avg),
                jnp.exp(log_step_size),
            )
            # account the the case log_step_size is an extreme number
            finfo = jnp.finfo(jnp.result_type(step_size))
            step_size = jnp.clip(step_size, a_min=finfo.tiny, a_max=finfo.max)

        # update mass matrix state
        is_middle_window = (0 < window_idx) & (window_idx < (num_windows - 1))
        if adapt_mass_matrix:
            z = z_info[0]
            mm_state = cond(
                is_middle_window,
                (z, mm_state),
                lambda args: mm_update(*args),
                mm_state,
                identity,
            )

        t_at_window_end = t == adaptation_schedule[window_idx, 1]
        window_idx = jnp.where(t_at_window_end, window_idx + 1, window_idx)
        state = HMCAdaptState(
            step_size,
            inverse_mass_matrix,
            mass_matrix_sqrt,
            mass_matrix_sqrt_inv,
            ss_state,
            mm_state,
            window_idx,
            rng_key,
        )
        state = cond(
            t_at_window_end & is_middle_window,
            (z_info, rng_key_ss, state),
            lambda args: _update_at_window_end(*args),
            state,
            identity,
        )
        return state

    return init_fn, update_fn


def _momentum_angle(inverse_mass_matrix, r_left, r_right, r_sum):
    if isinstance(inverse_mass_matrix, dict):
        left_angle, right_angle = jnp.zeros(()), jnp.zeros(())
        for site_names, inverse_mm in inverse_mass_matrix.items():
            r_left_b = tuple(r_left[k] for k in site_names)
            r_right_b = tuple(r_right[k] for k in site_names)
            r_sum_b = tuple(r_sum[k] for k in site_names)
            left_a, right_a = _momentum_angle(inverse_mm, r_left_b, r_right_b, r_sum_b)
            left_angle = left_angle + left_a
            right_angle = right_angle + right_a
        return left_angle, right_angle

    r_left, _ = ravel_pytree(r_left)
    r_right, _ = ravel_pytree(r_right)
    r_sum, _ = ravel_pytree(r_sum)

    if inverse_mass_matrix.ndim == 2:
        v_left = jnp.matmul(inverse_mass_matrix, r_left)
        v_right = jnp.matmul(inverse_mass_matrix, r_right)
    elif inverse_mass_matrix.ndim == 1:
        v_left = jnp.multiply(inverse_mass_matrix, r_left)
        v_right = jnp.multiply(inverse_mass_matrix, r_right)
    else:
        raise ValueError("inverse_mass_matrix should have 1 or 2 dimensions.")

    # This implements dynamic termination criterion (ref [2], section A.4.2).
    r_sum = r_sum - (r_left + r_right) / 2
    return jnp.dot(v_left, r_sum), jnp.dot(v_right, r_sum)


def _is_turning(inverse_mass_matrix, r_left, r_right, r_sum):
    left_angle, right_angle = _momentum_angle(
        inverse_mass_matrix, r_left, r_right, r_sum
    )
    turning_at_left = left_angle <= 0
    turning_at_right = right_angle <= 0
    return turning_at_left | turning_at_right


def _uniform_transition_kernel(current_tree, new_tree):
    # This function computes transition prob for subtrees (ref [2], section A.3.1).
    # e^new_weight / (e^new_weight + e^current_weight)
    transition_prob = expit(new_tree.weight - current_tree.weight)
    return transition_prob


def _biased_transition_kernel(current_tree, new_tree):
    # This function computes transition prob for main trees (ref [2], section A.3.2).
    transition_prob = jnp.exp(new_tree.weight - current_tree.weight)
    # If new tree is turning or diverging, we won't move the proposal
    # to the new tree.
    transition_prob = jnp.where(
        new_tree.turning | new_tree.diverging, 0.0, jnp.clip(transition_prob, a_max=1.0)
    )
    return transition_prob


def _combine_tree(
    current_tree, new_tree, inverse_mass_matrix, going_right, rng_key, biased_transition
):
    # Now we combine the current tree and the new tree. Note that outside
    # leaves of the combined tree are determined by the direction.
    z_left, r_left, z_left_grad, z_right, r_right, r_right_grad = cond(
        going_right,
        (current_tree, new_tree),
        lambda trees: (
            trees[0].z_left,
            trees[0].r_left,
            trees[0].z_left_grad,
            trees[1].z_right,
            trees[1].r_right,
            trees[1].z_right_grad,
        ),
        (new_tree, current_tree),
        lambda trees: (
            trees[0].z_left,
            trees[0].r_left,
            trees[0].z_left_grad,
            trees[1].z_right,
            trees[1].r_right,
            trees[1].z_right_grad,
        ),
    )
    r_sum = tree_multimap(jnp.add, current_tree.r_sum, new_tree.r_sum)

    if biased_transition:
        transition_prob = _biased_transition_kernel(current_tree, new_tree)
        turning = new_tree.turning | _is_turning(
            inverse_mass_matrix, r_left, r_right, r_sum
        )
    else:
        transition_prob = _uniform_transition_kernel(current_tree, new_tree)
        turning = current_tree.turning

    transition = random.bernoulli(rng_key, transition_prob)
    z_proposal, z_proposal_pe, z_proposal_grad, z_proposal_energy = cond(
        transition,
        new_tree,
        lambda tree: (
            tree.z_proposal,
            tree.z_proposal_pe,
            tree.z_proposal_grad,
            tree.z_proposal_energy,
        ),
        current_tree,
        lambda tree: (
            tree.z_proposal,
            tree.z_proposal_pe,
            tree.z_proposal_grad,
            tree.z_proposal_energy,
        ),
    )

    tree_depth = current_tree.depth + 1
    tree_weight = jnp.logaddexp(current_tree.weight, new_tree.weight)
    diverging = new_tree.diverging

    sum_accept_probs = current_tree.sum_accept_probs + new_tree.sum_accept_probs
    num_proposals = current_tree.num_proposals + new_tree.num_proposals

    return TreeInfo(
        z_left,
        r_left,
        z_left_grad,
        z_right,
        r_right,
        r_right_grad,
        z_proposal,
        z_proposal_pe,
        z_proposal_grad,
        z_proposal_energy,
        tree_depth,
        tree_weight,
        r_sum,
        turning,
        diverging,
        sum_accept_probs,
        num_proposals,
    )


def _build_basetree(
    vv_update,
    kinetic_fn,
    z,
    r,
    z_grad,
    inverse_mass_matrix,
    step_size,
    going_right,
    energy_current,
    max_delta_energy,
):
    step_size = jnp.where(going_right, step_size, -step_size)
    z_new, r_new, potential_energy_new, z_new_grad = vv_update(
        step_size, inverse_mass_matrix, (z, r, energy_current, z_grad)
    )

    energy_new = potential_energy_new + kinetic_fn(inverse_mass_matrix, r_new)
    delta_energy = energy_new - energy_current
    # Handles the NaN case.
    delta_energy = jnp.where(jnp.isnan(delta_energy), jnp.inf, delta_energy)
    tree_weight = -delta_energy

    diverging = delta_energy > max_delta_energy
    accept_prob = jnp.clip(jnp.exp(-delta_energy), a_max=1.0)
    return TreeInfo(
        z_new,
        r_new,
        z_new_grad,
        z_new,
        r_new,
        z_new_grad,
        z_new,
        potential_energy_new,
        z_new_grad,
        energy_new,
        depth=0,
        weight=tree_weight,
        r_sum=r_new,
        turning=False,
        diverging=diverging,
        sum_accept_probs=accept_prob,
        num_proposals=1,
    )


def _get_leaf(tree, going_right):
    return cond(
        going_right,
        tree,
        lambda tree: (tree.z_right, tree.r_right, tree.z_right_grad),
        tree,
        lambda tree: (tree.z_left, tree.r_left, tree.z_left_grad),
    )


def _double_tree(
    current_tree,
    vv_update,
    kinetic_fn,
    inverse_mass_matrix,
    step_size,
    going_right,
    rng_key,
    energy_current,
    max_delta_energy,
    r_ckpts,
    r_sum_ckpts,
):
    key, transition_key = random.split(rng_key)

    new_tree = _iterative_build_subtree(
        current_tree,
        vv_update,
        kinetic_fn,
        inverse_mass_matrix,
        step_size,
        going_right,
        key,
        energy_current,
        max_delta_energy,
        r_ckpts,
        r_sum_ckpts,
    )

    return _combine_tree(
        current_tree, new_tree, inverse_mass_matrix, going_right, transition_key, True
    )


def _leaf_idx_to_ckpt_idxs(n):
    # computes the number of non-zero bits except the last bit
    # e.g. 6 -> 2, 7 -> 2, 13 -> 2
    _, idx_max = while_loop(
        lambda nc: nc[0] > 0, lambda nc: (nc[0] >> 1, nc[1] + (nc[0] & 1)), (n >> 1, 0)
    )
    # computes the number of contiguous last non-zero bits
    # e.g. 6 -> 0, 7 -> 3, 13 -> 1
    _, num_subtrees = while_loop(
        lambda nc: (nc[0] & 1) != 0, lambda nc: (nc[0] >> 1, nc[1] + 1), (n, 0)
    )
    # TODO: explore the potential of setting idx_min=0 to allow more turning checks
    # It will be useful in case: e.g. assume a tree 0 -> 7 is a circle,
    # subtrees 0 -> 3, 4 -> 7 are half-circles, which two leaves might not
    # satisfy turning condition;
    # the full tree 0 -> 7 is a circle, which two leaves might also not satisfy
    # turning condition;
    # however, we can check the turning condition of the subtree 0 -> 5, which
    # likely satisfies turning condition because its trajectory 3/4 of a circle.
    # XXX: make sure that detailed balance is satisfied if we follow this direction
    idx_min = idx_max - num_subtrees + 1
    return idx_min, idx_max


def _is_iterative_turning(
    inverse_mass_matrix,
    r,
    r_sum,
    r_ckpts,
    r_sum_ckpts,
    idx_min,
    idx_max,
    unravel_fn=identity,
):
    def _body_fn(state):
        i, _ = state
        subtree_r_sum = r_sum - r_sum_ckpts[i] + r_ckpts[i]
        subtree_r_sum = unravel_fn(subtree_r_sum)
        r_left = unravel_fn(r_ckpts[i])
        return i - 1, _is_turning(inverse_mass_matrix, r_left, r, subtree_r_sum)

    _, turning = while_loop(
        lambda it: (it[0] >= idx_min) & ~it[1], _body_fn, (idx_max, False)
    )
    return turning


def _iterative_build_subtree(
    prototype_tree,
    vv_update,
    kinetic_fn,
    inverse_mass_matrix,
    step_size,
    going_right,
    rng_key,
    energy_current,
    max_delta_energy,
    r_ckpts,
    r_sum_ckpts,
):
    max_num_proposals = 2 ** prototype_tree.depth

    def _cond_fn(state):
        tree, turning, _, _, _ = state
        return (tree.num_proposals < max_num_proposals) & ~turning & ~tree.diverging

    def _body_fn(state):
        current_tree, _, r_ckpts, r_sum_ckpts, rng_key = state
        rng_key, transition_rng_key = random.split(rng_key)
        # If we are going to the right, start from the right leaf of the current tree.
        z, r, z_grad = _get_leaf(current_tree, going_right)
        new_leaf = _build_basetree(
            vv_update,
            kinetic_fn,
            z,
            r,
            z_grad,
            inverse_mass_matrix,
            step_size,
            going_right,
            energy_current,
            max_delta_energy,
        )
        new_tree = cond(
            current_tree.num_proposals == 0,
            new_leaf,
            identity,
            (
                current_tree,
                new_leaf,
                inverse_mass_matrix,
                going_right,
                transition_rng_key,
            ),
            lambda x: _combine_tree(*x, False),
        )

        leaf_idx = current_tree.num_proposals
        # NB: in the special case leaf_idx=0, ckpt_idx_min=1 and ckpt_idx_max=0,
        # the following logic is still valid for that case
        ckpt_idx_min, ckpt_idx_max = _leaf_idx_to_ckpt_idxs(leaf_idx)
        r, unravel_fn = ravel_pytree(new_leaf.r_right)
        r_sum, _ = ravel_pytree(new_tree.r_sum)
        # we update checkpoints when leaf_idx is even
        r_ckpts, r_sum_ckpts = cond(
            leaf_idx % 2 == 0,
            (r_ckpts, r_sum_ckpts),
            lambda x: (
                index_update(x[0], ckpt_idx_max, r),
                index_update(x[1], ckpt_idx_max, r_sum),
            ),
            (r_ckpts, r_sum_ckpts),
            identity,
        )

        turning = _is_iterative_turning(
            inverse_mass_matrix,
            new_leaf.r_right,
            r_sum,
            r_ckpts,
            r_sum_ckpts,
            ckpt_idx_min,
            ckpt_idx_max,
            unravel_fn,
        )
        return new_tree, turning, r_ckpts, r_sum_ckpts, rng_key

    basetree = prototype_tree._replace(num_proposals=0)

    tree, turning, _, _, _ = while_loop(
        _cond_fn, _body_fn, (basetree, False, r_ckpts, r_sum_ckpts, rng_key)
    )
    # update depth and turning condition
    return TreeInfo(
        tree.z_left,
        tree.r_left,
        tree.z_left_grad,
        tree.z_right,
        tree.r_right,
        tree.z_right_grad,
        tree.z_proposal,
        tree.z_proposal_pe,
        tree.z_proposal_grad,
        tree.z_proposal_energy,
        prototype_tree.depth,
        tree.weight,
        tree.r_sum,
        turning,
        tree.diverging,
        tree.sum_accept_probs,
        tree.num_proposals,
    )


def build_tree(
    verlet_update,
    kinetic_fn,
    verlet_state,
    inverse_mass_matrix,
    step_size,
    rng_key,
    max_delta_energy=1000.0,
    max_tree_depth=10,
):
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
    :param jax.random.PRNGKey rng_key: random key to be used as the source of
        randomness.
    :param float max_delta_energy: A threshold to decide if the new state diverges
        (based on the energy difference) too much from the initial integrator state.
    :param int max_tree_depth: Max depth of the binary tree created during the doubling
        scheme of NUTS sampler. Defaults to 10. This argument also accepts a tuple of
        integers `(d1, d2)`, where `d1` is the max tree depth at the current MCMC
        step and `d2` is the global max tree depth for all MCMC steps.
    :return: information of the tree.
    :rtype: :data:`TreeInfo`
    """
    if isinstance(max_tree_depth, tuple):
        max_tree_depth_current, max_tree_depth = max_tree_depth
    else:
        max_tree_depth_current = max_tree_depth
    z, r, potential_energy, z_grad = verlet_state
    energy_current = potential_energy + kinetic_fn(inverse_mass_matrix, r)
    latent_size = jnp.size(ravel_pytree(r)[0])
    r_ckpts = jnp.zeros((max_tree_depth, latent_size))
    r_sum_ckpts = jnp.zeros((max_tree_depth, latent_size))

    tree = TreeInfo(
        z,
        r,
        z_grad,
        z,
        r,
        z_grad,
        z,
        potential_energy,
        z_grad,
        energy_current,
        depth=0,
        weight=jnp.zeros(()),
        r_sum=r,
        turning=jnp.array(False),
        diverging=jnp.array(False),
        sum_accept_probs=jnp.zeros(()),
        num_proposals=jnp.array(0, dtype=jnp.result_type(int)),
    )

    def _cond_fn(state):
        tree, _ = state
        return (tree.depth < max_tree_depth_current) & ~tree.turning & ~tree.diverging

    def _body_fn(state):
        tree, key = state
        key, direction_key, doubling_key = random.split(key, 3)
        going_right = random.bernoulli(direction_key)
        tree = _double_tree(
            tree,
            verlet_update,
            kinetic_fn,
            inverse_mass_matrix,
            step_size,
            going_right,
            doubling_key,
            energy_current,
            max_delta_energy,
            r_ckpts,
            r_sum_ckpts,
        )
        return tree, key

    state = (tree, rng_key)
    tree, _ = while_loop(_cond_fn, _body_fn, state)
    return tree


def euclidean_kinetic_energy(inverse_mass_matrix, r):
    if isinstance(inverse_mass_matrix, dict):
        ke = jnp.zeros(())
        for site_names, inverse_mm in inverse_mass_matrix.items():
            r_block = tuple(r[k] for k in site_names)
            ke = ke + euclidean_kinetic_energy(inverse_mm, r_block)
        return ke

    r, _ = ravel_pytree(r)

    if inverse_mass_matrix.ndim == 2:
        v = jnp.matmul(inverse_mass_matrix, r)
    elif inverse_mass_matrix.ndim == 1:
        v = jnp.multiply(inverse_mass_matrix, r)
    else:
        raise ValueError("inverse_mass_matrix should have 1 or 2 dimensions.")

    return 0.5 * jnp.dot(v, r)


def _euclidean_kinetic_energy_grad(inverse_mass_matrix, r):
    if isinstance(inverse_mass_matrix, dict):
        r_grad = {}
        for site_names, inverse_mm in inverse_mass_matrix.items():
            r_block = OrderedDict([(k, r[k]) for k in site_names])
            r_grad.update(_euclidean_kinetic_energy_grad(inverse_mm, r_block))
        return r_grad

    r, unravel_fn = ravel_pytree(r)

    if inverse_mass_matrix.ndim == 2:
        v = jnp.matmul(inverse_mass_matrix, r)
    elif inverse_mass_matrix.ndim == 1:
        v = jnp.multiply(inverse_mass_matrix, r)
    else:
        raise ValueError("inverse_mass_matrix should have 1 or 2 dimensions.")

    return unravel_fn(v)


euclidean_kinetic_energy._kinetic_grad = _euclidean_kinetic_energy_grad


def consensus(subposteriors, num_draws=None, diagonal=False, rng_key=None):
    """
    Merges subposteriors following consensus Monte Carlo algorithm.

    **References:**

    1. *Bayes and big data: The consensus Monte Carlo algorithm*,
       Steven L. Scott, Alexander W. Blocker, Fernando V. Bonassi, Hugh A. Chipman,
       Edward I. George, Robert E. McCulloch

    :param list subposteriors: a list in which each element is a collection of samples.
    :param int num_draws: number of draws from the merged posterior.
    :param bool diagonal: whether to compute weights using variance or covariance, defaults to
        `False` (using covariance).
    :param jax.random.PRNGKey rng_key: source of the randomness, defaults to `jax.random.PRNGKey(0)`.
    :return: if `num_draws` is None, merges subposteriors without resampling; otherwise, returns
        a collection of `num_draws` samples with the same data structure as each subposterior.
    """
    # stack subposteriors
    joined_subposteriors = tree_multimap(lambda *args: jnp.stack(args), *subposteriors)
    # shape of joined_subposteriors: n_subs x n_samples x sample_shape
    joined_subposteriors = vmap(vmap(lambda sample: ravel_pytree(sample)[0]))(
        joined_subposteriors
    )

    if num_draws is not None:
        rng_key = random.PRNGKey(0) if rng_key is None else rng_key
        # randomly gets num_draws from subposteriors
        n_subs = len(subposteriors)
        n_samples = tree_flatten(subposteriors[0])[0][0].shape[0]
        # shape of draw_idxs: n_subs x num_draws x sample_shape
        draw_idxs = random.randint(
            rng_key, shape=(n_subs, num_draws), minval=0, maxval=n_samples
        )
        joined_subposteriors = vmap(lambda x, idx: x[idx])(
            joined_subposteriors, draw_idxs
        )

    if diagonal:
        # compute weights for each subposterior (ref: Section 3.1 of [1])
        weights = vmap(lambda x: 1 / jnp.var(x, ddof=1, axis=0))(joined_subposteriors)
        normalized_weights = weights / jnp.sum(weights, axis=0)
        # get weighted samples
        samples_flat = jnp.einsum(
            "ij,ikj->kj", normalized_weights, joined_subposteriors
        )
    else:
        weights = vmap(lambda x: jnp.linalg.inv(jnp.cov(x.T)))(joined_subposteriors)
        normalized_weights = jnp.matmul(
            jnp.linalg.inv(jnp.sum(weights, axis=0)), weights
        )
        samples_flat = jnp.einsum(
            "ijk,ilk->lj", normalized_weights, joined_subposteriors
        )

    # unravel_fn acts on 1 sample of a subposterior
    _, unravel_fn = ravel_pytree(tree_map(lambda x: x[0], subposteriors[0]))
    return vmap(lambda x: unravel_fn(x))(samples_flat)


def parametric(subposteriors, diagonal=False):
    """
    Merges subposteriors following (embarrassingly parallel) parametric Monte Carlo algorithm.

    **References:**

    1. *Asymptotically Exact, Embarrassingly Parallel MCMC*,
       Willie Neiswanger, Chong Wang, Eric Xing

    :param list subposteriors: a list in which each element is a collection of samples.
    :param bool diagonal: whether to compute weights using variance or covariance, defaults to
        `False` (using covariance).
    :return: the estimated mean and variance/covariance parameters of the joined posterior
    """
    joined_subposteriors = tree_multimap(lambda *args: jnp.stack(args), *subposteriors)
    joined_subposteriors = vmap(vmap(lambda sample: ravel_pytree(sample)[0]))(
        joined_subposteriors
    )

    submeans = jnp.mean(joined_subposteriors, axis=1)
    if diagonal:
        weights = vmap(lambda x: 1 / jnp.var(x, ddof=1, axis=0))(joined_subposteriors)
        var = 1 / jnp.sum(weights, axis=0)
        normalized_weights = var * weights

        # comparing to consensus implementation, we compute weighted mean here
        mean = jnp.einsum("ij,ij->j", normalized_weights, submeans)
        return mean, var
    else:
        weights = vmap(lambda x: jnp.linalg.inv(jnp.cov(x.T)))(joined_subposteriors)
        cov = jnp.linalg.inv(jnp.sum(weights, axis=0))
        normalized_weights = jnp.matmul(cov, weights)

        # comparing to consensus implementation, we compute weighted mean here
        mean = jnp.einsum("ijk,ik->j", normalized_weights, submeans)
        return mean, cov


def parametric_draws(subposteriors, num_draws, diagonal=False, rng_key=None):
    """
    Merges subposteriors following (embarrassingly parallel) parametric Monte Carlo algorithm.

    **References:**

    1. *Asymptotically Exact, Embarrassingly Parallel MCMC*,
       Willie Neiswanger, Chong Wang, Eric Xing

    :param list subposteriors: a list in which each element is a collection of samples.
    :param int num_draws: number of draws from the merged posterior.
    :param bool diagonal: whether to compute weights using variance or covariance, defaults to
        `False` (using covariance).
    :param jax.random.PRNGKey rng_key: source of the randomness, defaults to `jax.random.PRNGKey(0)`.
    :return: a collection of `num_draws` samples with the same data structure as each subposterior.
    """
    rng_key = random.PRNGKey(0) if rng_key is None else rng_key
    if diagonal:
        mean, var = parametric(subposteriors, diagonal=True)
        samples_flat = dist.Normal(mean, jnp.sqrt(var)).sample(rng_key, (num_draws,))
    else:
        mean, cov = parametric(subposteriors, diagonal=False)
        samples_flat = dist.MultivariateNormal(mean, cov).sample(rng_key, (num_draws,))

    _, unravel_fn = ravel_pytree(tree_map(lambda x: x[0], subposteriors[0]))
    return vmap(lambda x: unravel_fn(x))(samples_flat)
