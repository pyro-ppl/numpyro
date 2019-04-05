import math

import jax.numpy as np
from jax import partial, random
from jax.flatten_util import ravel_pytree
from jax.random import PRNGKey

import numpyro.distributions as dist
from numpyro.hmc_util import (
    IntegratorState,
    build_tree,
    find_reasonable_step_size,
    velocity_verlet,
    warmup_adapter
)
from numpyro.util import cond, fori_loop, laxtuple

HMCState = laxtuple('HMCState', ['z', 'z_grad', 'potential_energy', 'num_steps', 'accept_prob',
                                 'step_size', 'inverse_mass_matrix', 'rng'])


def _get_num_steps(step_size, trajectory_length):
    num_steps = np.array(trajectory_length / step_size, dtype=np.int32)
    return np.where(num_steps < 1, np.array(1, dtype=np.int32), num_steps)


def _sample_momentum(unpack_fn, inverse_mass_matrix, rng):
    if inverse_mass_matrix.ndim == 1:
        r = dist.norm(0., np.sqrt(np.reciprocal(inverse_mass_matrix))).rvs(random_state=rng)
        return unpack_fn(r)
    elif inverse_mass_matrix.ndim == 2:
        raise NotImplementedError


def _euclidean_ke(inverse_mass_matrix, r):
    r, _ = ravel_pytree(r)

    if inverse_mass_matrix.ndim == 2:
        v = np.matmul(inverse_mass_matrix, r)
    elif inverse_mass_matrix.ndim == 1:
        v = np.multiply(inverse_mass_matrix, r)

    return 0.5 * np.dot(v, r)


def hmc_kernel(potential_fn, kinetic_fn=None, algo='NUTS'):
    if kinetic_fn is None:
        kinetic_fn = _euclidean_ke
    vv_init, vv_update = velocity_verlet(potential_fn, kinetic_fn)
    trajectory_length = None
    momentum_generator = None
    wa_update = None

    def init_kernel(init_samples,
                    num_warmup_steps,
                    step_size=1.0,
                    num_steps=None,
                    adapt_step_size=True,
                    adapt_mass_matrix=True,
                    diag_mass=True,
                    target_accept_prob=0.8,
                    run_warmup=True,
                    rng=PRNGKey(0)):
        step_size = float(step_size)
        nonlocal trajectory_length, momentum_generator, wa_update

        if num_steps is None:
            trajectory_length = 2 * math.pi
        else:
            trajectory_length = num_steps * step_size

        z = init_samples
        z_flat, unravel_fn = ravel_pytree(z)
        momentum_generator = partial(_sample_momentum, unravel_fn)

        find_reasonable_ss = partial(find_reasonable_step_size,
                                     potential_fn, kinetic_fn, momentum_generator)

        wa_init, wa_update = warmup_adapter(num_warmup_steps,
                                            find_reasonable_step_size=find_reasonable_ss,
                                            adapt_step_size=adapt_step_size,
                                            adapt_mass_matrix=adapt_mass_matrix,
                                            diag_mass=diag_mass,
                                            target_accept_prob=target_accept_prob)

        rng_hmc, rng_wa = random.split(rng)
        wa_state = wa_init(z, rng_wa, mass_matrix_size=np.size(z_flat))
        r = momentum_generator(wa_state.inverse_mass_matrix, rng)
        vv_state = vv_init(z, r)
        hmc_state = HMCState(vv_state.z, vv_state.z_grad, vv_state.potential_energy, 0, 0.,
                             wa_state.step_size, wa_state.inverse_mass_matrix, rng_hmc)

        if run_warmup:
            hmc_state, _ = fori_loop(0, num_warmup_steps, warmup_update, (hmc_state, wa_state))
            return hmc_state
        else:
            return hmc_state, wa_state, warmup_update

    def warmup_update(t, states):
        hmc_state, wa_state = states
        hmc_state = sample_kernel(hmc_state)
        wa_state = wa_update(t, hmc_state.accept_prob, hmc_state.z, wa_state)
        hmc_state = hmc_state.update(step_size=wa_state.step_size,
                                     inverse_mass_matrix=wa_state.inverse_mass_matrix)
        return hmc_state, wa_state

    def _hmc_next(step_size, inverse_mass_matrix, vv_state, rng):
        num_steps = _get_num_steps(step_size, trajectory_length)
        vv_state_new = fori_loop(0, num_steps,
                                 lambda i, val: vv_update(step_size, inverse_mass_matrix, val),
                                 vv_state)
        energy_old = vv_state.potential_energy + kinetic_fn(inverse_mass_matrix, vv_state.r)
        energy_new = vv_state_new.potential_energy + kinetic_fn(inverse_mass_matrix, vv_state_new.r)
        delta_energy = energy_new - energy_old
        delta_energy = np.where(np.isnan(delta_energy), np.inf, delta_energy)
        accept_prob = np.clip(np.exp(-delta_energy), a_max=1.0)
        transition = random.bernoulli(rng, accept_prob)
        vv_state = cond(transition,
                        vv_state_new, lambda state: state,
                        vv_state, lambda state: state)
        return vv_state, num_steps, accept_prob

    def _nuts_next(step_size, inverse_mass_matrix, vv_state, rng):
        binary_tree = build_tree(vv_update, kinetic_fn, vv_state,
                                 inverse_mass_matrix, step_size, rng)
        accept_prob = binary_tree.sum_accept_probs / binary_tree.num_proposals
        num_steps = binary_tree.num_proposals
        vv_state = vv_state.update(z=binary_tree.z_proposal,
                                   potential_energy=binary_tree.z_proposal_pe,
                                   z_grad=binary_tree.z_proposal_grad)
        return vv_state, num_steps, accept_prob

    _next = _nuts_next if algo == 'NUTS' else _hmc_next

    def sample_kernel(hmc_state):
        rng, rng_momentum, rng_transition = random.split(hmc_state.rng, 3)
        r = momentum_generator(hmc_state.inverse_mass_matrix, rng_momentum)
        vv_state = IntegratorState(hmc_state.z, r, hmc_state.potential_energy, hmc_state.z_grad)
        vv_state, num_steps, accept_prob = _next(hmc_state.step_size,
                                                 hmc_state.inverse_mass_matrix,
                                                 vv_state, rng_transition)
        return HMCState(vv_state.z, vv_state.z_grad, vv_state.potential_energy, num_steps,
                        accept_prob, hmc_state.step_size, hmc_state.inverse_mass_matrix, rng)

    return init_kernel, sample_kernel
