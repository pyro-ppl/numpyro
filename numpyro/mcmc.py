import math

import jax.numpy as np
from jax import partial, random
from jax.flatten_util import ravel_pytree
from jax.random import PRNGKey

import numpyro.distributions as dist
from numpyro.hmc_util import IntegratorState, find_reasonable_step_size, velocity_verlet, warmup_adapter
from numpyro.util import cond, fori_loop, laxtuple

HMCState = laxtuple('HMCState', ['z', 'z_grad', 'potential_energy', 'num_steps',
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


def hmc_kernel(potential_fn, kinetic_fn):
    vv_init, vv_update = velocity_verlet(potential_fn, kinetic_fn)
    momentum_generator = None

    def init_kernel(init_samples,
                    num_warmup_steps,
                    step_size=1.0,
                    num_steps=None,
                    adapt_step_size=True,
                    adapt_mass_matrix=True,
                    diag_mass=True,
                    target_accept_prob=0.8,
                    rng=PRNGKey(0)):
        assert isinstance(step_size, float)
        if num_steps is None:
            trajectory_length = 2 * math.pi
        else:
            trajectory_length = num_steps * step_size

        z = init_samples
        z_flat, unravel_fn = ravel_pytree(z)
        nonlocal momentum_generator
        momentum_generator = partial(_sample_momentum, unravel_fn)

        find_reasonable_ss = partial(find_reasonable_step_size,
                                     potential_fn, kinetic_fn, momentum_generator)

        rng, rng_wa, rng_momentum = random.split(rng, 3)
        wa_init, wa_update = warmup_adapter(num_warmup_steps,
                                            find_reasonable_step_size=find_reasonable_ss,
                                            adapt_step_size=adapt_step_size,
                                            adapt_mass_matrix=adapt_mass_matrix,
                                            diag_mass=diag_mass,
                                            target_accept_prob=target_accept_prob)

        wa_state = wa_init(z, rng_wa, mass_matrix_size=np.size(z_flat))
        r = momentum_generator(wa_state.inverse_mass_matrix, rng_momentum)
        vv_state = vv_init(z, r)

        def body_fn(t, args):
            vv_state, wa_state, rng = args
            rng, rng_momentum, rng_transition, rng_wa = random.split(rng, 4)
            r = momentum_generator(wa_state.inverse_mass_matrix, rng_momentum)
            vv_state = vv_state.update(r=r)
            num_steps = _get_num_steps(wa_state.step_size, trajectory_length)
            accept_prob, vv_state_new = _next(num_steps, wa_state.step_size,
                                              wa_state.inverse_mass_matrix, vv_state)
            transition = random.bernoulli(rng_transition, accept_prob)
            vv_state = cond(transition,
                            vv_state_new, lambda state: state,
                            vv_state, lambda state: state)
            wa_state = wa_update(t, accept_prob, z, rng_wa, wa_state)
            return vv_state, wa_state, rng

        vv_state, wa_state, rng = fori_loop(0, num_warmup_steps, body_fn, (vv_state, wa_state, rng))
        num_steps = _get_num_steps(wa_state.step_size, trajectory_length)
        return HMCState(vv_state.z, vv_state.z_grad, vv_state.potential_energy, num_steps,
                        wa_state.step_size, wa_state.inverse_mass_matrix, rng)

    def _next(num_steps, step_size, inverse_mass_matrix, vv_state):
        vv_state_new = fori_loop(0, num_steps,
                                 lambda i, val: vv_update(step_size, inverse_mass_matrix, val),
                                 vv_state)
        energy_old = vv_state.potential_energy + kinetic_fn(vv_state.r, inverse_mass_matrix)
        energy_new = vv_state_new.potential_energy + kinetic_fn(vv_state_new.r, inverse_mass_matrix)
        delta_energy = energy_new - energy_old
        delta_energy = np.where(np.isnan(delta_energy), np.inf, delta_energy)
        accept_prob = np.clip(np.exp(-delta_energy), a_max=1.0)
        return accept_prob, vv_state_new

    def sample_kernel(hmc_state, i):
        rng, rng_momentum, rng_transition = random.split(hmc_state.rng, 3)
        r = momentum_generator(hmc_state.inverse_mass_matrix, rng_momentum)
        vv_state = IntegratorState(hmc_state.z, r, hmc_state.potential_energy, hmc_state.z_grad)
        accept_prob, vv_state_new = _next(hmc_state.num_steps, hmc_state.step_size,
                                          hmc_state.inverse_mass_matrix, vv_state)
        transition = random.bernoulli(rng_transition, accept_prob)
        vv_state = cond(transition,
                        vv_state_new, lambda state: state,
                        vv_state, lambda state: state)
        return HMCState(vv_state.z, vv_state.z_grad, vv_state.potential_energy, hmc_state.num_steps,
                        hmc_state.step_size, hmc_state.inverse_mass_matrix, rng)

    return init_kernel, sample_kernel
