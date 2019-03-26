import math

from jax.flatten_util import ravel_pytree
import jax.numpy as np
from jax import lax, random
from numpyro.hmc_util import warmup_adapter, build_adaptation_schedule, velocity_verlet
import numpyro.distributions as dist


def _get_num_steps(step_size, trajectory_length):
    return max(1, int(trajectory_length / step_size))


def _sample_momentum(inv_mass_matrix, rng):
    rng, = random.split(rng)
    if inv_mass_matrix.ndim == 1:
        return dist.norm(0., np.sqrt(np.reciprocal(inv_mass_matrix))), rng
    elif inv_mass_matrix.ndim == 2:
        return dist.mvn(0., np.sqrt()), rng


def hmc(potential_fn, kinetic_fn):
    vv_init, vv_update = velocity_verlet(potential_fn, kinetic_fn)

    def init_kernel(init_samples,
                    num_warmup_steps,
                    step_size=1,
                    num_steps=None,
                    adapt_step_size=True,
                    adapt_mass_matrix=True,
                    diag_mass=True,
                    target_accept_prob=0.8):
        wa_init, wa_update = warmup_adapter(num_steps,
                                            adapt_step_size=adapt_step_size,
                                            adapt_mass_matrix=adapt_mass_matrix,
                                            diag_mass=diag_mass,
                                            target_accept_prob=target_accept_prob)
        if num_steps is None:
            trajectory_length = 2 * math.pi
            num_steps = _get_num_steps(trajectory_length)
        else:
            trajectory_length = num_steps * step_size
        num_sites = np.numel(ravel_pytree(init_samples)[0])
        step_size, inverse_mass_matrix, ss_state, mm_state, window_idx = wa_init(
            step_size, num_sites)
        vv_state = vv_init()

        def _body_fn(t, val):
            rng, accept_prob, wa_state, vv_state = val
            step_size, inv_mass_matrix, _, _, _ = wa_state
            r, rng = _sample_momentum(inv_mass_matrix, rng)
            num_steps = _get_num_steps(step_size, trajectory_length)
            accept_prob, vv_state = _next(num_steps, step_size, inv_mass_matrix, vv_state)
            z, _, pe, z_grad = vv_state
            wa_state = wa_update(t, accept_prob, z, wa_state)

            return accept_prob, wa_state

        lax.fori_loop(0, num_warmup_steps, )
        z, _, _, _ = _next(vv_state, num_steps, step_size)
        return

    def _next(num_steps, step_size, inv_mass_matrix, vv_state):
        z_i, r_i, pe_i, _ = vv_state
        z_f, r_f, pe_f, z_grad = lax.fori_loop(0, num_steps,
                                               lambda i, val: vv_update(step_size, inv_mass_matrix, val),
                                               vv_state)
        energy_old = pe_i + kinetic_fn(r_i, inv_mass_matrix)
        energy_new = pe_f + kinetic_fn(r_f, inv_mass_matrix)
        delta_energy = energy_new - energy_old
        np.where(np.isnan(delta_energy), np.inf, delta_energy)
        accept_prob = np.clip(np.exp(-delta_energy), a_max=1.0)
        return accept_prob, (z_f, r_f, pe_f, z_grad)

    return init_kernel,
