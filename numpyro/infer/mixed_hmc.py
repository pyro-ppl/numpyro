# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

from jax import grad, lax, ops, random
import jax.numpy as jnp

from numpyro.infer.hmc import momentum_generator
from numpyro.infer.hmc_gibbs import DiscreteHMCGibbs, HMCGibbsState
from numpyro.infer.hmc_util import euclidean_kinetic_energy, warmup_adapter
from numpyro.util import cond, fori_loop, identity, ravel_pytree


class MixedHMC(DiscreteHMCGibbs):
    """
    Implementation of Mixed Hamiltonian Monte Carlo (reference [1]).

    **References**

    1. *Mixed Hamiltonian Monte Carlo for Mixed Discrete and Continuous Variables*,
       Guangyao Zhou (2020)
    """

    def __init__(self, inner_kernel, *, num_trajectories=20, random_walk=False, modified=True):
        super().__init__(inner_kernel, random_walk=random_walk, modified=modified)
        self._num_trajectories = num_trajectories

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        rng_key, rng_r = random.split(rng_key)
        state = super().init(rng_key, num_warmup, init_params, model_args, model_kwargs)
        support_sizes = {
            name: jnp.broadcast_to(site["fn"].enumerate_support(False).shape[0], jnp.shape(site["value"]))
            for name, site in self._prototype_trace.items()
            if site["type"] == "sample" and site["fn"].has_enumerate_support and not site["is_observed"]
        }
        self._support_sizes_flat, _ = ravel_pytree({k: support_sizes[k] for k in self._gibbs_sites})
        self._num_warmup = num_warmup
        _, self._wa_update = warmup_adapter(num_warmup,
                                            adapt_step_size=self.inner_kernel._adapt_step_size,
                                            adapt_mass_matrix=self.inner_kernel._adapt_mass_matrix,
                                            dense_mass=self.inner_kernel._dense_mass,
                                            target_accept_prob=self.inner_kernel._target_accept_prob,
                                            find_reasonable_step_size=None)

        r = momentum_generator(state.hmc_state.z, state.hmc_state.adapt_state.mass_matrix_sqrt, rng_r)
        return state._replace(hmc_state=state.hmc_state._replace(r=r))

    def sample(self, state, model_args, model_kwargs):
        model_kwargs = {} if model_kwargs is None else model_kwargs
        num_discretes = self._support_sizes_flat.shape[0]

        def potential_fn(z_gibbs, z_hmc):
            return self.inner_kernel._potential_fn_gen(
                *model_args, _gibbs_sites=z_gibbs, **model_kwargs)(z_hmc)

        def update_discrete(idx, rng_key, hmc_state, z_discrete, ke_discrete):
            rng_key, z_discrete_new, pe_new, log_accept_ratio = self._discrete_proposal_fn(
                rng_key, z_discrete, hmc_state.potential_energy,
                partial(potential_fn, z_hmc=hmc_state.z), idx, self._support_sizes_flat[idx])
            ke_discrete_i_new = ke_discrete[idx] + log_accept_ratio

            z_discrete, pe, ke_discrete_i, z_grad = lax.cond(
                ke_discrete_i_new > 0,
                (z_discrete_new, pe_new, ke_discrete_i_new),
                lambda vals: vals + (grad(partial(potential_fn, vals[0]))(hmc_state.z),),
                (z_discrete, hmc_state.potential_energy, ke_discrete[idx], hmc_state.z_grad),
                identity)

            ke_discrete = ops.index_update(ke_discrete, idx, ke_discrete_i)
            hmc_state = hmc_state._replace(potential_energy=pe, z_grad=z_grad)
            return rng_key, hmc_state, z_discrete, ke_discrete

        def update_continuous(hmc_state, z_discrete):
            model_kwargs_ = model_kwargs.copy()
            model_kwargs_["_gibbs_sites"] = z_discrete
            hmc_state_new = self.inner_kernel.sample(hmc_state, model_args, model_kwargs_)

            hmc_state = hmc_state_new._replace(i=hmc_state.i,
                                               adapt_state=hmc_state.adapt_state,
                                               num_steps=hmc_state.num_steps + hmc_state_new.num_steps)
            return hmc_state

        def body_fn(i, vals):
            rng_key, hmc_state, z_discrete, ke_discrete, arrival_times = vals
            idx = jnp.argmin(arrival_times)
            trajectory_length = trajectory_length_ratio * arrival_times[idx]
            arrival_times = arrival_times - arrival_times[idx]
            arrival_times = ops.index_update(arrival_times, idx, 1.)

            # in a sub-trajectory of HMC, we always accept the new proposal
            pe = jnp.inf if self.inner_kernel._algo == "HMC" else hmc_state.potential_energy
            hmc_state = hmc_state._replace(trajectory_length=trajectory_length, potential_energy=pe)
            hmc_state = update_continuous(hmc_state, z_discrete)
            rng_key, hmc_state, z_discrete, ke_discrete = update_discrete(
                idx, rng_key, hmc_state, z_discrete, ke_discrete)
            return rng_key, hmc_state, z_discrete, ke_discrete, arrival_times

        z_discrete = {k: v for k, v in state.z.items() if k not in state.hmc_state.z}
        rng_key, rng_ke, rng_time, rng_r, rng_accept = random.split(state.rng_key, 5)
        ke_discrete = random.exponential(rng_ke, (num_discretes,))
        # NB: velocity = +-1 / mass
        arrival_times = random.uniform(rng_time, (num_discretes,))
        # compute the amount of time to make num_trajectory discrete updates
        total_time = (self._num_trajectories - 1) // num_discretes \
            + jnp.sort(arrival_times)[(self._num_trajectories - 1) % num_discretes]
        trajectory_length_ratio = state.hmc_state.trajectory_length / total_time

        r = momentum_generator(state.hmc_state.r, state.hmc_state.adapt_state.mass_matrix_sqrt, rng_r)
        hmc_state = state.hmc_state._replace(r=r, num_steps=0)
        hmc_ke = euclidean_kinetic_energy(hmc_state.adapt_state.inverse_mass_matrix, r)
        energy_old = ke_discrete.sum() + hmc_ke + hmc_state.potential_energy

        init_val = (rng_key, hmc_state, z_discrete, ke_discrete, arrival_times)
        rng_key, hmc_state_new, z_discrete_new, ke_discrete, _ = fori_loop(
            0, self._num_trajectories, body_fn, init_val)
        hmc_ke = euclidean_kinetic_energy(hmc_state.adapt_state.inverse_mass_matrix, hmc_state_new.r)
        energy_new = ke_discrete.sum() + hmc_ke + hmc_state_new.potential_energy
        delta_energy = energy_new - energy_old
        delta_energy = jnp.where(jnp.isnan(delta_energy), jnp.inf, delta_energy)
        accept_prob = jnp.clip(jnp.exp(-delta_energy), a_max=1.0)

        # record the correct new num_steps
        hmc_state = hmc_state._replace(num_steps=hmc_state_new.num_steps)
        # reset the trajectory length
        hmc_state_new = hmc_state_new._replace(trajectory_length=hmc_state.trajectory_length)
        hmc_state, z_discrete = cond(random.bernoulli(rng_key, accept_prob),
                                     (hmc_state_new, z_discrete_new), identity,
                                     (hmc_state, z_discrete), identity)

        # perform hmc adapting
        adapt_state = cond(hmc_state.i < self._num_warmup,
                           (hmc_state.i, accept_prob, (hmc_state.z,), hmc_state.adapt_state),
                           lambda args: self._wa_update(*args),
                           hmc_state.adapt_state,
                           identity)

        itr = hmc_state.i + 1
        n = jnp.where(hmc_state.i < self._num_warmup, itr, itr - self._num_warmup)
        mean_accept_prob_prev = state.hmc_state.mean_accept_prob
        mean_accept_prob = mean_accept_prob_prev + (accept_prob - mean_accept_prob_prev) / n
        hmc_state = hmc_state._replace(i=itr,
                                       accept_prob=accept_prob,
                                       mean_accept_prob=mean_accept_prob,
                                       adapt_state=adapt_state)

        z = {**z_discrete, **hmc_state.z}
        return HMCGibbsState(z, hmc_state, rng_key)
