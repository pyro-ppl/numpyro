# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from functools import partial

from jax import grad, lax, ops, random, tree_map, vmap
import jax.numpy as jnp
# from jax.scipy.special import expit, logit

from numpyro.infer.hmc import momentum_generator
from numpyro.infer.hmc_gibbs import DiscreteHMCGibbs, HMCGibbsState
from numpyro.infer.hmc_util import build_adaptation_schedule, dual_averaging, euclidean_kinetic_energy, warmup_adapter
from numpyro.util import cond, fori_loop, identity, ravel_pytree, while_loop


class MixedHMC(DiscreteHMCGibbs):

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


ClockState = namedtuple("ClockState", "t, window_idx, discrete_mass, da_state")
AdaptiveMixedHMCState = namedtuple("AdaptiveMixedHMCState", "z, hmc_state, rng_key, clock_state")


class _ClockAdapter:
    """ some utilities to manage clock dynamics """
    def __init__(self, num_warmup, target_accept_prob=0.44):
        self._num_warmup = num_warmup
        self._da_init, self._da_update = dual_averaging()
        self._adaptation_schedule = jnp.array(build_adaptation_schedule(num_warmup))
        # Here we fix target accept prob = 0.44
        # https://m-clark.github.io/docs/ld_mcmc/index_onepage.html#adaptive_metropolis-within-gibbs
        # Neal 0.1353 https://projecteuclid.org/euclid.aoap/1350067989
        self._target_accept_prob = target_accept_prob

    def init_stats(self, num_discretes):
        # currently, stats is the sum of accept probs
        return jnp.zeros(num_discretes)

    def update_stats(self, stats, idx, log_accept_ratio):
        accept_prob = jnp.exp(log_accept_ratio)
        return ops.index_update(stats, idx, stats[idx] + accept_prob)

    def init_state(self, discrete_mass):
        da_state = vmap(self._da_init)(jnp.log(10 * discrete_mass))
        return ClockState(jnp.array(0), jnp.array(0), discrete_mass, da_state)

    def update_state(self, state, stats):
        (t, window_idx, discrete_mass, da_state) = state
        accept_prob = jnp.clip(stats, a_max=1.)
        da_state = vmap(lambda x, y: self._da_update(self._target_accept_prob - x, y))(
            accept_prob, da_state)
        # TODO: instead of log, use sigmoid to control the upper/lower bound for the mass
        log_mass, log_mass_avg = da_state[:2]
        discrete_mass = jnp.where(t == (self._num_warmup - 1),
                                  jnp.exp(log_mass),
                                  jnp.exp(log_mass_avg))
        # NB: probably set some threshold for the min and max value
        finfo = jnp.finfo(discrete_mass)
        discrete_mass = jnp.clip(discrete_mass, a_min=finfo.tiny, a_max=finfo.max)

        t_at_window_end = t == self._adaptation_schedule[window_idx, 1]
        window_idx = jnp.where(t_at_window_end, window_idx + 1, window_idx)
        da_state = cond(t_at_window_end,
                        discrete_mass, lambda x: vmap(self._da_init)(jnp.log(10 * x)),
                        da_state, identity)
        return ClockState(t + 1, window_idx, discrete_mass, da_state)


class AdaptiveMixedHMC(MixedHMC):

    def __init__(self, inner_kernel, *, num_trajectories=20, random_walk=False, modified=True):
        super().__init__(inner_kernel, num_trajectories=num_trajectories,
                         random_walk=random_walk, modified=modified)

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        init_state = super().init(rng_key, num_warmup, init_params, model_args, model_kwargs)
        discrete_mass = jnp.ones(self._support_sizes_flat.shape[0])
        # TODO: we can also adapt hmc_mass by using some statistics;
        # num_trajectories, which equals to int(1/hmc_mass), can be skipped by adapting hmc_mass.
        # The interaction between discrete_mass and hmc_mass can be learned
        # during warmup phase. So the number of discrete/hmc steps might change for each MCMC step.
        # This might also explain why U-turn behavior happens in MixedHMC paper.
        self._hmc_mass = jnp.array(1 / self._num_trajectories)
        self._clock_adapter = _ClockAdapter(num_warmup)
        clock_state = self._clock_adapter.init_state(discrete_mass)
        return AdaptiveMixedHMCState(init_state.z, init_state.hmc_state, init_state.rng_key, clock_state)

    def sample(self, state, model_args, model_kwargs):
        discrete_mass = jnp.array([1 / self._num_trajectories])  # state.clock_state.discrete_mass
        clock_mass = jnp.concatenate([discrete_mass, self._hmc_mass[None]])

        model_kwargs = {} if model_kwargs is None else model_kwargs
        num_discretes = self._support_sizes_flat.shape[0]

        def potential_fn(z_gibbs, z_hmc):
            return self.inner_kernel._potential_fn_gen(
                *model_args, _gibbs_sites=z_gibbs, **model_kwargs)(z_hmc)

        def update_discrete(idx, rng_key, hmc_state, z_discrete, ke_discrete, clock_stats):
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
            clock_stats = self._clock_adapter.update_stats(clock_stats, idx, log_accept_ratio)
            return rng_key, hmc_state, z_discrete, ke_discrete, clock_stats

        def update_continuous(rng_key, hmc_state, z_discrete, ke_discrete, clock_stats):
            model_kwargs_ = model_kwargs.copy()
            model_kwargs_["_gibbs_sites"] = z_discrete
            hmc_state_new = self.inner_kernel.sample(hmc_state, model_args, model_kwargs_)

            # XXX: keep kinetic energy but make direction diffuse
            rng_key, rng_r = random.split(rng_key)
            r = momentum_generator(hmc_state.r, hmc_state.adapt_state.mass_matrix_sqrt, rng_r)
            # scale to keep the same kinetic energy
            prev_ke = euclidean_kinetic_energy(hmc_state.adapt_state.inverse_mass_matrix, hmc_state.r)
            curr_ke = euclidean_kinetic_energy(hmc_state.adapt_state.inverse_mass_matrix, r)
            r = tree_map(lambda x: x * jnp.sqrt(prev_ke / curr_ke), r)

            # update accept prob information
            accept_prob_sum = hmc_state.accept_prob * hmc_state.num_steps
            accept_prob_sum = accept_prob_sum + hmc_state_new.accept_prob * hmc_state_new.num_steps
            num_steps = hmc_state.num_steps + hmc_state_new.num_steps
            hmc_state = hmc_state_new._replace(r=r,
                                               i=hmc_state.i,
                                               accept_prob=(accept_prob_sum / num_steps),
                                               num_steps=num_steps)
            return rng_key, hmc_state, z_discrete, ke_discrete, clock_stats

        def cond_fn(vals):
            arrival_times, time_left = vals[-2:]
            return jnp.any(arrival_times <= time_left)

        def body_fn(vals):
            rng_key, hmc_state, z_discrete, ke_discrete, clock_stats, arrival_times, time_left = vals
            idx = jnp.argmin(arrival_times)
            time_left = time_left - arrival_times[idx]
            arrival_times = arrival_times - arrival_times[idx]
            # amount of time for z[idx] arrive 0/1 again is its mass
            arrival_times = ops.index_update(arrival_times, idx, clock_mass[idx])
            rng_key, hmc_state, z_discrete, ke_discrete, clock_stats = cond(
                idx < num_discretes,
                (idx, rng_key, hmc_state, z_discrete, ke_discrete, clock_stats),
                lambda vals: update_discrete(*vals),
                (rng_key, hmc_state, z_discrete, ke_discrete, clock_stats),
                lambda vals: update_continuous(*vals))
            return rng_key, hmc_state, z_discrete, ke_discrete, clock_stats, arrival_times, time_left

        z_discrete = {k: v for k, v in state.z.items() if k not in state.hmc_state.z}
        rng_key, rng_ke, rng_time, rng_r = random.split(state.rng_key, 4)
        ke_discrete = random.exponential(rng_ke, (num_discretes,))
        # NB: velocity = +-1 / mass
        arrival_times = random.uniform(rng_time, (num_discretes + 1,)) * clock_mass
        total_time = 1.

        r = momentum_generator(state.hmc_state.r, state.hmc_state.adapt_state.mass_matrix_sqrt, rng_r)
        hmc_state = state.hmc_state._replace(r=r, num_steps=0)
        clock_stats = self._clock_adapter.init_stats(num_discretes)
        init_val = (rng_key, hmc_state, z_discrete, ke_discrete, clock_stats, arrival_times, total_time)
        rng_key, hmc_state, z_discrete, _, clock_stats, _, _ = while_loop(cond_fn, body_fn, init_val)
        accept_prob = hmc_state.accept_prob

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

        # perform clock adapting
        clock_state = cond(state.clock_state.t <= (self._num_warmup - 1),
                           (state.clock_state, clock_stats),
                           lambda vals: self._clock_adapter.update_state(*vals),
                           state.clock_state,
                           identity)
        z = {**z_discrete, **hmc_state.z}
        return AdaptiveMixedHMCState(z, hmc_state, rng_key, clock_state)
