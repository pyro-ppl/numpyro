# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from functools import partial

from jax import grad, lax, ops, random, tree_map, vmap
import jax.numpy as jnp
# from jax.scipy.special import expit, logit

from numpyro.handlers import seed, trace
from numpyro.infer.hmc import momentum_generator
from numpyro.infer.hmc_gibbs import DiscreteHMCGibbs, HMCGibbsState, _discrete_modified_gibbs_proposal
from numpyro.infer.hmc_util import build_adaptation_schedule, dual_averaging, euclidean_kinetic_energy
from numpyro.util import cond, fori_loop, identity, ravel_pytree, while_loop


class MixedHMC(DiscreteHMCGibbs):

    def __init__(self, inner_kernel, num_trajectories=20):
        super().__init__(inner_kernel, random_walk=False, modified=True)
        self._num_trajectories = num_trajectories
        self.inner_kernel._adapt_step_size = False
        self.inner_kernel._adapt_mass_matrix = False

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        model_kwargs = {} if model_kwargs is None else model_kwargs
        prototype_trace = trace(seed(self.model, rng_seed=0)).get_trace(*model_args, **model_kwargs)
        support_sizes = {
            name: jnp.broadcast_to(site["fn"].enumerate_support(False).shape[0], jnp.shape(site["value"]))
            for name, site in prototype_trace.items()
            if site["type"] == "sample" and site["fn"].has_enumerate_support and not site["is_observed"]
        }
        self._support_sizes_flat, _ = ravel_pytree({k: support_sizes[k] for k in self._gibbs_sites})
        self._num_warmup = num_warmup
        return super().init(rng_key, num_warmup, init_params, model_args, model_kwargs)

    def sample(self, state, model_args, model_kwargs):
        # NB: this adjusts MixedHMC algorithm a bit to be more compatible to NUTS sampling
        model_kwargs = {} if model_kwargs is None else model_kwargs
        num_discretes = self._support_sizes_flat.shape[0]

        def potential_fn(z_gibbs, z_hmc):
            return self.inner_kernel._potential_fn_gen(
                *model_args, _gibbs_sites=z_gibbs, **model_kwargs)(z_hmc)

        def update_discrete(idx, rng_key, hmc_state, z_discrete, ke_discrete):
            rng_key, z_discrete_new, pe_new, log_accept_ratio = _discrete_modified_gibbs_proposal(
                rng_key, z_discrete, hmc_state.potential_energy,
                partial(potential_fn, z_hmc=hmc_state.z), idx, self._support_sizes_flat[idx])
            ke_discrete_i_new = ke_discrete[idx] + hmc_state.potential_energy - pe_new

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
            hmc_state = self.inner_kernel.sample(hmc_state, model_args, model_kwargs_)
            return hmc_state

        def body_fn(i, vals):
            rng_key, hmc_state, z_discrete, ke_discrete, arrival_times = vals
            idx = jnp.argmin(arrival_times)
            trajectory_length = trajectory_length_ratio * arrival_times[idx]
            arrival_times = arrival_times - arrival_times[idx]
            # amount of time for z[idx] arrive 0/1 again is its mass
            arrival_times = ops.index_update(arrival_times, idx, 1.)

            hmc_state = hmc_state._replace(trajectory_length=trajectory_length,
                                           potential_energy=jnp.inf)
            hmc_state_new = update_continuous(hmc_state, z_discrete)
            hmc_state = hmc_state_new._replace(i=hmc_state.i,
                                               num_steps=hmc_state_new.num_steps + hmc_state.num_steps)
            rng_key, hmc_state, z_discrete, ke_discrete = update_discrete(
                idx, rng_key, hmc_state, z_discrete, ke_discrete)
            return rng_key, hmc_state, z_discrete, ke_discrete, arrival_times

        z_discrete = {k: v for k, v in state.z.items() if k not in state.hmc_state.z}
        rng_key, rng_ke, rng_time, rng_r, rng_accept = random.split(state.rng_key, 5)
        ke_discrete = random.exponential(rng_ke, (num_discretes,))
        # NB: velocity = +-1 / mass
        arrival_times = random.uniform(rng_time, (num_discretes,))
        total_time = (self._num_trajectories - 1) // num_discretes \
            + jnp.sort(arrival_times)[(self._num_trajectories - 1) % num_discretes]
        trajectory_length_ratio = state.hmc_state.trajectory_length / total_time

        r = momentum_generator(state.hmc_state.r, state.hmc_state.adapt_state.mass_matrix_sqrt, rng_r)
        hmc_state = state.hmc_state._replace(r=r, reset_momentum=False, num_steps=0)
        hmc_ke = euclidean_kinetic_energy(hmc_state.adapt_state.inverse_mass_matrix, r)
        energy_old = ke_discrete.sum() + hmc_ke + hmc_state.potential_energy

        init_val = (rng_key, hmc_state, z_discrete, ke_discrete, arrival_times)
        rng_key, hmc_state, z_discrete_new, ke_discrete, _ = fori_loop(
            0, self._num_trajectories, body_fn, init_val)
        energy_new = ke_discrete.sum() + hmc_state.energy
        delta_energy = energy_new - energy_old
        delta_energy = jnp.where(jnp.isnan(delta_energy), jnp.inf, delta_energy)
        accept_prob = jnp.clip(jnp.exp(-delta_energy), a_max=1.0)
        num_steps = hmc_state.num_steps
        hmc_state, z_discrete = cond(random.bernoulli(rng_key, accept_prob),
                                     (hmc_state, z_discrete_new), identity,
                                     (state.hmc_state, z_discrete), identity)

        itr = hmc_state.i + 1
        n = jnp.where(hmc_state.i < self._num_warmup, itr, itr - self._num_warmup)
        mean_accept_prob_prev = state.hmc_state.mean_accept_prob
        mean_accept_prob = mean_accept_prob_prev + (accept_prob - mean_accept_prob_prev) / n
        hmc_state = hmc_state._replace(i=itr,
                                       num_steps=num_steps,
                                       mean_accept_prob=mean_accept_prob,
                                       trajectory_length=state.hmc_state.trajectory_length)
        z = {**z_discrete, **hmc_state.z}
        return HMCGibbsState(z, hmc_state, rng_key)


ClockAdaptState = namedtuple("ClockAdaptState", "t, window_idx, discrete_mass, da_state")
MixedHMCGibbsState = namedtuple("HMCGibbsState", "z, hmc_state, rng_key, adapt_state")


class MixedHMCGibbs(DiscreteHMCGibbs):
    """Works for NUTS"""

    def __init__(self, inner_kernel, num_trajectories=20):
        # NB: num_trajectories, which equals to int(1/hmc_mass), can be skipped by adapting hmc_mass
        # The interaction between discrete_mass and hmc_mass can be learned
        # during warmup phase. So the number of discrete/hmc steps might change for each MCMC step.
        # This might also explain why U-turn behavior happens in MixedHMC paper.
        super().__init__(inner_kernel)
        self._num_trajectories = num_trajectories
        # target accept prob = 0.44
        # https://m-clark.github.io/docs/ld_mcmc/index_onepage.html#adaptive_metropolis-within-gibbs
        # Neal 0.1353 https://projecteuclid.org/euclid.aoap/1350067989
        self._target_accept_prob = 0.44

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        model_kwargs = {} if model_kwargs is None else model_kwargs
        prototype_trace = trace(seed(self.model, rng_seed=0)).get_trace(*model_args, **model_kwargs)
        support_sizes = {
            name: jnp.broadcast_to(site["fn"].enumerate_support(False).shape[0], jnp.shape(site["value"]))
            for name, site in prototype_trace.items()
            if site["type"] == "sample" and site["fn"].has_enumerate_support and not site["is_observed"]
        }
        self._support_sizes_flat, _ = ravel_pytree({k: support_sizes[k] for k in self._gibbs_sites})
        num_discretes = self._support_sizes_flat.shape[0]
        discrete_mass = jnp.ones(num_discretes)
        # TODO: we can also adapt hmc_mass by using some statistics
        self._num_warmup = num_warmup
        self._hmc_mass = jnp.array(2 * jnp.pi / self._num_trajectories)
        self._da_init, self._da_update = dual_averaging()
        da_state = vmap(self._da_init)(jnp.log(10 * discrete_mass))
        hmc_num_warmup = num_warmup * self._num_trajectories
        # TODO: use the same num_warmup for hmc
        state = super().init(rng_key, hmc_num_warmup, init_params, model_args, model_kwargs)
        self._adaptation_schedule = jnp.array(build_adaptation_schedule(num_warmup))
        return MixedHMCGibbsState(state.z, state.hmc_state, state.rng_key,
                                  ClockAdaptState(jnp.array(0), discrete_mass, da_state))

    def sample(self, state, model_args, model_kwargs):
        clock_mass = jnp.concatenate([state.adapt_state.discrete_mass, self._hmc_mass[None]])

        # NB: this adjusts MixedHMC algorithm a bit to be more compatible to NUTS sampling
        model_kwargs = {} if model_kwargs is None else model_kwargs
        num_discretes = self._support_sizes_flat.shape[0]

        def potential_fn(z_gibbs, z_hmc):
            return self.inner_kernel._potential_fn_gen(
                *model_args, _gibbs_sites=z_gibbs, **model_kwargs)(z_hmc)

        def update_discrete(idx, rng_key, hmc_state, z_discrete, ke_discrete, accept_prob):
            rng_key, z_discrete_new, pe_new, log_accept_ratio = _discrete_modified_gibbs_proposal(
                rng_key, z_discrete, hmc_state.potential_energy,
                partial(potential_fn, z_hmc=hmc_state.z), idx, self._support_sizes_flat[idx])
            delta_pe = pe_new - hmc_state.potential_energy
            ke_discrete_i_new = ke_discrete[idx] - delta_pe

            z_discrete, pe, ke_discrete_i, z_grad = lax.cond(
                ke_discrete_i_new > 0,
                (z_discrete_new, pe_new, ke_discrete_i_new),
                lambda vals: vals + (grad(partial(potential_fn, vals[0]))(hmc_state.z),),
                (z_discrete, hmc_state.potential_energy, ke_discrete[idx], hmc_state.z_grad),
                identity)

            accept_prob_i = jnp.exp(-delta_pe)
            accept_prob = ops.index_update(accept_prob, idx, accept_prob[idx] + accept_prob_i)
            ke_discrete = ops.index_update(ke_discrete, idx, ke_discrete_i)
            hmc_state = hmc_state._replace(potential_energy=pe, z_grad=z_grad)
            return rng_key, hmc_state, z_discrete, ke_discrete

        def update_continuous(rng_key, hmc_state, z_discrete, ke_discerete):
            model_kwargs_ = model_kwargs.copy()
            model_kwargs_["_gibbs_sites"] = z_discrete
            hmc_state = self.inner_kernel.sample(hmc_state, model_args, model_kwargs_)
            # XXX: keep kinetic energy but make direction diffuse
            rng_key, rng_r = random.split(rng_key)
            r = momentum_generator(hmc_state.r, hmc_state.adapt_state.mass_matrix_sqrt, rng_r)
            # scale to keep the same kinetic energy
            prev_ke = euclidean_kinetic_energy(hmc_state.adapt_state.inverse_mass_matrix, hmc_state.r)
            curr_ke = euclidean_kinetic_energy(hmc_state.adapt_state.inverse_mass_matrix, r)
            r = tree_map(lambda x: x * jnp.sqrt(prev_ke / curr_ke), r)
            hmc_state = hmc_state._replace(r=r)
            return rng_key, hmc_state, z_discrete, ke_discrete

        def cond_fn(vals):
            arrival_times, time_left = vals[-2:]
            return jnp.any(arrival_times <= time_left)

        def body_fn(vals):
            rng_key, hmc_state, z_discrete, ke_discrete, accept_prob, arrival_times, time_left = vals
            idx = jnp.argmin(arrival_times)
            time_left = time_left - arrival_times[idx]
            arrival_times = arrival_times - arrival_times[idx]
            # amount of time for z[idx] arrive 0/1 again is its mass
            arrival_times = ops.index_update(arrival_times, idx, clock_mass[idx])
            rng_key, hmc_state, z_discrete, ke_discrete = cond(
                idx < num_discretes,
                (idx, rng_key, hmc_state, z_discrete, ke_discrete, accept_prob),
                lambda vals: update_discrete(*vals),
                (rng_key, hmc_state, z_discrete, ke_discrete),
                lambda vals: update_continuous(*vals)
            )
            return rng_key, hmc_state, z_discrete, ke_discrete, arrival_times, time_left

        z_discrete = {k: v for k, v in state.z.items() if k not in state.hmc_state.z}
        rng_key, rng_ke, rng_time, rng_r = random.split(state.rng_key, 4)
        ke_discrete = random.exponential(rng_ke, (num_discretes,))
        # NB: velocity = +-1 / mass
        arrival_times = random.uniform(rng_time, (num_discretes + 1,)) * clock_mass
        total_time = 2 * jnp.pi

        r = momentum_generator(state.hmc_state.r, state.hmc_state.adapt_state.mass_matrix_sqrt, rng_r)
        hmc_state = state.hmc_state._replace(r=r, reset_momentum=False)
        accept_prob = jnp.zeros(num_discretes)
        init_val = (rng_key, hmc_state, z_discrete, ke_discrete, accept_prob, arrival_times, total_time)
        rng_key, hmc_state, z_discrete, _, accept_prob, _, _ = while_loop(cond_fn, body_fn, init_val)
        z = {**z_discrete, **hmc_state.z}

        def update_adapt_state(val):
            (t, window_idx, discrete_mass, da_state), accept_prob = val
            accept_prob = jnp.clip(accept_prob, a_max=1.)
            da_state = vmap(lambda x, y: self._da_update(self._target_accept_prob - x, y))(
                accept_prob, da_state)
            # TODO: instead of log, use sigmoid to control the upper/lower bound for the mass
            log_mass, log_mass_avg, *_ = da_state
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
            return ClockAdaptState(t + 1, window_idx, discrete_mass, da_state)

        adapt_state = cond(state.adapt_state.t <= (self._num_warmup - 1),
                           (state.adapt_state, accept_prob), update_adapt_state,
                           state.adapt_state, identity)
        return MixedHMCGibbsState(z, hmc_state, rng_key, adapt_state)
