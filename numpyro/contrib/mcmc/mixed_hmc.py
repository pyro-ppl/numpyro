# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
import copy
from functools import partial

import matplotlib.pyplot as plt
import seaborn as sns

from jax import device_put, disable_jit, grad, lax, ops, random, tree_map
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.handlers import condition, seed, trace
from numpyro.infer import HMC, MCMC, NUTS, Predictive
from numpyro.infer.hmc import momentum_generator
from numpyro.infer.hmc_util import euclidean_kinetic_energy
from numpyro.infer.mcmc import MCMCKernel
from numpyro.util import (control_flow_prims_disabled, fori_loop, identity, optional,
                          ravel_pytree, while_loop)


MixedHMC_State = namedtuple("MixedHMC_State", "z, hmc_state, rng_key")
"""
 - **z** - a dict of the current latent values (include discrete sites)
 - **hmc_state** - current hmc_state
 - **rng_key** - random key for the current step
"""


def _wrap_model(model):
    def fn(*args, **kwargs):
        discrete_values = kwargs.pop("_discrete_sites", {})
        with condition(data=discrete_values):
            model(*args, **kwargs)

    return fn


def _discrete_step(rng_key, z_discrete, pe, ke_discrete, time_to_go, max_time,
                   potential_fn, z_continuous, support_sizes):
    rng_key, rng_proposal = random.split(rng_key)

    # get z proposal
    z_discrete_flat, unravel_fn = ravel_pytree(z_discrete)
    idx = jnp.argmin(time_to_go)
    proposal = random.choice(rng_proposal, 3)
    proposal = jnp.where(proposal >= z_discrete_flat[idx], proposal + 1, proposal)
    z_discrete_new_flat = ops.index_update(z_discrete_flat, idx, proposal)
    z_discrete_new = unravel_fn(z_discrete_new_flat)

    pe_new = potential_fn(z_discrete_new, z_continuous)
    ke_discrete_i_new = ke_discrete[idx] + pe - pe_new

    z_discrete, pe, ke_discrete_i = lax.cond(ke_discrete_i_new > 0,
                                             (z_discrete_new, pe_new, ke_discrete_i_new), identity,
                                             (z_discrete, pe, ke_discrete[idx]), identity)
    ke_discrete = ops.index_update(ke_discrete, idx, ke_discrete_i)

    max_time = max_time - time_to_go[idx]
    time_to_go = time_to_go - time_to_go[idx]
    time_to_go = ops.index_update(time_to_go, idx, 1)
    return rng_key, z_discrete, pe, ke_discrete, time_to_go, max_time


class MixedHMC(MCMCKernel):
    sample_field = "z"

    def __init__(self, inner_kernel, *, discrete_sites=None, num_trajectories=1, num_discrete_steps=None,
                 max_times=None, discrete_mass=1.):
        self.inner_kernel = copy.copy(inner_kernel)
        self.inner_kernel._model = _wrap_model(inner_kernel.model)
        self.num_trajectories = num_trajectories
        self._support_sizes = None
        self._discrete_sites = discrete_sites
        self._num_discrete_steps = num_discrete_steps
        self._max_times = max_times  # TODO: for testing, should be removed
        self._discrete_mass = discrete_mass

    @property
    def model(self):
        return self.inner_kernel._model

    def postprocess_fn(self, args, kwargs):
        def fn(z):
            model_kwargs = {} if kwargs is None else kwargs.copy()
            cont_sites = {k: v for k, v in z.items() if k not in self._discrete_sites}
            discrete_sites = {k: v for k, v in z.items() if k in self._discrete_sites}
            model_kwargs["_discrete_sites"] = discrete_sites
            cont_sites = self.inner_kernel.postprocess_fn(args, model_kwargs)(cont_sites)
            return {**discrete_sites, **cont_sites}

        return fn

    def get_diagnostics_str(self, state):
        state = state.hmc_state
        return '{} steps of size {:.2e}. acc. prob={:.2f}'.format(state.num_steps,
                                                                  state.adapt_state.step_size,
                                                                  state.mean_accept_prob)

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        model_kwargs = {} if model_kwargs is None else model_kwargs.copy()
        rng_key, key_u, key_z = random.split(rng_key, 3)
        prototype_trace = trace(seed(self.model, key_u)).get_trace(*model_args, **model_kwargs)
        # by default, mixed sites will include discrete sites with finite support
        # and having event_dim > 0; sites with event_dim = 0 can be added through
        # `sites` keyword in the constructor.
        discrete_sites = {name: site["value"] for name, site in prototype_trace.items()
                          if site["type"] == "sample" and site["fn"].has_enumerate_support
                          and (site["fn"].event_dim > 0 or
                              (self._discrete_sites is not None and name in self._discrete_sites))}
        # store support sizes of flatten discrete latent values
        supports = {name: jnp.broadcast_to(
                        prototype_trace[name]["fn"].enumerate_support(False).shape[0],
                        jnp.shape(value))
                    for name, value in discrete_sites.items()}
        self._support_sizes, _ = ravel_pytree(supports)
        if self._num_discrete_steps is None:
            self._num_discrete_steps = max(self._support_sizes.shape[0], self.num_trajectories + 1)
        model_kwargs["_discrete_sites"] = discrete_sites
        hmc_state = self.inner_kernel.init(key_z, num_warmup * self.num_trajectories,
                                           init_params, model_args, model_kwargs)
        z = {**discrete_sites, **hmc_state.z}
        return device_put(MixedHMC_State(z, hmc_state, rng_key))

    def sample(self, state, model_args, model_kwargs):
        # NB: this adjusts MixedHMC algorithm a bit to be more compatible to NUTS sampling
        model_kwargs = {} if model_kwargs is None else model_kwargs
        rng_key, rng_ke, rng_time = random.split(state.rng_key, 3)

        # TODO: adaptive mass for discrete particles
        ke_discrete = random.exponential(rng_ke, self._support_sizes.shape) / self._discrete_mass

        # TODO: relax the assumption that each size of the torus is proportional to velocity
        time_to_go = random.uniform(rng_time, self._support_sizes.shape)

        # run discrete steps for those time < max_time
        if self._max_times is None:
            max_time = self._num_discrete_steps / (self.num_trajectories + 1) / time_to_go.shape[0]
            max_times = [max_time, max_time]
        else:
            max_times = self._max_times

        def potential_fn(z_discrete, z_continous):
            return self.inner_kernel._potential_fn_gen(
                *model_args, _discrete_sites=z_discrete, **model_kwargs)(z_continous)

        def discrete_cond_fn(vals):
            *_, time_to_go, max_time = vals
            return jnp.amin(time_to_go) <= max_time

        def discrete_body_fn(z_continuous, vals):
            return _discrete_step(*vals,
                                  potential_fn=potential_fn,
                                  z_continuous=z_continuous,
                                  support_sizes=self._support_sizes)

        z_discrete = {k: v for k, v in state.z.items() if k not in state.hmc_state.z}
        # TODO: it is also fine to run hmc -> gibbs -> hmc so consider implementing it
        # to compare
        hmc_state = state.hmc_state

        def body_fn(i, vals):
            rng_key, z_discrete, ke_discrete, time_to_go, hmc_state = vals
            pe = hmc_state.potential_energy
            max_time = max_times[0]
            discrete_step_vals = (rng_key, z_discrete, pe, ke_discrete, time_to_go, max_time)
            rng_key, z_discrete, pe, ke_discrete, time_to_go, max_time = while_loop(
                discrete_cond_fn, partial(discrete_body_fn, hmc_state.z), discrete_step_vals)
            time_to_go = time_to_go - max_time

            # build a trajectory
            model_kwargs_ = model_kwargs.copy()
            model_kwargs_["_discrete_sites"] = z_discrete
            z_grad = grad(partial(potential_fn, z_discrete))(hmc_state.z)
            # XXX: keep kinetic energy but make direction diffuse
            rng_key, rng_r = random.split(rng_key)
            r = momentum_generator(hmc_state.r, hmc_state.adapt_state.mass_matrix_sqrt, rng_r)
            # scale to keep the same kinetic energy
            prev_ke = euclidean_kinetic_energy(hmc_state.adapt_state.inverse_mass_matrix, hmc_state.r)
            curr_ke = euclidean_kinetic_energy(hmc_state.adapt_state.inverse_mass_matrix, r)
            r = tree_map(lambda x: x * prev_ke / curr_ke, r)
            hmc_state = hmc_state._replace(r=r, z_grad=z_grad, potential_energy=pe)
            hmc_state = self.inner_kernel.sample(hmc_state, model_args, model_kwargs_,
                                                 reset_momentum=(i == 0))
            return rng_key, z_discrete, ke_discrete, time_to_go, hmc_state

        vals = (rng_key, z_discrete, ke_discrete, time_to_go, hmc_state)
        rng_key, z_discrete, ke_discerete, time_to_go, hmc_state = fori_loop(
            0, self.num_trajectories, body_fn, vals)

        # run discrete steps for those time < max_time
        pe = hmc_state.potential_energy
        vals = (rng_key, z_discrete, pe, ke_discrete, time_to_go, max_times[1])
        rng_key, z_discrete, pe, ke_discrete, *_ = while_loop(
            discrete_cond_fn, partial(discrete_body_fn, hmc_state.z), vals)
        hmc_state = hmc_state._replace(potential_energy=pe)

        z = {**z_discrete, **hmc_state.z}
        return MixedHMC_State(z, hmc_state, rng_key)


def model(probs, mu_list):
    c = numpyro.sample("c", dist.Categorical(probs))
    numpyro.sample("x", dist.Normal(mu_list[c], jnp.sqrt(0.1)))


kernel = MixedHMC(NUTS(model), discrete_sites=["c"], num_trajectories=20,
                  num_discrete_steps=None, discrete_mass=0.1)
mcmc = MCMC(kernel, int(1e4), int(1e5))
probs = jnp.array([0.15, 0.3, 0.3, 0.25])
mu_list = jnp.array([-2, 0, 2, 4])
with optional(__debug__, disable_jit()), optional(__debug__, control_flow_prims_disabled()):
    mcmc.run(random.PRNGKey(0), probs, mu_list)
mcmc.print_summary()
actual_samples = Predictive(model, {}, num_samples=int(1e7), return_sites=["x"])(
    random.PRNGKey(1), probs, mu_list)["x"]
sns.kdeplot(actual_samples, color="r")
sns.kdeplot(mcmc.get_samples()["x"])
plt.show()
