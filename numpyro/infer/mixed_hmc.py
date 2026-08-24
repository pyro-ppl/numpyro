# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
import copy
from functools import partial
from typing import Any

import numpy as np

import jax
from jax import random
import jax.numpy as jnp

from numpyro._typing import ConstrainFn, ModelArgs, ModelKwargs, ModelT, SiteValues
from numpyro.infer.gibbs import (
    ModelWrapper,
    _flat_support_sizes,
    conditioned,
    discrete_latent_sites,
    prototype_trace,
    select_discrete_proposal,
    with_conditioning,
)
from numpyro.infer.hmc import HMC, NUTS, momentum_generator
from numpyro.infer.hmc_util import euclidean_kinetic_energy, warmup_adapter
from numpyro.infer.mcmc import MCMCKernel
from numpyro.util import cond, fori_loop, identity

MixedHMCState = namedtuple("MixedHMCState", "z, hmc_state, rng_key, accept_prob")


class MixedHMC(MCMCKernel):
    """
    Implementation of Mixed Hamiltonian Monte Carlo (reference [1]). An HMC-family wrapper
    that interleaves discrete updates inside the HMC trajectory; it can also be used as a
    block of :class:`~numpyro.infer.gibbs.Gibbs`.

    .. note:: The number of discrete sites to update at each MCMC iteration
        (`n_D` in reference [1]) is fixed at value 1.

    **References**

    1. *Mixed Hamiltonian Monte Carlo for Mixed Discrete and Continuous Variables*,
       Guangyao Zhou (2020)
    2. *Peskun's theorem and a modified discrete-state Gibbs sampler*,
       Liu, J. S. (1996)

    :param inner_kernel: A :class:`~numpyro.infer.hmc.HMC` kernel.
    :param int num_discrete_updates: Number of times to update discrete variables.
        Defaults to the number of discrete latent variables.
    :param bool random_walk: If False, Gibbs sampling will be used to draw a sample from the
        conditional `p(gibbs_site | remaining sites)`, where `gibbs_site` is one of the
        discrete sample sites in the model. Otherwise, a sample will be drawn uniformly
        from the domain of `gibbs_site`. Defaults to False.
    :param bool modified: whether to use a modified proposal, as suggested in reference [2], which
        always proposes a new state for the current Gibbs site (i.e. discrete site). Defaults to False.
        The modified scheme appears in the literature under the name "modified Gibbs sampler" or
        "Metropolised Gibbs sampler".

    **Example**

    .. doctest::

        >>> from jax import random
        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.infer import HMC, MCMC, MixedHMC
        ...
        >>> def model(probs, locs):
        ...     c = numpyro.sample("c", dist.Categorical(probs))
        ...     numpyro.sample("x", dist.Normal(locs[c], 0.5))
        ...
        >>> probs = jnp.array([0.15, 0.3, 0.3, 0.25])
        >>> locs = jnp.array([-2, 0, 2, 4])
        >>> kernel = MixedHMC(HMC(model, trajectory_length=1.2), num_discrete_updates=20)
        >>> mcmc = MCMC(kernel, num_warmup=1000, num_samples=100000, progress_bar=False)
        >>> mcmc.run(random.key(0), probs, locs)
        >>> mcmc.print_summary()  # doctest: +SKIP
        >>> samples = mcmc.get_samples()
        >>> assert "x" in samples and "c" in samples
        >>> assert abs(jnp.mean(samples["x"]) - 1.3) < 0.1
        >>> assert abs(jnp.var(samples["x"]) - 4.36) < 0.5
    """

    sample_field: str = "z"

    def __init__(
        self,
        inner_kernel: HMC,
        *,
        num_discrete_updates: int | None = None,
        random_walk: bool = False,
        modified: bool = False,
    ) -> None:
        if not isinstance(inner_kernel, HMC):
            raise ValueError("inner_kernel must be an HMC sampler.")
        if isinstance(inner_kernel, NUTS):
            raise ValueError(
                "The algorithm only works with HMC and and does not support NUTS."
            )
        if inner_kernel.model is None:
            raise ValueError(
                "MixedHMC does not support models specified via a potential function."
            )
        self._model = inner_kernel.model
        self.inner_kernel = inner_kernel.wrap_model(conditioned)
        self._num_discrete_updates = num_discrete_updates
        self._random_walk = random_walk
        self._modified = modified
        self._discrete_proposal_fn = select_discrete_proposal(random_walk, modified)
        # static metadata resolved at `init`
        self._gibbs_sites: tuple[str, ...] = ()
        self._support_sizes_flat: np.ndarray | None = None
        self._num_warmup = None
        self._wa_update = None

    @property
    def model(self) -> ModelT:
        return self._model

    def get_diagnostics_str(self, state: MixedHMCState) -> str:
        return self.inner_kernel.get_diagnostics_str(state.hmc_state)

    def _split(self, z: SiteValues) -> tuple[SiteValues, SiteValues]:
        z_discrete = {k: v for k, v in z.items() if k in self._gibbs_sites}
        z_hmc = {k: v for k, v in z.items() if k not in self._gibbs_sites}
        return z_discrete, z_hmc

    def postprocess_fn(
        self, model_args: ModelArgs, model_kwargs: ModelKwargs | None
    ) -> ConstrainFn:
        def fn(z: SiteValues) -> SiteValues:
            z_discrete, z_hmc = self._split(z)
            z_hmc = self.inner_kernel.postprocess_fn(
                model_args, with_conditioning(model_kwargs, z_discrete)
            )(z_hmc)
            return {**z_discrete, **z_hmc}

        return fn

    def get_constrain_fn(
        self, model_args: ModelArgs, model_kwargs: ModelKwargs | None
    ) -> ConstrainFn:
        def fn(z: SiteValues) -> SiteValues:
            z_discrete, z_hmc = self._split(z)
            z_hmc = self.inner_kernel.get_constrain_fn(
                model_args, with_conditioning(model_kwargs, z_discrete)
            )(z_hmc)
            return {**z_discrete, **z_hmc}

        return fn

    def init(
        self,
        rng_key: jax.Array,
        num_warmup: int,
        init_params: SiteValues | None,
        model_args: ModelArgs,
        model_kwargs: ModelKwargs | None,
    ) -> MixedHMCState:
        model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        rng_key, key_u, key_z, rng_r = random.split(rng_key, 4)
        model_trace = prototype_trace(
            conditioned(self._model), key_u, model_args, model_kwargs
        )
        self._gibbs_sites = discrete_latent_sites(model_trace)
        if not self._gibbs_sites:
            raise ValueError(
                "Cannot detect any discrete latent variables in the model."
            )
        self._support_sizes_flat = _flat_support_sizes(model_trace, self._gibbs_sites)
        if self._num_discrete_updates is None:
            self._num_discrete_updates = self._support_sizes_flat.shape[0]
        self._num_warmup = num_warmup

        init_params = None if init_params is None else dict(init_params)
        z_discrete = {}
        for name in self._gibbs_sites:
            if init_params and name in init_params:
                z_discrete[name] = init_params.pop(name)
            else:
                z_discrete[name] = model_trace[name]["value"]
        z_discrete = {k: jnp.asarray(v) for k, v in z_discrete.items()}
        hmc_state = self.inner_kernel.init(
            key_z,
            num_warmup,
            init_params or None,
            model_args,
            with_conditioning(model_kwargs, z_discrete),
        )

        # NB: the warmup adaptation can not be performed in sub-trajectories (i.e. the hmc trajectory
        # between two discrete updates), so we will do it here, at the end of each MixedHMC step.
        _, self._wa_update = warmup_adapter(
            num_warmup,
            adapt_step_size=self.inner_kernel._adapt_step_size,
            adapt_mass_matrix=self.inner_kernel._adapt_mass_matrix,
            dense_mass=self.inner_kernel._dense_mass,
            target_accept_prob=self.inner_kernel._target_accept_prob,
            find_reasonable_step_size=None,
        )

        # In HMC, when `hmc_state.r` is not None, we will skip drawing a random momentum at the
        # beginning of an HMC step. The reason is we need to maintain `r` between each sub-trajectories.
        r = momentum_generator(
            hmc_state.z, hmc_state.adapt_state.mass_matrix_sqrt, rng_r
        )
        z = {**z_discrete, **hmc_state.z}
        return MixedHMCState(z, hmc_state._replace(r=r), rng_key, jnp.zeros(()))

    def refresh(
        self,
        state: MixedHMCState,
        model_args: ModelArgs,
        model_kwargs: ModelKwargs | None,
    ) -> MixedHMCState:
        """Delegates to the inner kernel with the current discrete values in the kwargs."""
        z_discrete, _ = self._split(state.z)
        hmc_state = self.inner_kernel.refresh(
            state.hmc_state, model_args, with_conditioning(model_kwargs, z_discrete)
        )
        return state._replace(hmc_state=hmc_state)

    def wrap_model(self, wrapper: ModelWrapper) -> "MixedHMC":
        kernel = copy.copy(self)
        kernel._model = wrapper(self._model)
        kernel.inner_kernel = self.inner_kernel.wrap_model(wrapper)
        return kernel

    def sample(
        self,
        state: MixedHMCState,
        model_args: ModelArgs,
        model_kwargs: ModelKwargs | None,
    ) -> MixedHMCState:
        model_kwargs = {} if model_kwargs is None else model_kwargs
        assert self._support_sizes_flat is not None, (
            "`init` must be called before `sample`."
        )
        num_discretes = self._support_sizes_flat.shape[0]
        support_sizes_flat = jnp.asarray(self._support_sizes_flat)

        def potential_fn(z_gibbs, z_hmc):
            return self.inner_kernel.get_potential_fn(
                model_args, with_conditioning(model_kwargs, z_gibbs)
            )(z_hmc)

        def update_discrete(
            idx, rng_key, hmc_state, z_discrete, ke_discrete, delta_pe_sum
        ):
            # Algo 1, line 19: get a new discrete proposal
            (
                rng_key,
                z_discrete_new,
                pe_new,
                log_accept_ratio,
            ) = self._discrete_proposal_fn(
                rng_key,
                z_discrete,
                hmc_state.potential_energy,
                partial(potential_fn, z_hmc=hmc_state.z),
                idx,
                support_sizes_flat[idx],
            )
            # Algo 1, line 20: depending on reject or refract, we will update
            # the discrete variable and its corresponding kinetic energy. In case of
            # refract, we will need to update the potential energy and its grad w.r.t. hmc_state.z
            ke_discrete_i_new = ke_discrete[idx] + log_accept_ratio

            def refract(vals):
                z_discrete_new, _, ke_discrete_i_new = vals
                refreshed = self.inner_kernel.refresh(
                    hmc_state,
                    model_args,
                    with_conditioning(model_kwargs, z_discrete_new),
                )
                return (
                    z_discrete_new,
                    refreshed.potential_energy,
                    ke_discrete_i_new,
                    refreshed.z_grad,
                )

            z_discrete, pe, ke_discrete_i, z_grad = cond(
                ke_discrete_i_new > 0,
                (z_discrete_new, pe_new, ke_discrete_i_new),
                refract,
                (
                    z_discrete,
                    hmc_state.potential_energy,
                    ke_discrete[idx],
                    hmc_state.z_grad,
                ),
                identity,
            )

            delta_pe_sum = delta_pe_sum + pe - hmc_state.potential_energy
            ke_discrete = ke_discrete.at[idx].set(ke_discrete_i)
            hmc_state = hmc_state._replace(potential_energy=pe, z_grad=z_grad)
            return rng_key, hmc_state, z_discrete, ke_discrete, delta_pe_sum

        def update_continuous(hmc_state, z_discrete):
            hmc_state_new = self.inner_kernel.sample(
                hmc_state, model_args, with_conditioning(model_kwargs, z_discrete)
            )

            # each time a sub-trajectory is performed, we need to reset i and adapt_state
            # (we will only update them at the end of HMCGibbs step)
            # For `num_steps`, we will record its cumulative sum for diagnostics
            hmc_state = hmc_state_new._replace(
                i=hmc_state.i,
                adapt_state=hmc_state.adapt_state,
                num_steps=hmc_state.num_steps + hmc_state_new.num_steps,
            )
            return hmc_state

        def body_fn(i, vals):
            (
                rng_key,
                hmc_state,
                z_discrete,
                ke_discrete,
                delta_pe_sum,
                arrival_times,
            ) = vals
            idx = jnp.argmin(arrival_times)
            # NB: length of each sub-trajectory is scaled from the current min(arrival_times)
            # (see the note at total_time below)
            trajectory_length = arrival_times[idx] * time_unit
            arrival_times = arrival_times - arrival_times[idx]
            arrival_times = arrival_times.at[idx].set(1.0)

            # this is a trick, so that in a sub-trajectory of HMC, we always accept the new proposal
            pe = jnp.inf
            hmc_state = hmc_state._replace(
                trajectory_length=trajectory_length, potential_energy=pe
            )
            # Algo 1, line 7: perform a sub-trajectory
            hmc_state = update_continuous(hmc_state, z_discrete)
            # Algo 1, line 8: perform a discrete update
            rng_key, hmc_state, z_discrete, ke_discrete, delta_pe_sum = update_discrete(
                idx, rng_key, hmc_state, z_discrete, ke_discrete, delta_pe_sum
            )
            return (
                rng_key,
                hmc_state,
                z_discrete,
                ke_discrete,
                delta_pe_sum,
                arrival_times,
            )

        z_discrete, _ = self._split(state.z)
        rng_key, rng_ke, rng_time, rng_r, rng_accept = random.split(state.rng_key, 5)
        # Algo 1, line 2: sample discrete kinetic energy
        ke_discrete = random.exponential(rng_ke, (num_discretes,))
        # Algo 1, line 4 and 5: sample the initial amount of time that each discrete site visits
        # the point 0/1. The logic in GetStepSizesNSteps(...) is more complicated but does
        # the same job: the sub-trajectory length eta_t * M_t is the lag between two arrival time.
        arrival_times = random.uniform(rng_time, (num_discretes,))
        # compute the amount of time to make `num_discrete_updates` discrete updates
        num_discrete_updates = self._num_discrete_updates
        assert num_discrete_updates is not None
        total_time = (num_discrete_updates - 1) // num_discretes + jnp.sort(
            arrival_times
        )[(num_discrete_updates - 1) % num_discretes]
        # NB: total_time can be different from the HMC trajectory length, so we need to scale
        # the time unit so that total_time * time_unit = hmc_trajectory_length
        time_unit = state.hmc_state.trajectory_length / total_time

        # Algo 1, line 2: sample hmc momentum
        r = momentum_generator(
            state.hmc_state.r, state.hmc_state.adapt_state.mass_matrix_sqrt, rng_r
        )
        hmc_state = state.hmc_state._replace(r=r, num_steps=0)
        hmc_ke = euclidean_kinetic_energy(hmc_state.adapt_state.inverse_mass_matrix, r)
        # Algo 1, line 10: compute the initial energy
        energy_old = hmc_ke + hmc_state.potential_energy

        # Algo 1, line 3: set initial values
        delta_pe_sum = 0.0
        init_val = (
            rng_key,
            hmc_state,
            z_discrete,
            ke_discrete,
            delta_pe_sum,
            arrival_times,
        )
        # Algo 1, line 6-9: perform the update loop
        rng_key, hmc_state_new, z_discrete_new, _, delta_pe_sum, _ = fori_loop(
            0, self._num_discrete_updates, body_fn, init_val
        )
        # Algo 1, line 10: compute the proposal energy
        hmc_ke = euclidean_kinetic_energy(
            hmc_state.adapt_state.inverse_mass_matrix, hmc_state_new.r
        )
        energy_new = hmc_ke + hmc_state_new.potential_energy
        # Algo 1, line 11: perform MH correction
        delta_energy = energy_new - energy_old - delta_pe_sum
        delta_energy = jnp.where(jnp.isnan(delta_energy), jnp.inf, delta_energy)
        accept_prob = jnp.clip(jnp.exp(-delta_energy), None, 1.0)

        # record the correct new num_steps
        hmc_state = hmc_state._replace(num_steps=hmc_state_new.num_steps)
        # reset the trajectory length
        hmc_state_new = hmc_state_new._replace(
            trajectory_length=hmc_state.trajectory_length
        )
        hmc_state, z_discrete = cond(
            random.bernoulli(rng_accept, accept_prob),
            (hmc_state_new, z_discrete_new),
            identity,
            (hmc_state, z_discrete),
            identity,
        )

        # perform hmc adapting (similar to the implementation in hmc)
        wa_update = self._wa_update
        assert wa_update is not None
        adapt_state = cond(
            hmc_state.i < self._num_warmup,
            (hmc_state.i, accept_prob, (hmc_state.z,), hmc_state.adapt_state),
            lambda args: wa_update(*args),
            hmc_state.adapt_state,
            identity,
        )

        itr = hmc_state.i + 1
        n = jnp.where(hmc_state.i < self._num_warmup, itr, itr - self._num_warmup)
        mean_accept_prob_prev = state.hmc_state.mean_accept_prob
        mean_accept_prob = (
            mean_accept_prob_prev + (accept_prob - mean_accept_prob_prev) / n
        )
        hmc_state = hmc_state._replace(
            i=itr,
            accept_prob=accept_prob,
            mean_accept_prob=mean_accept_prob,
            adapt_state=adapt_state,
        )

        z = {**z_discrete, **hmc_state.z}
        return MixedHMCState(z, hmc_state, rng_key, accept_prob)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_wa_update"] = None
        return state
