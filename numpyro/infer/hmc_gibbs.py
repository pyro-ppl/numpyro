from collections import namedtuple
import copy
import math
from functools import partial

from jax import device_put, random, value_and_grad
import jax.numpy as jnp

import numpyro
from numpyro.handlers import condition, seed, trace
from numpyro.infer.mcmc import MCMCKernel
from numpyro.util import ravel_pytree


HMCGibbs_State = namedtuple("HMCGibbs_State", "z, hmc_state, rng_key")
"""
 - **z** - a dict of the current latent values (both HMC and Gibbs sites)
 - **hmc_state** - current hmc_state
 - **rng_key** - random key for the current step
"""


def _wrap_model(model):
    def fn(*args, **kwargs):
        gibbs_values = kwargs.pop("_gibbs_sites", {})
        with condition(data=gibbs_values):
            model(*args, **kwargs)
    return fn


class HMCGibbs(MCMCKernel):
    """
    HMC-within-Gibbs. This inference algorithm allows the user to combine
    general purpose gradient-based inference (HMC or NUTS) with custom
    Gibbs samplers.

    :param inner_kernel: Python callable containing Pyro :mod:`~numpyro.primitives`.
        If model is provided, `potential_fn` will be inferred using the model.
    :param potential_fn: Python callable that computes the potential energy
        given input parameters. The input parameters to `potential_fn` can be
        any python collection type, provided that `init_params` argument to
        :meth:`init` has the same type.
    :param callable init_strategy: a per-site initialization function.
        See :ref:`init_strategy` section for available functions.
    """

    sample_field = "z"

    def __init__(self, inner_kernel, gibbs_fn, gibbs_sites):
        self.inner_kernel = copy.copy(inner_kernel)
        self.inner_kernel._model = _wrap_model(inner_kernel.model)
        self._gibbs_sites = gibbs_sites
        self._gibbs_fn = gibbs_fn

    @property
    def model(self):
        return self.inner_kernel._model

    def postprocess_fn(self, args, kwargs):
        def fn(z):
            model_kwargs = {} if kwargs is None else kwargs.copy()
            hmc_sites = {k: v for k, v in z.items() if k not in self._gibbs_sites}
            gibbs_sites = {k: v for k, v in z.items() if k in self._gibbs_sites}
            model_kwargs["_gibbs_sites"] = gibbs_sites
            hmc_sites = self.inner_kernel.postprocess_fn(args, model_kwargs)(hmc_sites)
            return {**gibbs_sites, **hmc_sites}

        return fn

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        model_kwargs = {} if model_kwargs is None else model_kwargs.copy()
        rng_key, key_u, key_z = random.split(rng_key, 3)
        prototype_trace = trace(seed(self.model, key_u)).get_trace(*model_args, **model_kwargs)

        gibbs_sites = {name: site["value"] for name, site in prototype_trace.items() if name in self._gibbs_sites}
        model_kwargs["_gibbs_sites"] = gibbs_sites
        hmc_state = self.inner_kernel.init(key_z, num_warmup, init_params, model_args, model_kwargs)

        z = {**gibbs_sites, **hmc_state.z}
        _, self._unravel_fn = ravel_pytree(gibbs_sites)
        return device_put(HMCGibbs_State(z, hmc_state, rng_key))

    def sample(self, state, model_args, model_kwargs):
        model_kwargs = {} if model_kwargs is None else model_kwargs
        rng_key, rng_gibbs = random.split(state.rng_key)

        def potential_fn(z_gibbs, z_hmc):
            return self.inner_kernel._potential_fn_gen(
                *model_args, _gibbs_sites=z_gibbs, **model_kwargs)(z_hmc)

        z_gibbs = {k: v for k, v in state.z.items() if k not in state.hmc_state.z}
        z_hmc = {k: v for k, v in state.z.items() if k in state.hmc_state.z}

        z_gibbs = self._gibbs_fn(rng_key=rng_gibbs, **z_gibbs, **z_hmc)

        pe, z_grad = value_and_grad(partial(potential_fn, z_gibbs))(state.hmc_state.z)
        hmc_state = state.hmc_state._replace(z_grad=z_grad, potential_energy=pe)

        model_kwargs_ = model_kwargs.copy()
        model_kwargs_["_gibbs_sites"] = z_gibbs
        hmc_state = self.inner_kernel.sample(hmc_state, model_args, model_kwargs_)

        z = {**z_gibbs, **hmc_state.z}

        return HMCGibbs_State(z, hmc_state, rng_key)
