# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
import copy
from functools import partial

from jax import device_put, random, value_and_grad

from numpyro.handlers import condition, seed, trace, substitute
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.hmc import HMC


HMCGibbsState = namedtuple("HMCGibbsState", "z, hmc_state, rng_key")
"""
 - **z** - a dict of the current latent values (both HMC and Gibbs sites)
 - **hmc_state** - current hmc_state
 - **rng_key** - random key for the current step
"""


def _wrap_model(model):
    def fn(*args, **kwargs):
        gibbs_values = kwargs.pop("_gibbs_sites", {})
        with condition(data=gibbs_values), substitute(data=gibbs_values):
            model(*args, **kwargs)
    return fn


class HMCGibbs(MCMCKernel):
    """
    [EXPERIMENTAL INTERFACE]

    HMC-within-Gibbs. This inference algorithm allows the user to combine
    general purpose gradient-based inference (HMC or NUTS) with custom
    Gibbs samplers.

    Note that it is the user's responsibility to provide a correct implementation
    of `gibbs_fn` that samples from the corresponding posterior conditional.

    :param inner_kernel: One of :class:`~numpyro.infer.hmc.HMC` or :class:`~numpyro.infer.hmc.NUTS`.
    :param gibbs_fn: A Python callable that returns a dictionary of Gibbs samples conditioned
        on the HMC sites. Must include an argument `rng_key` that should be used for all sampling.
        Must also include arguments `hmc_sites` and `gibbs_sites`, each of which is a dictionary
        with keys that are site names and values that are sample values. Note that a given `gibbs_fn`
        may not need make use of all these sample values.
    :param gibbs_sites: a list of site names for the latent variables that are covered by the Gibbs sampler.

    **Example**

    .. doctest::

        >>> from jax import random
        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.infer import MCMC, NUTS, HMCGibbs
        ...
        >>> def model():
        ...     x = numpyro.sample("x", dist.Normal(0.0, 2.0))
        ...     y = numpyro.sample("y", dist.Normal(0.0, 2.0))
        ...     numpyro.sample("obs", dist.Normal(x + y, 1.0), obs=jnp.array([1.0]))
        ...
        >>> def gibbs_fn(rng_key, gibbs_sites, hmc_sites):
        ...    y = hmc_sites['y']
        ...    new_x = dist.Normal(0.8 * (1-y), jnp.sqrt(0.8)).sample(rng_key)
        ...    return {'x': new_x}
        ...
        >>> hmc_kernel = NUTS(model)
        >>> kernel = HMCGibbs(hmc_kernel, gibbs_fn=gibbs_fn, gibbs_sites=['x'])
        >>> mcmc = MCMC(kernel, 100, 100, progress_bar=False)
        >>> mcmc.run(random.PRNGKey(0))
        >>> mcmc.print_summary()  # doctest: +SKIP

    """

    sample_field = "z"

    def __init__(self, inner_kernel, gibbs_fn, gibbs_sites):
        if not isinstance(inner_kernel, HMC):
            raise ValueError("inner_kernel must be a HMC or NUTS sampler.")
        if not callable(gibbs_fn):
            raise ValueError("gibbs_fn must be a callable")
        assert inner_kernel.model is not None, "HMCGibbs does not support models specified via a potential function."

        self.inner_kernel = copy.copy(inner_kernel)
        self.inner_kernel._model = _wrap_model(inner_kernel.model)
        self._gibbs_sites = gibbs_sites
        self._gibbs_fn = gibbs_fn

    @property
    def model(self):
        return self.inner_kernel._model

    def get_diagnostics_str(self, state):
        state = state.hmc_state
        return '{} steps of size {:.2e}. acc. prob={:.2f}'.format(state.num_steps,
                                                                  state.adapt_state.step_size,
                                                                  state.mean_accept_prob)

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

        return device_put(HMCGibbsState(z, hmc_state, rng_key))

    def sample(self, state, model_args, model_kwargs):
        model_kwargs = {} if model_kwargs is None else model_kwargs
        rng_key, rng_gibbs = random.split(state.rng_key)

        def potential_fn(z_gibbs, z_hmc):
            return self.inner_kernel._potential_fn_gen(
                *model_args, _gibbs_sites=z_gibbs, **model_kwargs)(z_hmc)

        z_gibbs = {k: v for k, v in state.z.items() if k not in state.hmc_state.z}
        z_hmc = {k: v for k, v in state.z.items() if k in state.hmc_state.z}
        model_kwargs_ = model_kwargs.copy()
        model_kwargs_["_gibbs_sites"] = z_gibbs
        # TODO: give the user more control over which sites are transformed from unconstrained to constrained space
        z_hmc = self.inner_kernel.postprocess_fn(model_args, model_kwargs_)(z_hmc)

        z_gibbs = self._gibbs_fn(rng_key=rng_gibbs, gibbs_sites=z_gibbs, hmc_sites=z_hmc)

        pe, z_grad = value_and_grad(partial(potential_fn, z_gibbs))(state.hmc_state.z)
        hmc_state = state.hmc_state._replace(z_grad=z_grad, potential_energy=pe)

        model_kwargs_["_gibbs_sites"] = z_gibbs
        hmc_state = self.inner_kernel.sample(hmc_state, model_args, model_kwargs_)

        z = {**z_gibbs, **hmc_state.z}

        return HMCGibbsState(z, hmc_state, rng_key)


#def discrete_gibbs_fn(...)


#def subsample_gibbs_fn(model):
    """
    Returns a gibbs_fn to be used in :class:`HMCGibbs`, which works for subsampling
    statements using :class:`~numpyro.plate` primitive. This implements the Algorithm 1
    of reference [1] but uses a naive estimation of log likelihood, hence might incur a
    high bias (more explanation can be found in [2]).

    **References:**

    1. *Hamiltonian Monte Carlo with energy conserving subsampling*,
       Dang, K. D., Quiroz, M., Kohn, R., Minh-Ngoc, T., & Villani, M. (2019)
    2. *The Fundamental Incompatibility ofScalable Hamiltonian Monte Carlo and Naive Data Subsampling*,
       Michael Betancourt
    """
