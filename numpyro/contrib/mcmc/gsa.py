from collections import namedtuple
import copy
import math
from functools import partial

from jax import device_put, grad, lax, ops, random, jit, value_and_grad
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.handlers import condition, seed, trace
from numpyro.infer import MCMC, SA
from numpyro.infer.mcmc import MCMCKernel
from numpyro.util import identity, ravel_pytree, control_flow_prims_disabled


GibbsSA_State = namedtuple("GibbsSA_State", "z, sa_state, rng_key")
"""
 - **z** - a dict of the current latent values (include discrete sites)
 - **sa_state** - current sa_state
 - **rng_key** - random key for the current step
"""


def _wrap_model(model):
    def fn(*args, **kwargs):
        gibbs_values = kwargs.pop("_gibbs_sites", {})
        with condition(data=gibbs_values):
            model(*args, **kwargs)

    return fn


class GibbsSA(MCMCKernel):
    sample_field = "z"

    def __init__(self, inner_kernel, gibbs_fn, gibbs_sites):
        self.inner_kernel = copy.copy(inner_kernel)
        self.inner_kernel._model = _wrap_model(inner_kernel._model)
        self._gibbs_sites = gibbs_sites
        self._gibbs_fn = gibbs_fn

    @property
    def model(self):
        return self.inner_kernel._model

    def postprocess_fn(self, args, kwargs):
        def fn(z):
            model_kwargs = {} if kwargs is None else kwargs.copy()
            sa_sites = {k: v for k, v in z.items() if k not in self._gibbs_sites}
            gibbs_sites = {k: v for k, v in z.items() if k in self._gibbs_sites}
            model_kwargs["_gibbs_sites"] = gibbs_sites
            sa_sites = self.inner_kernel.postprocess_fn(args, model_kwargs)(sa_sites)
            return {**gibbs_sites, **sa_sites}

        return fn

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        model_kwargs = {} if model_kwargs is None else model_kwargs.copy()
        rng_key, key_u, key_z = random.split(rng_key, 3)
        prototype_trace = trace(seed(self.model, key_u)).get_trace(*model_args, **model_kwargs)

        gibbs_sites = {name: site["value"] for name, site in prototype_trace.items() if name in self._gibbs_sites}
        model_kwargs["_gibbs_sites"] = gibbs_sites
        sa_state = self.inner_kernel.init(key_z, num_warmup, init_params, model_args, model_kwargs)

        z = {**gibbs_sites, **sa_state.z}
        _, self._unravel_fn = ravel_pytree(gibbs_sites)
        return device_put(GibbsSA_State(z, sa_state, rng_key))

    def sample(self, state, model_args, model_kwargs):
        model_kwargs = {} if model_kwargs is None else model_kwargs
        rng_key, rng_gibbs = random.split(state.rng_key)

        def potential_fn(z_gibbs, z_sa):
            return self.inner_kernel._potential_fn_gen(
                *model_args, _gibbs_sites=z_gibbs, **model_kwargs)(z_sa)

        z_gibbs = {k: v for k, v in state.z.items() if k not in state.sa_state.z}
        z_sa = {k: v for k, v in state.z.items() if k in state.sa_state.z}

        z_gibbs = self._gibbs_fn(rng_key=rng_gibbs, **z_gibbs, **z_sa)

        pe = potential_fn(z_gibbs, state.sa_state.z)
        sa_state = state.sa_state._replace(potential_energy=pe)

        pes = lax.map(lambda z: potential_fn(z_gibbs, {'y': z}), state.sa_state.adapt_state.zs)
        adapt_state = sa_state.adapt_state._replace(pes=pes)
        sa_state = sa_state._replace(adapt_state=adapt_state)

        model_kwargs_ = model_kwargs.copy()
        model_kwargs_["_gibbs_sites"] = z_gibbs
        sa_state = self.inner_kernel.sample(sa_state, model_args, model_kwargs_)

        z = {**z_gibbs, **sa_state.z}
        return GibbsSA_State(z, sa_state, rng_key)


def model():
    x = numpyro.sample("x", dist.Normal(0.0, 2.0))
    y = numpyro.sample("y", dist.Normal(0.0, 2.0))
    numpyro.sample("obs", dist.Normal(x + y, 1.0), obs=jnp.array([1.0]))

def gibbs_fn(rng_key, x, y):
    new_x = dist.Normal(0.8 * (1-y), math.sqrt(0.8)).sample(rng_key)
    return {'x': new_x}


print("===  GIBBS-SA  ===")
sa = SA(model=model, adapt_state_size=32, dense_mass=False)
kernel = GibbsSA(sa, gibbs_fn=gibbs_fn, gibbs_sites=['x'])
mcmc = MCMC(kernel, 5 * 1000, 5 * 1000, progress_bar=False)
mcmc.run(random.PRNGKey(0))
mcmc.print_summary(exclude_deterministic=False)

do_hmc = False

if do_hmc:
    print("===  HMC  ===")
    mcmc = MCMC(NUTS(model), 1000, 1000)
    mcmc.run(random.PRNGKey(0))
    mcmc.print_summary(exclude_deterministic=False)
