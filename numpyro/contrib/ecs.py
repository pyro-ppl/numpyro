""" Based on fehiepsi implementation: https://gist.github.com/fehiepsi/b4a5a80b245600b99467a0264be05fd5 """
import copy
from collections import namedtuple

import jax.numpy as jnp
from jax import device_put, lax, random, partial, jit, jacobian, hessian

from numpyro.contrib.ecs_utils import (
    init_near_values,
    estimator,
    subsample_size,
    _tangent_curve
)
from numpyro.contrib.ecs_utils import taylor_proxy, variational_proxy, difference_estimator_fn
from numpyro.handlers import substitute, trace, seed, block
from numpyro.infer import log_likelihood
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import _predictive, log_density
from numpyro.util import identity

HMC_ECS_State = namedtuple("HMC_ECS_State", "uz, hmc_state, accept_prob, rng_key")
"""
 - **uz** - a dict of current subsample indices and the current latent values
 - **hmc_state** - current hmc_stat    log_like += j.T @ z_diff + .5 * z_diff.T @ h.reshape(k, k) @ z_diff
e
 - **accept_prob** - acceptance probability of the proposal subsample indices
 - **rng_key** - random key to generate new subsample indices
"""

""" Notes:
- [x] init(...) ]
sample(...)
    will use check_potential handler method!
"""


def _wrap_est_model(model, estimators, plate_sizes):
    def fn(*args, **kwargs):
        subsample_values = kwargs.pop("_subsample_sites", {})
        with substitute(data=subsample_values):
            with estimator(model, estimators, plate_sizes):
                model(*args, **kwargs)

    return fn


@partial(jit, static_argnums=(2, 3, 4))
def _update_block(rng_key, u, n, m, g):
    """Returns indexes of the new subsample. The update mechanism selects blocks of indices within the subsample to be updated.
     The number of indexes to be updated depend on the block size, higher block size more correlation among elements in the subsample.
    :param rng_key:
    :param u: subsample indexes
    :param n: total number of data
    :param m: subsample size
    :param g: number of subsample blocks
    """

    rng_key_block, rng_key_index = random.split(rng_key)

    chosen_block = random.randint(rng_key_block, shape=(), minval=0, maxval=g + 1)
    new_idx = random.randint(rng_key_index, minval=0, maxval=n, shape=(m,))
    block_mask = (jnp.arange(m) // g == chosen_block).astype(int)
    rest_mask = (block_mask - 1) ** 2

    u_new = u * rest_mask + block_mask * new_idx
    return u_new


class ECS(MCMCKernel):
    """ Energy conserving subsampling as first described in [1].

    ** Reference: **
      1. *Hamiltonian Monte Carlo with Energy ConservingSubsampling* by Dang, Khue-Dang et al.
    """
    sample_field = "uz"

    def __init__(self, inner_kernel, proxy, ref=None, guide=None):
        self.inner_kernel = copy.copy(inner_kernel)
        self.inner_kernel._model = inner_kernel.model
        self._guide = guide
        self._proxy = proxy
        self._ref = ref
        self._plate_sizes = None
        self._estimator = difference_estimator_fn

    @property
    def model(self):
        return self.inner_kernel._model

    def postprocess_fn(self, args, kwargs):
        def fn(uz):
            z = {k: v for k, v in uz.items() if k not in self._plate_sizes}
            return self.inner_kernel.postprocess_fn(args, kwargs)(z)

        return fn

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        model_kwargs = {} if model_kwargs is None else model_kwargs.copy()
        rng_key, key_u, key_z = random.split(rng_key, 3)

        prototype_trace = trace(seed(self.model, key_u)).get_trace(*model_args, **model_kwargs)
        u = {name: site["value"] for name, site in prototype_trace.items()
             if site["type"] == "plate" and site["args"][0] > site["args"][1]}

        # TODO: estimate good block size
        self._plate_sizes = {name: prototype_trace[name]["args"] + (min(prototype_trace[name]["args"][1] // 2, 100),)
                             for name in u}

        # Precompute Jaccobian and Hessian for Taylor Proxy
        # TODO: check proxy type and branch
        plate_sizes_all = {name: (prototype_trace[name]["args"][0], prototype_trace[name]["args"][0]) for name in u}
        if self._proxy == 'taylor':
            with subsample_size(self.model, plate_sizes_all):
                ref_trace = trace(substitute(self.model, data=self._z_ref)).get_trace(*model_args, **model_kwargs)
                jac_all = {name: _tangent_curve(site['fn'], site['value'], jacobian) for name, site in ref_trace.items()
                           if (site['type'] == 'sample' and site['is_observed'] and site['cond_indep_stack'])}
                hess_all = {name: _tangent_curve(site['fn'], site['value'], hessian) for name, site in ref_trace.items()
                            if (site['type'] == 'sample' and site['is_observed'] and site['cond_indep_stack'])}
                ll_ref = {name: site['fn'].log_prob(site['value']) for name, site in ref_trace.items() if
                          (site['type'] == 'sample' and site['is_observed'] and site['cond_indep_stack'])}

            ref_trace = trace(substitute(self.model, data={**self._z_ref, **u})).get_trace(*model_args,
                                                                                           **model_kwargs)  # TODO: check reparam
            proxy_fn, uproxy_fn = taylor_proxy(ref_trace, ll_ref, jac_all, hess_all)
        elif self._proxy == 'variational':
            num_samples = 10  # TODO: heuristic for this
            guide = substitute(self._guide, self._ref)
            posterior_samples = _predictive(random.PRNGKey(2), guide, {},
                                            (num_samples,), return_sites='', parallel=True,
                                            model_args=model_args, model_kwargs=model_kwargs)
            with subsample_size(self.model, plate_sizes_all):
                model = subsample_size(self.model, plate_sizes_all)
                ll = log_likelihood(model, posterior_samples, *model_args, **model_kwargs)
            # TODO: fix multiple likehoods
            weights = {name: jnp.mean((value.T / value.sum(1).T).T, 0) for name, value in
                       ll.items()}  # TODO: fix broadcast
            prior, _ = log_density(block(model, hide_fn=lambda site: site['type'] == 'sample' and site['is_observed']),
                                   model_args, model_kwargs, posterior_samples)
            variational, _ = log_density(guide, model_args, model_kwargs, posterior_samples)
            evidence = {name: variational / num_samples - prior / num_samples - ll.mean(1).sum() for name, ll in
                        ll.items()}

            proxy_fn, uproxy_fn = variational_proxy(self.model, self._guide, evidence, weights, model_args,
                                                    model_kwargs)

        estimators = {name: partial(self._estimator_fn, proxy_fn=proxy_fn, uproxy_fn=uproxy_fn)
                      for name, site in prototype_trace.items() if
                      (site['type'] == 'sample' and site['is_observed'] and site['cond_indep_stack'])}
        self.inner_kernel._model = _wrap_est_model(self.model, estimators, self._plate_sizes)
        init_params = {name: init_near_values(site, self._z_ref) for name, site in prototype_trace.items()}
        model_kwargs["_subsample_sites"] = u

        hmc_state = self.inner_kernel.init(key_z, num_warmup, init_params, model_args, model_kwargs)
        uz = {**u, **hmc_state.z}
        return device_put(HMC_ECS_State(uz, hmc_state, 1., rng_key))

    def sample(self, state, model_args, model_kwargs):
        model_kwargs = {} if model_kwargs is None else model_kwargs.copy()
        rng_key, key_u = random.split(state.rng_key)
        u = {k: v for k, v in state.uz.items() if k in self._plate_sizes}
        u_new = {}
        for name, (size, subsample_size, num_blocks) in self._plate_sizes.items():
            key_u, subkey = random.split(key_u)
            u_new[name] = _update_block(subkey, u[name], size, subsample_size,
                                        num_blocks)  # TODO: dynamically adjust block size
        sample = self.postprocess_fn(model_args, model_kwargs)(state.hmc_state.z)
        u_loglik = log_likelihood(self.model, sample, *model_args, batch_ndims=0, **model_kwargs, _subsample_sites=u)
        u_loglik = sum(v.sum() for v in u_loglik.values())
        u_new_loglik = log_likelihood(self.model, sample, *model_args, batch_ndims=0, **model_kwargs,
                                      _subsample_sites=u_new)
        u_new_loglik = sum(v.sum() for v in u_new_loglik.values())
        accept_prob = jnp.clip(jnp.exp(u_new_loglik - u_loglik), a_max=1.0)
        u = lax.cond(random.bernoulli(key_u, accept_prob), u_new, identity, u, identity)
        model_kwargs["_subsample_sites"] = u
        hmc_state = self.inner_kernel.sample(state.hmc_state, model_args, model_kwargs)
        uz = {**u, **hmc_state.z}
        return HMC_ECS_State(uz, hmc_state, accept_prob, rng_key)
