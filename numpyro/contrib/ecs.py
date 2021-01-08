""" Based on fehiepsi implementation: https://gist.github.com/fehiepsi/b4a5a80b245600b99467a0264be05fd5 """
import copy
from collections import namedtuple

import jax.numpy as jnp
from jax import device_put, lax, random, partial, jit, jacobian, hessian, make_jaxpr

import numpyro
import numpyro.distributions as dist
from check_potential import my_estimator, my_taylor, estimator, subsample_size, _tangent_curve
from numpyro.contrib.hmcecs_utils import init_near_values
from numpyro.handlers import substitute, trace, seed
from numpyro.infer import MCMC, NUTS, log_likelihood
from numpyro.infer.mcmc import MCMCKernel
from numpyro.util import identity

HMC_ECS_State = namedtuple("HMC_ECS_State", "uz, hmc_state, accept_prob, rng_key")
"""
 - **uz** - a dict of current subsample indices and the current latent values
 - **hmc_state** - current hmc_state
 - **accept_prob** - acceptance probability of the proposal subsample indices
 - **rng_key** - random key to generate new subsample indices
"""

""" Notes:
- [x] init(...) ]
sample(...)
    will use check_potential handler method!
"""


def _wrap_model(model):
    def fn(*args, **kwargs):
        subsample_values = kwargs.pop("_subsample_sites", {})
        with substitute(data=subsample_values):
            model(*args, **kwargs)

    return fn


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
    sample_field = "uz"

    def __init__(self, inner_kernel, estimator_fn=None, proxy_gen_fn=None, z_ref=None):
        self.inner_kernel = copy.copy(inner_kernel)
        self.inner_kernel._model = inner_kernel.model  # Removed wrapper!
        self._proxy_gen_fn = proxy_gen_fn
        self._estimator_fn = estimator_fn
        self._z_ref = z_ref
        self._plate_sizes = None

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

        self._plate_sizes = {name: prototype_trace[name]["args"] + (min(prototype_trace[name]["args"][1] // 2, 100),)
                             for name in u}

        plate_sizes_all = {name: (prototype_trace[name]["args"][0], prototype_trace[name]["args"][0]) for name in u}
        with subsample_size(model, plate_sizes_all):
            ref_trace = trace(substitute(model, data=z_ref)).get_trace(*model_args, **model_kwargs)
            jac_all = {name: _tangent_curve(site['fn'], site['value'], jacobian) for name, site in ref_trace.items()
                       if
                       (site['type'] == 'sample' and site['is_observed'] and site['cond_indep_stack'])}
            hess_all = {name: _tangent_curve(site['fn'], site['value'], hessian) for name, site in ref_trace.items()
                        if
                        (site['type'] == 'sample' and site['is_observed'] and site['cond_indep_stack'])}
            ll_ref = {name: site['fn'].log_prob(site['value']) for name, site in ref_trace.items() if
                      (site['type'] == 'sample' and site['is_observed'] and site['cond_indep_stack'])}

        ref_trace = trace(substitute(model, data={**z_ref, **u})).get_trace(*model_args,
                                                                            **model_kwargs)  # TODO: check reparam
        proxy_fn, uproxy_fn = self._proxy_gen_fn(ref_trace, ll_ref, jac_all, hess_all)

        estimators = {name: partial(self._estimator_fn, proxy_fn=proxy_fn, uproxy_fn=uproxy_fn)
                      for name, site in prototype_trace.items() if
                      (site['type'] == 'sample' and site['is_observed'] and site['cond_indep_stack'])}
        self.inner_kernel._model = _wrap_est_model(model, estimators, self._plate_sizes)
        init_params = {name: init_near_values(site, self._z_ref) for name, site in prototype_trace.items()}
        model_kwargs["_subsample_sites"] = u
        hmc_state = self.inner_kernel.init(key_z, num_warmup, init_params,
                                           model_args, model_kwargs)
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


def model(data, *args, **kwargs):
    x = numpyro.sample("x", dist.Normal(0., 1.))
    with numpyro.plate("N", data.shape[0], subsample_size=1000):
        batch = numpyro.subsample(data, event_dim=0)
        numpyro.sample("obs", dist.Normal(x, 1.), obs=batch)


def plain_model(data, *args, **kwargs):
    x = numpyro.sample("x", dist.Normal(0., 1.))
    numpyro.sample("obs", dist.Normal(x, 1.), obs=data)


if __name__ == '__main__':
    data = random.normal(random.PRNGKey(1), (10_000,)) + 1
    kernel = NUTS(plain_model)
    state = kernel.init(random.PRNGKey(1), 500, None, (data,), {})
    print(make_jaxpr(kernel.sample)(state, (data,), {}), file=open('nuts_jaxpr.txt', 'w'))
    mcmc = MCMC(kernel, 500, 500)
    mcmc.run(random.PRNGKey(1), data)
    mcmc.print_summary(exclude_deterministic=False)
    z_ref = {k: v.mean() for k, v in mcmc.get_samples().items()}

    kernel = ECS(NUTS(model), estimator_fn=my_estimator, proxy_gen_fn=my_taylor, z_ref=z_ref)
    state = kernel.init(random.PRNGKey(1), 500, None, (data,), {})
    print(make_jaxpr(kernel.sample)(state, (data,), {}), file=open('ecs_jaxpr.txt', 'w'))
    mcmc = MCMC(kernel, 1500, 1500)
    mcmc.run(random.PRNGKey(0), data, extra_fields=("accept_prob",))
    # there is a bug when exclude_deterministic=True, which will be fixed upstream
    mcmc.print_summary(exclude_deterministic=False)
