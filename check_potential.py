from functools import partial

import jax
import jax.numpy as jnp
from jax import random, hessian, jacfwd

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.hmcecs_utils import log_density_obs_hmcecs
from numpyro.contrib.hmcecs_utils import potential_est, taylor_proxy
from numpyro.handlers import substitute, trace, seed
from numpyro.infer.util import log_density
from numpyro.primitives import Messenger, _subsample_fn
from numpyro.util import ravel_pytree


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


def model(data):
    x = numpyro.sample("x", dist.Normal(0., 1.))
    with numpyro.plate("N", data.shape[0], subsample_size=100):
        batch = numpyro.subsample(data, event_dim=0)
        obs = numpyro.sample("obs", dist.Normal(x, 1.), obs=batch)


def plain_model(data):
    x = numpyro.sample("x", dist.Normal(0., 1.))
    obs = numpyro.sample("obs", dist.Normal(x, 1.), obs=data)


class estimator(Messenger):
    def __init__(self, fn, estimators, plate_sizes):
        self.estimators = estimators
        self.plate_sizes = plate_sizes
        super(estimator, self).__init__(fn)

    def process_message(self, msg):
        if msg['type'] == 'sample' and msg['is_observed'] and msg['cond_indep_stack']:
            log_prob = msg['fn'].log_prob
            msg['scale'] = 1.
            msg['fn'].log_prob = lambda *args, **kwargs: \
                self.estimators[msg['name']](*args, name=msg['name'], z=_extract_params(msg['fn']), log_prob=log_prob,
                                             sizes=self.plate_sizes[msg['cond_indep_stack'][0].name],
                                             **kwargs)  # TODO: check multiple levels


def my_estimator(value, name, z, sizes, log_prob, proxy_fn=lambda x, y: x, uproxy_fn=lambda x: x, **kwargs, ):
    n, m, g = sizes
    ll_sub = log_prob(value).sum()
    diff = ll_sub - uproxy_fn(name, value, z)
    l_hat = proxy_fn(name, z) + n / m * diff
    sigma = n ** 2 / m * jnp.var(diff)
    return l_hat - .5 * sigma


def _extract_params(distribution):
    params, _ = distribution.tree_flatten()
    return params


def my_taylor(ref_trace, ll_ref, jac_all, hess_all):
    def proxy(name, z):
        z_ref = _extract_params(ref_trace[name]['fn'])
        jac, hess = jac_all[name], hess_all[name]
        log_like = jnp.array(0.)
        for argnum in range(len(z_ref)):
            z_diff = z[argnum] - z_ref[argnum]
            j, h = jac[argnum], hess[argnum]
            k, = j.shape
            log_like += j.T @ z_diff + .5 * z_diff.T @ h.reshape(k, k) @ z_diff
        return ll_ref[name].sum() + log_like

    def uproxy(name, value, z):
        ref_dist = ref_trace[name]['fn']
        z_ref, aux_data = ref_dist.tree_flatten()

        log_prob = lambda *params: ref_dist.tree_unflatten(aux_data, params).log_prob(value).sum()
        log_like = jnp.array(0.)
        for argnum in range(len(z_ref)):
            z_diff = z[argnum] - z_ref[argnum]
            jac = jax.jacobian(log_prob, argnum)(*z_ref)
            k, = jac.shape
            hess = jax.hessian(log_prob, argnum)(*z_ref)
            log_like += jac @ z_diff + .5 * z_diff @ hess.reshape(k, k) @ z_diff.T

        return log_prob(*z_ref).sum() + log_like

    return proxy, uproxy


class subsample_size(Messenger):
    def __init__(self, fn, plate_sizes, rng_key=None):
        super(subsample_size, self).__init__(fn)
        self.plate_sizes = plate_sizes
        self.rng_key = rng_key

    def process_message(self, msg):
        if msg['type'] == 'plate' and msg['args'] and msg["args"][0] > msg["args"][1]:
            if msg['name'] in self.plate_sizes:
                msg['args'] = self.plate_sizes[msg['name']]
                msg['value'] = _subsample_fn(*msg['args'], self.rng_key) if msg["args"][1] < msg["args"][
                    0] else jnp.arange(msg["args"][0])


def _tangent_curve(dist, value, tangent_fn):
    z, aux_data = dist.tree_flatten()
    log_prob = lambda *params: dist.tree_unflatten(aux_data, params).log_prob(value).sum()
    return tuple(tangent_fn(log_prob, argnum)(*z) for argnum in range(len(z)))


def check_handler():
    data = random.normal(random.PRNGKey(1), (10000,)) + 1
    z = {'x': jnp.array(0.9511842)}
    model_trace = trace(seed(model, random.PRNGKey(2))).get_trace(data)
    u = {name: site["value"] for name, site in model_trace.items()
         if site["type"] == "plate" and site["args"][0] > site["args"][1]}
    z_ref = {k: v + .1 for k, v in z.items()}
    proxy_fn, uproxy_fn = my_taylor(model, (data,), {}, model_trace, z_ref, u)
    plate_sizes = {name: model_trace[name]["args"] + (1,) for name in u}
    wrapped_model = _wrap_est_model(model, {'obs': partial(my_estimator, proxy_fn=proxy_fn, uproxy_fn=uproxy_fn)},
                                    plate_sizes)
    new_potential, _ = log_density(wrapped_model, (data,), {"_subsample_sites": u}, z)
    print('new potential', new_potential)

    ld_fn = lambda args: jnp.sum(partial(log_density_obs_hmcecs, plain_model, (data,), {})(args)[0])
    jac_all, _ = ravel_pytree(jacfwd(ld_fn)(z_ref))
    print('ref jac all', jac_all)
    hess_all, _ = ravel_pytree(hessian(ld_fn)(z_ref))
    k, = jac_all.shape
    hess_all = hess_all.reshape((k, k))
    print('ref hess all', hess_all)
    ll_ref = ld_fn(z_ref)
    print('ref ll', ll_ref)
    proxy_fn, uproxy_fn = taylor_proxy(z_ref, plain_model, ll_ref, jac_all, hess_all)

    print('reference potential', potential_est(plain_model, (data[u['N']],), {}, z, 10000, 100, proxy_fn, uproxy_fn))


if __name__ == '__main__':
    check_handler()
