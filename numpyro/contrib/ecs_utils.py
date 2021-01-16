import jax
import jax.numpy as jnp

from numpyro.infer.util import log_density
from numpyro.primitives import Messenger, _subsample_fn


def init_near_values(site=None, values={}):
    """Initialize the sampling to a noisy map estimate of the parameters"""
    from functools import partial

    from numpyro.distributions.continuous import Normal
    from numpyro.infer.initialization import init_to_uniform

    if site is None:
        return partial(init_near_values(values=values))

    if site['type'] == 'sample' and not site['is_observed']:
        if site['name'] in values:
            try:
                rng_key = site['kwargs'].get('rng_key')
                sample_shape = site['kwargs'].get('sample_shape')
                return values[site['name']] + Normal(0., 1e-3).sample(rng_key, sample_shape)
            except:
                return init_to_uniform(site)


def variational_proxy(model, guide, evidence, weights, model_args, model_kwargs, ):
    # TODO: fuse computation for S + log_posterior_prob(z) - log_prior_prob(z)?
    log_posterior_prob = lambda params: log_density(guide, model_args, model_kwargs, params)
    log_prior_prob = lambda params: log_density(model, model_args, model_kwargs, params)

    def proxy(name, z):
        return evidence[name] + log_posterior_prob(z) - log_prior_prob(z)

    def uproxy(name, z, subsample):
        return evidence[name] + weights[subsample].sum() + log_posterior_prob(z) - log_prior_prob(z)

    return proxy, uproxy


def _extract_params(distribution):
    params, _ = distribution.tree_flatten()
    return params


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


def taylor_proxy(ref_trace, ll_ref, jac_all, hess_all):
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


def difference_estimator_fn(value, name, z, sizes, log_prob, proxy_fn, uproxy_fn, *args, **kwargs, ):
    n, m, g = sizes
    ll_sub = log_prob(value).sum()
    diff = ll_sub - uproxy_fn(name, value, z)
    l_hat = proxy_fn(name, z) + n / m * diff
    sigma = n ** 2 / m * jnp.var(diff)
    return l_hat - .5 * sigma


def _tangent_curve(dist, value, tangent_fn):
    z, aux_data = dist.tree_flatten()
    log_prob = lambda *params: dist.tree_unflatten(aux_data, params).log_prob(value).sum()
    return tuple(tangent_fn(log_prob, argnum)(*z) for argnum in range(len(z)))
