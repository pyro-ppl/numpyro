from collections import OrderedDict, defaultdict

import jax
import jax.numpy as jnp

from numpyro.primitives import Messenger, _subsample_fn


def _tangent_curve(dist, value, tangent_fn):
    z, aux_data = dist.tree_flatten()
    log_prob = lambda *params: dist.tree_unflatten(aux_data, params).log_prob(value).sum()
    return tuple(tangent_fn(log_prob, argnum)(*z) for argnum in range(len(z)))


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


def _extract_params(distribution):
    params, _ = distribution.tree_flatten()
    return params


class estimator(Messenger):
    def __init__(self, fn, estimators, predecessors):
        self.estimators = estimators
        self.predecessors = predecessors
        self.predecessor_sites = defaultdict(OrderedDict)
        self._successors = None

        super(estimator, self).__init__(fn)

    @property
    def successors(self):
        if getattr(self, '_successors') is None:
            successors = {}
            for site_name, preds in self.predecessors.items():
                successors.update({pred_name: site_name for pred_name in preds})  # TODO: handle shared priors
            self._successors = successors
        return self._successors

    def postprocess_message(self, msg):
        name = msg['name']
        if name in self.successors:
            self.predecessor_sites[self.successors[name]][name] = msg.copy()

        if msg['type'] == 'sample' and msg['is_observed'] and msg['cond_indep_stack']:  # TODO: is subsampled
            msg['fn'] = self.estimators[name](msg['fn'], self.predecessor_sites[name])


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


class DifferenceEstimator:
    def __init__(self, name, proxy, uproxy, plate_name, plate_size):
        self._name = name
        self.plate_name = plate_name
        self.size = plate_size
        self.proxy = proxy
        self.uproxy = uproxy
        self.subsample = None
        self._dist = None
        self._predecessors = None

    def __call__(self, dist, predecessors):
        self.dist = dist
        self.predecessors = predecessors

    def log_prob(self, value):
        n, m, g = self.size
        ll_sub = self.dist.log_prob(value).sum()
        diff = ll_sub - self.uproxy(name=self._name,
                                    value=value,
                                    subsample=self.predecessors[self.plate_name],
                                    predecessors=self.predecessors)
        l_hat = self.proxy(self._name) + n / m * diff
        sigma = n ** 2 / m * jnp.var(diff)
        return l_hat - .5 * sigma


def variational_proxy(guide_trace, evidence, weights):
    def _log_like(predecessors):
        log_prob = jnp.array(0.)
        for pred in predecessors:
            if pred['type'] == 'sample':
                val = pred['value']
                name = pred['name']
                log_prob += guide_trace[name]['fn'].log_prob(val) - pred['fn'].log_prob(val)
        return log_prob

    def proxy(name, predecessors, *args, **kwargs):
        return evidence[name] + _log_like(predecessors)

    def uproxy(name, predecessors, subsample, *args, **kwargs):
        return evidence[name] + weights[name][subsample].sum() * _log_like(predecessors)

    return proxy, uproxy
