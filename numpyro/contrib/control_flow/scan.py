# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from jax import lax, random, tree_flatten
import jax.numpy as np
from jax.tree_util import register_pytree_node_class

from numpyro import handlers
from numpyro.primitives import _PYRO_STACK, apply_stack


@register_pytree_node_class
class PytreeTrace:
    def __init__(self, trace):
        self.trace = trace

    def tree_flatten(self):
        trace, aux_trace = {}, {}
        for name, site in self.trace.items():
            if site['type'] == 'sample':
                site_main, site_aux = {}, {}
                for key in site:
                    if key in ['type', 'name', 'scale', 'is_observed', 'cond_indep_stack']:
                        site_aux[key] = site[key]
                    elif key == 'kwargs':
                        # XXX: should we record a batch of rng_keys? maybe unnecessary
                        site_aux['kwargs'] = {'rng_key': None, 'sample_shape': site['kwargs']['sample_shape']}
                    elif key == 'mask':
                        if isinstance(site['mask'], bool):
                            site_aux['mask'] = site['mask']
                        else:
                            site_main['mask'] = site['mask']
                    elif key != 'stop':
                        site_main[key] = site[key]
                trace[name] = site_main
                aux_trace[name] = site_aux
            elif site['type'] == 'deterministic':
                aux_trace[name] = {'type': 'sample', 'name': name}
                trace[name] = {'value': site['value']}
        return (trace,), (aux_trace, tuple(self.trace))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        trace, = children
        aux_trace, keys = aux_data
        for name, site in trace.items():
            site.update(aux_trace[name])
        return cls({name: trace[name] for name in keys})


def scan_wrapper(fn, init_value, xs, rng_key=None,
                 condition_map=None, condition_fn=None,
                 substitute_map=None, substitute_fn=None):

    def body_fn(wrapped_carry, wrapped_x):
        rng_key, carry = wrapped_carry
        cond_map, subs_map, x = wrapped_x

        if rng_key is not None:
            rng_key, subkey = random.split(rng_key)
        else:
            subkey = None

        with handlers.block():
            trace_fn = fn
            if subkey is not None:
                trace_fn = handlers.seed(trace_fn, subkey)

            if cond_map is not None or condition_fn is not None:
                # NB: we substitute here to avoid overwriting `is_observed` field;
                # If we condition on a user-defined observed site,
                # the error "Cannot condition an already observed site" will not happen here
                # instead, it will happen outside this wrapper: when we send the returned msgs
                # to the stack.
                # If we use condition handler here, those sites will become observed sites,
                # which will trigger "Cannot condition an already observed site" when a condition
                # handler applies on those sites.
                trace_fn = handlers.substitute(trace_fn,
                                               param_map=cond_map,
                                               substitute_fn=condition_fn)

            if subs_map is not None or substitute_fn is not None:
                trace_fn = handlers.substitute(trace_fn,
                                               param_map=subs_map,
                                               substitute_fn=substitute_fn)
            traced_fn = handlers.trace(trace_fn)
            carry, y = traced_fn(carry, x)

        return (rng_key, carry), (PytreeTrace(traced_fn.trace), y)

    length = tree_flatten(xs)[0][0].shape[0]
    # only keeps sites have the same leading dimension as xs
    if condition_map is not None:
        condition_map = condition_map.copy()
    if substitute_map is not None:
        substitute_map = substitute_map.copy()
    for param_map in (condition_map, substitute_map):
        if param_map is not None:
            for site_name in list(param_map.keys()):
                if np.ndim(param_map[site_name]) == 0 or np.shape(param_map[site_name])[0] != length:
                    param_map.pop(site_name)
    return lax.scan(body_fn, (rng_key, init_value), (condition_map, substitute_map, xs))


def scan(name, fn, init_value, xs, rng_key=None):
    # if there are no active Messengers, we just run and return it as expected:
    if not _PYRO_STACK:
        (rng_key, carry), (pytree_trace, ys) = scan_wrapper(fn, init_value, xs, rng_key=rng_key)
    else:
        # Otherwise, we initialize a message...
        initial_msg = {
            'type': 'control_flow',
            'name': name,
            'fn': scan_wrapper,
            'args': (fn, init_value, xs),
            'kwargs': {'rng_key': rng_key,
                       'condition_map': None,
                       'condition_fn': None,
                       'substitute_map': None,
                       'substitute_fn': None},
            'value': None,
        }

        # ...and use apply_stack to send it to the Messengers
        msg = apply_stack(initial_msg)
        (rng_key, carry), (pytree_trace, ys) = msg['value']

    for msg in pytree_trace.trace.values():
        apply_stack(msg)

    return carry, ys
