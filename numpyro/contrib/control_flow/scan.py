from jax import lax, random, tree_flatten
import jax.numpy as np

import numpyro
from numpyro import handlers
from numpyro.primitives import _PYRO_STACK, apply_stack


def scan_wrapper(fn, init_value, xs, rng_key=None, param_map=None, substitute_fn=None):

    def body_fn(wrapped_carry, wrapped_x):
        rng_key, carry = wrapped_carry
        site_values, x = wrapped_x

        if rng_key is not None:
            rng_key, subkey = random.split(rng_key)
            seeded_fn = fn if subkey is None else handlers.seed(fn, subkey)
        else:
            seeded_fn = fn

        with handlers.block():
            traced_fn = handlers.trace(handlers.substitute(seeded_fn, param_map=site_values,
                                                           substitute_fn=substitute_fn))
            carry, y = traced_fn(carry, x)
        # we return 3 informations: distribution, value, is_subtituted
        site_values = {name: site["value"] for name, site in traced_fn.trace.items()}
        site_dists = {name: site["fn"] for name, site in traced_fn.trace.items()}
        return (rng_key, carry), (site_values, site_dists, y)

    length = tree_flatten(xs)[0][0].shape[0]
    param_map = {} if (param_map is None and substitute_fn is None) else param_map
    # only keeps sites have the same leading dimension as xs:
    if param_map is not None:
        param_map = param_map.copy()
        for site_name in list(param_map.keys()):
            if np.ndim(param_map[site_name]) == 0 or np.shape(param_map[site_name])[0] != length:
                param_map.pop(site_name)
    return lax.scan(body_fn, (rng_key, init_value), (param_map, xs))


def scan(name, fn, init_value, xs, rng_key=None):
    # if there are no active Messengers, we just run and return it as expected:
    if not _PYRO_STACK:
        (rng_key, carry), (site_values, site_dists, ys) = scan_wrapper(fn, init_value, xs, rng_key=rng_key)
    else:
        # Otherwise, we initialize a message...
        initial_msg = {
            'type': 'control_flow',
            'name': name,
            'fn': scan_wrapper,
            'args': (fn, init_value, xs),
            'kwargs': {'rng_key': rng_key},
            'value': None,
        }

        # ...and use apply_stack to send it to the Messengers
        msg = apply_stack(initial_msg)
        (rng_key, carry), (site_values, site_dists, ys) = msg['value']

    with handlers.substitute(param_map=site_values):
        for site_name, dist in site_dists.items():
            numpyro.sample(site_name, dist)

    return carry, ys
