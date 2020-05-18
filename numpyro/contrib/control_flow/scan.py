from jax import lax, random

import numpyro
from numpyro import handlers
from numpyro.distributions import PRNGIdentity
from numpyro.primitives import _PYRO_STACK, apply_stack


def scan_wrapper(fn, init_value, xs, rng_key=None, param_map=None):

    def body_fn(wrapped_carry, wrapped_x):
        rng_key, carry = wrapped_carry
        site_values, x = wrapped_x

        if rng_key is not None:
            rng_key, subkey = random.split(rng_key)
            seeded_fn = fn if subkey is None else handlers.seed(fn, subkey)
        else:
            seeded_fn = fn

        with handlers.block():
            traced_fn = handlers.trace(handlers.substitute(seeded_fn, param_map=site_values))
            carry, y = traced_fn(carry, x)
        # we return 3 informations: distribution, value, is_subtituted
        site_values = {name: site["value"] for name, site in traced_fn.trace.items()}
        site_dists = {name: site["fn"] for name, site in traced_fn.trace.items()}
        return (rng_key, carry), (site_values, site_dists, y)

    param_map = {} if param_map is None else param_map
    return lax.scan(body_fn, (rng_key, init_value), (param_map, xs))


def scan(name, fn, init_value, xs, rng_key=None):
    # if there are no active Messengers, we just run and return it as expected:
    if not _PYRO_STACK:
        (rng_key, carry), (site_values, site_dists, ys) = scan_wrapper(fn, init_value, xs, rng_key=rng_key)
    else:
        if rng_key is None:
            rng_key = numpyro.sample(name + '$rng_key', PRNGIdentity())

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
