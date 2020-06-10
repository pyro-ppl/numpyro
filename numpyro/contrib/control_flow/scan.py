# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

from jax import lax, random
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
                    # XXX: we block the scan body_fn so those sites will have 'stop' field;
                    # here we remove that field!
                    elif key != 'stop':
                        site_main[key] = site[key]
                trace[name] = site_main
                aux_trace[name] = site_aux
            elif site['type'] == 'deterministic':
                trace[name] = {'value': site['value']}
                aux_trace[name] = {'type': 'deterministic', 'name': name}
        return (trace,), aux_trace

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        trace, = children
        for name, site in trace.items():
            site.update(aux_data[name])
        return cls(trace)


def _subs_wrapper(subs_map, i, site):
    value = None
    if isinstance(subs_map, dict) and site['name'] in subs_map:
        value = subs_map[site['name']]
    elif callable(subs_map):
        rng_key = site['kwargs'].get('rng_key')
        subs_map = handlers.seed(subs_map, rng_seed=rng_key) if rng_key is not None else subs_map
        # we only collect the output and block any new sites created in substitute_fn
        # those new sites will be addressed when we apply `apply_stack` on scanned messages.
        with handlers.block():
            value = subs_map(site)

    if value is not None:
        if np.ndim(value) > len(site['kwargs']['sample_shape']) + len(site['fn'].shape()):
            return value[i]
        else:
            return value


def scan_wrapper(fn, init_value, xs, length, rng_key=None, substitute_stack=[]):

    def body_fn(wrapped_carry, x):
        i, rng_key, carry = wrapped_carry
        rng_key, subkey = random.split(rng_key) if rng_key is not None else (None, None)

        with handlers.block():
            seeded_fn = handlers.seed(fn, subkey) if subkey is not None else fn
            for subs_map in substitute_stack:
                seeded_fn = handlers.substitute(seeded_fn,
                                                substitute_fn=partial(_subs_wrapper, subs_map, i))

            with handlers.trace() as trace:
                carry, y = seeded_fn(carry, x)

        return (i + 1, rng_key, carry), (PytreeTrace(trace), y)

    return lax.scan(body_fn, (np.array(0), rng_key, init_value), xs, length=length)


def scan(name, fn, init_value, xs, length=None, rng_key=None):
    """
    This primitive scans a function over the leading array axes of
    `xs` while carrying along state. See :func:`jax.lax.scan` for more
    information.

    **Usage**::

    .. doctest::

       >>> import jax.numpy as np
       >>> import numpyro
       >>> import numpyro.distributions as dist
       >>> from numpyro.contrib.control_flow import scan
       >>>
       >>> def gaussian_hmm(y=None, T=10):
       ...     def transition(x_prev, y_curr):
       ...         x_curr = numpyro.sample('x', dist.Normal(x_prev, 1))
       ...         y_curr = numpyro.sample('y', dist.Normal(x_curr, 1), obs=y_curr)
       ...         return x_curr, (x_curr, y_curr)
       ...
       ...     x0 = numpyro.sample('x_0', dist.Normal(0, 1))
       ...     _, (x, y) = scan('scan', transition, x0, y, length=T)
       ...     return (x, y)
       >>>
       >>> # here we do some quick tests
       >>> with numpyro.handlers.seed(rng_seed=0):
       ...     x, y = gaussian_hmm(np.arange(10.))
       >>> assert x.shape == (10,) and y.shape == (10,)
       >>> assert np.all(y == np.arange(10))
       >>>
       >>> with numpyro.handlers.seed(rng_seed=0):  # generative
       ...     x, y = gaussian_hmm()
       >>> assert x.shape == (10,) and y.shape == (10,)

    :param str name: name of this primitive
    :param callable fn: a function to be scanned.
    :param init_value: the initial carrying state
    :param xs: the values over which we scan along the leading axis. This can
        be any JAX pytree (e.g. list/dict of arrays).
    :param init length: optional value specifying the length of `xs`
        but can be used when `xs` is an empty pytree (e.g. None)
    :param jax.random.PRNGKey rng_key: an optional random key to seed `fn`.
    :return: output of scan, quoted from :func:`jax.lax.scan` docs:
        "pair of type (c, [b]) where the first element represents the final loop
        carry value and the second element represents the stacked outputs of the
        second output of f when scanned over the leading axis of the inputs".
    """
    # if there are no active Messengers, we just run and return it as expected:
    if not _PYRO_STACK:
        (length, rng_key, carry), (pytree_trace, ys) = scan_wrapper(
            fn, init_value, xs, length, rng_key=rng_key)
    else:
        # Otherwise, we initialize a message...
        initial_msg = {
            'type': 'control_flow',
            'name': name,
            'fn': scan_wrapper,
            'args': (fn, init_value, xs, length),
            'kwargs': {'rng_key': rng_key,
                       'substitute_stack': []},
            'value': None,
        }

        # ...and use apply_stack to send it to the Messengers
        msg = apply_stack(initial_msg)
        (length, rng_key, carry), (pytree_trace, ys) = msg['value']

    for msg in pytree_trace.trace.values():
        # XXX: it would be best to have a mechanism to block condition, substitute
        # handlers from processing those messages; so we can
        #   + use `condition` handler in scan_wrapper (currently, we use `substitute`
        #     instead of `condition` there to avoid overwriting `is_observed` field)
        #   + remove `block` handler in `subs_handler` (i.e. we also scanned those
        #     extra sites, e.g. `log_det` sites, created in `substitute_fn`)
        apply_stack(msg)

    return carry, ys
