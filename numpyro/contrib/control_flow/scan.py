# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

from jax import lax, random, tree_flatten
import jax.numpy as jnp
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
            if site['type'] in ['sample', 'deterministic']:
                trace[name], aux_trace[name] = {}, {'_control_flow_done': True}
                for key in site:
                    if key in ['fn', 'args', 'value', 'intermediates']:
                        trace[name][key] = site[key]
                    # scanned sites have stop field because we trace them inside a block handler
                    elif key != 'stop':
                        aux_trace[name][key] = site[key]
        return (trace,), aux_trace

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        trace, = children
        for name, site in trace.items():
            site.update(aux_data[name])
        return cls(trace)


def _subs_wrapper(subs_map, i, length, site):
    value = None
    if isinstance(subs_map, dict) and site['name'] in subs_map:
        value = subs_map[site['name']]
    elif callable(subs_map):
        rng_key = site['kwargs'].get('rng_key')
        subs_map = handlers.seed(subs_map, rng_seed=rng_key) if rng_key is not None else subs_map
        value = subs_map(site)

    if value is not None:
        value_ndim = jnp.ndim(value)
        sample_shape = site['kwargs']['sample_shape']
        fn_ndim = len(sample_shape + site['fn'].shape())
        if value_ndim == fn_ndim:
            # this branch happens when substitute_fn is init_strategy,
            # where we apply init_strategy to each element in the scanned series
            return value
        elif value_ndim == fn_ndim + 1:
            # this branch happens when we substitute a series of values
            shape = jnp.shape(value)
            if shape[0] == length:
                return value[i]
            elif shape[0] < length:
                rng_key = site['kwargs']['rng_key']
                assert rng_key is not None
                # we use the substituted values if i < shape[0]
                # and generate a new sample otherwise
                return lax.cond(i < shape[0],
                                (value, i),
                                lambda val: val[0][val[1]],
                                rng_key,
                                lambda val: site['fn'](rng_key=val, sample_shape=sample_shape))
            else:
                raise RuntimeError(f"Substituted value for site {site['name']} "
                                   "requires length greater than or equal to scan length."
                                   f" Expected length >= {length}, but got {shape[0]}.")
        else:
            raise RuntimeError(f"Something goes wrong. Expected ndim = {fn_ndim} or {fn_ndim+1},"
                               f" but got {value_ndim}. Please report the issue to us!")


def scan_wrapper(f, init, xs, length, reverse, rng_key=None, substitute_stack=[]):

    def body_fn(wrapped_carry, x):
        i, rng_key, carry = wrapped_carry
        rng_key, subkey = random.split(rng_key) if rng_key is not None else (None, None)

        with handlers.block():
            seeded_fn = handlers.seed(f, subkey) if subkey is not None else f
            for subs_type, subs_map in substitute_stack:
                subs_fn = partial(_subs_wrapper, subs_map, i, length)
                if subs_type == 'condition':
                    seeded_fn = handlers.condition(seeded_fn, condition_fn=subs_fn)
                elif subs_type == 'substitute':
                    seeded_fn = handlers.substitute(seeded_fn, substitute_fn=subs_fn)

            with handlers.trace() as trace:
                carry, y = seeded_fn(carry, x)

        return (i + 1, rng_key, carry), (PytreeTrace(trace), y)

    if length is None:
        length = tree_flatten(xs)[0][0].shape[0]
    return lax.scan(body_fn, (jnp.array(0), rng_key, init), xs, length=length, reverse=reverse)


def scan(f, init, xs, length=None, reverse=False):
    """
    This primitive scans a function over the leading array axes of
    `xs` while carrying along state. See :func:`jax.lax.scan` for more
    information.

    **Usage**::

    .. doctest::

       >>> import numpy as np
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
       ...     _, (x, y) = scan(transition, x0, y, length=T)
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

    .. warning:: This is an experimental utility function that allows users to use
        JAX control flow with NumPyro's effect handlers. Currently, `sample` and
        `deterministic` sites within the scan body `f` are supported. If you notice
        that any effect handlers or distributions are unsupported, please file an issue.

    .. note:: It is ambiguous to align `scan` dimension inside a `plate` context.
        So the following pattern won't be supported

            with numpyro.plate('N', 10):
                last, ys = scan(f, init, xs)

        All `plate` statements should be put inside `f`. For example, the corresponding
        working code is

            def g(*args, **kwargs):
                with numpyro.plate('N', 10):
                    return f(*arg, **kwargs)

            last, ys = scan(g, init, xs)

    :param callable f: a function to be scanned.
    :param init: the initial carrying state
    :param xs: the values over which we scan along the leading axis. This can
        be any JAX pytree (e.g. list/dict of arrays).
    :param length: optional value specifying the length of `xs`
        but can be used when `xs` is an empty pytree (e.g. None)
    :param bool reverse: optional boolean specifying whether to run the scan iteration
        forward (the default) or in reverse
    :return: output of scan, quoted from :func:`jax.lax.scan` docs:
        "pair of type (c, [b]) where the first element represents the final loop
        carry value and the second element represents the stacked outputs of the
        second output of f when scanned over the leading axis of the inputs".
    """
    # if there are no active Messengers, we just run and return it as expected:
    if not _PYRO_STACK:
        (length, rng_key, carry), (pytree_trace, ys) = scan_wrapper(
            f, init, xs, length=length, reverse=reverse)
    else:
        # Otherwise, we initialize a message...
        initial_msg = {
            'type': 'control_flow',
            'fn': scan_wrapper,
            'args': (f, init, xs, length, reverse),
            'kwargs': {'rng_key': None,
                       'substitute_stack': []},
            'value': None,
        }

        # ...and use apply_stack to send it to the Messengers
        msg = apply_stack(initial_msg)
        (length, rng_key, carry), (pytree_trace, ys) = msg['value']

    for msg in pytree_trace.trace.values():
        apply_stack(msg)

    return carry, ys
