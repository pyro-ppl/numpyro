# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from functools import partial

from jax import lax, random, tree_flatten, tree_map, tree_multimap, tree_unflatten
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from numpyro import handlers
from numpyro.primitives import _PYRO_STACK, Messenger, apply_stack
from numpyro.util import not_jax_tracer


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
        # keep the site order information because in JAX, flatten and unflatten do not preserve
        # the order of keys in a dict
        site_names = list(trace.keys())
        return (trace,), (aux_trace, site_names)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        aux_trace, site_names = aux_data
        trace, = children
        trace_with_aux = {}
        for name in site_names:
            trace[name].update(aux_trace[name])
            trace_with_aux[name] = trace[name]
        return cls(trace_with_aux)


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
                                   "requires length less than or equal to scan length."
                                   f" Expected length <= {length}, but got {shape[0]}.")
        else:
            raise RuntimeError(f"Something goes wrong. Expected ndim = {fn_ndim} or {fn_ndim+1},"
                               f" but got {value_ndim}. This might happen when you use nested scan,"
                               " which is currently not supported. Please report the issue to us!")


class promote_shapes(Messenger):
    # a helper messenger to promote shapes of `fn` and `value`
    #   + msg: fn.batch_shape = (2, 3), value.shape = (3,) + fn.event_shape
    #     process_message(msg): promote value so that value.shape = (1, 3) + fn.event_shape
    #   + msg: fn.batch_shape = (3,), value.shape = (2, 3) + fn.event_shape
    #     process_message(msg): promote fn so that fn.batch_shape = (1, 3).
    def process_message(self, msg):
        if msg["type"] == "sample" and msg["value"] is not None:
            fn, value = msg["fn"], msg["value"]
            value_batch_ndims = jnp.ndim(value) - fn.event_dim
            fn_batch_ndim = len(fn.batch_shape)
            prepend_shapes = (1,) * abs(fn_batch_ndim - value_batch_ndims)
            if fn_batch_ndim > value_batch_ndims:
                msg["value"] = jnp.reshape(value, prepend_shapes + jnp.shape(value))
            elif fn_batch_ndim < value_batch_ndims:
                msg["fn"] = tree_map(lambda x: jnp.reshape(x, prepend_shapes + jnp.shape(x)), fn)


def scan_enum(f, init, xs, length, reverse, rng_key=None, substitute_stack=None, history=1):
    from numpyro.contrib.funsor import config_enumerate, enum, markov
    from numpyro.contrib.funsor import trace as packed_trace

    history = min(history, length)
    if reverse:
        x0 = tree_map(lambda x: x[-history:][::-1], xs)
        xs_ = tree_map(lambda x: x[:-history], xs)
    else:
        x0 = tree_map(lambda x: x[:history], xs)
        xs_ = tree_map(lambda x: x[history:], xs)

    carry_shapes = []

    def body_fn(wrapped_carry, x, prefix=None):
        i, rng_key, carry = wrapped_carry
        init = True if (not_jax_tracer(i) and i in range(history)) else False
        rng_key, subkey = random.split(rng_key) if rng_key is not None else (None, None)

        # we need to tell unconstrained messenger in potential energy computation
        # that only the item at time `i` is needed when transforming
        fn = handlers.infer_config(f, config_fn=lambda msg: {'_scan_current_index': i})

        seeded_fn = handlers.seed(fn, subkey) if subkey is not None else fn
        for subs_type, subs_map in substitute_stack:
            subs_fn = partial(_subs_wrapper, subs_map, i, length)
            if subs_type == 'condition':
                seeded_fn = handlers.condition(seeded_fn, condition_fn=subs_fn)
            elif subs_type == 'substitute':
                seeded_fn = handlers.substitute(seeded_fn, substitute_fn=subs_fn)

        if init:
            # handler the name to match the pattern of sakkar_bilmes product
            with handlers.scope(prefix='P' * (history - i), divider='_'):
                new_carry, y = seeded_fn(carry, x)
                trace = {}
        else:
            with handlers.block(), packed_trace() as trace, promote_shapes(), enum(), markov():
                # Like scan_wrapper, we collect the trace of scan's transition function
                # `seeded_fn` here. To put time dimension to the correct position, we need to
                # promote shapes to make `fn` and `value`
                # at each site have the same batch dims (e.g. if `fn.batch_shape = (2, 3)`,
                # and value's batch_shape is (3,), then we promote shape of
                # value so that its batch shape is (1, 3)).
                with handlers.scope(divider='_'):
                    new_carry, y = config_enumerate(seeded_fn)(carry, x)

            # store shape of new_carry at a global variable
            if len(carry_shapes) < (history + 1):
                carry_shapes.append([jnp.shape(x) for x in tree_flatten(new_carry)[0]])
            # make new_carry have the same shape as carry
            # FIXME: is this rigorous?
            new_carry = tree_multimap(lambda a, b: jnp.reshape(a, jnp.shape(b)),
                                      new_carry, carry)
        return (i + jnp.array(1), rng_key, new_carry), (PytreeTrace(trace), y)

    with markov(history=history):
        wrapped_carry = (0, rng_key, init)
        y0s = []
        for i in range(history):
            wrapped_carry, (_, y0) = body_fn(wrapped_carry, tree_map(lambda z: z[i], x0))
            if i > 0:
                # reshape y1, y2,... to have the same shape as y0
                y0 = tree_multimap(lambda z0, z: jnp.reshape(z, jnp.shape(z0)), y0s[0], y0)
            y0s.append(y0)
            carry_shapes.append([jnp.shape(x) for x in tree_flatten(wrapped_carry[-1])[0]])
        y0s = tree_multimap(lambda *z: jnp.stack(z, axis=0), *y0s)
        if length == history:
            return wrapped_carry, (PytreeTrace({}), y0s)
        wrapped_carry, (pytree_trace, ys) = lax.scan(body_fn, wrapped_carry, xs_, length - history, reverse)

    first_var = None
    for name, site in pytree_trace.trace.items():
        # currently, we only record sample or deterministic in the trace
        # we don't need to adjust `dim_to_name` for deterministic site
        if site['type'] not in ('sample',):
            continue
        # add `time` dimension, the name will be '_time_{first variable in the trace}'
        if first_var is None:
            first_var = name

        # XXX: site['infer']['dim_to_name'] is not enough to determine leftmost dimension because
        # we don't record 1-size dimensions in this field
        time_dim = -min(len(site['fn'].batch_shape), jnp.ndim(site['value']) - site['fn'].event_dim)
        site['infer']['dim_to_name'][time_dim] = '_time_{}'.format(first_var)

    # similar to carry, we need to reshape due to shape alternating in markov
    ys = tree_multimap(lambda z0, z: jnp.reshape(z, z.shape[:1] + jnp.shape(z0)[1:]), y0s, ys)
    # then join with y0s
    ys = tree_multimap(lambda z0, z: jnp.concatenate([z0, z], axis=0), y0s, ys)
    # we also need to reshape `carry` to match sequential behavior
    i = (length - 1) % (history + 1)
    if i != history - 1:  # no need to reshape if i == history - 1 (i.e. before scanning)
        t, rng_key, carry = wrapped_carry
        carry_shape = carry_shapes[i]
        flatten_carry, treedef = tree_flatten(carry)
        flatten_carry = [jnp.reshape(x, t1_shape)
                         for x, t1_shape in zip(flatten_carry, carry_shape)]
        carry = tree_unflatten(treedef, flatten_carry)
        wrapped_carry = (t, rng_key, carry)
    return wrapped_carry, (pytree_trace, ys)


def scan_wrapper(f, init, xs, length, reverse, rng_key=None, substitute_stack=[], enum=False, history=1):
    if length is None:
        length = tree_flatten(xs)[0][0].shape[0]

    if enum and history > 0:
        return scan_enum(f, init, xs, length, reverse, rng_key, substitute_stack, history)

    def body_fn(wrapped_carry, x):
        i, rng_key, carry = wrapped_carry
        rng_key, subkey = random.split(rng_key) if rng_key is not None else (None, None)

        with handlers.block():

            # we need to tell unconstrained messenger in potential energy computation
            # that only the item at time `i` is needed when transforming
            fn = handlers.infer_config(f, config_fn=lambda msg: {'_scan_current_index': i})

            seeded_fn = handlers.seed(fn, subkey) if subkey is not None else fn
            for subs_type, subs_map in substitute_stack:
                subs_fn = partial(_subs_wrapper, subs_map, i, length)
                if subs_type == 'condition':
                    seeded_fn = handlers.condition(seeded_fn, condition_fn=subs_fn)
                elif subs_type == 'substitute':
                    seeded_fn = handlers.substitute(seeded_fn, substitute_fn=subs_fn)

            with handlers.trace() as trace:
                carry, y = seeded_fn(carry, x)

        return (i + 1, rng_key, carry), (PytreeTrace(trace), y)

    return lax.scan(body_fn, (jnp.array(0), rng_key, init), xs, length=length, reverse=reverse)


def scan(f, init, xs, length=None, reverse=False, history=1):
    """
    This primitive scans a function over the leading array axes of
    `xs` while carrying along state. See :func:`jax.lax.scan` for more
    information.

    **Usage**:

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

        .. code-block:: python

            with numpyro.plate('N', 10):
                last, ys = scan(f, init, xs)

        All `plate` statements should be put inside `f`. For example, the corresponding
        working code is

        .. code-block:: python

            def g(*args, **kwargs):
                with numpyro.plate('N', 10):
                    return f(*arg, **kwargs)

            last, ys = scan(g, init, xs)

    .. note:: Nested scan is currently not supported.

    .. note:: We can scan over discrete latent variables in `f`. The joint density is
        evaluated using parallel-scan (reference [1]) over time dimension, which
        reduces parallel complexity to `O(log(length))`.

        A :class:`~numpyro.handlers.trace` of `scan` with discrete latent
        variables will contain the following sites:

            + init sites: those sites belong to the first `history` traces of `f`.
                Sites at the `i`-th trace will have name prefixed with `P` * (history - i).
            + scanned sites: those sites collect the values of the remaining scan
                loop over `f`. An addition time dimension `_time_foo` will be
                added to those sites, where `foo` is the name of the first site
                appeared in `f`.

        Not all transition functions `f` are supported. All of the restrictions from
        Pyro's enumeration tutorial [2] still apply here. In addition, there should
        not have any site outside of `scan` depend on the first output of `scan`
        (the last carry value).

    ** References **

    1. *Temporal Parallelization of Bayesian Smoothers*,
       Simo Sarkka, Angel F. Garcia-Fernandez
       (https://arxiv.org/abs/1905.13002)

    2. *Inference with Discrete Latent Variables*
       (http://pyro.ai/examples/enumeration.html#Dependencies-among-plates)

    :param callable f: a function to be scanned.
    :param init: the initial carrying state
    :param xs: the values over which we scan along the leading axis. This can
        be any JAX pytree (e.g. list/dict of arrays).
    :param length: optional value specifying the length of `xs`
        but can be used when `xs` is an empty pytree (e.g. None)
    :param bool reverse: optional boolean specifying whether to run the scan iteration
        forward (the default) or in reverse
    :param int history: The number of previous contexts visible from the current context.
        Defaults to 1. If zero, this is similar to :class:`numpyro.plate`.
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
                       'substitute_stack': [],
                       'history': history},
            'value': None,
        }

        # ...and use apply_stack to send it to the Messengers
        msg = apply_stack(initial_msg)
        (length, rng_key, carry), (pytree_trace, ys) = msg['value']

    if not msg["kwargs"].get("enum", False):
        for msg in pytree_trace.trace.values():
            apply_stack(msg)
    else:
        from numpyro.contrib.funsor import to_funsor
        from numpyro.contrib.funsor.enum_messenger import LocalNamedMessenger

        for msg in pytree_trace.trace.values():
            with LocalNamedMessenger():
                dim_to_name = msg["infer"].get("dim_to_name")
                to_funsor(msg["value"], dim_to_name=OrderedDict([(k, dim_to_name[k]) for k in sorted(dim_to_name)]))
                apply_stack(msg)

    return carry, ys
