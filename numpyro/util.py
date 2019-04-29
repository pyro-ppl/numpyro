import random
from collections import namedtuple
from contextlib import contextmanager

import numpy as onp
import tqdm

import jax
import jax.numpy as np
from jax import jit, lax, ops, vmap
from jax.flatten_util import ravel_pytree
from jax.tree_util import register_pytree_node, tree_flatten, tree_map, tree_multimap, tree_unflatten

_DATA_TYPES = {}
_DISABLE_CONTROL_FLOW_PRIM = False


def set_rng_seed(rng_seed):
    random.seed(rng_seed)
    onp.random.seed(rng_seed)


# let JAX recognize _TreeInfo structure
# ref: https://github.com/google/jax/issues/446
# TODO: remove this when namedtuple is supported in JAX
def register_pytree(cls):
    if not getattr(cls, '_registered', False):
        register_pytree_node(
            cls,
            lambda xs: (tuple(xs), None),
            lambda _, xs: cls(*xs)
        )
    cls._registered = True


def laxtuple(name, fields):
    key = (name,) + tuple(fields)
    if key in _DATA_TYPES:
        return _DATA_TYPES[key]
    cls = namedtuple(name, fields)
    register_pytree(cls)
    cls.update = cls._replace
    _DATA_TYPES[key] = cls
    return cls


@contextmanager
def optional(condition, context_manager):
    """
    Optionally wrap inside `context_manager` if condition is `True`.
    """
    if condition:
        with context_manager:
            yield
    else:
        yield


@contextmanager
def control_flow_prims_disabled():
    global _DISABLE_CONTROL_FLOW_PRIM
    stored_flag = _DISABLE_CONTROL_FLOW_PRIM
    try:
        _DISABLE_CONTROL_FLOW_PRIM = True
        yield
    finally:
        _DISABLE_CONTROL_FLOW_PRIM = stored_flag


def cond(pred, true_operand, true_fun, false_operand, false_fun):
    if _DISABLE_CONTROL_FLOW_PRIM:
        if pred:
            return true_fun(true_operand)
        else:
            return false_fun(false_operand)
    else:
        return lax.cond(pred, true_operand, true_fun, false_operand, false_fun)


def while_loop(cond_fun, body_fun, init_val):
    if _DISABLE_CONTROL_FLOW_PRIM:
        val = init_val
        while cond_fun(val):
            val = body_fun(val)
        return val
    else:
        return lax.while_loop(cond_fun, body_fun, init_val)


def fori_loop(lower, upper, body_fun, init_val):
    if _DISABLE_CONTROL_FLOW_PRIM:
        val = init_val
        for i in range(int(lower), int(upper)):
            val = body_fun(i, val)
        return val
    else:
        return lax.fori_loop(lower, upper, body_fun, init_val)


def scan(f, a, bs):
    if _DISABLE_CONTROL_FLOW_PRIM:
        length = tree_flatten(bs)[0][0].shape[0]
        for i in range(length):
            b = tree_map(lambda x: x[i], bs)
            a = f(a, b)
            a_out = tree_map(lambda x: np.expand_dims(x, axis=0), a)
            if i == 0:
                out = a_out
            else:
                out = tree_multimap(lambda x, y: np.concatenate((x, y)), out, a_out)
        return out
    else:
        return lax.scan(f, a, bs)


def _identity(x):
    return x


def fori_append(f, a, n, transform=_identity, jit=True):
    init_val, a_treedef = tree_flatten(a)
    a, trans_treedef = tree_flatten(transform(a))
    state = [lax.full((n,) + np.shape(i), 0, lax._dtype(i)) for i in a]

    def body_fun(i, vals):
        a, state = vals
        a = tree_unflatten(a_treedef, a)
        a_out = f(a)
        state_out = transform(a_out)
        a_out, state_out = tree_flatten(a_out)[0], tree_flatten(state_out)[0]
        state_out = [lax.dynamic_update_index_in_dim(s, t[None, ...], i, axis=0)
                     for t, s in zip(state_out, state)]
        return a_out, state_out

    def fn_loop(n, x): return lax.fori_loop(0, n, body_fun, x)

    if jit:
        # JIT on a single step.
        fn_loop = jax.jit(fn_loop)
        fn_loop(1, (init_val, state))
    _, state = fn_loop(n, (init_val, state))
    return tree_unflatten(trans_treedef, state)


def fori_collect(n, body_fun, init_val, transform=_identity, progbar=True, use_prims=True):
    # works like lax.fori_loop but ignores i in body_fn, supports
    # postprocessing `transform`, and collects values during the loop
    init_val_flat, unravel_fn = ravel_pytree(transform(init_val))
    ravel_fn = lambda x: ravel_pytree(transform(x))[0]  # noqa: E731

    if use_prims:
        collection = np.zeros((n,) + init_val_flat.shape, dtype=init_val_flat.dtype)

        def _body_fn(i, vals):
            val, collection = vals
            val = body_fun(val)
            collection = ops.index_update(collection, i, ravel_fn(val))
            return val, collection

        _, collection = jit(lax.fori_loop, static_argnums=(2,))(0, n, _body_fn,
                                                                (init_val, collection))
    else:
        collection = []

        val = init_val
        n_iter = tqdm.trange(n) if progbar else range(n)
        for i in n_iter:
            val = body_fun(val)
            collection.append(jit(ravel_fn)(val))

        # XXX: jax.numpy.stack/concatenate is currently so slow
        collection = onp.stack(collection)

    return vmap(unravel_fn)(collection)
