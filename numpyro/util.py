import random
from collections import namedtuple
from contextlib import contextmanager

import numpy as onp

import jax.numpy as np
from jax import core, lax
from jax.abstract_arrays import ShapedArray
from jax.api_util import pytree_fun_to_flatjaxtuple_fun, pytree_to_flatjaxtuple
from jax.interpreters import partial_eval, xla
from jax.linear_util import wrap_init
from jax.tree_util import register_pytree_node, tree_flatten, tree_map, tree_multimap, tree_unflatten
from jax.util import partial

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
            a_out = tree_map(lambda x: x[None, ...], a)
            if i == 0:
                out = a_out
            else:
                out = tree_multimap(lambda x, y: np.concatenate((x, y)), out, a_out)
        return out
    else:
        return lax.scan(f, a, bs)


def tscan(f, a, bs, fields=(0,)):
    if _DISABLE_CONTROL_FLOW_PRIM:
        length = tree_flatten(bs)[0][0].shape[0]
        for i in range(length):
            b = tree_map(lambda x: x[i], bs)
            a = f(a, b)
            a_out = tree_map(lambda x: x[None, ...], a)

            # the following three lines are necessary for tscan;
            # it might be useful in the case you want to use `transform` instead of `fields`
            a_flat, a_tree = tree_flatten(a_out)
            a_selected = [field if i in fields else None for i, field in enumerate(a_out)]
            a_out = tree_unflatten(a_tree, a_selected)

            if i == 0:
                out = a_out
            else:
                out = tree_multimap(lambda x, y: np.concatenate((x, y)) if x is not None else None,
                                    out, a_out)
        return out
    else:
        return _tscan(f, a, bs, fields)


def _tscan(f, a, bs, fields=(0,)):
    """
    Works as jax.lax.scan but has additional `fields` argument to select only
    necessary fields from `a`'s structure. Defaults to selecting only the first
    field. Other fields will be filled by None.
    """
    # Note: code is copied and modified from lax.scan implementation in
    # [JAX](https://github.com/google/jax) to support the additional `fields`
    # arg. Original code has the following copyright:
    #
    # Copyright 2018 Google LLC
    #
    # Licensed under the Apache License, Version 2.0 (the "License")

    # convert pytree to flat jaxtuple
    a, a_tree = pytree_to_flatjaxtuple(a)
    bs, b_tree = pytree_to_flatjaxtuple(bs)
    fields, _ = pytree_to_flatjaxtuple(fields)
    f, out_tree = pytree_fun_to_flatjaxtuple_fun(wrap_init(f), (a_tree, b_tree))

    # convert arrays to abstract values
    a_aval, _ = lax._abstractify(a)
    bs_aval, _ = lax._abstractify(bs)
    # convert bs to b
    b_aval = core.AbstractTuple([ShapedArray(b.shape[1:], b.dtype) for b in bs_aval])

    # convert abstract values to partial values (?) then evaluate to get jaxpr
    a_pval = partial_eval.PartialVal((a_aval, core.unit))
    b_pval = partial_eval.PartialVal((b_aval, core.unit))
    jaxpr, pval_out, consts = partial_eval.trace_to_jaxpr(f, (a_pval, b_pval))
    aval_out, _ = pval_out
    consts = core.pack(consts)

    out = tscan_p.bind(a, bs, fields, consts, aval_out=aval_out, jaxpr=jaxpr)
    return tree_unflatten(out_tree(), out)


def _tscan_impl(a, bs, fields, consts, aval_out, jaxpr):
    length = tuple(bs)[0].shape[0]
    state = [lax.full((length,) + a[i].shape, 0, lax._dtype(a[i])) for i in fields]

    def body_fun(i, vals):
        a, state = vals
        # select i-th element from each b
        b = [lax.dynamic_index_in_dim(b, i, keepdims=False) for b in bs]
        a_out = core.eval_jaxpr(jaxpr, consts, (), a, core.pack(b))
        # select fields from a_out and update state
        state_out = [lax.dynamic_update_index_in_dim(s, a[None, ...], i, axis=0)
                     for a, s in zip([tuple(a_out)[j] for j in fields], state)]
        return a_out, state_out

    _, state = lax.fori_loop(0, length, body_fun, (a, state))

    # set None for non-selected fields
    out = [None] * len(a)
    for field, i in zip(fields, range(len(fields))):
        out[field] = state[i]
    return core.pack(out)


def _tscan_abstract_eval(a, bs, fields, consts, aval_out, jaxpr):
    return lax.maybe_tracer_tuple_to_abstract_tuple(aval_out)


tscan_p = core.Primitive('tscan')
tscan_p.def_impl(_tscan_impl)
tscan_p.def_abstract_eval(_tscan_abstract_eval)
xla.translations[tscan_p] = partial(xla.lower_fun, _tscan_impl)
