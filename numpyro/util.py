import random
from collections import namedtuple
from contextlib import contextmanager

import numpy as onp
import tqdm

import jax.numpy as np
from jax import jit, lax, ops, vmap
from jax.flatten_util import ravel_pytree
from jax.tree_util import register_pytree_node

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


def identity(x):
    return x


def fori_collect(n, body_fun, init_val, transform=identity, progbar=True, **progbar_opts):
    """
    This looping construct works like :func:`~jax.lax.fori_loop` but with the additional
    effect of collecting values from the loop body. In addition, this allows for
    post-processing of these samples via `transform`, and progress bar updates.
    Note that, in some cases, `progbar=False` can be faster, when collecting a
    lot of samples. Refer to example usage in :func:`~numpyro.mcmc.hmc`.

    :param int n: number of times to run the loop body.
    :param body_fun: a callable that takes a collection of
        `np.ndarray` and returns a collection with the same shape and
        `dtype`.
    :param init_val: initial value to pass as argument to `body_fun`. Can
        be any Python collection type containing `np.ndarray` objects.
    :param transform: A callable
    :param progbar: whether to post progress bar updates.
    :param `**progbar_opts`: optional additional progress bar arguments. A
        `diagnostics_fn` can be supplied which when passed the current value
        from `body_fun` returns a string that is used to update the progress
        bar postfix. Also a `progbar_desc` keyword argument can be supplied
        which is used to label the progress bar.
    :return: collection with the same type as `init_val` with values
        collected along the leading axis of `np.ndarray` objects.
    """
    init_val_flat, unravel_fn = ravel_pytree(transform(init_val))
    ravel_fn = lambda x: ravel_pytree(transform(x))[0]  # noqa: E731

    if not progbar:
        collection = np.zeros((n,) + init_val_flat.shape, dtype=init_val_flat.dtype)

        def _body_fn(i, vals):
            val, collection = vals
            val = body_fun(val)
            collection = ops.index_update(collection, i, ravel_fn(val))
            return val, collection

        _, collection = fori_loop(0, n, _body_fn, (init_val, collection))
    else:
        diagnostics_fn = progbar_opts.pop('diagnostics_fn', None)
        progbar_desc = progbar_opts.pop('progbar_desc', '')
        collection = []

        val = init_val
        with tqdm.trange(n, desc=progbar_desc) as t:
            for _ in t:
                val = body_fun(val)
                collection.append(jit(ravel_fn)(val))
                if diagnostics_fn:
                    t.set_postfix_str(diagnostics_fn(val), refresh=False)

        collection = np.stack(collection)

    return vmap(unravel_fn)(collection)


def copy_docs_from(source_class, full_text=False):
    """
    Decorator to copy class and method docs from source to destin class.
    """

    def decorator(destin_class):
        # This works only in python 3.3+:
        # if not destin_class.__doc__:
        #     destin_class.__doc__ = source_class.__doc__
        for name in dir(destin_class):
            if name.startswith('_'):
                continue
            destin_attr = getattr(destin_class, name)
            destin_attr = getattr(destin_attr, '__func__', destin_attr)
            source_attr = getattr(source_class, name, None)
            source_doc = getattr(source_attr, '__doc__', None)
            if source_doc and not getattr(destin_attr, '__doc__', None):
                if full_text or source_doc.startswith('See '):
                    destin_doc = source_doc
                else:
                    destin_doc = 'See :meth:`{}.{}.{}`'.format(
                        source_class.__module__, source_class.__name__, name)
                if isinstance(destin_attr, property):
                    # Set docs for object properties.
                    # Since __doc__ is read-only, we need to reset the property
                    # with the updated doc.
                    updated_property = property(destin_attr.fget,
                                                destin_attr.fset,
                                                destin_attr.fdel,
                                                destin_doc)
                    setattr(destin_class, name, updated_property)
                else:
                    destin_attr.__doc__ = destin_doc
        return destin_class

    return decorator
