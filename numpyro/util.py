from collections import namedtuple
from contextlib import contextmanager
import os
import random
import re

import numpy as onp
import tqdm

import jax
from jax import jit, lax, ops, vmap
from jax.lib.xla_bridge import canonicalize_dtype
import jax.numpy as np
from jax.tree_util import tree_flatten, tree_map, tree_unflatten

_DATA_TYPES = {}
_DISABLE_CONTROL_FLOW_PRIM = False


def set_rng_seed(rng_seed):
    random.seed(rng_seed)
    onp.random.seed(rng_seed)


def set_platform(platform=None):
    """
    Changes platform to CPU, GPU, or TPU. This utility only takes
    effect at the beginning of your program.

    :param str platform: either 'cpu', 'gpu', or 'tpu'.
    """
    if platform is None:
        platform = os.getenv('JAX_PLATFORM_NAME', 'cpu')
    jax.config.update('jax_platform_name', platform)


def set_host_devices(n):
    """
    By default, XLA considers all CPU cores as one device. This utility tells XLA
    that there are `n` host (CPU) devices available to use. As a consequence, this
    allows parallel mapping in JAX :func:`jax.pmap` to work in CPU platform.

    .. note:: This utility only takes effect at the beginning of your program.
        Under the hood, this sets the environment variable
        `XLA_FLAGS=--xla_force_host_platform_device_count=[num_devices]`, where
        `[num_device]` is the desired number of CPU devices `n`.

    .. warning:: We do not understand much the side effects when using
        `xla_force_host_platform_device_count` flag. If you observe some strange
        phenomenon when using this utility, please let us know through our issue
        or forum page. Here we quote from XLA source code the meaning of this flag:
        "Force the host platform to pretend that there are these many host
        'devices'. All of these host devices are backed by the same threadpool.
        Setting this to anything other than 1 can increase overhead from context
        switching but we let the user override this behavior to help run tests
        on the host that run models in parallel across multiple devices."

    :param int n: number of CPU devices to use.
    """
    xla_flags = os.getenv('XLA_FLAGS', '').lstrip('--')
    xla_flags = re.sub('xla_force_host_platform_device_count=.+\s', '', xla_flags).split()
    os.environ['XLA_FLAGS'] = ' '.join(['--xla_force_host_platform_device_count={}'.format(n)]
                                       + xla_flags)


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


def fori_collect(lower, upper, body_fun, init_val, transform=identity, progbar=True, **progbar_opts):
    """
    This looping construct works like :func:`~jax.lax.fori_loop` but with the additional
    effect of collecting values from the loop body. In addition, this allows for
    post-processing of these samples via `transform`, and progress bar updates.
    Note that, `progbar=False` will be faster, especially when collecting a
    lot of samples. Refer to example usage in :func:`~numpyro.mcmc.hmc`.

    :param int lower: the index to start the collective work. In other words,
        we will skip collecting the first `lower` values.
    :param int upper: number of times to run the loop body.
    :param body_fun: a callable that takes a collection of
        `np.ndarray` and returns a collection with the same shape and
        `dtype`.
    :param init_val: initial value to pass as argument to `body_fun`. Can
        be any Python collection type containing `np.ndarray` objects.
    :param transform: a callable to post-process the values returned by `body_fn`.
    :param progbar: whether to post progress bar updates.
    :param `**progbar_opts`: optional additional progress bar arguments. A
        `diagnostics_fn` can be supplied which when passed the current value
        from `body_fun` returns a string that is used to update the progress
        bar postfix. Also a `progbar_desc` keyword argument can be supplied
        which is used to label the progress bar.
    :return: collection with the same type as `init_val` with values
        collected along the leading axis of `np.ndarray` objects.
    """
    assert lower < upper
    init_val_flat, unravel_fn = ravel_pytree(transform(init_val))
    ravel_fn = lambda x: ravel_pytree(transform(x))[0]  # noqa: E731

    if not progbar:
        collection = np.zeros((upper - lower,) + init_val_flat.shape)

        def _body_fn(i, vals):
            val, collection = vals
            val = body_fun(val)
            i = np.where(i >= lower, i - lower, 0)
            collection = ops.index_update(collection, i, ravel_fn(val))
            return val, collection

        _, collection = fori_loop(0, upper, _body_fn, (init_val, collection))
    else:
        diagnostics_fn = progbar_opts.pop('diagnostics_fn', None)
        progbar_desc = progbar_opts.pop('progbar_desc', lambda x: '')
        collection = []

        val = init_val
        with tqdm.trange(upper) as t:
            for i in t:
                val = jit(body_fun)(val)
                if i >= lower:
                    collection.append(jit(ravel_fn)(val))
                t.set_description(progbar_desc(i), refresh=False)
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


pytree_metadata = namedtuple('pytree_metadata', ['flat', 'shape', 'size', 'dtype'])


def _ravel_list(*leaves):
    leaves_metadata = tree_map(lambda l: pytree_metadata(
        np.ravel(l), np.shape(l), np.size(l), canonicalize_dtype(lax.dtype(l))), leaves)
    leaves_idx = np.cumsum(np.array((0,) + tuple(d.size for d in leaves_metadata)))

    def unravel_list(arr):
        return [np.reshape(lax.dynamic_slice_in_dim(arr, leaves_idx[i], m.size),
                           m.shape).astype(m.dtype)
                for i, m in enumerate(leaves_metadata)]

    flat = np.concatenate([m.flat for m in leaves_metadata]) if leaves_metadata else np.array([])
    return flat, unravel_list


def ravel_pytree(pytree):
    leaves, treedef = tree_flatten(pytree)
    flat, unravel_list = _ravel_list(*leaves)

    def unravel_pytree(arr):
        return tree_unflatten(treedef, unravel_list(arr))

    return flat, unravel_pytree
