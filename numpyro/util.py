# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from contextlib import contextmanager
import os
import random
import re
import warnings

import numpy as np
import tqdm
from tqdm.auto import tqdm as tqdm_auto

import jax
from jax import device_put, jit, lax, ops, vmap
from jax.core import Tracer
from jax.experimental import host_callback
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_map

_DISABLE_CONTROL_FLOW_PRIM = False
_CHAIN_RE = re.compile(r"(?<=_)\d+$")  # e.g. get '3' from 'TFRT_CPU_3'


def set_rng_seed(rng_seed):
    """
    Initializes internal state for the Python and NumPy random number generators.

    :param int rng_seed: seed for Python and NumPy random states.
    """
    random.seed(rng_seed)
    np.random.seed(rng_seed)


def enable_x64(use_x64=True):
    """
    Changes the default array type to use 64 bit precision as in NumPy.

    :param bool use_x64: when `True`, JAX arrays will use 64 bits by default;
        else 32 bits.
    """
    if not use_x64:
        use_x64 = os.getenv("JAX_ENABLE_X64", 0)
    jax.config.update("jax_enable_x64", use_x64)


def set_platform(platform=None):
    """
    Changes platform to CPU, GPU, or TPU. This utility only takes
    effect at the beginning of your program.

    :param str platform: either 'cpu', 'gpu', or 'tpu'.
    """
    if platform is None:
        platform = os.getenv("JAX_PLATFORM_NAME", "cpu")
    jax.config.update("jax_platform_name", platform)


def set_host_device_count(n):
    """
    By default, XLA considers all CPU cores as one device. This utility tells XLA
    that there are `n` host (CPU) devices available to use. As a consequence, this
    allows parallel mapping in JAX :func:`jax.pmap` to work in CPU platform.

    .. note:: This utility only takes effect at the beginning of your program.
        Under the hood, this sets the environment variable
        `XLA_FLAGS=--xla_force_host_platform_device_count=[num_devices]`, where
        `[num_device]` is the desired number of CPU devices `n`.

    .. warning:: Our understanding of the side effects of using the
        `xla_force_host_platform_device_count` flag in XLA is incomplete. If you
        observe some strange phenomenon when using this utility, please let us
        know through our issue or forum page. More information is available in this
        `JAX issue <https://github.com/google/jax/issues/1408>`_.

    :param int n: number of CPU devices to use.
    """
    xla_flags = os.getenv("XLA_FLAGS", "")
    xla_flags = re.sub(
        r"--xla_force_host_platform_device_count=\S+", "", xla_flags
    ).split()
    os.environ["XLA_FLAGS"] = " ".join(
        ["--xla_force_host_platform_device_count={}".format(n)] + xla_flags
    )


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


def not_jax_tracer(x):
    """
    Checks if `x` is not an array generated inside `jit`, `pmap`, `vmap`, or `lax_control_flow`.
    """
    return not isinstance(x, Tracer)


def identity(x, *args, **kwargs):
    return x


def cached_by(outer_fn, *keys):
    # Restrict cache size to prevent ref cycles.
    max_size = 8
    outer_fn._cache = getattr(outer_fn, "_cache", OrderedDict())

    def _wrapped(fn):
        fn_cache = outer_fn._cache
        if keys in fn_cache:
            fn = fn_cache[keys]
            # update position
            del fn_cache[keys]
            fn_cache[keys] = fn
        else:
            fn_cache[keys] = fn
        if len(fn_cache) > max_size:
            fn_cache.popitem(last=False)
        return fn

    return _wrapped


def progress_bar_factory(num_samples, num_chains):
    """Factory that builds a progress bar decorator along
    with the `set_tqdm_description` and `close_tqdm` functions
    """

    if num_samples > 20:
        print_rate = int(num_samples / 20)
    else:
        print_rate = 1

    remainder = num_samples % print_rate

    tqdm_bars = {}
    finished_chains = []
    for chain in range(num_chains):
        tqdm_bars[chain] = tqdm_auto(range(num_samples), position=chain)
        tqdm_bars[chain].set_description("Compiling.. ", refresh=True)

    def _update_tqdm(arg, transform, device):
        chain_match = _CHAIN_RE.search(str(device))
        assert chain_match
        chain = int(chain_match.group())
        tqdm_bars[chain].set_description(f"Running chain {chain}", refresh=False)
        tqdm_bars[chain].update(arg)

    def _close_tqdm(arg, transform, device):
        chain_match = _CHAIN_RE.search(str(device))
        assert chain_match
        chain = int(chain_match.group())
        tqdm_bars[chain].update(arg)
        finished_chains.append(chain)
        if len(finished_chains) == num_chains:
            for chain in range(num_chains):
                tqdm_bars[chain].close()

    def _update_progress_bar(iter_num):
        """Updates tqdm progress bar of a JAX loop only if the iteration number is a multiple of the print_rate
        Usage: carry = progress_bar((iter_num, print_rate), carry)
        """

        _ = lax.cond(
            iter_num == 1,
            lambda _: host_callback.id_tap(
                _update_tqdm, 0, result=iter_num, tap_with_device=True
            ),
            lambda _: iter_num,
            operand=None,
        )
        _ = lax.cond(
            iter_num % print_rate == 0,
            lambda _: host_callback.id_tap(
                _update_tqdm, print_rate, result=iter_num, tap_with_device=True
            ),
            lambda _: iter_num,
            operand=None,
        )
        _ = lax.cond(
            iter_num == num_samples,
            lambda _: host_callback.id_tap(
                _close_tqdm, remainder, result=iter_num, tap_with_device=True
            ),
            lambda _: iter_num,
            operand=None,
        )

    def progress_bar_fori_loop(func):
        """Decorator that adds a progress bar to `body_fun` used in `lax.fori_loop`.
        Note that `body_fun` must be looping over a tuple who's first element is `np.arange(num_samples)`.
        This means that `iter_num` is the current iteration number
        """

        def wrapper_progress_bar(i, vals):
            result = func(i, vals)
            _update_progress_bar(i + 1)
            return result

        return wrapper_progress_bar

    return progress_bar_fori_loop


def fori_collect(
    lower,
    upper,
    body_fun,
    init_val,
    transform=identity,
    progbar=True,
    return_last_val=False,
    collection_size=None,
    thinning=1,
    **progbar_opts,
):
    """
    This looping construct works like :func:`~jax.lax.fori_loop` but with the additional
    effect of collecting values from the loop body. In addition, this allows for
    post-processing of these samples via `transform`, and progress bar updates.
    Note that, `progbar=False` will be faster, especially when collecting a
    lot of samples. Refer to example usage in :func:`~numpyro.infer.mcmc.hmc`.

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
    :param bool return_last_val: If `True`, the last value is also returned.
        This has the same type as `init_val`.
    :param thinning: Positive integer that controls the thinning ratio for retained
        values. Defaults to 1, i.e. no thinning.
    :param int collection_size: Size of the returned collection. If not
        specified, the size will be ``(upper - lower) // thinning``. If the
        size is larger than ``(upper - lower) // thinning``, only the top
        ``(upper - lower) // thinning`` entries will be non-zero.
    :param `**progbar_opts`: optional additional progress bar arguments. A
        `diagnostics_fn` can be supplied which when passed the current value
        from `body_fun` returns a string that is used to update the progress
        bar postfix. Also a `progbar_desc` keyword argument can be supplied
        which is used to label the progress bar.
    :return: collection with the same type as `init_val` with values
        collected along the leading axis of `np.ndarray` objects.
    """
    assert lower <= upper
    assert thinning >= 1
    collection_size = (
        (upper - lower) // thinning if collection_size is None else collection_size
    )
    assert collection_size >= (upper - lower) // thinning
    init_val_flat, unravel_fn = ravel_pytree(transform(init_val))
    start_idx = lower + (upper - lower) % thinning
    num_chains = progbar_opts.pop("num_chains", 1)
    # host_callback does not work yet with multi-GPU platforms
    # See: https://github.com/google/jax/issues/6447
    if num_chains > 1 and jax.default_backend() == "gpu":
        warnings.warn(
            "We will disable progress bar because it does not work yet on multi-GPUs platforms."
        )
        progbar = False

    @cached_by(fori_collect, body_fun, transform)
    def _body_fn(i, vals):
        val, collection, start_idx, thinning = vals
        val = body_fun(val)
        idx = (i - start_idx) // thinning
        collection = cond(
            idx >= 0,
            collection,
            lambda x: ops.index_update(x, idx, ravel_pytree(transform(val))[0]),
            collection,
            identity,
        )
        return val, collection, start_idx, thinning

    collection = jnp.zeros((collection_size,) + init_val_flat.shape)
    if not progbar:
        last_val, collection, _, _ = fori_loop(
            0, upper, _body_fn, (init_val, collection, start_idx, thinning)
        )
    elif num_chains > 1:
        progress_bar_fori_loop = progress_bar_factory(upper, num_chains)
        _body_fn_pbar = progress_bar_fori_loop(_body_fn)
        last_val, collection, _, _ = fori_loop(
            0, upper, _body_fn_pbar, (init_val, collection, start_idx, thinning)
        )
    else:
        diagnostics_fn = progbar_opts.pop("diagnostics_fn", None)
        progbar_desc = progbar_opts.pop("progbar_desc", lambda x: "")

        vals = (init_val, collection, device_put(start_idx), device_put(thinning))
        if upper == 0:
            # special case, only compiling
            jit(_body_fn)(0, vals)
        else:
            with tqdm.trange(upper) as t:
                for i in t:
                    vals = jit(_body_fn)(i, vals)
                    t.set_description(progbar_desc(i), refresh=False)
                    if diagnostics_fn:
                        t.set_postfix_str(diagnostics_fn(vals[0]), refresh=False)

        last_val, collection, _, _ = vals

    unravel_collection = vmap(unravel_fn)(collection)
    return (unravel_collection, last_val) if return_last_val else unravel_collection


def soft_vmap(fn, xs, batch_ndims=1, chunk_size=None):
    """
    Vectorizing map that maps a function `fn` over `batch_ndims` leading axes
    of `xs`. This uses jax.vmap over smaller chunks of the batch dimensions
    to keep memory usage constant.

    :param callable fn: The function to map over.
    :param xs: JAX pytree (e.g. an array, a list/tuple/dict of arrays,...)
    :param int batch_ndims: The number of leading dimensions of `xs`
        to apply `fn` element-wise over them.
    :param int chunk_size: Size of each chunk of `xs`.
        Defaults to the size of batch dimensions.
    :returns: output of `fn(xs)`.
    """
    flatten_xs = tree_flatten(xs)[0]
    batch_shape = np.shape(flatten_xs[0])[:batch_ndims]
    for x in flatten_xs[1:]:
        assert np.shape(x)[:batch_ndims] == batch_shape

    # we'll do map(vmap(fn), xs) and make xs.shape = (num_chunks, chunk_size, ...)
    num_chunks = batch_size = int(np.prod(batch_shape))
    prepend_shape = (-1,) if batch_size > 1 else ()
    xs = tree_map(
        lambda x: jnp.reshape(x, prepend_shape + jnp.shape(x)[batch_ndims:]), xs
    )
    # XXX: probably for the default behavior with chunk_size=None,
    # it is better to catch OOM error and reduce chunk_size by half until OOM disappears.
    chunk_size = batch_size if chunk_size is None else min(batch_size, chunk_size)
    if chunk_size > 1:
        pad = chunk_size - (batch_size % chunk_size)
        xs = tree_map(
            lambda x: jnp.pad(x, ((0, pad),) + ((0, 0),) * (np.ndim(x) - 1)), xs
        )
        num_chunks = batch_size // chunk_size + int(pad > 0)
        prepend_shape = (-1,) if num_chunks > 1 else ()
        xs = tree_map(
            lambda x: jnp.reshape(x, prepend_shape + (chunk_size,) + jnp.shape(x)[1:]),
            xs,
        )
        fn = vmap(fn)

    ys = lax.map(fn, xs) if num_chunks > 1 else fn(xs)
    map_ndims = int(num_chunks > 1) + int(chunk_size > 1)
    ys = tree_map(
        lambda y: jnp.reshape(y, (-1,) + jnp.shape(y)[map_ndims:])[:batch_size], ys
    )
    return tree_map(lambda y: jnp.reshape(y, batch_shape + jnp.shape(y)[1:]), ys)
