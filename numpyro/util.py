# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from contextlib import contextmanager
from functools import partial
import inspect
from itertools import zip_longest
import os
import random
import re
import warnings

import numpy as np
import tqdm
from tqdm.auto import tqdm as tqdm_auto

import jax
from jax import device_put, jit, lax, vmap
from jax.core import Tracer
from jax.experimental import host_callback
import jax.numpy as jnp

_DISABLE_CONTROL_FLOW_PRIM = False
_CHAIN_RE = re.compile(r"\d+$")  # e.g. get '3' from 'TFRT_CPU_3'


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
    jax.config.update("jax_enable_x64", bool(use_x64))


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


def maybe_jit(fn, *args, **kwargs):
    if _DISABLE_CONTROL_FLOW_PRIM:
        return fn
    else:
        return jit(fn, *args, **kwargs)


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


def is_prng_key(key):
    try:
        if jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key):
            return key.shape == ()
        return key.shape == (2,) and key.dtype == np.uint32
    except AttributeError:
        return False


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
        hashkeys = (*keys, fn.__name__)
        if hashkeys in fn_cache:
            fn = fn_cache[hashkeys]
            # update position
            del fn_cache[hashkeys]
            fn_cache[hashkeys] = fn
        else:
            fn_cache[hashkeys] = fn
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
    init_val_transformed = transform(init_val)
    start_idx = lower + (upper - lower) % thinning
    num_chains = progbar_opts.pop("num_chains", 1)
    # host_callback does not work yet with multi-GPU platforms
    # See: https://github.com/google/jax/issues/6447
    if num_chains > 1 and jax.default_backend() == "gpu":
        warnings.warn(
            "We will disable progress bar because it does not work yet on multi-GPUs platforms.",
            stacklevel=find_stack_level(),
        )
        progbar = False

    @partial(maybe_jit, donate_argnums=2)
    @cached_by(fori_collect, body_fun, transform)
    def _body_fn(i, val, collection, start_idx, thinning):
        val = body_fun(val)
        idx = (i - start_idx) // thinning

        def update_fn(collect_array, new_val):
            return cond(
                idx >= 0,
                collect_array,
                lambda x: x.at[idx].set(new_val),
                collect_array,
                identity,
            )

        def update_collection(collection, val):
            return jax.tree.map(update_fn, collection, transform(val))

        collection = update_collection(collection, val)
        return val, collection, start_idx, thinning

    def map_fn(x):
        nx = jnp.asarray(x)
        return jnp.zeros((collection_size, *nx.shape), dtype=nx.dtype) * nx[None, ...]

    collection = jax.tree.map(map_fn, init_val_transformed)

    if not progbar:

        def loop_fn(collection):
            return fori_loop(
                0,
                upper,
                lambda i, vals: _body_fn(i, *vals),
                (init_val, collection, start_idx, thinning),
            )

        last_val, collection, _, _ = maybe_jit(loop_fn, donate_argnums=0)(collection)

    elif num_chains > 1:
        progress_bar_fori_loop = progress_bar_factory(upper, num_chains)
        _body_fn_pbar = progress_bar_fori_loop(lambda i, vals: _body_fn(i, *vals))

        def loop_fn(collection):
            return fori_loop(
                0, upper, _body_fn_pbar, (init_val, collection, start_idx, thinning)
            )

        last_val, collection, _, _ = maybe_jit(loop_fn, donate_argnums=0)(collection)

    else:
        diagnostics_fn = progbar_opts.pop("diagnostics_fn", None)
        progbar_desc = progbar_opts.pop("progbar_desc", lambda x: "")

        vals = (init_val, collection, device_put(start_idx), device_put(thinning))

        if upper == 0:
            # special case, only compiling
            val, collection, start_idx, thinning = vals
            _, collection, _, _ = _body_fn(-1, val, collection, start_idx, thinning)
            vals = (val, collection, start_idx, thinning)
        else:
            with tqdm.trange(upper) as t:
                for i in t:
                    vals = _body_fn(i, *vals)

                    t.set_description(progbar_desc(i), refresh=False)
                    if diagnostics_fn:
                        t.set_postfix_str(diagnostics_fn(vals[0]), refresh=False)

        last_val, collection, _, _ = vals

    return (collection, last_val) if return_last_val else collection


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
    flatten_xs = jax.tree.flatten(xs)[0]
    batch_shape = np.shape(flatten_xs[0])[:batch_ndims]
    for x in flatten_xs[1:]:
        assert np.shape(x)[:batch_ndims] == batch_shape

    # we'll do map(vmap(fn), xs) and make xs.shape = (num_chunks, chunk_size, ...)
    num_chunks = batch_size = int(np.prod(batch_shape))
    prepend_shape = (batch_size,) if batch_size > 1 else ()
    xs = jax.tree.map(
        lambda x: jnp.reshape(x, prepend_shape + jnp.shape(x)[batch_ndims:]), xs
    )
    # XXX: probably for the default behavior with chunk_size=None,
    # it is better to catch OOM error and reduce chunk_size by half until OOM disappears.
    chunk_size = batch_size if chunk_size is None else min(batch_size, chunk_size)
    if chunk_size > 1:
        pad = chunk_size - (batch_size % chunk_size)
        xs = jax.tree.map(
            lambda x: jnp.pad(x, ((0, pad),) + ((0, 0),) * (np.ndim(x) - 1)), xs
        )
        num_chunks = batch_size // chunk_size + int(pad > 0)
        prepend_shape = (-1,) if num_chunks > 1 else ()
        xs = jax.tree.map(
            lambda x: jnp.reshape(x, prepend_shape + (chunk_size,) + jnp.shape(x)[1:]),
            xs,
        )
        fn = vmap(fn)

    ys = lax.map(fn, xs) if num_chunks > 1 else fn(xs)
    map_ndims = int(num_chunks > 1) + int(chunk_size > 1)
    ys = jax.tree.map(
        lambda y: jnp.reshape(
            y, (int(np.prod(jnp.shape(y)[:map_ndims])),) + jnp.shape(y)[map_ndims:]
        )[:batch_size],
        ys,
    )
    return jax.tree.map(lambda y: jnp.reshape(y, batch_shape + jnp.shape(y)[1:]), ys)


def format_shapes(
    trace,
    *,
    compute_log_prob=False,
    title="Trace Shapes:",
    last_site=None,
):
    """
    Given the trace of a function, returns a string showing a table of the shapes of
    all sites in the trace.

    Use :class:`~numpyro.handlers.trace` handler (or funsor
    :class:`~numpyro.contrib.funsor.enum_messenger.trace` handler for enumeration) to
    produce the trace.

    :param dict trace: The model trace to format.
    :param compute_log_prob: Compute log probabilities and display the shapes in the
        table. Accepts True / False or a function which when given a dictionary
        containing site-level metadata returns whether the log probability should be
        calculated and included in the table.
    :param str title: Title for the table of shapes.
    :param str last_site: Name of a site in the model. If supplied, subsequent sites
        are not displayed in the table.

    Usage::

        def model(*args, **kwargs):
            ...

        with numpyro.handlers.seed(rng_seed=1):
            trace = numpyro.handlers.trace(model).get_trace(*args, **kwargs)
        print(numpyro.util.format_shapes(trace))
    """
    if not trace.keys():
        return title
    rows = [[title]]

    rows.append(["Param Sites:"])
    for name, site in trace.items():
        if site["type"] == "param":
            rows.append(
                [name, None]
                + [str(size) for size in getattr(site["value"], "shape", ())]
            )
        if name == last_site:
            break

    rows.append(["Sample Sites:"])
    for name, site in trace.items():
        if site["type"] == "sample":
            # param shape
            batch_shape = getattr(site["fn"], "batch_shape", ())
            event_shape = getattr(site["fn"], "event_shape", ())
            rows.append(
                [f"{name} dist", None]
                + [str(size) for size in batch_shape]
                + ["|", None]
                + [str(size) for size in event_shape]
            )

            # value shape
            event_dim = len(event_shape)
            shape = getattr(site["value"], "shape", ())
            batch_shape = shape[: len(shape) - event_dim]
            event_shape = shape[len(shape) - event_dim :]
            rows.append(
                ["value", None]
                + [str(size) for size in batch_shape]
                + ["|", None]
                + [str(size) for size in event_shape]
            )

            # log_prob shape
            if (not callable(compute_log_prob) and compute_log_prob) or (
                callable(compute_log_prob) and compute_log_prob(site)
            ):
                batch_shape = getattr(site["fn"].log_prob(site["value"]), "shape", ())
                rows.append(
                    ["log_prob", None]
                    + [str(size) for size in batch_shape]
                    + ["|", None]
                )
        elif site["type"] == "plate":
            shape = getattr(site["value"], "shape", ())
            rows.append(
                [f"{name} plate", None] + [str(size) for size in shape] + ["|", None]
            )

        if name == last_site:
            break

    return _format_table(rows)


# TODO: follow pyro.util.check_site_shape logics for more complete validation
def _validate_model(model_trace, plate_warning="loose"):
    # TODO: Consider exposing global configuration for those strategies.
    assert plate_warning in ["loose", "strict", "error"]
    enum_dims = set(
        [
            site["infer"]["name_to_dim"][name]
            for name, site in model_trace.items()
            if site["type"] == "sample"
            and site["infer"].get("enumerate") == "parallel"
            and site["infer"].get("name_to_dim") is not None
        ]
    )
    # Check if plate is missing in the model.
    for name, site in model_trace.items():
        if site["type"] == "sample":
            value_ndim = jnp.ndim(site["value"])
            batch_shape = lax.broadcast_shapes(
                tuple(site["fn"].batch_shape),
                jnp.shape(site["value"])[: value_ndim - len(site["fn"].event_shape)],
            )
            plate_dims = set(f.dim for f in site["cond_indep_stack"])
            batch_ndim = len(batch_shape)
            for i in range(batch_ndim):
                dim = -i - 1
                if batch_shape[dim] > 1 and (dim not in (plate_dims | enum_dims)):
                    # Skip checking if it is the `scan` dimension.
                    if dim == -batch_ndim and site.get("_control_flow_done", False):
                        continue
                    message = (
                        f"Missing a plate statement for batch dimension {dim}"
                        f" at site '{name}'. You can use `numpyro.util.format_shapes`"
                        " utility to check shapes at all sites of your model."
                    )

                    if plate_warning == "error":
                        raise ValueError(message)
                    elif plate_warning == "strict" or (len(plate_dims) > 0):
                        warnings.warn(message, stacklevel=find_stack_level())


def check_model_guide_match(model_trace, guide_trace):
    """
    Checks the following assumptions:

    1. Each sample site in the model also appears in the guide and is not
        marked auxiliary.
    2. Each sample site in the guide either appears in the model or is marked,
        auxiliary via ``infer={'is_auxiliary': True}``.
    3. Each :class:`~numpyro.primitives.plate` statement in the guide also
        appears in the model.
    4. At each sample site that appears in both the model and guide, the model
        and guide agree on sample shape.

    :param dict model_trace: The model trace to check.
    :param dict guide_trace: The guide trace to check.
    :raises: RuntimeWarning, ValueError
    """
    # Check ordinary sample sites.
    guide_vars = set(
        name
        for name, site in guide_trace.items()
        if site["type"] == "sample" and not site.get("is_observed", False)
    )
    aux_vars = set(
        name
        for name, site in guide_trace.items()
        if site["type"] == "sample"
        if site["infer"].get("is_auxiliary")
    )
    model_vars = set(
        name
        for name, site in model_trace.items()
        if site["type"] == "sample"
        and not site.get("is_observed", False)
        and not (
            name not in guide_trace and site["infer"].get("enumerate") == "parallel"
        )
    )
    enum_vars = set(
        [
            name
            for name, site in model_trace.items()
            if site["type"] == "sample"
            and not site.get("is_observed", False)
            and name not in guide_trace
            and site["infer"].get("enumerate") == "parallel"
        ]
    )

    if aux_vars & model_vars:
        warnings.warn(
            "Found auxiliary vars in the model: {}".format(aux_vars & model_vars),
            stacklevel=find_stack_level(),
        )
    if not (guide_vars <= model_vars | aux_vars):
        warnings.warn(
            "Found non-auxiliary vars in guide but not model, "
            "consider marking these infer={{'is_auxiliary': True}}:\n{}".format(
                guide_vars - aux_vars - model_vars
            ),
            stacklevel=find_stack_level(),
        )
    if not (model_vars <= guide_vars | enum_vars):
        warnings.warn(
            "Found vars in model but not guide: {}".format(
                model_vars - guide_vars - enum_vars
            ),
            stacklevel=find_stack_level(),
        )

    # Check shapes agree.
    for name in model_vars & guide_vars:
        model_site = model_trace[name]
        guide_site = guide_trace[name]

        if hasattr(model_site["fn"], "event_dim") and hasattr(
            guide_site["fn"], "event_dim"
        ):
            if model_site["fn"].event_dim != guide_site["fn"].event_dim:
                raise ValueError(
                    "Model and guide event_dims disagree at site '{}': {} vs {}".format(
                        name, model_site["fn"].event_dim, guide_site["fn"].event_dim
                    )
                )

        if hasattr(model_site["fn"], "shape") and hasattr(guide_site["fn"], "shape"):
            model_shape = model_site["fn"].shape(model_site["kwargs"]["sample_shape"])
            guide_shape = guide_site["fn"].shape(guide_site["kwargs"]["sample_shape"])
            if model_shape == guide_shape:
                continue

            for model_size, guide_size in zip_longest(
                reversed(model_shape), reversed(guide_shape), fillvalue=1
            ):
                if model_size != guide_size:
                    raise ValueError(
                        "Model and guide shapes disagree at site '{}': {} vs {}".format(
                            name, model_shape, guide_shape
                        )
                    )

    # Check subsample sites introduced by plate.
    model_vars = set(
        name for name, site in model_trace.items() if site["type"] == "plate"
    )
    guide_vars = set(
        name for name, site in guide_trace.items() if site["type"] == "plate"
    )
    if not (guide_vars <= model_vars):
        warnings.warn(
            "Found plate statements in guide but not model: {}".format(
                guide_vars - model_vars
            ),
            stacklevel=find_stack_level(),
        )


def _format_table(rows):
    """
    Formats a right justified table using None as column separator.
    """
    # compute column widths
    column_widths = [0, 0, 0]
    for row in rows:
        widths = [0, 0, 0]
        j = 0
        for cell in row:
            if cell is None:
                j += 1
            else:
                widths[j] += 1
        for j in range(3):
            column_widths[j] = max(column_widths[j], widths[j])

    # justify columns
    for i, row in enumerate(rows):
        cols = [[], [], []]
        j = 0
        for cell in row:
            if cell is None:
                j += 1
            else:
                cols[j].append(cell)
        cols = [
            [""] * (width - len(col)) + col
            if direction == "r"
            else col + [""] * (width - len(col))
            for width, col, direction in zip(column_widths, cols, "rrl")
        ]
        rows[i] = sum(cols, [])

    # compute cell widths
    cell_widths = [0] * len(rows[0])
    for row in rows:
        for j, cell in enumerate(row):
            cell_widths[j] = max(cell_widths[j], len(cell))

    # justify cells
    return "\n".join(
        " ".join(cell.rjust(width) for cell, width in zip(row, cell_widths))
        for row in rows
    )


def find_stack_level() -> int:
    """
    Find the first place in the stack that is not inside numpyro
    (tests notwithstanding).

    Source:
    https://github.com/pandas-dev/pandas/blob/ccb25ab1d24c4fb9691270706a59c8d319750870/pandas/util/_exceptions.py#L27-L48
    """
    import numpyro

    pkg_dir = os.path.dirname(numpyro.__file__)

    # https://stackoverflow.com/questions/17407119/python-inspect-stack-is-slow
    frame = inspect.currentframe()
    n = 0
    while frame:
        fname = inspect.getfile(frame)
        if fname.startswith(pkg_dir):
            frame = frame.f_back
            n += 1
        else:
            break
    return n
