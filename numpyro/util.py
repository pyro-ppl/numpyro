# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict, namedtuple
from contextlib import contextmanager
import itertools
import os
from pathlib import Path
import random
import re

import numpy as np
import tqdm

import jax
from jax import device_put, jit, lax, ops, vmap
from jax.core import Tracer
from jax.dtypes import canonicalize_dtype
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_map, tree_unflatten

_DISABLE_CONTROL_FLOW_PRIM = False


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
        use_x64 = os.getenv('JAX_ENABLE_X64', 0)
    jax.config.update('jax_enable_x64', use_x64)


def set_platform(platform=None):
    """
    Changes platform to CPU, GPU, or TPU. This utility only takes
    effect at the beginning of your program.

    :param str platform: either 'cpu', 'gpu', or 'tpu'.
    """
    if platform is None:
        platform = os.getenv('JAX_PLATFORM_NAME', 'cpu')
    jax.config.update('jax_platform_name', platform)


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
    xla_flags = os.getenv('XLA_FLAGS', '').lstrip('--')
    xla_flags = re.sub(r'xla_force_host_platform_device_count=.+\s', '', xla_flags).split()
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
    outer_fn._cache = getattr(outer_fn, '_cache', OrderedDict())

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


def fori_collect(lower, upper, body_fun, init_val, transform=identity,
                 progbar=True, return_last_val=False, collection_size=None,
                 thinning=1, **progbar_opts):
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
    collection_size = (upper - lower) // thinning if collection_size is None else collection_size
    assert collection_size >= (upper - lower) // thinning
    init_val_flat, unravel_fn = ravel_pytree(transform(init_val))
    start_idx = lower + (upper - lower) % thinning

    @cached_by(fori_collect, body_fun, transform)
    def _body_fn(i, vals):
        val, collection, start_idx, thinning = vals
        val = body_fun(val)
        idx = (i - start_idx) // thinning
        collection = cond(idx >= 0,
                          collection,
                          lambda x: ops.index_update(x, idx, ravel_pytree(transform(val))[0]),
                          collection,
                          identity)
        return val, collection, start_idx, thinning

    collection = jnp.zeros((collection_size,) + init_val_flat.shape)
    if not progbar:
        last_val, collection, _, _ = fori_loop(0, upper, _body_fn, (init_val, collection, start_idx, thinning))
    else:
        diagnostics_fn = progbar_opts.pop('diagnostics_fn', None)
        progbar_desc = progbar_opts.pop('progbar_desc', lambda x: '')

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


pytree_metadata = namedtuple('pytree_metadata', ['flat', 'shape', 'size', 'dtype'])


def _ravel_list(*leaves):
    leaves_metadata = tree_map(lambda l: pytree_metadata(
        jnp.ravel(l), jnp.shape(l), jnp.size(l), canonicalize_dtype(lax.dtype(l))), leaves)
    leaves_idx = jnp.cumsum(jnp.array((0,) + tuple(d.size for d in leaves_metadata)))

    def unravel_list(arr):
        return [jnp.reshape(lax.dynamic_slice_in_dim(arr, leaves_idx[i], m.size),
                            m.shape).astype(m.dtype)
                for i, m in enumerate(leaves_metadata)]

    flat = jnp.concatenate([m.flat for m in leaves_metadata]) if leaves_metadata else jnp.array([])
    return flat, unravel_list


def ravel_pytree(pytree):
    leaves, treedef = tree_flatten(pytree)
    flat, unravel_list = _ravel_list(*leaves)

    def unravel_pytree(arr):
        return tree_unflatten(treedef, unravel_list(arr))

    return flat, unravel_pytree


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
    xs = tree_map(lambda x: jnp.reshape(x, prepend_shape + jnp.shape(x)[batch_ndims:]), xs)
    # XXX: probably for the default behavior with chunk_size=None,
    # it is better to catch OOM error and reduce chunk_size by half until OOM disappears.
    chunk_size = batch_size if chunk_size is None else min(batch_size, chunk_size)
    if chunk_size > 1:
        pad = chunk_size - (batch_size % chunk_size)
        xs = tree_map(lambda x: jnp.pad(x, ((0, pad),) + ((0, 0),) * (np.ndim(x) - 1)), xs)
        num_chunks = batch_size // chunk_size + int(pad > 0)
        prepend_shape = (-1,) if num_chunks > 1 else ()
        xs = tree_map(lambda x: jnp.reshape(x, prepend_shape + (chunk_size,) + jnp.shape(x)[1:]), xs)
        fn = vmap(fn)

    ys = lax.map(fn, xs) if num_chunks > 1 else fn(xs)
    map_ndims = int(num_chunks > 1) + int(chunk_size > 1)
    ys = tree_map(lambda y: jnp.reshape(y, (-1,) + jnp.shape(y)[map_ndims:])[:batch_size], ys)
    return tree_map(lambda y: jnp.reshape(y, batch_shape + jnp.shape(y)[1:]), ys)


def get_model_relations(model, model_args=None, model_kwargs=None, num_tries=10):
    """
    Infer relations of RVs and plates from given model and optionally data.
    See issue #949 on pyro-ppl/numpyro for more details.

    :param int num_tries: times to trace model to detect discrete -> continuous dependency.
    """
    # TODO: put import in more sensible location
    from numpyro import handlers

    model_args = model_args or ()
    model_kwargs = model_kwargs or {}

    trace = handlers.trace(handlers.seed(model, 0)).get_trace(*model_args, **model_kwargs)
    obs_sites = [name for name, site in trace.items()
                 if site['type'] == 'sample' and site['is_observed']]

    sample_plates = {name: [frame.name for frame in site['cond_indep_stack']]
                     for name, site in trace.items() if site['type'] == 'sample'}
    plate_samples = {k: {name for name, plates in sample_plates.items() if k in plates}
                     for k in trace if trace[k]['type'] == 'plate'}

    def _resolve_plate_samples(plate_samples):
        for p, pv in plate_samples.items():
            for q, qv in plate_samples.items():
                if len(pv & qv) > 0 and len(pv - qv) > 0 and len(qv - pv) > 0:
                    plate_samples_ = plate_samples.copy()
                    plate_samples_[q] = pv & qv
                    plate_samples_[q + '__CLONE'] = qv - pv
                    return _resolve_plate_samples(plate_samples_)
        return plate_samples

    plate_samples = _resolve_plate_samples(plate_samples)
    # convert set to list to keep order of variables
    plate_samples = {k: [name for name in trace if name in v]
                     for k, v in plate_samples.items()}

    def get_log_probs(sample, seed=0):
        with handlers.trace() as tr, handlers.seed(model, seed), handlers.substitute(data=sample):
            model(*model_args, **model_kwargs)
        return {name: site['fn'].log_prob(site['value'])
                for name, site in tr.items() if site['type'] == 'sample'}

    samples = {name: site['value'] for name, site in trace.items()
               if site['type'] == 'sample' and not site['is_observed']
               and not site['fn'].is_discrete}
    log_prob_grads = jax.jacobian(get_log_probs)(samples)
    sample_deps = {}
    for name, grads in log_prob_grads.items():
        sample_deps[name] = {n for n in grads if n != name and (grads[n] != 0).any()}

    # find discrete -> continuous dependency
    samples = {name: site['value'] for name, site in trace.items() if site['type'] == 'sample'}
    discrete_sites = [name for name, site in trace.items() if site['type'] == 'sample'
                      and not site['is_observed'] and site['fn'].is_discrete]
    log_probs_prototype = get_log_probs(samples)
    for name in discrete_sites:
        samples_ = samples.copy()
        samples_.pop(name)
        for i in range(num_tries):
            log_probs = get_log_probs(samples_, seed=i + 1)
            for var in samples:
                if var == name:
                    continue
                if (log_probs[var] != log_probs_prototype[var]).any():
                    sample_deps[var] |= {name}
    sample_sample = {}
    for name in samples:
        sample_sample[name] = [var for var in samples if var in sample_deps[name]]
    return {
        'sample_sample': sample_sample,
        'plate_sample': plate_samples,
        'observed': obs_sites
    }


def generate_graph_specification(model_relations):
    """
    Convert model relations into data structure which can be readily
    converted into a network.
    """
    # group nodes by plate
    plate_groups = dict(model_relations['plate_sample'])
    plate_rvs = {rv for rvs in plate_groups.values() for rv in rvs}
    plate_groups[None] = [
        rv for rv in model_relations['sample_sample'] if rv not in plate_rvs
    ]  # RVs which are in no plate

    # retain node metadata
    node_data = {}
    for rv in model_relations['sample_sample']:
        node_data[rv] = {
            'is_observed': rv in model_relations['observed'],
        }

    # infer plate structure
    # TODO: subset relation might not always hold, raise exception if not
    plate_data = {}
    for plate1, plate2 in list(itertools.permutations(plate_groups, 2)):
        if plate1 is None or plate2 is None:
            continue

        if set(plate_groups[plate1]) < set(plate_groups[plate2]):
            plate_data[plate1] = {'parent': plate2}

    for plate in plate_groups:
        if plate is None:
            continue

        if plate not in plate_data:
            plate_data[plate] = {'parent': None}

    # infer RV edges
    edge_list = []
    for target, source_list in model_relations['sample_sample'].items():
        edge_list.extend([(source, target) for source in source_list])

    return {
        'plate_groups': plate_groups,
        'plate_data': plate_data,
        'node_data': node_data,
        'edge_list': edge_list,
    }


def render_graph(graph_specification):
    """
    Create a graphviz object given a graph specification.
    """
    # TODO: plate_graph_dict and plate_data assume that "deepest" plates come first. This will break!

    try:
        import graphviz  # noqa: F401
    except ImportError as e:
        raise ImportError(
            'Looks like you want to use graphviz to render your model. '
            'You need to install `graphviz` to be able to use this feature. '
            'It can be installed with `pip install graphviz`.'
        ) from e

    plate_groups = graph_specification['plate_groups']
    plate_data = graph_specification['plate_data']
    node_data = graph_specification['node_data']
    edge_list = graph_specification['edge_list']

    graph = graphviz.Digraph()

    # add plates
    # TODO: order may not always be as expected (parents before children)
    plate_graph_dict = {
        plate: graphviz.Digraph(name=f'cluster_{plate}')
        for plate in plate_groups
        if plate is not None
    }
    for plate, plate_graph in plate_graph_dict.items():
        plate_graph.attr(label=plate.split('__CLONE')[0], labeljust='r', labelloc='b')

    # add nodes
    for plate, rv_list in plate_groups.items():
        cur_graph = graph if plate is None else plate_graph_dict[plate]

        for rv in rv_list:
            color = 'grey' if node_data[rv]['is_observed'] else 'white'
            cur_graph.node(
                rv, label=rv, shape='circle', style='filled', fillcolor=color
            )

    for plate, data in plate_data.items():
        parent_plate = data['parent']

        if parent_plate is None:
            graph.subgraph(plate_graph_dict[plate])
        else:
            plate_graph_dict[parent_plate].subgraph(plate_graph_dict[plate])

    # add edges
    for source, target in edge_list:
        graph.edge(source, target)

    # return whole graph
    return graph


def render_model(
    model,
    model_args=None,
    model_kwargs=None,
    filename=None,
    num_tries=10,
):
    """
    Wrap all functions needed to automatically render a model.

    :param model: Model to render.
    :param str filename: File to save rendered model in.
    """
    relations = get_model_relations(
        model,
        model_args=model_args,
        model_kwargs=model_kwargs,
        num_tries=num_tries,
    )
    graph_spec = generate_graph_specification(relations)
    graph = render_graph(graph_spec)

    if filename is not None:
        filename = Path(filename)
        graph.render(
            filename.stem,
            view=False,
            cleanup=True,
            format=filename.suffix[1:],  # remove leading period from suffix
        )

    return graph
