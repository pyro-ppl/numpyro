# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import jax
from jax.interpreters import partial_eval
from jax.linear_util import wrap_init
import jax.numpy as jnp


class _ProvenanceJaxprTrace(partial_eval.DynamicJaxprTrace):
    """A JAX class to control the behavior of primitives on tracers."""

    def process_primitive(self, primitive, tracers, params):
        # remove "_provenance" dimension in arguments before executing the function
        provenances = [
            t.aval.named_shape.pop("_provenance", frozenset()) for t in tracers
        ]
        out_tracers = super().process_primitive(primitive, tracers, params)
        # add "_provenance" dimensions to arguments again
        for t, p in zip(tracers, provenances):
            if p:
                t.aval.named_shape["_provenance"] = p

        # update outputs' provenance
        out_provenance = frozenset().union(*provenances)
        if out_provenance:
            out_tracers = out_tracers if primitive.multiple_results else [out_tracers]
            for t in out_tracers:
                t.aval.named_shape["_provenance"] = out_provenance
                # Also update provenance of the cached tracer -> aval dict.
                aval_cache = self.frame.tracer_to_var[id(t)].aval
                aval_cache.named_shape["_provenance"] = out_provenance
            out_tracers = out_tracers if primitive.multiple_results else out_tracers[0]
        return out_tracers


class ProvenanceArray:
    """
    Provenance tracking implementation in JAX.

    This class wraps an ndarray to track provenance through JAX ops,
    where provenance is a user-defined frozenset of objects. The
    provenance of the output arrays of any op is the union of provenances
    of input arrays.

    -   To start tracking provenance in a function, wrap input arrays in
        :class:`ProvenanceArray` with user-defined initial provenance,
        then use :func:`eval_provenance` to get the provenance output array.
    -   To read the provenance of an ndarray use :func:`get_provenance` .

    Example::

        >>> a = ProvenanceArray(jnp.zeros(3), frozenset({"a"}))
        >>> b = ProvenanceArray(jnp.ones(3), frozenset({"b"}))
        >>> c = jnp.arange(3)
        >>> f = lambda a, b, c: a + b + c
        >>> o = eval_provenance(f, a, b, c)
        >>> assert get_provenance(o) == frozenset({"a", "b"})

    **References**

    [1] David Wingate, Noah Goodman, Andreas Stuhlmüller, Jeffrey Siskind (2011)
        Nonstandard Interpretations of Probabilistic Programs for Efficient Inference
        http://papers.neurips.cc/paper/4309-nonstandard-interpretations-of-probabilistic-programs-for-efficient-inference.pdf

    :param data: An initial data to start tracking. The data needs
        to have attributes `shape` and `dtype`.
    :param frozenset provenance: An initial provenance set.
    """

    def __init__(self, data, provenance=frozenset()):
        self.shape = jnp.shape(data)
        self.dtype = jnp.result_type(data)
        self.named_shape = {"_provenance": provenance}


def get_provenance(data):
    """
    Reads the provenance of a recursive data structure possibly containing ndarray.

    :param data: An input data.
    :returns: A provenance frozenset.
    :rtype: frozenset
    """
    return jax.tree_util.tree_map(
        lambda a: a.named_shape.get("_provenance", frozenset()), data
    )


def eval_provenance(fun, *args, **kwargs):
    """
    Compute the provenance output of ``fun`` using JAX's abstract
    interpretation machinery. There is no actual array computation performed.

    :param fun: A callable to track provenance of its (keyword) arguments.
    :param args: Positional arguments of `fun`.
    :param kwargs: Keyword arguments of `fun`.
    :returns: A pytree of :class:`ProvenanceArray`.
    """
    # flatten the function and its arguments
    args_flat, in_tree = jax.tree_util.tree_flatten((args, kwargs))
    wrapped_fun, out_tree = jax.api_util.flatten_fun(wrap_init(fun), in_tree)
    fun = wrap_init(wrapped_fun.call_wrapped)
    avals = jax.util.safe_map(jax.api_util.shaped_abstractify, args_flat)

    # execute the function and trace provenance
    with jax.core.new_main(_ProvenanceJaxprTrace, dynamic=True) as main:
        main.jaxpr_stack = ()
        out = partial_eval.trace_to_subjaxpr_dynamic(fun, main, avals)[1]

    # unflatten the output and get its provenance
    out = [jax.ShapeDtypeStruct(x.shape, x.dtype, x.named_shape) for x in out]
    out = jax.tree_util.tree_unflatten(out_tree(), out)
    return jax.tree_util.tree_map(
        lambda x: ProvenanceArray(x, x.named_shape.get("_provenance", frozenset())),
        out,
    )
