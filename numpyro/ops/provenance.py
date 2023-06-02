# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import jax
from jax.api_util import flatten_fun, shaped_abstractify
import jax.core as core
from jax.experimental.pjit import pjit_p
from jax.interpreters.partial_eval import trace_to_jaxpr_dynamic
from jax.interpreters.pxla import xla_pmap_p
import jax.linear_util as lu
import jax.numpy as jnp


def eval_provenance(fn, **kwargs):
    """
    Compute the provenance output of ``fun`` using JAX's abstract
    interpretation machinery. There is no actual array computation performed.

    Example::

        >>> o = eval_provenance(lambda x, y, z: x + y, x=1, y=2, z=3)
        >>> assert o == frozenset({"x", "y"})

    **References**

    [1] David Wingate, Noah Goodman, Andreas Stuhlmüller, Jeffrey Siskind (2011)
        Nonstandard Interpretations of Probabilistic Programs for Efficient Inference
        http://papers.neurips.cc/paper/4309-nonstandard-interpretations-of-probabilistic-programs-for-efficient-inference.pdf
    [2] https://jax.readthedocs.io/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html

    :param fun: A callable to track provenance of its (keyword) arguments.
    :param kwargs: Keyword arguments of `fun`.
    :returns: A pytree of :class:`frozenset` indicating the dependency on the inputs.
    """
    # Flatten the function and its arguments
    args, in_tree = jax.tree_util.tree_flatten(((), kwargs))
    wrapped_fun, out_tree = flatten_fun(lu.wrap_init(fn), in_tree)
    # Abstract eval to get output pytree
    avals = core.safe_map(shaped_abstractify, args)
    # XXX: we split out the process of abstract evaluation and provenance tracking
    # for simplicity. In principle, they can be merged so that we only need to walk
    # through the equations once.
    jaxpr, avals_out, _ = trace_to_jaxpr_dynamic(
        lu.wrap_init(wrapped_fun.call_wrapped, {}), avals
    )

    # get provenances of flatten kwargs
    aval_kwargs = {}
    for n, v in kwargs.items():
        aval = jax.ShapeDtypeStruct((), jnp.bool_, {"provenance": frozenset({n})})
        aval_kwargs[n] = jax.tree_util.tree_map(lambda _: aval, v)
    aval_args, _ = jax.tree_util.tree_flatten(((), aval_kwargs))
    provenance_inputs = jax.tree_util.tree_map(
        lambda x: x.named_shape["provenance"], aval_args
    )

    provenance_outputs = track_deps_jaxpr(jaxpr, provenance_inputs)
    out_flat = []
    for v, p in zip(avals_out, provenance_outputs):
        val = jax.ShapeDtypeStruct(jnp.shape(v), jnp.result_type(v), {"provenance": p})
        out_flat.append(val)
    out = jax.tree_util.tree_unflatten(out_tree(), out_flat)
    return jax.tree_util.tree_map(lambda x: x.named_shape["provenance"], out)


def track_deps_jaxpr(jaxpr, provenance_inputs):
    # Mapping from variable -> provenance
    env = {}

    def read(v):
        if isinstance(v, core.Literal):
            return frozenset()
        return env.get(v, frozenset())

    def write(v, p):
        if isinstance(v, core.Literal):
            return
        env[v] = read(v) | p

    core.safe_map(write, jaxpr.invars, provenance_inputs)
    for eqn in jaxpr.eqns:
        provenance_inputs = core.safe_map(read, eqn.invars)
        rule = track_deps_rules.get(eqn.primitive, _default_track_deps_rules)
        provenance_outputs = rule(eqn, provenance_inputs)
        core.safe_map(write, eqn.outvars, provenance_outputs)

    return core.safe_map(read, jaxpr.outvars)


track_deps_rules = {}


# XXX: Currently, we use default rule for scan_p, cond_p, while_p, remat_p
def _default_track_deps_rules(eqn, provenance_inputs):
    provenance_outputs = frozenset().union(*provenance_inputs)
    return [provenance_outputs] * len(eqn.outvars)


def track_deps_call_rule(eqn, provenance_inputs):
    return track_deps_jaxpr(eqn.params["call_jaxpr"], provenance_inputs)


track_deps_rules[core.call_p] = track_deps_call_rule
track_deps_rules[xla_pmap_p] = track_deps_call_rule


def track_deps_closed_call_rule(eqn, provenance_inputs):
    return track_deps_jaxpr(eqn.params["call_jaxpr"].jaxpr, provenance_inputs)


track_deps_rules[core.closed_call_p] = track_deps_closed_call_rule


def track_deps_pjit_rule(eqn, provenance_inputs):
    return track_deps_jaxpr(eqn.params["jaxpr"].jaxpr, provenance_inputs)


track_deps_rules[pjit_p] = track_deps_pjit_rule
