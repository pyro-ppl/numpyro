# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import jax
from jax.api_util import flatten_fun, shaped_abstractify
from jax.experimental.pjit import pjit_p

try:
    import jax.extend.linear_util as lu
except ImportError:
    import jax.linear_util as lu

try:
    from jax.extend.core import Literal
except ImportError:
    from jax.core import Literal

try:
    from jax.extend.core.primitives import call_p, closed_call_p
except ImportError:
    from jax.core import call_p, closed_call_p

try:
    from jax.api_util import debug_info
except ImportError:
    debug_info = None

from jax.interpreters.partial_eval import trace_to_jaxpr_dynamic
from jax.interpreters.pxla import xla_pmap_p


# Adapted from definition in jax v0.5.0
def _safe_map(f, *args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, f"length mismatch: {list(map(len, args))}"
    return list(map(f, *args))


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
    args, in_tree = jax.tree.flatten(((), kwargs))
    fn_info = (
        dict(debug_info=debug_info("eval_provenance fn", fn, (), kwargs))
        if debug_info is not None
        else {}
    )
    wrapped_fun, out_tree = flatten_fun(lu.wrap_init(fn, **fn_info), in_tree)
    # Abstract eval to get output pytree
    avals = _safe_map(shaped_abstractify, args)
    # XXX: we split out the process of abstract evaluation and provenance tracking
    # for simplicity. In principle, they can be merged so that we only need to walk
    # through the equations once.

    wrapped_info = (
        dict(
            debug_info=debug_info(
                "eval_provenance wrapped", wrapped_fun.call_wrapped, args, {}
            )
        )
        if debug_info is not None
        else {}
    )
    jaxpr, avals_out, _ = trace_to_jaxpr_dynamic(
        lu.wrap_init(wrapped_fun.call_wrapped, {}, **wrapped_info), avals
    )

    # get provenances of flatten kwargs
    aval_kwargs = {}
    for n, v in kwargs.items():
        aval_kwargs[n] = jax.tree.map(lambda _: frozenset({n}), v)
    provenance_inputs, _ = jax.tree.flatten(((), aval_kwargs))

    provenance_outputs = track_deps_jaxpr(jaxpr, provenance_inputs)
    return jax.tree.unflatten(out_tree(), provenance_outputs)


def track_deps_jaxpr(jaxpr, provenance_inputs):
    # Mapping from variable -> provenance
    env = {}

    def read(v):
        if isinstance(v, Literal):
            return frozenset()
        return env.get(v, frozenset())

    def write(v, p):
        if isinstance(v, Literal):
            return
        env[v] = read(v) | p

    _safe_map(write, jaxpr.invars, provenance_inputs)
    for eqn in jaxpr.eqns:
        provenance_inputs = _safe_map(read, eqn.invars)
        rule = track_deps_rules.get(eqn.primitive, _default_track_deps_rules)
        provenance_outputs = rule(eqn, provenance_inputs)
        _safe_map(write, eqn.outvars, provenance_outputs)

    return _safe_map(read, jaxpr.outvars)


track_deps_rules = {}


# XXX: Currently, we use default rule for scan_p, cond_p, while_p, remat_p
def _default_track_deps_rules(eqn, provenance_inputs):
    provenance_outputs = frozenset().union(*provenance_inputs)
    return [provenance_outputs] * len(eqn.outvars)


def track_deps_call_rule(eqn, provenance_inputs):
    return track_deps_jaxpr(eqn.params["call_jaxpr"], provenance_inputs)


track_deps_rules[call_p] = track_deps_call_rule
track_deps_rules[xla_pmap_p] = track_deps_call_rule


def track_deps_closed_call_rule(eqn, provenance_inputs):
    return track_deps_jaxpr(eqn.params["call_jaxpr"].jaxpr, provenance_inputs)


track_deps_rules[closed_call_p] = track_deps_closed_call_rule


def track_deps_pjit_rule(eqn, provenance_inputs):
    return track_deps_jaxpr(eqn.params["jaxpr"].jaxpr, provenance_inputs)


track_deps_rules[pjit_p] = track_deps_pjit_rule
