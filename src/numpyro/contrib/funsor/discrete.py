# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict, defaultdict
import functools

from jax import random
import jax.numpy as jnp

import funsor
from numpyro.contrib.funsor.enum_messenger import enum
from numpyro.contrib.funsor.infer_util import _enum_log_density, _get_shift, _shift_name
from numpyro.handlers import block, seed, substitute, trace
from numpyro.infer.util import _guess_max_plate_nesting


@functools.singledispatch
def _get_support_value(funsor_dist, name, **kwargs):
    raise ValueError(
        "Could not extract point from {} at name {}".format(funsor_dist, name)
    )


@_get_support_value.register(funsor.cnf.Contraction)
def _get_support_value_contraction(funsor_dist, name, **kwargs):
    delta_terms = [
        v
        for v in funsor_dist.terms
        if isinstance(v, funsor.delta.Delta) and name in v.fresh
    ]
    assert len(delta_terms) == 1
    return _get_support_value(delta_terms[0], name, **kwargs)


@_get_support_value.register(funsor.delta.Delta)
def _get_support_value_delta(funsor_dist, name, **kwargs):
    assert name in funsor_dist.fresh
    return OrderedDict(funsor_dist.terms)[name][0]


def _sample_posterior(
    model, first_available_dim, temperature, rng_key, *args, **kwargs
):
    if temperature == 0:
        sum_op, prod_op = funsor.ops.max, funsor.ops.add
        approx = funsor.approximations.argmax_approximate
    elif temperature == 1:
        sum_op, prod_op = funsor.ops.logaddexp, funsor.ops.add
        rng_key, sub_key = random.split(rng_key)
        approx = funsor.montecarlo.MonteCarlo(rng_key=sub_key)
    else:
        raise ValueError("temperature must be 0 (map) or 1 (sample) for now")

    if first_available_dim is None:
        with block():
            model_trace = trace(seed(model, rng_key)).get_trace(*args, **kwargs)
        first_available_dim = -_guess_max_plate_nesting(model_trace) - 1

    with funsor.adjoint.AdjointTape() as tape:
        with block(), enum(first_available_dim=first_available_dim):
            log_prob, model_tr, log_measures = _enum_log_density(
                model, args, kwargs, {}, sum_op, prod_op
            )

    with approx:
        approx_factors = tape.adjoint(sum_op, prod_op, log_prob)

    # construct a result trace to replay against the model
    sample_tr = model_tr.copy()
    for name, node in sample_tr.items():
        if node["type"] != "sample":
            continue
        if node["infer"].get("enumerate") == "parallel":
            log_measure = approx_factors[log_measures[name]]
            value = _get_support_value(log_measure, name)
            node["value"] = funsor.to_data(
                value, name_to_dim=node["infer"]["name_to_dim"]
            )

    data = {
        name: site["value"]
        for name, site in sample_tr.items()
        if site["type"] == "sample"
    }

    # concatenate _PREV_foo to foo
    time_vars = defaultdict(list)
    for name in data:
        if name.startswith("_PREV_"):
            root_name = _shift_name(name, -_get_shift(name))
            time_vars[root_name].append(name)
    for name in time_vars:
        if name in data:
            time_vars[name].append(name)
        time_vars[name] = sorted(time_vars[name], key=len, reverse=True)

    for root_name, vars in time_vars.items():
        prototype_shape = model_trace[root_name]["value"].shape
        values = [data.pop(name) for name in vars]
        if len(values) == 1:
            data[root_name] = values[0].reshape(prototype_shape)
        else:
            assert len(prototype_shape) >= 1
            values = [v.reshape((-1,) + prototype_shape[1:]) for v in values]
            data[root_name] = jnp.concatenate(values)

    return data


def infer_discrete(fn=None, first_available_dim=None, temperature=1, rng_key=None):
    """
    A handler that samples discrete sites marked with
    ``site["infer"]["enumerate"] = "parallel"`` from the posterior,
    conditioned on observations.

    Example::

        @infer_discrete(first_available_dim=-1, temperature=0)
        @config_enumerate
        def viterbi_decoder(data, hidden_dim=10):
            transition = 0.3 / hidden_dim + 0.7 * jnp.eye(hidden_dim)
            means = jnp.arange(float(hidden_dim))
            states = [0]
            for t in markov(range(len(data))):
                states.append(numpyro.sample("states_{}".format(t),
                                             dist.Categorical(transition[states[-1]])))
                numpyro.sample("obs_{}".format(t),
                               dist.Normal(means[states[-1]], 1.),
                               obs=data[t])
            return states  # returns maximum likelihood states

    .. warning: This does not yet support :func:`numpyro.contrib.control_flow.scan`
        primitive.

    .. warning: The ``log_prob``s of the inferred model's trace are not
        meaningful, and may be changed in a future release.

    :param fn: a stochastic function (callable containing NumPyro primitive calls)
    :param int first_available_dim: The first tensor dimension (counting
        from the right) that is available for parallel enumeration. This
        dimension and all dimensions left may be used internally by Pyro.
        This should be a negative integer.
    :param int temperature: Either 1 (sample via forward-filter backward-sample)
        or 0 (optimize via Viterbi-like MAP inference). Defaults to 1 (sample).
    :param jax.random.PRNGKey rng_key: a random number generator key, to be used in
        cases ``temperature=1`` or ``first_available_dim is None``.
    """
    if temperature == 1 or first_available_dim is None:
        assert rng_key is not None
    if fn is None:  # support use as a decorator
        return functools.partial(
            infer_discrete,
            first_available_dim=first_available_dim,
            temperature=temperature,
            rng_key=rng_key,
        )

    def wrap_fn(*args, **kwargs):
        samples = _sample_posterior(
            fn, first_available_dim, temperature, rng_key, *args, **kwargs
        )
        with substitute(data=samples):
            return fn(*args, **kwargs)

    return wrap_fn
