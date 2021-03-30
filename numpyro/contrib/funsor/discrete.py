# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
import functools

from jax import random

import funsor
from numpyro.contrib.funsor.enum_messenger import enum
from numpyro.contrib.funsor.enum_messenger import trace as packed_trace
from numpyro.contrib.funsor.infer_util import plate_to_enum_plate
from numpyro.distributions.util import is_identically_one
from numpyro.handlers import block, replay, seed, trace
from numpyro.infer.util import _guess_max_plate_nesting


@functools.singledispatch
def _get_support_value(funsor_dist, name, **kwargs):
    raise ValueError("Could not extract point from {} at name {}".format(funsor_dist, name))


@_get_support_value.register(funsor.cnf.Contraction)
def _get_support_value_contraction(funsor_dist, name, **kwargs):
    delta_terms = [v for v in funsor_dist.terms if isinstance(v, funsor.delta.Delta) and name in v.fresh]
    assert len(delta_terms) == 1
    return _get_support_value(delta_terms[0], name, **kwargs)


@_get_support_value.register(funsor.delta.Delta)
def _get_support_value_delta(funsor_dist, name, **kwargs):
    assert name in funsor_dist.fresh
    return OrderedDict(funsor_dist.terms)[name][0]


def terms_from_trace(tr):
    """Helper function to extract elbo components from execution traces."""
    log_factors = {}
    log_measures = {}
    sum_vars, prod_vars = frozenset(), frozenset()
    for site in tr.values():
        if site["type"] == "sample":
            value = site["value"]
            intermediates = site["intermediates"]
            scale = site["scale"]
            if intermediates:
                log_prob = site["fn"].log_prob(value, intermediates)
            else:
                log_prob = site["fn"].log_prob(value)

            if (scale is not None) and (not is_identically_one(scale)):
                log_prob = scale * log_prob

            dim_to_name = site["infer"]["dim_to_name"]
            log_prob_factor = funsor.to_funsor(log_prob, output=funsor.Real, dim_to_name=dim_to_name)

            if site["is_observed"]:
                log_factors[site["name"]] = log_prob_factor
            else:
                log_measures[site["name"]] = log_prob_factor
                sum_vars |= frozenset({site["name"]})
            prod_vars |= frozenset(f.name for f in site["cond_indep_stack"] if f.dim is not None)

    return {"log_factors": log_factors, "log_measures": log_measures, "measure_vars": sum_vars, "plate_vars": prod_vars}


def _sample_posterior(model, first_available_dim, temperature, rng_key, *args, **kwargs):

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

    with block(), enum(first_available_dim=first_available_dim):
        with plate_to_enum_plate():
            model_tr = packed_trace(model).get_trace(*args, **kwargs)

    terms = terms_from_trace(model_tr)
    # terms["log_factors"] = [log p(x) for each observed or latent sample site x]
    # terms["log_measures"] = [log p(z) or other Dice factor
    #                          for each latent sample site z]

    with funsor.interpretations.lazy:
        log_prob = funsor.sum_product.sum_product(sum_op, prod_op, list(terms["log_factors"].values()) + list(terms["log_measures"].values()), eliminate=terms["measure_vars"] | terms["plate_vars"], plates=terms["plate_vars"])
        log_prob = funsor.optimizer.apply_optimizer(log_prob)

    with approx:
        approx_factors = funsor.adjoint.adjoint(sum_op, prod_op, log_prob)

    # construct a result trace to replay against the model
    sample_tr = model_tr.copy()
    sample_subs = {}
    for name, node in sample_tr.items():
        if node["type"] != "sample":
            continue
        if node["is_observed"]:
            # "observed" values may be collapsed samples that depend on enumerated
            # values, so we have to slice them down
            # TODO this should really be handled entirely under the hood by adjoint
            output = funsor.Reals[node["fn"].event_shape]
            value = funsor.to_funsor(node["value"], output, dim_to_name=node["infer"]["dim_to_name"])
            value = value(**sample_subs)
            node["value"] = funsor.to_data(value, name_to_dim=node["infer"]["name_to_dim"])
        else:
            log_measure = approx_factors[terms["log_measures"][name]]
            sample_subs[name] = _get_support_value(log_measure, name)
            node["value"] = funsor.to_data(sample_subs[name], name_to_dim=node["infer"]["name_to_dim"])

    with replay(guide_trace=sample_tr):
        return model(*args, **kwargs)


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
        return functools.partial(infer_discrete, first_available_dim=first_available_dim, temperature=temperature, rng_key=rng_key)
    return functools.partial(_sample_posterior, fn, first_available_dim, temperature, rng_key)
