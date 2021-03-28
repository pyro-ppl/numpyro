# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
import functools

import funsor
from numpyro.contrib.funsor import enum, plate_to_enum_plate
from numpyro.contrib.funsor import trace as packed_trace
from numpyro.distributions.util import is_identically_one
from numpyro.handlers import block, replay, trace
from numpyro.infer.util import _guess_max_plate_nesting


@functools.singledispatch
def _get_support_value(funsor_dist, name, **kwargs):
    raise ValueError("Could not extract point from {} at name {}".format(funsor_dist, name))


@_get_support_value.register(funsor.cnf.Contraction)
def _get_support_value_contraction(funsor_dist, name, **kwargs):
    delta_terms = [v for v in funsor_dist.terms
                   if isinstance(v, funsor.delta.Delta) and name in v.fresh]
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
        if site['type'] == 'sample':
            value = site['value']
            intermediates = site['intermediates']
            scale = site['scale']
            if intermediates:
                log_prob = site['fn'].log_prob(value, intermediates)
            else:
                log_prob = site['fn'].log_prob(value)

            if (scale is not None) and (not is_identically_one(scale)):
                log_prob = scale * log_prob

            dim_to_name = site["infer"]["dim_to_name"]
            log_prob_factor = funsor.to_funsor(log_prob, output=funsor.Real, dim_to_name=dim_to_name)

            if site['is_observed']:
                log_factors[site["name"]] = log_prob_factor
            else:
                log_measures[site["name"]] = log_prob_factor
                sum_vars |= frozenset({site['name']})
            prod_vars |= frozenset(f.name for f in site['cond_indep_stack'] if f.dim is not None)

    return {"log_factors": log_factors, "log_measures": log_measures,
            "measure_vars": sum_vars, "plate_vars": prod_vars}


def _sample_posterior(model, first_available_dim, temperature, rng_key, *args, **kwargs):

    if temperature == 0:
        sum_op, prod_op = funsor.ops.max, funsor.ops.add
        approx = funsor.approximations.argmax_approximate
    elif temperature == 1:
        sum_op, prod_op = funsor.ops.logaddexp, funsor.ops.add
        approx = funsor.montecarlo.MonteCarlo(rng_key=rng_key)
    else:
        raise ValueError("temperature must be 0 (map) or 1 (sample) for now")

    if first_available_dim is None:
        with block():
            model_trace = trace(model).get_trace(*args, **kwargs)
        first_available_dim = -_guess_max_plate_nesting(model_trace) - 1

    with block(), enum(first_available_dim=first_available_dim):
        with plate_to_enum_plate():
            model_tr = packed_trace(model).get_trace(*args, **kwargs)

    terms = terms_from_trace(model_tr)
    # terms["log_factors"] = [log p(x) for each observed or latent sample site x]
    # terms["log_measures"] = [log p(z) or other Dice factor
    #                          for each latent sample site z]

    with funsor.interpretations.lazy:
        log_prob = funsor.sum_product.sum_product(
            sum_op, prod_op,
            list(terms["log_factors"].values()) + list(terms["log_measures"].values()),
            eliminate=terms["measure_vars"] | terms["plate_vars"],
            plates=terms["plate_vars"]
        )
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


def infer_discrete(model, first_available_dim=None, temperature=1, rng_key=None):
    if temperature == 1:
        assert rng_key is not None
    return functools.partial(_sample_posterior, model, first_available_dim, temperature, rng_key)
