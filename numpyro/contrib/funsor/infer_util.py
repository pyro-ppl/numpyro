# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import funsor

from numpyro.contrib.funsor.enum_messenger import enum, trace as packed_trace
from numpyro.distributions.util import is_identically_one
from numpyro.handlers import substitute

funsor.set_backend("jax")


def enum_log_density(model, model_args, model_kwargs, params):
    model = substitute(model, param_map=params)
    model_trace = packed_trace(enum(model)).get_trace(*model_args, **model_kwargs)
    log_factors = []
    sum_vars, prod_vars = frozenset(), frozenset()
    for site in model_trace.values():
        if site['type'] == 'sample':
            value = site['value']
            intermediates = site['intermediates']
            scale = site['scale']
            if intermediates:
                log_prob = site['fn'].log_prob(intermediates)
            else:
                log_prob = site['fn'].log_prob(value)

            if (scale is not None) and (not is_identically_one(scale)):
                log_prob = scale * log_prob

            log_prob = funsor.to_funsor(log_prob, output=funsor.reals(), dim_to_name=site['infer']['dim_to_name'])
            log_factors.append(log_prob)
            sum_vars |= frozenset({site['name']})
            prod_vars |= frozenset(f.name for f in site['cond_indep_stack'] if f.dim is not None)

    with funsor.interpreter.interpretation(funsor.terms.lazy):
        lazy_result = funsor.sum_product.sum_product(
            funsor.ops.logaddexp, funsor.ops.add, log_factors,
            eliminate=sum_vars | prod_vars, plates=prod_vars)
    result = funsor.optimizer.apply_optimizer(lazy_result)
    return -result.data
