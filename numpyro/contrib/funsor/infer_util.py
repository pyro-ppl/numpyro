# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from functools import partial

import funsor

import numpyro
from numpyro.contrib.funsor.enum_messenger import enum, plate as enum_plate, trace as packed_trace
from numpyro.distributions.util import is_identically_one
from numpyro.handlers import substitute

funsor.set_backend("jax")


@contextmanager
def plate_to_enum_plate():
    try:
        numpyro.plate.__new__ = lambda *args, **kwargs: object.__new__(enum_plate)
        yield
    finally:
        numpyro.plate.__new__ = lambda *args, **kwargs: object.__new__(numpyro.plate)


class infer_config(numpyro.primitives.Messenger):
    def __init__(self, fn=None, config_fn=None):
        assert config_fn is not None
        self.config_fn = config_fn
        super().__init__(fn)

    def process_message(self, msg):
        if msg['type'] != 'sample':
            return

        msg["infer"].update(self.config_fn(msg))
        return None


def _config_fn(default, site):
    if site['type'] == 'sample' and (not site['is_observed']) \
            and site['fn'].has_enumerate_support:
        return {'enumerate': site['infer'].get('enumerate', default)}
    return {}


def config_enumerate(fn, default='parallel'):
    return infer_config(fn, partial(_config_fn, default))


def enum_log_density(model, model_args, model_kwargs, params):
    model = substitute(config_enumerate(model), param_map=params)
    with plate_to_enum_plate():
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
