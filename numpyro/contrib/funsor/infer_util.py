# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from contextlib import contextmanager

import funsor
import numpyro
from numpyro.contrib.funsor.enum_messenger import infer_config
from numpyro.contrib.funsor.enum_messenger import plate as enum_plate
from numpyro.contrib.funsor.enum_messenger import trace as packed_trace
from numpyro.distributions.util import is_identically_one
from numpyro.handlers import substitute

funsor.set_backend("jax")


@contextmanager
def plate_to_enum_plate():
    """
    A context manager to replace `numpyro.plate` statement by a funsor-based
    :class:`~numpyro.contrib.funsor.enum_messenger.plate`.

    This is useful when doing inference for the usual NumPyro programs with
    `numpyro.plate` statements. For example, to get trace of a `model` whose discrete
    latent sites are enumerated, we can use

        enum_model = numpyro.contrib.funsor.enum(model)
        with plate_to_enum_plate():
            model_trace = numpyro.contrib.funsor.trace(enum_model).get_trace(
                *model_args, **model_kwargs)

    """
    try:
        numpyro.plate.__new__ = lambda cls, *args, **kwargs: enum_plate(*args, **kwargs)
        yield
    finally:
        numpyro.plate.__new__ = lambda *args, **kwargs: object.__new__(numpyro.plate)


def config_enumerate(fn, default='parallel'):
    """
    Configures enumeration for all relevant sites in a NumPyro model.

    When configuring for exhaustive enumeration of discrete variables, this
    configures all sample sites whose distribution satisfies
    ``.has_enumerate_support == True``.

    This can be used as either a function::

        model = config_enumerate(model)

    or as a decorator::

        @config_enumerate
        def model(*args, **kwargs):
            ...

    .. note:: Currently, only ``default='parallel'`` is supported.

    :param callable fn: Python callable with NumPyro primitives.
    :param str default: Which enumerate strategy to use, one of
        "sequential", "parallel", or None. Defaults to "parallel".
    """
    def config_fn(site):
        if site['type'] == 'sample' and (not site['is_observed']) \
                and site['fn'].has_enumerate_support:
            return {'enumerate': site['infer'].get('enumerate', default)}
        return {}

    return infer_config(fn, config_fn)


def compute_markov_factors(time_to_factors, time_to_init_vars, time_to_markov_dims, sum_vars, prod_vars):
    """
    :param dict time_to_factors: a map from time variable to the log prob factors.
    :param dict time_to_init_vars: a map from time variable to init discrete sites.
    :param dict time_to_markov_dims: a map from time variable to dimensions at markov sites
        (discrete sites that depend on previous steps).
    :param frozenset sum_vars: all plate and enum dimensions in the trace.
    :param frozenset prod_vars: all plate dimensions in the trace.
    :returns: a list of factors after eliminate time dimensions
    """
    markov_factors = []
    for time_var, log_factors in time_to_factors.items():
        prev_vars = time_to_init_vars[time_var]
        # remove `_init/` prefix to convert prev to curr
        prev_to_curr = {k: "/".join(k.split("/")[1:]) for k in prev_vars}
        # we eliminate all plate and enum dimensions not available at markov sites.
        eliminate_vars = (sum_vars | prod_vars) - time_to_markov_dims[time_var]
        with funsor.interpreter.interpretation(funsor.terms.lazy):
            lazy_result = funsor.sum_product.sum_product(
                funsor.ops.logaddexp, funsor.ops.add, log_factors,
                eliminate=eliminate_vars, plates=prod_vars)
        trans = funsor.optimizer.apply_optimizer(lazy_result)
        markov_factors.append(funsor.sum_product.sequential_sum_product(
            funsor.ops.logaddexp, funsor.ops.add, trans, time_var, prev_to_curr))
    return markov_factors


def log_density(model, model_args, model_kwargs, params):
    """
    Similar to :func:`numpyro.infer.util.log_density` but works for models
    with discrete latent variables. Internally, this uses :mod:`funsor`
    to marginalize discrete latent sites and evaluate the joint log probability.

    :param model: Python callable containing NumPyro primitives. Typically,
        the model has been enumerated by using
        :class:`~numpyro.contrib.funsor.enum_messenger.enum` handler::

            def model(*args, **kwargs):
                ...

            log_joint = log_density(enum(config_enumerate(model)), args, kwargs, params)

    :param tuple model_args: args provided to the model.
    :param dict model_kwargs: kwargs provided to the model.
    :param dict params: dictionary of current parameter values keyed by site
        name.
    :return: log of joint density and a corresponding model trace
    """
    model = substitute(model, data=params)
    with plate_to_enum_plate():
        model_trace = packed_trace(model).get_trace(*model_args, **model_kwargs)
    log_factors = []
    time_to_factors = defaultdict(list)  # log prob factors
    time_to_init_vars = defaultdict(frozenset)  # _init/... variables
    time_to_markov_dims = defaultdict(frozenset)  # dimensions at markov sites
    sum_vars, prod_vars = frozenset(), frozenset()
    for site in model_trace.values():
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
            log_prob = funsor.to_funsor(log_prob, output=funsor.reals(), dim_to_name=dim_to_name)

            time_dim = None
            for dim, name in dim_to_name.items():
                if name.startswith("_time"):
                    time_dim = funsor.Variable(name, funsor.domains.bint(site["value"].shape[dim]))
                    time_to_factors[time_dim].append(log_prob)
                    time_to_init_vars[time_dim] |= frozenset(
                        s for s in dim_to_name.values() if s.startswith("_init"))
                    break
            if time_dim is None:
                log_factors.append(log_prob)

            if not site['is_observed']:
                sum_vars |= frozenset({site['name']})
            prod_vars |= frozenset(f.name for f in site['cond_indep_stack'] if f.dim is not None)

    for time_dim, init_vars in time_to_init_vars.items():
        for var in init_vars:
            curr_var = "/".join(var.split("/")[1:])
            dim_to_name = model_trace[curr_var]["infer"]["dim_to_name"]
            if var in dim_to_name.values():  # i.e. _init (i.e. prev) in dim_to_name
                time_to_markov_dims[time_dim] |= frozenset(name for name in dim_to_name.values())

    if len(time_to_factors) > 0:
        markov_factors = compute_markov_factors(time_to_factors, time_to_init_vars,
                                                time_to_markov_dims, sum_vars, prod_vars)
        log_factors = log_factors + markov_factors

    with funsor.interpreter.interpretation(funsor.terms.lazy):
        lazy_result = funsor.sum_product.sum_product(
            funsor.ops.logaddexp, funsor.ops.add, log_factors,
            eliminate=sum_vars | prod_vars, plates=prod_vars)
    result = funsor.optimizer.apply_optimizer(lazy_result)
    return result.data, model_trace
