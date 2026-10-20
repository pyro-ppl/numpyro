# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""Effect-handler and model-tracing benchmarks.

Several of these run entirely in Python with no ``jax.jit`` involved, so their
compile-time column is expected to sit at zero and their run-time column
measures pure interpreter overhead in ``numpyro/primitives.py`` and
``numpyro/handlers.py``. Because every inference algorithm traces the model at
least once, a regression here is felt everywhere -- including in the
compile-time column of every other suite.
"""

import jax
from jax import random

from benchmarks.harness import benchmark
from benchmarks.models import (
    hierarchical_glm,
    hierarchical_glm_data,
    logistic_regression,
    logistic_regression_data,
)
from numpyro.handlers import condition, seed, substitute, trace
from numpyro.infer import Predictive
from numpyro.infer.util import initialize_model, log_density, potential_energy


@benchmark(suite="handlers", warm_repeats=20)
def trace_seeded_model():
    """trace(seed(model)).get_trace on logistic regression -- pure Python."""
    data = logistic_regression_data()
    key = random.key(0)

    def run():
        return trace(seed(logistic_regression, key)).get_trace(**data)

    return run


@benchmark(suite="handlers", warm_repeats=20)
def nested_handler_stack():
    """A four-deep substitute/condition/seed/trace stack on the hierarchical GLM."""
    data = hierarchical_glm_data()
    key = random.key(0)
    params = {
        site: value["value"]
        for site, value in trace(seed(hierarchical_glm, key)).get_trace(**data).items()
        if value["type"] == "sample" and not value.get("is_observed", False)
    }
    observed = {"obs": data["y"]}

    def run():
        model = substitute(hierarchical_glm, data=params)
        model = condition(model, data=observed)
        return trace(seed(model, key)).get_trace(**data)

    return run


@benchmark(suite="handlers", warm_repeats=20)
def log_density_hierarchical():
    """log_density on the hierarchical GLM -- untraced, un-jitted."""
    data = hierarchical_glm_data()
    key = random.key(0)
    params = {
        site: value["value"]
        for site, value in trace(seed(hierarchical_glm, key)).get_trace(**data).items()
        if value["type"] == "sample" and not value.get("is_observed", False)
    }

    def run():
        return log_density(hierarchical_glm, (), data, params)

    return run


@benchmark(suite="handlers", warm_repeats=20)
def potential_energy_and_grad():
    """jit(grad(potential_energy)) on logistic regression -- the NUTS inner loop."""
    data = logistic_regression_data()
    key = random.key(0)
    # ``initialize_model`` hands back unconstrained values, which is exactly
    # what ``potential_energy`` expects.
    params = initialize_model(key, logistic_regression, model_kwargs=data).param_info.z
    grad_fn = jax.jit(
        jax.grad(lambda p: potential_energy(logistic_regression, (), data, p))
    )

    def run():
        return grad_fn(params)

    return run


@benchmark(suite="handlers", warm_repeats=5)
def initialize_model_hierarchical():
    """initialize_model on the hierarchical GLM -- tracing plus init strategy."""
    data = hierarchical_glm_data()
    key = random.key(0)

    def run():
        info = initialize_model(key, hierarchical_glm, model_kwargs=data)
        return info.param_info.z

    return run


@benchmark(suite="handlers", warm_repeats=5)
def predictive_forward_sampling():
    """Predictive with 250 draws from the prior of the hierarchical GLM."""
    data = hierarchical_glm_data()
    key = random.key(0)
    model_kwargs = {k: v for k, v in data.items() if k != "y"}
    predictive = Predictive(hierarchical_glm, num_samples=250)

    def run():
        return predictive(key, **model_kwargs)

    return run
