# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""MCMC benchmarks: end-to-end ``mcmc.run`` for HMC and NUTS.

The cold call pays for tracing the model, building the ``lax.scan`` sampling
loop and compiling it; warm calls hit :attr:`MCMC._cache` and re-execute the
compiled loop. That split is exactly what the report shows as compile time
versus run time.
"""

from functools import partial

from jax import random

from benchmarks.harness import benchmark
from benchmarks.models import (
    eight_schools,
    eight_schools_data,
    hierarchical_glm,
    hierarchical_glm_data,
    logistic_regression,
    logistic_regression_data,
    neals_funnel,
)
from numpyro.infer import HMC, MCMC, NUTS

# Chains are short on purpose: this measures throughput and compilation, not
# inference quality, and the whole suite runs four times per comparison.
NUM_WARMUP = 200
NUM_SAMPLES = 200


def _mcmc(kernel, **kwargs):
    return MCMC(
        kernel,
        num_warmup=kwargs.pop("num_warmup", NUM_WARMUP),
        num_samples=kwargs.pop("num_samples", NUM_SAMPLES),
        progress_bar=False,
        **kwargs,
    )


def _runner(mcmc, key, **model_kwargs):
    def run():
        mcmc.run(key, **model_kwargs)
        return mcmc.get_samples()

    return run


@benchmark(suite="mcmc", warm_repeats=3)
def nuts_logistic_regression():
    """NUTS on a dense 1000x10 logistic regression."""
    data = logistic_regression_data()
    mcmc = _mcmc(NUTS(logistic_regression))
    return _runner(mcmc, random.key(0), **data)


@benchmark(suite="mcmc", warm_repeats=3)
def nuts_eight_schools():
    """NUTS on non-centred eight schools."""
    data = eight_schools_data()
    mcmc = _mcmc(NUTS(eight_schools), num_warmup=500, num_samples=500)
    return _runner(mcmc, random.key(0), **data)


@benchmark(suite="mcmc", warm_repeats=3)
def nuts_hierarchical_glm():
    """NUTS on a partially pooled regression with 20 groups."""
    data = hierarchical_glm_data()
    mcmc = _mcmc(NUTS(hierarchical_glm))
    return _runner(mcmc, random.key(0), **data)


@benchmark(suite="mcmc", warm_repeats=3)
def nuts_dense_mass_funnel():
    """NUTS with a dense mass matrix on Neal's funnel."""
    mcmc = _mcmc(NUTS(partial(neals_funnel, dim=10), dense_mass=True))
    return _runner(mcmc, random.key(0))


@benchmark(suite="mcmc", warm_repeats=3)
def hmc_logistic_regression():
    """HMC with a fixed trajectory length on logistic regression."""
    data = logistic_regression_data()
    mcmc = _mcmc(HMC(logistic_regression, trajectory_length=1.0))
    return _runner(mcmc, random.key(0), **data)


@benchmark(suite="mcmc", warm_repeats=2)
def nuts_vectorized_chains():
    """Four vmapped NUTS chains on eight schools."""
    data = eight_schools_data()
    mcmc = _mcmc(NUTS(eight_schools), num_chains=4, chain_method="vectorized")
    return _runner(mcmc, random.key(0), **data)
