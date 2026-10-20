# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""SVI benchmarks: ``svi.run`` with the common autoguides.

With ``progress_bar=False`` the whole optimisation loop is a single
``lax.scan``, so the cold call measures guide construction plus compilation of
that loop and the warm calls measure raw step throughput.
"""

from jax import random

from benchmarks.harness import benchmark
from benchmarks.models import (
    eight_schools,
    eight_schools_data,
    hierarchical_glm,
    hierarchical_glm_data,
    logistic_regression,
    logistic_regression_data,
)
from numpyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from numpyro.infer.autoguide import (
    AutoDelta,
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    AutoNormal,
)
import numpyro.optim as optim

NUM_STEPS = 1000


def _runner(svi, key, num_steps=NUM_STEPS, **model_kwargs):
    def run():
        return svi.run(key, num_steps, progress_bar=False, **model_kwargs)

    return run


@benchmark(suite="svi", warm_repeats=3)
def svi_autonormal_logistic():
    """AutoNormal + Trace_ELBO on logistic regression."""
    data = logistic_regression_data()
    svi = SVI(
        logistic_regression,
        AutoNormal(logistic_regression),
        optim.Adam(1e-3),
        Trace_ELBO(),
    )
    return _runner(svi, random.key(0), **data)


@benchmark(suite="svi", warm_repeats=3)
def svi_autodiagonalnormal_hierarchical():
    """AutoDiagonalNormal + Trace_ELBO on the hierarchical GLM."""
    data = hierarchical_glm_data()
    svi = SVI(
        hierarchical_glm,
        AutoDiagonalNormal(hierarchical_glm),
        optim.Adam(1e-3),
        Trace_ELBO(),
    )
    return _runner(svi, random.key(0), **data)


@benchmark(suite="svi", warm_repeats=3)
def svi_automultivariatenormal_eight_schools():
    """AutoMultivariateNormal on eight schools -- exercises the Cholesky path."""
    data = eight_schools_data()
    svi = SVI(
        eight_schools,
        AutoMultivariateNormal(eight_schools),
        optim.Adam(1e-3),
        Trace_ELBO(),
    )
    return _runner(svi, random.key(0), **data)


@benchmark(suite="svi", warm_repeats=3)
def svi_autodelta_map_logistic():
    """AutoDelta (MAP estimation) on logistic regression."""
    data = logistic_regression_data()
    svi = SVI(
        logistic_regression,
        AutoDelta(logistic_regression),
        optim.Adam(1e-2),
        Trace_ELBO(),
    )
    return _runner(svi, random.key(0), **data)


@benchmark(suite="svi", warm_repeats=3)
def svi_trace_mean_field_elbo():
    """TraceMeanField_ELBO on the hierarchical GLM -- analytic KL path."""
    data = hierarchical_glm_data()
    svi = SVI(
        hierarchical_glm,
        AutoNormal(hierarchical_glm),
        optim.Adam(1e-3),
        TraceMeanField_ELBO(),
    )
    return _runner(svi, random.key(0), **data)


@benchmark(suite="svi", warm_repeats=3)
def svi_multi_particle_elbo():
    """Trace_ELBO with 16 particles -- vmapped ELBO estimation."""
    data = logistic_regression_data()
    svi = SVI(
        logistic_regression,
        AutoNormal(logistic_regression),
        optim.Adam(1e-3),
        Trace_ELBO(num_particles=16),
    )
    return _runner(svi, random.key(0), num_steps=500, **data)
