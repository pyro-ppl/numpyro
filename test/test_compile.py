# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest

import jax
from jax import lax, random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDiagonalNormal
import numpyro.optim as optim

GLOBAL = {"count": 0}


def model(deterministic=True):
    GLOBAL["count"] += 1
    x = numpyro.sample("x", dist.Normal())
    if deterministic:
        numpyro.deterministic("x_copy", x)


@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("find_heuristic_step_size", [True, False])
def test_mcmc_one_chain(deterministic, find_heuristic_step_size):
    GLOBAL["count"] = 0
    mcmc = MCMC(
        NUTS(model, find_heuristic_step_size=find_heuristic_step_size),
        num_warmup=100,
        num_samples=100,
    )
    mcmc.run(random.PRNGKey(0), deterministic=deterministic)
    mcmc.get_samples()

    num_traces_for_heuristic = 2 if find_heuristic_step_size else 0
    if deterministic:
        assert GLOBAL["count"] == 4 + num_traces_for_heuristic
    else:
        assert GLOBAL["count"] == 3 + num_traces_for_heuristic


@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.skipif(jax.device_count() < 2, reason="only one device is available")
def test_mcmc_parallel_chain(deterministic):
    GLOBAL["count"] = 0
    mcmc = MCMC(NUTS(model), num_warmup=100, num_samples=100, num_chains=2)
    mcmc.run(random.PRNGKey(0), deterministic=deterministic)
    mcmc.get_samples()

    if deterministic:
        assert GLOBAL["count"] == 4
    else:
        assert GLOBAL["count"] == 3


@pytest.mark.parametrize("deterministic", [True, False])
def test_autoguide(deterministic):
    GLOBAL["count"] = 0
    guide = AutoDiagonalNormal(model)
    svi = SVI(model, guide, optim.Adam(0.1), Trace_ELBO(), deterministic=deterministic)
    svi_state = svi.init(random.PRNGKey(0))
    svi_state = lax.fori_loop(0, 100, lambda i, val: svi.update(val)[0], svi_state)
    params = svi.get_params(svi_state)
    guide.sample_posterior(random.PRNGKey(1), params, sample_shape=(100,))

    if deterministic:
        assert GLOBAL["count"] == 5
    else:
        assert GLOBAL["count"] == 4
