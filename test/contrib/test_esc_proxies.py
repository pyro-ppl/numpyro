# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest

from jax import numpy as jnp, random

from numpyro import plate, prng_key, sample
from numpyro.contrib.ecs_proxies import block_update
from numpyro.contrib.module import random_haiku_module
from numpyro.distributions import Cauchy, Normal
from numpyro.handlers import seed
from numpyro.infer import HMC, HMCECS, MCMC, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.optim import Adam


@pytest.mark.parametrize("num_blocks", [1, 2, 50, 100])
def test_block_update_partitioning(num_blocks):
    plate_size = 10000, 100

    plate_sizes = {"N": plate_size}
    gibbs_sites = {"N": jnp.arange(plate_size[1])}
    gibbs_state = {}

    new_gibbs_sites, new_gibbs_state = block_update(
        plate_sizes, num_blocks, random.PRNGKey(2), gibbs_sites, gibbs_state
    )
    block_size = 100 // num_blocks
    for name in gibbs_sites:
        assert (
            block_size == jnp.not_equal(gibbs_sites[name], new_gibbs_sites[name]).sum()
        )

    assert gibbs_state == new_gibbs_state


def test_haiku_compatiable():
    try:
        import haiku as hk  # noqa: F401

        data_points = 6
        x_dim = 4

        def model(x, y):
            net = random_haiku_module(
                "net",
                hk.transform(lambda x: hk.Linear(1)(x)),
                prior={"linear.b": Cauchy(), "linear.w": Normal()},
                input_shape=(1, x_dim),
            )

            with plate("data", data_points, subsample_size=2) as idx:
                yb = y[idx]
                xb = x[idx]
                sample("y", Normal(net(xb).squeeze()), obs=yb)

        x = jnp.ones((data_points, x_dim))
        y = jnp.array((data_points, 0))

        with seed(rng_seed=0):
            svi = SVI(model, AutoDelta(model), Adam(step_size=1e-3), Trace_ELBO())
            svi_result = svi.run(prng_key(), 1, x, y)
            ref_params = {
                k.removesuffix("_auto_loc"): v for k, v in svi_result.params.items()
            }

            proxy = HMCECS.taylor_proxy(ref_params, degree=2)
            kernel = HMCECS(HMC(model), num_blocks=2, proxy=proxy)

            mcmc = MCMC(kernel, num_warmup=2, num_samples=2)
            mcmc.run(prng_key(), x, y)
    except ImportError:
        pass
