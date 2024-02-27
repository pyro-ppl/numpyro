# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest

from jax import numpy as jnp, random

from numpyro.contrib.ecs_proxies import block_update


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
