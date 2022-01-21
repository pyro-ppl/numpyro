# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numpy.testing import assert_allclose
import pytest

from jax import numpy as jnp
from jax.tree_util import tree_flatten, tree_multimap

from numpyro.contrib.einstein.util import batch_ravel_pytree


@pytest.mark.parametrize(
    "pytree",
    [
        {
            "a": np.array([[[1.0], [0.0], [3]], [[1], [0.0], [-1]]]),
        },
        {
            "b": np.array([[1.0, 0.0, 3], [1, 0.0, -1]]),
        },
        [
            np.array([[1.0, 0.0, 3], [1, 0.0, -1]]),
            np.array([[1.0, 0.0, 3], [1, 0.0, -1]]),
        ],
    ],
)
@pytest.mark.parametrize("nbatch_dims", [1, 2])
def test_ravel_pytree_batched(pytree, nbatch_dims):
    flat, unravel_fn = batch_ravel_pytree(pytree, nbatch_dims)
    unravel = unravel_fn(flat)
    tree_flatten(tree_multimap(lambda x, y: assert_allclose(x, y), unravel, pytree))
    assert all(
        tree_flatten(
            tree_multimap(
                lambda x, y: jnp.result_type(x) == jnp.result_type(y), unravel, pytree
            )
        )[0]
    )
