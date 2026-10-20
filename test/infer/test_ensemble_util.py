# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp

from numpyro.infer.ensemble_util import batch_ravel_pytree, get_nondiagonal_indices


def test_nondiagonal_indices():
    truth = jnp.array([[1, 0], [2, 0], [2, 1], [0, 1], [0, 2], [1, 2]], dtype=jnp.int32)

    assert jnp.all(get_nondiagonal_indices(3) == truth)


def test_batch_ravel_pytree():
    arr1 = jnp.arange(10).reshape((5, 2))
    arr2 = jnp.arange(15).reshape((5, 3))
    arr3 = jnp.arange(20).reshape((5, 4))

    tree = {"arr1": arr1, "arr2": arr2, "arr3": arr3}

    flattened, unravel_fn = batch_ravel_pytree(tree)
    unflattened = unravel_fn(flattened)

    assert flattened.shape == (5, 2 + 3 + 4)

    for unflattened_leaf, original_leaf in zip(
        jax.tree.leaves(unflattened), jax.tree.leaves(tree)
    ):
        assert jnp.all(unflattened_leaf == original_leaf)
