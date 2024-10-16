# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp


def get_nondiagonal_indices(n):
    """
    From https://github.com/dfm/emcee/blob/main/src/emcee/moves/de.py:

    Get the indices of a square matrix with size n, excluding the diagonal.
    """
    rows, cols = np.tril_indices(n, -1)  # -1 to exclude diagonal

    # Combine rows-cols and cols-rows pairs
    pairs = np.column_stack(
        [np.concatenate([rows, cols]), np.concatenate([cols, rows])]
    )

    return jnp.asarray(pairs)


def batch_ravel_pytree(pytree):
    """
    Ravel (flatten) a pytree of arrays with leading batch dimension down to a (batch_size, 1D) array.

    Args:
      pytree: a pytree of arrays and scalars to ravel.
    Returns:
      A pair where the first element is a (batch_size, 1D) array representing the flattened and
      concatenated leaf values, with dtype determined by promoting the dtypes of
      leaf values, and the second element is a callable for unflattening a (batch_size, 1D)
      array of the same length back to a pytree of the same structure as the
      input ``pytree``. If the input pytree is empty (i.e. has no leaves) then as
      a convention a 1D empty array of dtype float32 is returned in the first
      component of the output.
    """
    flat = jax.vmap(lambda x: ravel_pytree(x)[0])(pytree)
    unravel_fn = jax.vmap(ravel_pytree(jax.tree.map(lambda z: z[0], pytree))[1])

    return flat, unravel_fn
