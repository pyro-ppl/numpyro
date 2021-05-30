# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp


def posdef(m):
    """Map a matrix to a positive definite matrix, where all eigenvalues are >= 1e-5."""
    mlambda, mvec = jnp.linalg.eigh(m)
    if jnp.ndim(mlambda) >= 2:
        mlambda = jax.vmap(
            lambda ml: jnp.diag(jnp.maximum(ml, 1e-5)),
            in_axes=tuple(range(jnp.ndim(mlambda) - 1)),
        )(mlambda)
    else:
        mlambda = jnp.diag(jnp.maximum(mlambda, 1e-5))
    return mvec @ mlambda @ jnp.swapaxes(mvec, -2, -1)


def sqrth(m):
    """Map a matrix to a positive definite matrix and square root it."""
    mlambda, mvec = jnp.linalg.eigh(m)
    if jnp.ndim(mlambda) >= 2:
        mlambdasqrt = jax.vmap(
            lambda ml: jnp.diag(jnp.maximum(ml, 1e-5) ** 0.5),
            in_axes=tuple(range(jnp.ndim(mlambda) - 1)),
        )(mlambda)
        msqrt = mvec @ mlambdasqrt @ jnp.swapaxes(mvec, -2, -1)
    else:
        mlambdasqrt = jnp.diag(jnp.maximum(mlambda, 1e-5) ** 0.5)
        msqrt = mvec * mlambdasqrt * jnp.swapaxes(mvec, -2, -1)

    return msqrt


def safe_norm(a, ord=2, axis=None):
    if axis is not None:
        is_zero = jnp.expand_dims(jnp.isclose(jnp.sum(a, axis=axis), 0.0), axis=axis)
    else:
        is_zero = jnp.ones_like(a, dtype="bool")
    norm = jnp.linalg.norm(
        a + jnp.where(is_zero, jnp.ones_like(a) * 1e-5 ** ord, jnp.zeros_like(a)),
        ord=ord,
        axis=axis,
    )
    return norm
