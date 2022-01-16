# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import jax
from jax import numpy as jnp, vmap
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map, tree_multimap

from numpyro.distributions import biject_to
from numpyro.distributions.constraints import real
from numpyro.distributions.transforms import ComposeTransform, IdentityTransform


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


def get_parameter_transform(site):
    constraint = site["kwargs"].get("constraint", real)
    transform = site["kwargs"].get("particle_transform", IdentityTransform())
    return ComposeTransform([transform, biject_to(constraint)])


def batch_ravel_pytree(pytree, nbatch_dims=0):
    if nbatch_dims == 0:
        return ravel_pytree(pytree)

    shapes = tree_map(lambda x: x.shape, pytree)
    flat_pytree = tree_map(lambda x: x.reshape(*x.shape[:-nbatch_dims], -1), pytree)
    flat = vmap(lambda x: ravel_pytree(x)[0])(flat_pytree)
    unravel_fn = ravel_pytree(tree_map(lambda x: x[0], flat_pytree))[1]
    return flat, lambda _flat: tree_multimap(
        lambda x, shape: x.reshape(shape), vmap(unravel_fn)(_flat), shapes
    )
