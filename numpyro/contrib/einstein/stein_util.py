# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import jax
from jax import numpy as jnp, vmap
from jax.flatten_util import ravel_pytree

from numpyro.distributions import biject_to
from numpyro.distributions.constraints import real
from numpyro.distributions.transforms import ComposeTransform, IdentityTransform


def posdef(m):
    """Map a matrix to a positive definite matrix, where all eigenvalues are >= 1e-5."""
    mlambda, mvec = jnp.linalg.eigh(m)
    mlambda = jnp.maximum(mlambda, 1e-5)
    return (mvec * jnp.expand_dims(mlambda, -2)) @ jnp.swapaxes(mvec, -2, -1)


def sqrth(m):
    """Map a matrix to a positive definite matrix and square root it."""
    mlambda, mvec = jnp.linalg.eigh(m)
    mlambdasqrt = jnp.maximum(mlambda, 1e-5) ** 0.5
    msqrt = (mvec * jnp.expand_dims(mlambdasqrt, -2)) @ jnp.swapaxes(mvec, -2, -1)
    return msqrt


def all_pairs_eucl_dist(a, b):
    a_sqr = jnp.sum(a**2, 1)[None, :]
    b_sqr = jnp.sum(b**2, 1)[:, None]
    diff = jnp.matmul(b, a.T)
    return jnp.sqrt(jnp.maximum(a_sqr + b_sqr - 2 * diff, 0.0))


def median_bandwidth(particles, factor_fn):
    if particles.shape[0] == 1:
        return 1.0  # Median produces NaN for single particle
    dists = all_pairs_eucl_dist(particles, particles)
    bandwidth = (
        jnp.median(dists) ** 2 * factor_fn(particles.shape[0])
        + jnp.finfo(dists.dtype).eps
    )
    return bandwidth


def get_parameter_transform(site):
    constraint = site["kwargs"].get("constraint", real)
    transform = site["kwargs"].get("particle_transform", IdentityTransform())
    return ComposeTransform([transform, biject_to(constraint)])


def batch_ravel_pytree(pytree, nbatch_dims=0):
    """Ravel a pytree to a flat array. Assumes all leaves have the same shape. If `nbatch_dims>1' then the batch
    is flattened to one dimension.

    :param pytree: a pytree of arrays with same shape.
    :param nbatch_dims: number of batch dimensions in from right most.
    :return: A triple where the first element is a 1D array of flattened and concatenated leave values.
        The `dtype` is determined by promoting the dtypes of leaf values. The second element is a callable for
        converting 1D arrays (of an element in the batch) back to a pytree. The third element is a callable for
        converting the entire flattened array back to a pytree.
    """
    if nbatch_dims == 0:
        flat, unravel_fn = ravel_pytree(pytree)
        return flat, unravel_fn, unravel_fn

    shapes = jax.tree.map(lambda x: x.shape, pytree)
    flat_pytree = jax.tree.map(lambda x: x.reshape(*x.shape[:-nbatch_dims], -1), pytree)
    flat = vmap(lambda x: ravel_pytree(x)[0])(flat_pytree)
    unravel_fn = ravel_pytree(jax.tree.map(lambda x: x[0], flat_pytree))[1]
    return (
        flat,
        unravel_fn,
        lambda _flat: jax.tree.map(
            lambda x, shape: x.reshape(shape), vmap(unravel_fn)(_flat), shapes
        ),
    )
