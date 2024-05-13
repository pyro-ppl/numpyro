# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
This module contains functions for computing eigenvalues and eigenfunctions of the laplace operator.
"""

from jaxlib.xla_extension import ArrayImpl

import jax.numpy as jnp


# TODO: Adapt to dim >= 1.
def sqrt_eigenvalues(ell: float, m: int) -> ArrayImpl:
    """
    The first `m` square root of eigenvalues of the laplacian operator in `[-ell, ell]`. See Eq. (56) in [1].

    **References:**

    1. Solin, A., Särkkä, S. Hilbert space methods for reduced-rank Gaussian process regression.
    Stat Comput 30, 419-446 (2020)

    :param float ell: The length of the interval divided by 2.
    :param int m: The number of eigenvalues to compute.
    :returns: An array of the first `m` square root of eigenvalues.
    :rtype: ArrayImpl
    """
    return jnp.arange(1, 1 + m) * jnp.pi / 2 / ell


# TODO: Adapt to dim >= 1.
def eigenfunctions(x: ArrayImpl, ell: float, m: int) -> ArrayImpl:
    """
    The first `m` eigenfunctions of the laplacian operator in `[-ell, ell]`
    evaluated at `x`. See Eq. (56) in [1].

    **Example:**

    .. code-block:: python

        >>> import jax.numpy as jnp

        >>> from numpyro.contrib.hsgp.laplacian import eigenfunctions

        >>> n = 100
        >>> m = 10

        >>> x = jnp.linspace(-1, 1, n)

        >>> basis = eigenfunctions(x=x, ell=1.2, m=m)

        >>> assert basis.shape == (n, m)


    **References:**
    1. Solin, A., Särkkä, S. Hilbert space methods for reduced-rank Gaussian process regression.
    Stat Comput 30, 419-446 (2020)

    :param ArrayImpl x: The points at which to evaluate the eigenfunctions.
    :param float ell: The length of the interval divided by 2.
    :param int m: The number of eigenfunctions to compute.
    :returns: An array of the first `m` eigenfunctions evaluated at `x`.
    :rtype: ArrayImpl
    """
    m1 = (jnp.pi / (2 * ell)) * jnp.tile(ell + x[:, None], m)
    m2 = jnp.diag(jnp.linspace(1, m, num=m))
    num = jnp.sin(m1 @ m2)
    den = jnp.sqrt(ell)
    return num / den


def eigenfunctions_periodic(x, w0, m):
    """
    Basis functions for the approximation of the periodic kernel.

    :param x: The points at which to evaluate the eigenfunctions.
    :param w0: The frequency of the periodic kernel.
    :param m: The number of eigenfunctions to compute.

    .. note::
        If you want to parameterize it with respect to the period use `w0 = 2 * jnp.pi / period`.
    """
    m1 = jnp.tile(w0 * x[:, None], m)
    m2 = jnp.diag(jnp.arange(m, dtype=jnp.float32))
    mw0x = m1 @ m2
    cosines = jnp.cos(mw0x)
    sines = jnp.sin(mw0x)
    return cosines, sines
