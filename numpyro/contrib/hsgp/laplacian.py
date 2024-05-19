# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
This module contains functions for computing eigenvalues and eigenfunctions of the laplace operator.
"""

from jaxlib.xla_extension import ArrayImpl

import jax
import jax.numpy as jnp


def eigen_indices(m: list[int] | int, dim: int) -> ArrayImpl:
    """Returns the indices of the first `m_star x D` eigenvalues of the laplacian operator.

    .. math::

        m^\\star = prod_{i=1}^D m_i

    **References:**
    1. Riutort-Mayol, G., Bürkner, PC., Andersen, M.R. et al. Practical Hilbert space
       approximate Bayesian Gaussian processes for probabilistic programming. Stat Comput 33, 17 (2023).

    :param Sequence[int] | int m: The number of desired eigenvalue indices in each dimension.
    If an integer, the same number of eigenvalues is computed in each dimension.
    :param int dim: The dimension of the space.

    :returns: An array of the indices of the first `m_star x D` eigenvalues.
    :rtype: ArrayImpl

    **Examples:**

    .. code-block:: python

            >>> import jax.numpy as jnp

            >>> from numpyro.contrib.hsgp.laplacian import eigen_indices

            >>> m = 10
            >>> S = eigen_indices(m, 1)
            >>> assert S.shape == (1, m)
            >>> S
            Array([[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]], dtype=int32)

            >>> m = 10
            >>> S = eigen_indices(m, 2)
            >>> assert S.shape == (2, 100)

            >>> m = [2, 2, 3]  # Ruitort-Mayol et al eq (10)
            >>> S = eigen_indices(m, 3)
            >>> assert S.shape == (3, 12)
            >>> S
            Array([[1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
                   [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
                   [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]], dtype=int32)

    """
    if isinstance(m, int):
        m = [m] * dim
    elif len(m) != dim:
        raise ValueError("The length of m must be equal to the dimension of the space.")
    return (
        jnp.stack(
            jnp.meshgrid(*[jnp.arange(1, m_ + 1) for m_ in m], indexing="ij"), axis=-1
        )
        .reshape(-1, dim)
        .T
    )


def sqrt_eigenvalues(
    ell: float | list[float], m: list[int] | int, dim: int
) -> ArrayImpl:
    """
    The first `dim x m_star` square root of eigenvalues of the laplacian operator in
    `[-ell_1, ell_1] x ... x [-ell_D, ell_D]`. See Eq. (56) in [1].

    **References:**

    1. Solin, A., Särkkä, S. Hilbert space methods for reduced-rank Gaussian process regression.
    Stat Comput 30, 419-446 (2020)

    :param Sequence[float] | float ell: The length of the interval in each dimension divided by 2.
    If a float, the same length is used in each dimension.
    :param list[int] | int m: The number of eigenvalues to compute in each dimension.
    If an integer, the same number of eigenvalues is computed in each dimension.
    :param int dim: The dimension of the space.

    :returns: An array of the first `m` square root of eigenvalues.
    :rtype: ArrayImpl
    """
    ell_ = _convert_ell(ell, dim)
    S = eigen_indices(m, dim)
    return S * jnp.pi / 2 / ell_  # dim x prod(m) array of eigenvalues


def eigenfunctions(
    x: ArrayImpl, ell: float | list[float], m: int | list[int]
) -> ArrayImpl:
    """
    The first `m_star` eigenfunctions of the laplacian operator in `[-ell_1, ell_1] x ... x [-ell_D, ell_D]`
    evaluated at values of `x`. See Eq. (56) in [1].

    **Example:**

    .. code-block:: python

        >>> import jax.numpy as jnp

        >>> from numpyro.contrib.hsgp.laplacian import eigenfunctions

        >>> n = 100
        >>> m = 10

        >>> x = jnp.linspace(-1, 1, n)

        >>> basis = eigenfunctions(x=x, ell=1.2, m=m)

        >>> assert basis.shape == (n, m)

        >>> x = jnp.ones((n, 3))  # 2d input
        >>> basis = eigenfunctions(x=x, ell=1.2, m=[2, 2, 3])
        >>> assert basis.shape == (n, 12)


    **References:**
    1. Solin, A., Särkkä, S. Hilbert space methods for reduced-rank Gaussian process regression.
    Stat Comput 30, 419-446 (2020)

    :param ArrayImpl x: The points at which to evaluate the eigenfunctions.
    If x is 1D the problem is assumed unidimensional.
    Otherwise, the dimension of the input space is inferred as the last dimension of x.
    Other dimensions are treated as batch dimensions.
    :param float | list[float] ell: The length of the interval in each dimension divided by 2.
    If a float, the same length is used in each dimension.
    :param int | list[int] m: The number of eigenvalues to compute in each dimension.
    If an integer, the same number of eigenvalues is computed in each dimension.
    :returns: An array of the first `m_star` eigenfunctions evaluated at `x`.
    :rtype: ArrayImpl
    """
    if x.ndim == 1:
        x_ = x[..., None]
    else:
        x_ = x
    dim = x_.shape[-1]  # others assumed batch dims
    n_batch_dims = x_.ndim - 1
    ell_ = _convert_ell(ell, dim)
    a = jnp.expand_dims(ell_, tuple(range(n_batch_dims)))
    b = jnp.expand_dims(sqrt_eigenvalues(ell_, m, dim), tuple(range(n_batch_dims)))
    return jnp.prod(jnp.sqrt(1 / a) * jnp.sin(b * (x_[..., None] + a)), axis=-2)


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


def _convert_ell(ell: float | list[float] | ArrayImpl, dim: int) -> ArrayImpl:
    if isinstance(ell, float | int):
        ell = [ell] * dim
    if isinstance(ell, list):
        if len(ell) != dim:
            raise ValueError(
                "The length of ell must be equal to the dimension of the space."
            )
        ell_ = jnp.array(ell)[..., None]  # dim x 1 array
    elif isinstance(ell, jax.Array):
        ell_ = ell
    if ell_.shape != (dim, 1):
        raise ValueError("ell must be a scalar or a list of length dim.")
    return ell_
