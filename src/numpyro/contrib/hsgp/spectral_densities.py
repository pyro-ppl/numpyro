# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
This module contains spectral densities for various kernel functions.
"""

from __future__ import annotations

from jaxlib.xla_extension import ArrayImpl

from jax import vmap
import jax.numpy as jnp
from jax.scipy import special

from numpyro.contrib.hsgp.laplacian import sqrt_eigenvalues


def align_param(dim, param):
    return jnp.broadcast_to(param, jnp.broadcast_shapes(jnp.shape(param), (dim,)))


def spectral_density_squared_exponential(
    dim: int, w: ArrayImpl, alpha: float, length: float | ArrayImpl
) -> float:
    """
    Spectral density of the squared exponential kernel.

    See Section 4.2 in [1] and Section 2.1 in [2].

    .. math::

        S(\\boldsymbol{\\omega}) = \\alpha (\\sqrt{2\\pi})^D \\ell^D
            \\exp\\left(-\\frac{1}{2} \\ell^2 \\boldsymbol{\\omega}^{T} \\boldsymbol{\\omega}\\right)


    **References:**

        1. Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning.

        2. Riutort-Mayol, G., Bürkner, PC., Andersen, M.R. et al. Practical Hilbert space
           approximate Bayesian Gaussian processes for probabilistic programming. Stat Comput 33, 17 (2023).

    :param int dim: dimension
    :param ArrayImpl w: frequency
    :param float alpha: amplitude
    :param float length: length scale
    :return: spectral density value
    :rtype: float
    """
    length = align_param(dim, length)
    c = alpha * jnp.prod(jnp.sqrt(2 * jnp.pi) * length, axis=-1)
    e = jnp.exp(-0.5 * jnp.sum(w**2 * length**2, axis=-1))
    return c * e


def spectral_density_matern(
    dim: int, nu: float, w: ArrayImpl, alpha: float, length: float | ArrayImpl
) -> float:
    """
    Spectral density of the Matérn kernel.

    See Eq. (4.15) in [1] and Section 2.1 in [2].

    .. math::

        S(\\boldsymbol{\\omega}) = \\alpha
            \\frac{2^{D} \\pi^{D/2} \\Gamma(\\nu + D/2) (2 \\nu)^{\\nu}}{\\Gamma(\\nu) \\ell^{2 \\nu}}
            \\left(\\frac{2 \\nu}{\\ell^2} + \\boldsymbol{\\omega}^{T} \\boldsymbol{\\omega}\\right)^{-\\nu - D/2}


    **References:**

        1. Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning.

        2. Riutort-Mayol, G., Bürkner, PC., Andersen, M.R. et al. Practical Hilbert space
           approximate Bayesian Gaussian processes for probabilistic programming. Stat Comput 33, 17 (2023).

    :param int dim: dimension
    :param float nu: smoothness
    :param ArrayImpl w: frequency
    :param float alpha: amplitude
    :param float length: length scale
    :return: spectral density value
    :rtype: float
    """  # noqa: E501
    length = align_param(dim, length)
    c1 = (
        alpha
        * (2 ** (dim))
        * (jnp.pi ** (dim / 2))
        * ((2 * nu) ** nu)
        * special.gamma(nu + dim / 2)
    )
    s = jnp.sum(length**2 * w**2, axis=-1)
    c2 = jnp.prod(length, axis=-1) * (2 * nu + s) ** (-nu - dim / 2)
    c3 = special.gamma(nu)
    return c1 * c2 / c3


def diag_spectral_density_squared_exponential(
    alpha: float,
    length: float | list[float],
    ell: float | int | list[float | int],
    m: int | list[int],
    dim: int,
) -> ArrayImpl:
    """
    Evaluates the spectral density of the squared exponential kernel at the first :math:`D \\times m^\\star`
    square root eigenvalues of the laplacian operator in :math:`[-L_1, L_1] \\times ... \\times [-L_D, L_D]`.

    :param float alpha: amplitude of the squared exponential kernel
    :param float length: length scale of the squared exponential kernel
    :param float | int | list[float | int] ell: The length of the interval divided by 2 in each dimension.
        If a float or int, the same length is used in each dimension.
    :param int | list[int] m: The number of eigenvalues to compute for each dimension.
        If an integer, the same number of eigenvalues is computed in each dimension.
    :param int dim: The dimension of the space

    :return: spectral density vector evaluated at the first :math:`D \\times m^\\star` square root eigenvalues
    :rtype: ArrayImpl
    """

    def _spectral_density(w):
        return spectral_density_squared_exponential(
            dim=dim, w=w, alpha=alpha, length=length
        )

    sqrt_eigenvalues_ = sqrt_eigenvalues(ell=ell, m=m, dim=dim)  # dim x m
    return vmap(_spectral_density, in_axes=-1)(sqrt_eigenvalues_)


# TODO support length-D kernel hyperparameters
def diag_spectral_density_matern(
    nu: float,
    alpha: float,
    length: float,
    ell: float | int | list[float | int],
    m: int | list[int],
    dim: int,
) -> ArrayImpl:
    """
    Evaluates the spectral density of the Matérn kernel at the first :math:`D \\times m^\\star`
    square root eigenvalues of the laplacian operator in :math:`[-L_1, L_1] \\times ... \\times [-L_D, L_D]`.

    :param float nu: smoothness parameter
    :param float alpha: amplitude of the Matérn kernel
    :param float length: length scale of the Matérn kernel
    :param float | int | list[float | int] ell: The length of the interval divided by 2 in each dimension.
        If a float or int, the same length is used in each dimension.
    :param int | list[int] m: The number of eigenvalues to compute for each dimension.
        If an integer, the same number of eigenvalues is computed in each dimension.
    :param int dim: The dimension of the space

    :return: spectral density vector evaluated at the first :math:`D \\times m^\\star` square root eigenvalues
    :rtype: ArrayImpl
    """

    def _spectral_density(w):
        return spectral_density_matern(dim=dim, nu=nu, w=w, alpha=alpha, length=length)

    sqrt_eigenvalues_ = sqrt_eigenvalues(ell=ell, m=m, dim=dim)
    return vmap(_spectral_density, in_axes=-1)(sqrt_eigenvalues_)


def modified_bessel_first_kind(v, z):
    try:
        from tensorflow_probability.substrates import jax as tfp
    except ImportError as e:
        raise ImportError(
            "TensorFlow Probability is required for this function."
        ) from e

    v = jnp.asarray(v, dtype=float)
    z = jnp.asarray(z, dtype=float)
    return jnp.exp(jnp.abs(z)) * tfp.math.bessel_ive(v, z)


def diag_spectral_density_periodic(alpha: float, length: float, m: int) -> ArrayImpl:
    """
    Not actually a spectral density but these are used in the same
    way. These are simply the first `m` coefficients of the low rank
    approximation for the periodic kernel. See Appendix B in [1].

    **References:**

        1. Riutort-Mayol, G., Bürkner, PC., Andersen, M.R. et al. Practical Hilbert space
           approximate Bayesian Gaussian processes for probabilistic programming. Stat Comput 33, 17 (2023).

    :param float alpha: amplitude
    :param float length: length scale
    :param int m: number of eigenvalues
    :return: "spectral density" vector
    :rtype: ArrayImpl
    """
    a = length ** (-2)
    j = jnp.arange(0, m)
    c = jnp.where(j > 0, 2, 1)
    return (c * alpha**2 / jnp.exp(a)) * modified_bessel_first_kind(j, a)
