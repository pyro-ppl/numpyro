# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
This module contains spectral densities for various kernel functions.
"""

from jaxlib.xla_extension import ArrayImpl

from jax import vmap
import jax.numpy as jnp
from jax.scipy import special

from numpyro.contrib.hsgp.laplacian import sqrt_eigenvalues


def spectral_density_squared_exponential(
    dim: int, w: ArrayImpl, alpha: float, length: float
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
    c = alpha * (jnp.sqrt(2 * jnp.pi) * length) ** dim
    e = jnp.exp(-0.5 * (length**2) * jnp.dot(w, w))
    return c * e


def spectral_density_matern(
    dim: int, nu: float, w: ArrayImpl, alpha: float, length: float
) -> float:
    """
    Spectral density of the Matérn kernel.

    See Eq. (4.15) in [1] and Section 2.1 in [2].

    .. math::

        S(\\boldsymbol{\\omega}) = \\alpha
            \\frac{2^{D} \\pi^{D/2} \\Gamma(\\nu + D/2) (2 \\nu)^{\\nu}}{\\Gamma(\\nu) \\ell^{2 \\nu}}
            \\left(\\frac{2 \\nu}{\\ell^2} + 4 \\pi^2 \\boldsymbol{\\omega}^{T} \\boldsymbol{\\omega}\\right)^{-\\nu - D/2}


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
    c1 = (
        alpha
        * (2 ** (dim))
        * (jnp.pi ** (dim / 2))
        * ((2 * nu) ** nu)
        * special.gamma(nu + dim / 2)
    )
    c2 = ((2 * nu / (length**2)) + 4 * jnp.pi ** jnp.dot(w, w)) ** (-nu - dim / 2)
    c3 = special.gamma(nu) * length ** (2 * nu)
    return c1 * c2 / c3


# TODO: Adapt to dim >= 1.
def diag_spectral_density_squared_exponential(
    alpha: float, length: float, ell: float, m: int
) -> ArrayImpl:
    """
    Evaluates the spectral density of the squared exponential kernel at the first `m`
    square root eigenvalues of the laplacian operator in `[-ell, ell]`.

    :param float alpha: amplitude of the squared exponential kernel
    :param float length: length scale of the squared exponential kernel
    :param float ell: The length of the interval divided by 2
    :param int m: The number of eigenvalues to compute
    :return: spectral density vector evaluated at the first `m` square root eigenvalues
    :rtype: ArrayImpl
    """

    def _spectral_density(w):
        return spectral_density_squared_exponential(
            dim=1, w=w, alpha=alpha, length=length
        )

    sqrt_eigenvalues_ = sqrt_eigenvalues(ell=ell, m=m)
    return vmap(_spectral_density)(sqrt_eigenvalues_)


# TODO: Adapt to dim >= 1.
def diag_spectral_density_matern(
    nu: float, alpha: float, length: float, ell: float, m: int
) -> ArrayImpl:
    """
    Evaluates the spectral density of the Matérn kernel at the first `m`
    square root eigenvalues of the laplacian operator in `[-ell, ell]`.

    :param float nu: smoothness parameter
    :param float alpha: amplitude of the Matérn kernel
    :param float length: length scale of the Matérn kernel
    :param float ell: The length of the interval divided by 2
    :param int m: The number of eigenvalues to compute
    :return: spectral density vector evaluated at the first `m` square root eigenvalues
    :rtype: ArrayImpl
    """

    def _spectral_density(w):
        return spectral_density_matern(dim=1, nu=nu, w=w, alpha=alpha, length=length)

    sqrt_eigenvalues_ = sqrt_eigenvalues(ell=ell, m=m)
    return vmap(_spectral_density)(sqrt_eigenvalues_)


def modified_bessel_first_kind(v, z):
    try:
        from tensorflow_probability.substrates import jax as tfp
    except ImportError as e:
        raise ImportError(
            "TensorFlow Probability is required for this function."
        ) from e

    v = jnp.asarray(v, dtype=float)
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
