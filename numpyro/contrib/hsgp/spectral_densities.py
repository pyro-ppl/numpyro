# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
This module contains spectral densities for various kernel functions.
"""

from __future__ import annotations

from jax import Array, vmap
import jax.numpy as jnp
from jax.scipy import special
from jax.typing import ArrayLike

from numpyro.contrib.hsgp.laplacian import sqrt_eigenvalues


def align_param(dim, param):
    return jnp.broadcast_to(param, jnp.broadcast_shapes(jnp.shape(param), (dim,)))


def spectral_density_squared_exponential(
    dim: int, w: ArrayLike, alpha: float, length: float | ArrayLike
) -> Array:
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
    :param ArrayLike w: frequency
    :param float alpha: amplitude
    :param float length: length scale
    :return: spectral density value
    :rtype: Array
    """
    length = align_param(dim, length)
    c = alpha * jnp.prod(jnp.sqrt(2 * jnp.pi) * length, axis=-1)
    e = jnp.exp(-0.5 * jnp.sum(w**2 * length**2, axis=-1))
    return c * e


def spectral_density_matern(
    dim: int, nu: float, w: ArrayLike, alpha: float, length: float | ArrayLike
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
    :param ArrayLike w: frequency
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
) -> Array:
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
    :rtype: Array
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
) -> Array:
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
    :rtype: Array
    """

    def _spectral_density(w):
        return spectral_density_matern(dim=dim, nu=nu, w=w, alpha=alpha, length=length)

    sqrt_eigenvalues_ = sqrt_eigenvalues(ell=ell, m=m, dim=dim)
    return vmap(_spectral_density, in_axes=-1)(sqrt_eigenvalues_)


def modified_bessel_second_kind(v, z):
    """
    Modified Bessel function of the second kind K_v(z).

    Uses the exponentially scaled version from TensorFlow Probability for numerical stability.

    :param v: order of the Bessel function
    :param z: argument
    :return: K_v(z)
    """
    try:
        from tensorflow_probability.substrates import jax as tfp
    except ImportError as e:
        raise ImportError(
            "TensorFlow Probability is required for the Rational Quadratic kernel spectral density."
        ) from e

    v = jnp.asarray(v, dtype=float)
    z = jnp.asarray(z, dtype=float)
    # bessel_kve returns K_v(z) * exp(z), so we multiply by exp(-z) to get K_v(z)
    return jnp.exp(-z) * tfp.math.bessel_kve(v, z)


def spectral_density_rational_quadratic(
    dim: int,
    w: ArrayLike,
    alpha: float,
    length: float | ArrayLike,
    scale_mixture: float,
) -> Array:
    """
    Spectral density of the Rational Quadratic kernel.

    The Rational Quadratic kernel can be seen as a scale mixture (an infinite sum)
    of squared exponential kernels with different length scales. As the scale mixture
    parameter approaches infinity, the kernel converges to the squared exponential kernel.

    The spectral density involves the modified Bessel function of the second kind.
    For the kernel :math:`k(r) = (1 + r^2/(2 \\alpha_{\\text{mix}} \\ell^2))^{-\\alpha_{\\text{mix}}}`,
    the spectral density in :math:`D` dimensions is:

    .. math::

        S(\\boldsymbol{\\omega}) = \\sigma^2 (2\\pi)^{D/2} \\cdot 2^{1-\\alpha_{\\text{mix}}} \\cdot a \\cdot
            \\frac{(a|\\boldsymbol{\\omega}|)^{\\alpha_{\\text{mix}}-D/2}
            K_{\\alpha_{\\text{mix}}-D/2}(a|\\boldsymbol{\\omega}|)}{\\Gamma(\\alpha_{\\text{mix}})}

    where :math:`a = \\sqrt{2 \\alpha_{\\text{mix}}} \\cdot \\ell`, :math:`\\sigma^2` is the amplitude,
    and :math:`K_\\nu` is the modified Bessel function of the second kind.

    For :math:`\\boldsymbol{\\omega} \\to 0`, we use the asymptotic expansion
    :math:`z^\\nu K_\\nu(z) \\to \\Gamma(\\nu) 2^{\\nu-1}` as :math:`z \\to 0`, giving:

    .. math::

        S(0) = \\sigma^2 \\pi^{D/2} a^D \\frac{\\Gamma(\\alpha_{\\text{mix}} - D/2)}{\\Gamma(\\alpha_{\\text{mix}})}

    .. note::

        This implementation currently only supports 1-dimensional inputs (dim=1) with isotropic
        length scales, matching the limitation of sklearn's RationalQuadratic kernel.
        The HSGP approximation for the RQ kernel may require larger ``ell`` values compared
        to the Squared Exponential kernel due to the heavier tails of the RQ kernel.

    **References:**

        1. Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning.

        2. Riutort-Mayol, G., Bürkner, PC., Andersen, M.R. et al. Practical Hilbert space
           approximate Bayesian Gaussian processes for probabilistic programming. Stat Comput 33, 17 (2023).

    :param int dim: dimension (currently only dim=1 is fully supported)
    :param ArrayLike w: frequency
    :param float alpha: amplitude (σ² in the spectral density formula)
    :param float length: length scale (scalar)
    :param float scale_mixture: scale mixture parameter (α_mix in the RQ kernel formula).
        Controls the relative weighting of small-scale and large-scale variations.
        As scale_mixture → ∞, the kernel converges to the squared exponential kernel.
    :return: spectral density value
    :rtype: Array
    """
    # For now, only support isotropic (scalar) length scale
    length = jnp.atleast_1d(length)
    if length.shape[-1] > 1 and dim > 1:
        raise NotImplementedError(
            "Rational Quadratic spectral density currently only supports "
            "isotropic (scalar) length scales."
        )
    length_scalar = jnp.mean(length)  # Use scalar length

    # Compute scaling parameter: a = sqrt(2 * scale_mixture) * length
    a = jnp.sqrt(2 * scale_mixture) * length_scalar

    # Compute |ω| (magnitude of frequency vector)
    w = jnp.atleast_1d(w)
    abs_w = jnp.sqrt(jnp.sum(w**2, axis=-1))
    scaled_w = a * abs_w

    # Order of Bessel function: ν = α_mix - D/2
    nu = scale_mixture - dim / 2

    # For small ω, use asymptotic expansion: z^ν K_ν(z) → Γ(ν) 2^(ν-1) as z → 0
    # This gives: S(0) = α * π^(D/2) * a^D * Γ(α_mix - D/2) / Γ(α_mix)
    log_S_0 = (
        jnp.log(alpha)
        + (dim / 2) * jnp.log(jnp.pi)
        + dim * jnp.log(a)
        + special.gammaln(scale_mixture - dim / 2)
        - special.gammaln(scale_mixture)
    )
    S_0 = jnp.exp(log_S_0)

    # For regular case (ω ≠ 0):
    # S(ω) = α * (2π)^(D/2) * 2^(1-α_mix) * a * (a|ω|)^(α_mix-D/2) * K_{α_mix-D/2}(a|ω|) / Γ(α_mix)
    # Using log for numerical stability
    log_S_regular = (
        jnp.log(alpha)
        + (dim / 2) * jnp.log(2 * jnp.pi)
        + (1 - scale_mixture) * jnp.log(2)
        + jnp.log(a)
        + nu * jnp.log(scaled_w)
        + jnp.log(modified_bessel_second_kind(nu, scaled_w))
        - special.gammaln(scale_mixture)
    )
    S_regular = jnp.exp(log_S_regular)

    # Use S_0 for small ω, S_regular otherwise
    return jnp.where(abs_w < 1e-8, S_0, S_regular)


def diag_spectral_density_rational_quadratic(
    alpha: float,
    length: float | list[float],
    scale_mixture: float,
    ell: float | int | list[float | int],
    m: int | list[int],
    dim: int,
) -> Array:
    """
    Evaluates the spectral density of the Rational Quadratic kernel at the first :math:`D \\times m^\\star`
    square root eigenvalues of the laplacian operator in :math:`[-L_1, L_1] \\times ... \\times [-L_D, L_D]`.

    :param float alpha: amplitude of the Rational Quadratic kernel
    :param float length: length scale of the Rational Quadratic kernel
    :param float scale_mixture: scale mixture parameter (α in the RQ kernel formula).
        Controls the relative weighting of small-scale and large-scale variations.
        As scale_mixture → ∞, the kernel converges to the squared exponential kernel.
    :param float | int | list[float | int] ell: The length of the interval divided by 2 in each dimension.
        If a float or int, the same length is used in each dimension.
    :param int | list[int] m: The number of eigenvalues to compute for each dimension.
        If an integer, the same number of eigenvalues is computed in each dimension.
    :param int dim: The dimension of the space

    :return: spectral density vector evaluated at the first :math:`D \\times m^\\star` square root eigenvalues
    :rtype: Array
    """

    def _spectral_density(w):
        return spectral_density_rational_quadratic(
            dim=dim, w=w, alpha=alpha, length=length, scale_mixture=scale_mixture
        )

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


def diag_spectral_density_periodic(alpha: float, length: float, m: int) -> Array:
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
    :rtype: Array
    """
    a = length ** (-2)
    j = jnp.arange(0, m)
    c = jnp.where(j > 0, 2, 1)
    return (c * alpha**2 / jnp.exp(a)) * modified_bessel_first_kind(j, a)
