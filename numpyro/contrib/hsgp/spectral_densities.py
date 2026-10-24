# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
This module contains spectral densities for various kernel functions.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType

from jax import Array, vmap
import jax.numpy as jnp
from jax.scipy import special
from jax.typing import ArrayLike

from numpyro.contrib.hsgp.laplacian import sqrt_eigenvalues


def align_param(dim, param):
    return jnp.broadcast_to(param, jnp.broadcast_shapes(jnp.shape(param), (dim,)))


def _exp_floored(log_density: Array) -> Array:
    # Floor at the smallest positive normal so that ``jnp.sqrt`` of the density
    # keeps a finite gradient where the density underflows.
    density = jnp.exp(log_density)
    return jnp.maximum(density, jnp.finfo(density.dtype).tiny)


def _log_spectral_density_squared_exponential(
    dim: int, w: ArrayLike, alpha: ArrayLike, length: ArrayLike
) -> Array:
    length = align_param(dim, length)
    log_c = jnp.log(alpha) + jnp.sum(
        0.5 * jnp.log(2 * jnp.pi) + jnp.log(length), axis=-1
    )
    return log_c - 0.5 * jnp.sum(w**2 * length**2, axis=-1)


def spectral_density_squared_exponential(
    dim: int, w: ArrayLike, alpha: ArrayLike, length: ArrayLike
) -> Array:
    """
    Spectral density of the squared exponential kernel.

    See Section 4.2 in [`1 <spectral-density-squared-exponential-ref-1_>`__] and Section 2.1 in [`2
    <spectral-density-squared-exponential-ref-2_>`__].

    .. math::

        S(\\boldsymbol{\\omega}) = \\alpha (\\sqrt{2\\pi})^D \\ell^D
            \\exp\\left(-\\frac{1}{2} \\ell^2 \\boldsymbol{\\omega}^{T} \\boldsymbol{\\omega}\\right)


    **References:**

        1. .. _spectral-density-squared-exponential-ref-1:

           Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning.

        2. .. _spectral-density-squared-exponential-ref-2:

           Riutort-Mayol, G., Bürkner, PC., Andersen, M.R. et al. Practical Hilbert space
           approximate Bayesian Gaussian processes for probabilistic programming. Stat Comput 33, 17 (2023).

    :param int dim: dimension
    :param ArrayLike w: frequency
    :param ArrayLike alpha: amplitude
    :param ArrayLike length: length scale
    :return: spectral density value
    :rtype: Array
    """
    return _exp_floored(
        _log_spectral_density_squared_exponential(
            dim=dim, w=w, alpha=alpha, length=length
        )
    )


def _log_spectral_density_matern(
    dim: int, nu: float, w: ArrayLike, alpha: ArrayLike, length: ArrayLike
) -> Array:
    length = align_param(dim, length)
    log_c = (
        jnp.log(alpha)
        + dim * jnp.log(2.0)
        + (dim / 2) * jnp.log(jnp.pi)
        + nu * jnp.log(2 * nu)
        + special.gammaln(nu + dim / 2)
        - special.gammaln(nu)
        + jnp.sum(jnp.log(length), axis=-1)
    )
    s = jnp.sum(length**2 * w**2, axis=-1)
    return log_c - (nu + dim / 2) * jnp.log(2 * nu + s)


def spectral_density_matern(
    dim: int, nu: float, w: ArrayLike, alpha: ArrayLike, length: ArrayLike
) -> Array:
    """
    Spectral density of the Matérn kernel.

    See Eq. (4.15) in [`1 <spectral-density-matern-ref-1_>`__] and Section 2.1 in [`2 <spectral-density-matern-ref-2_>`__].

    .. math::

        S(\\boldsymbol{\\omega}) = \\alpha
            \\frac{2^{D} \\pi^{D/2} \\Gamma(\\nu + D/2) (2 \\nu)^{\\nu}}{\\Gamma(\\nu) \\ell^{2 \\nu}}
            \\left(\\frac{2 \\nu}{\\ell^2} + \\boldsymbol{\\omega}^{T} \\boldsymbol{\\omega}\\right)^{-\\nu - D/2}


    **References:**

        1. .. _spectral-density-matern-ref-1:

           Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning.

        2. .. _spectral-density-matern-ref-2:

           Riutort-Mayol, G., Bürkner, PC., Andersen, M.R. et al. Practical Hilbert space
           approximate Bayesian Gaussian processes for probabilistic programming. Stat Comput 33, 17 (2023).

    :param int dim: dimension
    :param float nu: smoothness
    :param ArrayLike w: frequency
    :param ArrayLike alpha: amplitude
    :param ArrayLike length: length scale
    :return: spectral density value
    :rtype: Array
    """  # noqa: E501
    return _exp_floored(
        _log_spectral_density_matern(dim=dim, nu=nu, w=w, alpha=alpha, length=length)
    )


def _eval_at_sqrt_eigenvalues(
    fn: Callable[[Array], Array],
    ell: float | int | list[float | int],
    m: int | list[int],
    dim: int,
) -> Array:
    """Evaluate ``fn`` at each of the first :math:`D \\times m^\\star` square root eigenvalues."""
    sqrt_eigenvalues_ = sqrt_eigenvalues(ell=ell, m=m, dim=dim)  # dim x prod(m)
    return vmap(fn, in_axes=-1)(sqrt_eigenvalues_)


def diag_spectral_density_squared_exponential(
    alpha: ArrayLike,
    length: ArrayLike,
    ell: float | int | list[float | int],
    m: int | list[int],
    dim: int,
) -> Array:
    """
    Evaluates the spectral density of the squared exponential kernel at the first :math:`D \\times m^\\star`
    square root eigenvalues of the laplacian operator in :math:`[-L_1, L_1] \\times ... \\times [-L_D, L_D]`.

    :param ArrayLike alpha: amplitude of the squared exponential kernel
    :param ArrayLike length: length scale of the squared exponential kernel
    :param float | int | list[float | int] ell: The length of the interval divided by 2 in each dimension.
        If a float or int, the same length is used in each dimension.
    :param int | list[int] m: The number of eigenvalues to compute for each dimension.
        If an integer, the same number of eigenvalues is computed in each dimension.
    :param int dim: The dimension of the space

    :return: spectral density vector evaluated at the first :math:`D \\times m^\\star` square root eigenvalues
    :rtype: Array
    """

    def _spectral_density(w: Array) -> Array:
        return spectral_density_squared_exponential(
            dim=dim, w=w, alpha=alpha, length=length
        )

    return _eval_at_sqrt_eigenvalues(_spectral_density, ell=ell, m=m, dim=dim)


def diag_spectral_density_matern(
    nu: float,
    alpha: ArrayLike,
    length: ArrayLike,
    ell: float | int | list[float | int],
    m: int | list[int],
    dim: int,
) -> Array:
    """
    Evaluates the spectral density of the Matérn kernel at the first :math:`D \\times m^\\star`
    square root eigenvalues of the laplacian operator in :math:`[-L_1, L_1] \\times ... \\times [-L_D, L_D]`.

    :param float nu: smoothness parameter
    :param ArrayLike alpha: amplitude of the Matérn kernel
    :param ArrayLike length: length scale of the Matérn kernel
    :param float | int | list[float | int] ell: The length of the interval divided by 2 in each dimension.
        If a float or int, the same length is used in each dimension.
    :param int | list[int] m: The number of eigenvalues to compute for each dimension.
        If an integer, the same number of eigenvalues is computed in each dimension.
    :param int dim: The dimension of the space

    :return: spectral density vector evaluated at the first :math:`D \\times m^\\star` square root eigenvalues
    :rtype: Array
    """

    def _spectral_density(w: Array) -> Array:
        return spectral_density_matern(dim=dim, nu=nu, w=w, alpha=alpha, length=length)

    return _eval_at_sqrt_eigenvalues(_spectral_density, ell=ell, m=m, dim=dim)


def _import_tfp() -> ModuleType:
    try:
        from tensorflow_probability.substrates import jax as tfp
    except ImportError as e:
        raise ImportError(
            "TensorFlow Probability is required for the rational quadratic and "
            "periodic spectral densities."
        ) from e
    return tfp


def _log_spectral_density_rational_quadratic(
    dim: int,
    w: ArrayLike,
    alpha: ArrayLike,
    length: ArrayLike,
    scale_mixture: ArrayLike,
) -> Array:
    tfp = _import_tfp()

    # Only support isotropic (scalar) length scale
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
    small = abs_w < 1e-8

    # Order of Bessel function: ν = α_mix - D/2. TFP requires the order and the
    # argument to share a dtype.
    nu = scale_mixture - dim / 2
    dtype = jnp.result_type(float, scaled_w, nu)
    nu = jnp.asarray(nu, dtype=dtype)

    # Dimension-dependent correction factor: (2 * α_mix)^((D-1)/2)
    log_dim_factor = ((dim - 1) / 2) * jnp.log(2 * scale_mixture)

    # For small ω, use asymptotic expansion: z^ν K_ν(z) → Γ(ν) 2^(ν-1) as z → 0
    # S(0) = α * π^(D/2) * a^D * (2*α_mix)^((D-1)/2) * Γ(α_mix - D/2) / Γ(α_mix)
    # Note: This requires α_mix > D/2 for Γ(α_mix - D/2) to be positive and finite
    log_S_0 = (
        jnp.log(alpha)
        + (dim / 2) * jnp.log(jnp.pi)
        + dim * jnp.log(a)
        + log_dim_factor
        + special.gammaln(scale_mixture - dim / 2)
        - special.gammaln(scale_mixture)
    )

    # For regular case (ω ≠ 0):
    # S(ω) = α * (2π)^(D/2) * 2^(1-α_mix) * a * (2*α_mix)^((D-1)/2)
    #        * (a|ω|)^(α_mix-D/2) * K_{α_mix-D/2}(a|ω|) / Γ(α_mix)
    # The branch is evaluated at a safe argument where ``small`` holds so that
    # log(0) cannot leak NaNs into the gradient through ``jnp.where``, and
    # log K_ν(z) = log(K_ν(z) exp(z)) - z is taken from the exponentially scaled
    # Bessel function so it stays finite where K_ν(z) underflows.
    safe_z = jnp.where(small, 1.0, scaled_w).astype(dtype)
    log_bessel = tfp.math.log_bessel_kve(nu, safe_z) - safe_z
    log_S_regular = (
        jnp.log(alpha)
        + (dim / 2) * jnp.log(2 * jnp.pi)
        + (1 - scale_mixture) * jnp.log(2)
        + jnp.log(a)
        + log_dim_factor
        + nu * jnp.log(safe_z)
        + log_bessel
        - special.gammaln(scale_mixture)
    )

    # Use S_0 for small ω, S_regular otherwise
    return jnp.where(small, log_S_0, log_S_regular)


def spectral_density_rational_quadratic(
    dim: int,
    w: ArrayLike,
    alpha: ArrayLike,
    length: ArrayLike,
    scale_mixture: ArrayLike,
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

        S(\\boldsymbol{\\omega}) = \\sigma^2 (2\\pi)^{D/2} \\cdot 2^{1-\\alpha_{\\text{mix}}} \\cdot a
            \\cdot (2\\alpha_{\\text{mix}})^{(D-1)/2} \\cdot
            \\frac{(a|\\boldsymbol{\\omega}|)^{\\alpha_{\\text{mix}}-D/2}
            K_{\\alpha_{\\text{mix}}-D/2}(a|\\boldsymbol{\\omega}|)}{\\Gamma(\\alpha_{\\text{mix}})}

    where :math:`a = \\sqrt{2 \\alpha_{\\text{mix}}} \\cdot \\ell`, :math:`\\sigma^2` is the amplitude,
    and :math:`K_\\nu` is the modified Bessel function of the second kind.

    For :math:`\\boldsymbol{\\omega} \\to 0`, we use the asymptotic expansion
    :math:`z^\\nu K_\\nu(z) \\to \\Gamma(\\nu) 2^{\\nu-1}` as :math:`z \\to 0`, giving:

    .. math::

        S(0) = \\sigma^2 \\pi^{D/2} a^D (2\\alpha_{\\text{mix}})^{(D-1)/2}
            \\frac{\\Gamma(\\alpha_{\\text{mix}} - D/2)}{\\Gamma(\\alpha_{\\text{mix}})}

    .. note::

        The S(0) formula requires :math:`\\alpha_{\\text{mix}} > D/2` to be well-defined.
        For :math:`\\alpha_{\\text{mix}} \\leq D/2`, the spectral density diverges at :math:`\\omega = 0`.
        The HSGP approximation for the RQ kernel may require larger ``ell`` values compared
        to the Squared Exponential kernel due to the heavier tails of the RQ kernel.
        Gradients with respect to ``scale_mixture`` are not supported: the modified
        Bessel function from TensorFlow Probability has no derivative with respect to
        its order, so that contribution is silently dropped.

    **References:**

        1. Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning.

        2. Riutort-Mayol, G., Bürkner, PC., Andersen, M.R. et al. Practical Hilbert space
           approximate Bayesian Gaussian processes for probabilistic programming. Stat Comput 33, 17 (2023).

    :param int dim: dimension
    :param ArrayLike w: frequency
    :param ArrayLike alpha: amplitude (σ² in the spectral density formula)
    :param ArrayLike length: length scale (scalar, isotropic)
    :param ArrayLike scale_mixture: scale mixture parameter (α_mix in the RQ kernel formula).
        Controls the relative weighting of small-scale and large-scale variations.
        As scale_mixture → ∞, the kernel converges to the squared exponential kernel.
    :return: spectral density value
    :rtype: Array
    """
    return _exp_floored(
        _log_spectral_density_rational_quadratic(
            dim=dim, w=w, alpha=alpha, length=length, scale_mixture=scale_mixture
        )
    )


def diag_spectral_density_rational_quadratic(
    alpha: ArrayLike,
    length: ArrayLike,
    scale_mixture: ArrayLike,
    ell: float | int | list[float | int],
    m: int | list[int],
    dim: int,
) -> Array:
    """
    Evaluates the spectral density of the Rational Quadratic kernel at the first :math:`D \\times m^\\star`
    square root eigenvalues of the laplacian operator in :math:`[-L_1, L_1] \\times ... \\times [-L_D, L_D]`.

    :param ArrayLike alpha: amplitude of the Rational Quadratic kernel
    :param ArrayLike length: length scale of the Rational Quadratic kernel
    :param ArrayLike scale_mixture: scale mixture parameter (α in the RQ kernel formula).
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

    def _spectral_density(w: Array) -> Array:
        return spectral_density_rational_quadratic(
            dim=dim, w=w, alpha=alpha, length=length, scale_mixture=scale_mixture
        )

    return _eval_at_sqrt_eigenvalues(_spectral_density, ell=ell, m=m, dim=dim)


def _log_diag_spectral_density_periodic(
    alpha: ArrayLike, length: ArrayLike, m: int
) -> Array:
    """
    Log of the periodic kernel variance coefficients.

    Computes the coefficients in log space to avoid underflow in the modified
    Bessel function for large basis sizes.
    """
    try:
        from tensorflow_probability.substrates import jax as tfp
    except ImportError as e:
        raise ImportError(
            "TensorFlow Probability is required for the periodic kernel spectral density."
        ) from e

    a = length ** (-2)
    j = jnp.arange(0, m, dtype=jnp.result_type(a, float))
    c = jnp.where(j > 0, 2, 1)
    alpha_nonzero = alpha != 0
    safe_alpha2 = jnp.where(alpha_nonzero, alpha**2, 1)
    log_alpha2 = jnp.where(alpha_nonzero, jnp.log(safe_alpha2), -jnp.inf)
    return jnp.log(c) + log_alpha2 + tfp.math.log_bessel_ive(j, a)


def diag_spectral_density_periodic(
    alpha: ArrayLike, length: ArrayLike, m: int
) -> Array:
    """
    Not actually a spectral density but these are used in the same
    way. These are simply the first `m` coefficients of the low rank
    approximation for the periodic kernel. See Appendix B in [`1 <diag-spectral-density-periodic-ref-1_>`__].

    **References:**

        1. .. _diag-spectral-density-periodic-ref-1:

           Riutort-Mayol, G., Bürkner, PC., Andersen, M.R. et al. Practical Hilbert space
           approximate Bayesian Gaussian processes for probabilistic programming. Stat Comput 33, 17 (2023).

    :param ArrayLike alpha: amplitude
    :param ArrayLike length: length scale
    :param int m: number of eigenvalues
    :return: "spectral density" vector
    :rtype: Array
    """
    tfp = _import_tfp()
    # q_j^2 = c_j alpha^2 I_j(a) / exp(a) with a = length^-2; the exponentially
    # scaled I_j(a) exp(-a) stays finite where exp(a) overflows (length <~ 0.1 in
    # single precision). TFP requires the order and the argument to share a dtype.
    a = jnp.asarray(length) ** -2.0
    dtype = jnp.result_type(float, a)
    a = a.astype(dtype)
    j = jnp.arange(0, m, dtype=dtype)
    c = jnp.where(j > 0, 2, 1)
    q2 = c * alpha**2 * tfp.math.bessel_ive(j, a)
    # Strictly positive, like the other densities (see ``_exp_floored``).
    return jnp.maximum(q2, jnp.finfo(q2.dtype).tiny)
