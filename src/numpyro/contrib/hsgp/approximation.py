# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
This module contains the low-rank approximation functions of the Hilbert space Gaussian process.
"""

from __future__ import annotations

from jaxlib.xla_extension import ArrayImpl

import jax.numpy as jnp

import numpyro
from numpyro.contrib.hsgp.laplacian import eigenfunctions, eigenfunctions_periodic
from numpyro.contrib.hsgp.spectral_densities import (
    diag_spectral_density_matern,
    diag_spectral_density_periodic,
    diag_spectral_density_squared_exponential,
)
import numpyro.distributions as dist


def _non_centered_approximation(phi: ArrayImpl, spd: ArrayImpl, m: int) -> ArrayImpl:
    with numpyro.plate("basis", m):
        beta = numpyro.sample("beta", dist.Normal(loc=0.0, scale=1.0))

    return phi @ (spd * beta)


def _centered_approximation(phi: ArrayImpl, spd: ArrayImpl, m: int) -> ArrayImpl:
    with numpyro.plate("basis", m):
        beta = numpyro.sample("beta", dist.Normal(loc=0.0, scale=spd))

    return phi @ beta


def linear_approximation(
    phi: ArrayImpl, spd: ArrayImpl, m: int | list[int], non_centered: bool = True
) -> ArrayImpl:
    """
    Linear approximation formula of the Hilbert space Gaussian process.

    See Eq. (8) in [1].

    **References:**

        1. Riutort-Mayol, G., Bürkner, PC., Andersen, M.R. et al. Practical Hilbert space
           approximate Bayesian Gaussian processes for probabilistic programming. Stat Comput 33, 17 (2023).

    :param ArrayImpl phi: laplacian eigenfunctions
    :param ArrayImpl spd: square root of the diagonal of the spectral density evaluated at square
        root of the first `m` eigenvalues.
    :param int | list[int] m: number of eigenfunctions in the approximation
    :param bool non_centered: whether to use a non-centered parameterization
    :return: The low-rank approximation linear model
    :rtype: ArrayImpl
    """
    if non_centered:
        return _non_centered_approximation(phi, spd, m)
    return _centered_approximation(phi, spd, m)


def hsgp_squared_exponential(
    x: ArrayImpl,
    alpha: float,
    length: float,
    ell: float | int | list[float | int],
    m: int | list[int],
    non_centered: bool = True,
) -> ArrayImpl:
    """
    Hilbert space Gaussian process approximation using the squared exponential kernel.

    The main idea of the approach is to combine the associated spectral density of the
    squared exponential kernel and the spectrum of the Dirichlet Laplacian operator to
    obtain a low-rank approximation of the Gram matrix. For more details see [1, 2].

    **References:**

        1. Solin, A., Särkkä, S. Hilbert space methods for reduced-rank Gaussian process regression.
           Stat Comput 30, 419-446 (2020).

        2. Riutort-Mayol, G., Bürkner, PC., Andersen, M.R. et al. Practical Hilbert space
           approximate Bayesian Gaussian processes for probabilistic programming. Stat Comput 33, 17 (2023).

    :param ArrayImpl x: input data
    :param float alpha: amplitude of the squared exponential kernel
    :param float length: length scale of the squared exponential kernel
    :param float | int | list[float | int] ell: positive value that parametrizes the length of the D-dimensional box so
        that the input data lies in the interval :math:`[-L_1, L_1] \\times ... \\times [-L_D, L_E]`.
        We expect the approximation to be valid within this interval
    :param int | list[m] m: number of eigenvalues to compute and include in the approximation for each dimension
        (:math:`\\left\\{1, ..., D\\right\\}`).
        If an integer, the same number of eigenvalues is computed in each dimension.
    :param bool non_centered: whether to use a non-centered parameterization. By default, it is set to True
    :return: the low-rank approximation linear model
    :rtype: ArrayImpl
    """
    dim = x.shape[-1] if x.ndim > 1 else 1
    phi = eigenfunctions(x=x, ell=ell, m=m)
    spd = jnp.sqrt(
        diag_spectral_density_squared_exponential(
            alpha=alpha, length=length, ell=ell, m=m, dim=dim
        )
    )
    return linear_approximation(
        phi=phi, spd=spd, m=phi.shape[-1], non_centered=non_centered
    )


def hsgp_matern(
    x: ArrayImpl,
    nu: float,
    alpha: float,
    length: float,
    ell: float | int | list[float | int],
    m: int | list[int],
    non_centered: bool = True,
):
    """
    Hilbert space Gaussian process approximation using the Matérn kernel.

    The main idea of the approach is to combine the associated spectral density of the
    Matérn kernel kernel and the spectrum of the Dirichlet Laplacian operator to obtain
    a low-rank approximation of the Gram matrix. For more details see [1, 2].

    **References:**

        1. Solin, A., Särkkä, S. Hilbert space methods for reduced-rank Gaussian process regression.
           Stat Comput 30, 419-446 (2020).

        2. Riutort-Mayol, G., Bürkner, PC., Andersen, M.R. et al. Practical Hilbert space
           approximate Bayesian Gaussian processes for probabilistic programming. Stat Comput 33, 17 (2023).

    :param ArrayImpl x: input data
    :param float nu: smoothness parameter
    :param float alpha: amplitude of the squared exponential kernel
    :param float length: length scale of the squared exponential kernel
    :param float | int | list[float | int] ell: positive value that parametrizes the length of the D-dimensional box so
        that the input data lies in the interval :math:`[-L_1, L_1] \\times ... \\times [-L_D, L_D]`.
        We expect the approximation to be valid within this interval
    :param int | list[m] m: number of eigenvalues to compute and include in the approximation for each dimension
        (:math:`\\left\\{1, ..., D\\right\\}`).
        If an integer, the same number of eigenvalues is computed in each dimension.
    :param bool non_centered: whether to use a non-centered parameterization. By default, it is set to True.
    :return: the low-rank approximation linear model
    :rtype: ArrayImpl
    """
    dim = x.shape[-1] if x.ndim > 1 else 1
    phi = eigenfunctions(x=x, ell=ell, m=m)
    spd = jnp.sqrt(
        diag_spectral_density_matern(
            nu=nu, alpha=alpha, length=length, ell=ell, m=m, dim=dim
        )
    )
    return linear_approximation(
        phi=phi, spd=spd, m=phi.shape[-1], non_centered=non_centered
    )


def hsgp_periodic_non_centered(
    x: ArrayImpl, alpha: float, length: float, w0: float, m: int
) -> ArrayImpl:
    """
    Low rank approximation for the periodic squared exponential kernel in the non-centered parametrization.

    See Appendix B in [1].

    **References:**

        1. Riutort-Mayol, G., Bürkner, PC., Andersen, M.R. et al. Practical Hilbert space
           approximate Bayesian Gaussian processes for probabilistic programming. Stat Comput 33, 17 (2023).

    :param ArrayImpl x: input data
    :param float alpha: amplitude
    :param float length: length scale
    :param float w0: frequency of the periodic kernel
    :param int m: number of eigenvalues to compute and include in the approximation
    :return: the low-rank approximation linear model
    :rtype: ArrayImpl
    """
    q2 = diag_spectral_density_periodic(alpha=alpha, length=length, m=m)
    cosines, sines = eigenfunctions_periodic(x=x, w0=w0, m=m)

    with numpyro.plate("cos_basis", m):
        beta_cos = numpyro.sample("beta_cos", dist.Normal(0, 1))

    with numpyro.plate("sin_basis", m - 1):
        beta_sin = numpyro.sample("beta_sin", dist.Normal(0, 1))

    # The first eigenfunction for the sine component
    # is zero, so the first parameter wouldn't contribute to the approximation.
    # We set it to zero to identify the model and avoid divergences.
    zero = jnp.array([0.0])
    beta_sin = jnp.concatenate((zero, beta_sin))

    return cosines @ (q2 * beta_cos) + sines @ (q2 * beta_sin)
