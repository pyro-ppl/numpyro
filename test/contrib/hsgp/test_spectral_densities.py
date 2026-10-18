# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import reduce
from operator import mul

import numpy as np
import pytest
from scipy.special import gammaln, ive, kve

import jax
import jax.numpy as jnp

from numpyro.contrib.hsgp.spectral_densities import (
    diag_spectral_density_matern,
    diag_spectral_density_periodic,
    diag_spectral_density_rational_quadratic,
    diag_spectral_density_squared_exponential,
    log_diag_spectral_density_matern,
    log_diag_spectral_density_rational_quadratic,
    log_diag_spectral_density_squared_exponential,
    spectral_density_matern,
    spectral_density_rational_quadratic,
    spectral_density_squared_exponential,
)


@pytest.mark.parametrize(
    argnames="dim, w, alpha, length",
    argvalues=[
        (1, 0.1, 1.0, 0.2),
        (2, np.array([0.1, 0.2]), 1.0, 0.2),
        (3, np.array([0.1, 0.2, 0.3]), 1.0, 5.0),
    ],
    ids=["dim=1", "dim=2", "dim=3"],
)
def test_spectral_density_squared_exponential(dim, w, alpha, length):
    spectral_density = spectral_density_squared_exponential(
        dim=dim, w=w, alpha=alpha, length=length
    )
    assert spectral_density.shape == ()
    assert spectral_density > 0.0


@pytest.mark.parametrize(
    argnames="dim, nu, w, alpha, length",
    argvalues=[
        (1, 3 / 2, 0.1, 1.0, 0.2),
        (2, 5 / 2, np.array([0.1, 0.2]), 1.0, 0.2),
        (3, 5 / 2, np.array([0.1, 0.2, 0.3]), 1.0, 5.0),
    ],
    ids=["dim=1", "dim=2", "dim=3"],
)
def test_spectral_density_matern(dim, nu, w, alpha, length):
    spectral_density = spectral_density_matern(
        dim=dim, nu=nu, w=w, alpha=alpha, length=length
    )
    assert spectral_density.shape == ()
    assert spectral_density > 0.0


@pytest.mark.parametrize(
    argnames="alpha, length, ell, m, dim",
    argvalues=[
        (1.0, 0.2, 0.1, 1, 1),
        (1.0, 0.2, 0.2, 2, 1),
        (1.0, 0.2, 0.3, 10, 1),
        (1.0, 0.2, 0.1, 100, 1),
        (1.0, 0.2, 0.1, 10, 2),
        (1.0, 0.2, 0.1, [2, 2, 3], 3),
    ],
    ids=["m=1,d=1", "m=2,d=1", "m=10,d=1", "m=100,d=1", "m=10,d=2", "m=[2,2,3],d=3"],
)
def test_diag_spectral_density_squared_exponential(alpha, length, ell, m, dim):
    diag_spectral_density = diag_spectral_density_squared_exponential(
        alpha=alpha, length=length, ell=ell, m=m, dim=dim
    )
    if isinstance(m, int):
        m = [m] * dim
    assert diag_spectral_density.shape == (reduce(mul, m),)
    assert jnp.all(diag_spectral_density >= 0.0)


@pytest.mark.parametrize(
    argnames="nu, alpha, length, ell, m, dim",
    argvalues=[
        (3 / 2, 1.0, 0.2, 0.1, 1, 1),
        (5 / 2, 1.0, 0.2, 0.2, 2, 1),
        (2, 1.0, 0.2, 0.3, 10, 1),
        (7 / 2, 1.0, 0.2, 0.1, 100, 1),
        (2, 1.0, 0.2, 0.3, 10, 2),
        (2, 1.0, 0.2, 0.3, [2, 2, 3], 3),
    ],
    ids=["m=1,d=1", "m=2,d=1", "m=10,d=1", "m=100,d=1", "m=10,d=2", "m=[2,2,3],d=3"],
)
def test_diag_spectral_density_matern(nu, alpha, length, ell, m, dim):
    diag_spectral_density = diag_spectral_density_matern(
        nu=nu, alpha=alpha, length=length, ell=ell, m=m, dim=dim
    )
    if isinstance(m, int):
        m = [m] * dim
    assert diag_spectral_density.shape == (reduce(mul, m),)
    assert jnp.all(diag_spectral_density >= 0.0)


@pytest.mark.parametrize("length", [0.2, 1.0, 2.0])
def test_log_diag_spectral_density_squared_exponential_no_underflow(length):
    """In float32 ``exp(-w^2 length^2 / 2)`` underflows to 0 for most of the first
    100 eigenvalues; the log density must stay finite and match a float64 reference.
    """
    m, ell, alpha = 100, 1.0, 1.0
    log_spd = log_diag_spectral_density_squared_exponential(
        alpha=alpha, length=length, ell=ell, m=m, dim=1
    )
    w = np.arange(1, m + 1, dtype=np.float64) * np.pi / 2 / ell
    expected = np.log(alpha * np.sqrt(2 * np.pi) * length) - 0.5 * w**2 * length**2
    assert np.all(np.isfinite(log_spd))
    assert np.allclose(log_spd, expected, rtol=1e-5)


def test_sqrt_spectral_density_squared_exponential_gradient():
    """``exp(0.5 * log S)`` must have a finite gradient w.r.t. ``length`` where ``S``
    underflows (``sqrt(S)`` gives ``inf * 0 = nan`` there); compare with the
    analytic float64 derivative of ``sum_j sqrt(S_j)``.
    """
    m, ell, alpha, length = 100, 1.0, 1.0, 1.0

    def sqrt_spd_sum(length):
        return jnp.exp(
            0.5
            * log_diag_spectral_density_squared_exponential(
                alpha=alpha, length=length, ell=ell, m=m, dim=1
            )
        ).sum()

    w = np.arange(1, m + 1, dtype=np.float64) * np.pi / 2 / ell
    sqrt_spd = np.sqrt(alpha * np.sqrt(2 * np.pi) * length) * np.exp(
        -0.25 * w**2 * length**2
    )
    expected = np.sum(sqrt_spd * (0.5 / length - 0.5 * w**2 * length))
    assert np.allclose(jax.grad(sqrt_spd_sum)(length), expected, rtol=1e-4)


@pytest.mark.parametrize("nu", [1.5, 30.5, 60.5])
def test_diag_spectral_density_matern_large_nu(nu):
    """``gamma(nu + dim / 2)`` and ``(2 nu)^nu`` overflow float32 for large ``nu``;
    the log-space evaluation must stay finite and match a float64 reference.
    """
    m, ell, alpha, length = 5, 1.0, 1.0, 1.0
    kwargs = {"nu": nu, "alpha": alpha, "length": length, "ell": ell, "m": m, "dim": 1}
    log_spd = log_diag_spectral_density_matern(**kwargs)
    w = np.arange(1, m + 1, dtype=np.float64) * np.pi / 2 / ell
    expected = (
        np.log(alpha)
        + np.log(2)
        + 0.5 * np.log(np.pi)
        + nu * np.log(2 * nu)
        + gammaln(nu + 0.5)
        - gammaln(nu)
        + np.log(length)
        - (nu + 0.5) * np.log(2 * nu + length**2 * w**2)
    )
    # float32 cancellation among O(100) terms limits absolute accuracy to ~1e-5
    assert np.allclose(log_spd, expected, rtol=1e-5, atol=1e-4)
    spd = diag_spectral_density_matern(**kwargs)
    assert np.all(np.isfinite(spd))
    assert np.allclose(spd, np.exp(expected), rtol=1e-4)


@pytest.mark.parametrize(
    argnames="dim, w, alpha, length, scale_mixture",
    argvalues=[
        (1, 0.1, 1.0, 0.2, 1.0),
        (1, 0.1, 1.0, 0.2, 10.0),
        (2, np.array([0.1, 0.2]), 1.0, 0.2, 1.5),
        (3, np.array([0.1, 0.2, 0.3]), 1.0, 5.0, 2.0),
    ],
    ids=["dim=1,sm=1.0", "dim=1,sm=10.0", "dim=2", "dim=3"],
)
def test_spectral_density_rational_quadratic(dim, w, alpha, length, scale_mixture):
    spectral_density = spectral_density_rational_quadratic(
        dim=dim, w=w, alpha=alpha, length=length, scale_mixture=scale_mixture
    )
    assert spectral_density.shape == ()
    assert spectral_density > 0.0


@pytest.mark.parametrize(
    argnames="alpha, length, scale_mixture, ell, m, dim",
    argvalues=[
        (1.0, 0.2, 1.0, 0.1, 1, 1),
        (1.0, 0.2, 1.5, 0.2, 2, 1),
        (1.0, 0.2, 2.0, 0.3, 10, 1),
        (1.0, 0.2, 10.0, 0.1, 100, 1),
        (1.0, 0.2, 1.5, 0.1, 10, 2),
        (1.0, 0.2, 2.0, 0.1, [2, 2, 3], 3),
    ],
    ids=["m=1,d=1", "m=2,d=1", "m=10,d=1", "m=100,d=1", "m=10,d=2", "m=[2,2,3],d=3"],
)
def test_diag_spectral_density_rational_quadratic(
    alpha, length, scale_mixture, ell, m, dim
):
    diag_spectral_density = diag_spectral_density_rational_quadratic(
        alpha=alpha, length=length, scale_mixture=scale_mixture, ell=ell, m=m, dim=dim
    )
    if isinstance(m, int):
        m = [m] * dim
    assert diag_spectral_density.shape == (reduce(mul, m),)
    assert jnp.all(diag_spectral_density >= 0.0)


def _log_spectral_density_rational_quadratic_reference(
    w: np.ndarray, alpha: float, length: float, scale_mixture: float
) -> tuple[np.ndarray, np.ndarray]:
    """float64 ``log S(w)`` of the 1d rational quadratic kernel and its derivative
    with respect to ``length``, via SciPy's exponentially scaled ``K_nu``.
    """
    a = np.sqrt(2 * scale_mixture) * length
    nu = scale_mixture - 0.5
    z = a * w
    log_spd = (
        np.log(alpha)
        + 0.5 * np.log(2 * np.pi)
        + (1 - scale_mixture) * np.log(2)
        + np.log(a)
        + nu * np.log(z)
        + np.log(kve(nu, z))
        - z
        - gammaln(scale_mixture)
    )
    # d/dz log K_nu(z) = -(K_{nu-1}(z) + K_{nu+1}(z)) / (2 K_nu(z))
    dlog_bessel_dz = -(kve(nu - 1, z) + kve(nu + 1, z)) / (2 * kve(nu, z))
    dlog_spd_dlength = (1 + nu) / length + np.sqrt(
        2 * scale_mixture
    ) * w * dlog_bessel_dz
    return log_spd, dlog_spd_dlength


def test_spectral_density_rational_quadratic_gradient_at_zero_frequency():
    """At ``w = 0`` the ``S(0)`` branch is selected; its gradient w.r.t. ``length``
    must not pick up NaNs from the (unselected) ``K_nu(0)`` branch.
    """

    def spd(length):
        return spectral_density_rational_quadratic(
            dim=1, w=np.array([0.0]), alpha=1.0, length=length, scale_mixture=1.0
        )

    # S(0) is linear in ``length`` (through a = sqrt(2 scale_mixture) length)
    length = 1.0
    assert np.allclose(jax.grad(spd)(length), spd(length) / length, rtol=1e-5)


@pytest.mark.parametrize(
    "length, scale_mixture",
    [(1.0, 1.0), (np.float64(1.0), np.float64(1.0))],
    ids=["python-float", "np-float64"],
)
def test_log_diag_spectral_density_rational_quadratic_no_underflow(
    length, scale_mixture
):
    """``K_nu(a w)`` underflows float32 for most of the first 100 eigenvalues; the
    log density must stay finite and match a float64 SciPy reference.
    """
    m, ell, alpha = 100, 1.0, 1.0
    log_spd = log_diag_spectral_density_rational_quadratic(
        alpha=alpha, length=length, scale_mixture=scale_mixture, ell=ell, m=m, dim=1
    )
    w = np.arange(1, m + 1, dtype=np.float64) * np.pi / 2 / ell
    expected, _ = _log_spectral_density_rational_quadratic_reference(
        w, alpha, float(length), float(scale_mixture)
    )
    assert np.all(np.isfinite(log_spd))
    assert np.allclose(log_spd, expected, rtol=1e-4)


def test_sqrt_spectral_density_rational_quadratic_gradient():
    """``exp(0.5 * log S)`` must have a finite gradient w.r.t. ``length`` where ``S``
    underflows; compare with the float64 derivative of ``sum_j sqrt(S_j)``.
    """
    m, ell, alpha, length, scale_mixture = 100, 1.0, 1.0, 1.0, 1.0

    def sqrt_spd_sum(length):
        return jnp.exp(
            0.5
            * log_diag_spectral_density_rational_quadratic(
                alpha=alpha,
                length=length,
                scale_mixture=scale_mixture,
                ell=ell,
                m=m,
                dim=1,
            )
        ).sum()

    w = np.arange(1, m + 1, dtype=np.float64) * np.pi / 2 / ell
    log_spd, dlog_spd = _log_spectral_density_rational_quadratic_reference(
        w, alpha, length, scale_mixture
    )
    expected = np.sum(np.exp(0.5 * log_spd) * 0.5 * dlog_spd)
    assert np.allclose(jax.grad(sqrt_spd_sum)(length), expected, rtol=1e-4)


@pytest.mark.parametrize(
    argnames="alpha, length, m",
    argvalues=[
        (1.0, 0.2, 1),
        (3.0, 0.4, 2),
        (2.0, 0.66, 10),
        (1.0, 0.2, 100),
    ],
    ids=["m=1", "m=2", "m=10", "m=100"],
)
def test_diag_spectral_density_periodic(alpha, length, m):
    diag_spectral_density = diag_spectral_density_periodic(
        alpha=alpha, length=length, m=m
    )
    assert diag_spectral_density.shape == (m,)
    assert jnp.all(diag_spectral_density >= 0.0)


@pytest.mark.parametrize(
    "length",
    [1, 1.0, 0.1, 0.05, np.float64(0.05)],
    ids=[
        "length=1-int",
        "length=1.0",
        "length=0.1",
        "length=0.05",
        "length=0.05-np-float64",
    ],
)
def test_diag_spectral_density_periodic_small_length(length):
    """``exp(1 / length^2)`` overflows float32 for ``length <~ 0.1``; the coefficients
    must stay finite and match SciPy's exponentially scaled ``I_j``.
    """
    m, alpha = 20, 1.0
    q2 = diag_spectral_density_periodic(alpha=alpha, length=length, m=m)
    j = np.arange(m)
    expected = np.where(j > 0, 2.0, 1.0) * alpha**2 * ive(j, float(length) ** (-2))
    assert np.all(np.isfinite(q2))
    assert np.allclose(q2, expected, rtol=1e-4)
