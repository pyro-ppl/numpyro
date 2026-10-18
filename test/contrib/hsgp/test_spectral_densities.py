# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import reduce
from operator import mul

import numpy as np
import pytest
from scipy.special import gammaln

import jax
import jax.numpy as jnp

from numpyro.contrib.hsgp.spectral_densities import (
    diag_spectral_density_matern,
    diag_spectral_density_periodic,
    diag_spectral_density_rational_quadratic,
    diag_spectral_density_squared_exponential,
    log_diag_spectral_density_matern,
    log_diag_spectral_density_squared_exponential,
    modified_bessel_first_kind,
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


@pytest.mark.parametrize(
    argnames="v, z",
    argvalues=[
        (0.5, 0.1),
        (1.0, 0.2),
        (2.0, 0.3),
        (3.0, 0.4),
    ],
    ids=["v=0.5-z=0.1", "v=1.0-z=0.2", "v=2.0-z=0.3", "v=3.0-z=0.4"],
)
def test_modified_bessel_first_kind_one_dim(v, z):
    assert modified_bessel_first_kind(v, z) > 0.0


@pytest.mark.parametrize(
    argnames="v, z",
    argvalues=[
        (np.linspace(0.1, 1.0, 10), np.array([0.1])),
        (np.linspace(0.1, 1.0, 10), np.linspace(0.1, 1.0, 10)),
    ],
    ids=["z=0.1", "z=0.2"],
)
def test_modified_bessel_first_kind_vect(v, z):
    assert jnp.all(modified_bessel_first_kind(v, z) > 0.0)


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
