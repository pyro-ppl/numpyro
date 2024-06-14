# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import reduce
from operator import mul

import numpy as np
import pytest

import jax.numpy as jnp

from numpyro.contrib.hsgp.spectral_densities import (
    diag_spectral_density_matern,
    diag_spectral_density_periodic,
    diag_spectral_density_squared_exponential,
    modified_bessel_first_kind,
    spectral_density_matern,
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
