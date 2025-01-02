# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import reduce
from operator import mul
from typing import Literal, Union

import numpy as np
import pytest
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, Matern

from jax import Array, random
import jax.numpy as jnp
from jax.typing import ArrayLike

import numpyro
from numpyro.contrib.hsgp.approximation import (
    hsgp_matern,
    hsgp_periodic_non_centered,
    hsgp_squared_exponential,
)
from numpyro.contrib.hsgp.laplacian import eigenfunctions, eigenfunctions_periodic
from numpyro.contrib.hsgp.spectral_densities import (
    diag_spectral_density_matern,
    diag_spectral_density_periodic,
    diag_spectral_density_squared_exponential,
)
import numpyro.distributions as dist
from numpyro.handlers import scope, seed, trace


def generate_synthetic_one_dim_data(
    rng_key: ArrayLike, start: float, stop: float, num: int, scale: float
) -> tuple[Array, Array]:
    x = jnp.linspace(start=start, stop=stop, num=num)
    y = jnp.sin(4 * jnp.pi * x) + jnp.sin(7 * jnp.pi * x)
    y_obs = y + scale * random.normal(rng_key, shape=(num,))
    return x, y_obs


@pytest.fixture
def synthetic_one_dim_data() -> tuple[Array, Array]:
    kwargs = {
        "rng_key": random.PRNGKey(0),
        "start": -0.2,
        "stop": 1.2,
        "num": 80,
        "scale": 0.3,
    }
    return generate_synthetic_one_dim_data(**kwargs)


def generate_synthetic_two_dim_data(
    rng_key: ArrayLike, start: float, stop: float, num: int, scale: float
) -> tuple[Array, Array]:
    x = random.uniform(rng_key, shape=(num, 2), minval=start, maxval=stop)
    y = jnp.sin(4 * jnp.pi * x[:, 0]) + jnp.sin(7 * jnp.pi * x[:, 1])
    y_obs = y + scale * random.normal(rng_key, shape=(num, num))
    return x, y_obs


@pytest.fixture
def synthetic_two_dim_data() -> tuple[Array, Array]:
    kwargs = {
        "rng_key": random.PRNGKey(0),
        "start": -0.2,
        "stop": 1.2,
        "num": 80,
        "scale": 0.3,
    }
    return generate_synthetic_two_dim_data(**kwargs)


@pytest.mark.parametrize(
    argnames="x1, x2, length, ell, xfail",
    argvalues=[
        (np.array([[1.0]]), np.array([[0.0]]), 1.0, 5.0, False),
        (
            np.array([[1.5, 1.25]]),
            np.array([[0.0, 0.0]]),
            1.0,
            5.0,
            False,
        ),
        (
            np.array([[1.5, 1.25]]),
            np.array([[0.0, 0.0]]),
            np.array([1.0, 0.5]),
            5.0,
            False,
        ),
        (
            np.array([[1.5, 1.25]]),
            np.array([[0.0, 0.0]]),
            np.array(
                [[1.0, 0.5], [0.5, 1.0]]
            ),  # different length scale for each point/dimension
            5.0,
            False,
        ),
        (
            np.array([[1.5, 1.25, 1.0]]),
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[1.0, 0.5], [0.5, 1.0]]),  # invalid length scale
            5.0,
            True,
        ),
    ],
    ids=[
        "1d,scalar-length",
        "2d,scalar-length",
        "2d,vector-length",
        "2d,matrix-length",
        "2d,invalid-length",
    ],
)
def test_kernel_approx_squared_exponential(
    x1: ArrayLike,
    x2: ArrayLike,
    length: Union[float, ArrayLike],
    ell: float,
    xfail: bool,
):
    """ensure that the approximation of the squared exponential kernel is accurate,
    matching the exact kernel implementation from sklearn.

    See Riutort-Mayol 2023 equation (13) for the approximation formula.
    """
    assert x1.shape == x2.shape
    m = 100  # large enough to ensure the approximation is accurate
    dim = x1.shape[-1]
    if xfail:
        with pytest.raises(ValueError):
            diag_spectral_density_squared_exponential(1.0, length, ell, m, dim)
        return
    spd = diag_spectral_density_squared_exponential(1.0, length, ell, m, dim)

    eig_f1 = eigenfunctions(x1, ell=ell, m=m)
    eig_f2 = eigenfunctions(x2, ell=ell, m=m)
    approx = (eig_f1 * eig_f2) @ spd

    def _exact_rbf(length):
        return RBF(length)(x1, x2).squeeze(axis=-1)

    if isinstance(length, int) | isinstance(length, float):
        exact = _exact_rbf(length)
    elif length.ndim == 1:
        exact = _exact_rbf(length)
    else:
        exact = np.apply_along_axis(_exact_rbf, axis=0, arr=length)
    assert jnp.isclose(approx, exact, rtol=1e-3).all()


@pytest.mark.parametrize(
    argnames="x1, x2, nu, length, ell",
    argvalues=[
        (np.array([[1.0]]), np.array([[0.0]]), 3 / 2, np.array([1.0]), 5.0),
        (np.array([[1.0]]), np.array([[0.0]]), 5 / 2, np.array([1.0]), 5.0),
        (
            np.array([[1.5, 1.25]]),
            np.array([[0.0, 0.0]]),
            3 / 2,
            np.array([0.25, 0.5]),
            5.0,
        ),
        (
            np.array([[1.5, 1.25]]),
            np.array([[0.0, 0.0]]),
            5 / 2,
            np.array([0.25, 0.5]),
            5.0,
        ),
        (
            np.array([[1.5, 1.25]]),
            np.array([[0.0, 0.0]]),
            3 / 2,
            np.array(
                [[1.0, 0.5], [0.5, 1.0]]
            ),  # different length scale for each point/dimension
            5.0,
        ),
        (
            np.array([[1.5, 1.25]]),
            np.array([[0.0, 0.0]]),
            5 / 2,
            np.array(
                [[1.0, 0.5], [0.5, 1.0]]
            ),  # different length scale for each point/dimension
            5.0,
        ),
    ],
    ids=[
        "1d,nu=3/2",
        "1d,nu=5/2",
        "2d,nu=3/2,1d-length",
        "2d,nu=5/2,1d-length",
        "2d,nu=3/2,2d-length",
        "2d,nu=5/2,2d-length",
    ],
)
def test_kernel_approx_squared_matern(
    x1: ArrayLike, x2: ArrayLike, nu: float, length: ArrayLike, ell: float
):
    """ensure that the approximation of the matern kernel is accurate,
    matching the exact kernel implementation from sklearn.

    See Riutort-Mayol 2023 equation (13) for the approximation formula.
    """
    assert x1.shape == x2.shape
    m = 100  # large enough to ensure the approximation is accurate
    dim = x1.shape[-1]
    spd = diag_spectral_density_matern(
        nu=nu, alpha=1.0, length=length, ell=ell, m=m, dim=dim
    )

    eig_f1 = eigenfunctions(x1, ell=ell, m=m)
    eig_f2 = eigenfunctions(x2, ell=ell, m=m)
    approx = (eig_f1 * eig_f2) @ spd

    def _exact_matern(length):
        return Matern(length_scale=length, nu=nu)(x1, x2).squeeze(axis=-1)

    if isinstance(length, float) | isinstance(length, int):
        exact = _exact_matern(length)
    elif length.ndim == 1:
        exact = _exact_matern(length)
    else:
        exact = np.apply_along_axis(_exact_matern, axis=0, arr=length)
    assert jnp.isclose(approx, exact, rtol=1e-3).all()


@pytest.mark.parametrize(
    argnames="x1, x2, w0, length",
    argvalues=[
        (np.array([1.0]), np.array([0.0]), 1.0, 1.0),
        (np.array([1.0]), np.array([0.0]), 1.5, 1.0),
    ],
    ids=[
        "1d,w0=1.0",
        "1d,w0=1.5",
    ],
)
def test_kernel_approx_periodic(
    x1: ArrayLike,
    x2: ArrayLike,
    w0: float,
    length: float,
):
    """ensure that the approximation of the periodic kernel is accurate,
    matching the exact kernel implementation from sklearn

    Note that the exact kernel implementation is parameterized with respect to the period,
    and the periodicity is w0**(-1). We adjust the input values by dividing by 2*pi.

    See Riutort-Mayol 2023 appendix B for the approximation formula.
    """
    assert x1.shape == x2.shape
    m = 100
    q2 = diag_spectral_density_periodic(alpha=1.0, length=length, m=m)
    q2_sine = jnp.concatenate([jnp.array([0.0]), q2[1:]])

    cosines_f1, sines_f1 = eigenfunctions_periodic(x1, w0=w0, m=m)
    cosines_f2, sines_f2 = eigenfunctions_periodic(x2, w0=w0, m=m)
    approx = (cosines_f1 * cosines_f2) @ q2 + (sines_f1 * sines_f2) @ q2_sine
    exact = ExpSineSquared(length_scale=length, periodicity=w0 ** (-1))(
        x1[..., None] / (2 * jnp.pi), x2[..., None] / (2 * jnp.pi)
    )
    assert jnp.isclose(approx, exact, rtol=1e-3)


@pytest.mark.parametrize(
    argnames="x, alpha, length, ell, m, non_centered",
    argvalues=[
        (np.linspace(0, 1, 10), 1.0, 0.2, 12, 10, True),
        (np.linspace(0, 1, 10), 1.0, 0.2, 12, 10, False),
        (np.linspace(0, 10, 100), 3.0, 0.5, 120, 100, True),
        (np.linspace(np.zeros(2), np.ones(2), 10), 1.0, 0.2, 12, [3, 3], True),
    ],
    ids=["non_centered", "centered", "non_centered-large-domain", "non_centered-2d"],
)
def test_approximation_squared_exponential(
    x: ArrayLike,
    alpha: float,
    length: float,
    ell: Union[int, float, list[Union[int, float]]],
    m: Union[int, list[int]],
    non_centered: bool,
):
    def model(x, alpha, length, ell, m, non_centered):
        numpyro.deterministic(
            "f",
            hsgp_squared_exponential(x, alpha, length, ell, m, non_centered),
        )

    rng_key = random.PRNGKey(0)
    approx_trace = trace(seed(model, rng_key)).get_trace(
        x, alpha, length, ell, m, non_centered
    )

    if x.ndim == 1:
        x_ = x[..., None]
    else:
        x_ = x
    if isinstance(m, int):
        m_ = [m] * x_.shape[-1]
    else:
        m_ = m

    assert approx_trace["f"]["value"].shape == x_.shape[:-1]
    assert approx_trace["beta"]["value"].shape == (reduce(mul, m_),)
    assert approx_trace["basis"]["value"].shape == (reduce(mul, m_),)


@pytest.mark.parametrize(
    argnames="x, nu, alpha, length, ell, m, non_centered",
    argvalues=[
        (np.linspace(0, 1, 10), 3 / 2, 1.0, 0.2, 12, 10, True),
        (np.linspace(0, 1, 10), 5 / 2, 1.0, 0.2, 12, 10, False),
        (np.linspace(0, 10, 100), 7 / 2, 3.0, 0.5, 120, 100, True),
        (
            np.linspace(np.zeros(2), np.ones(2), 10),
            3 / 2,
            1.0,
            0.2,
            12,
            [3, 3],
            True,
        ),
    ],
    ids=["non_centered", "centered", "non_centered-large-domain", "non_centered-2d"],
)
def test_approximation_matern(
    x: ArrayLike,
    nu: float,
    alpha: float,
    length: float,
    ell: Union[int, float, list[Union[int, float]]],
    m: Union[int, list[int]],
    non_centered: bool,
):
    def model(x, nu, alpha, length, ell, m, non_centered):
        numpyro.deterministic(
            "f", hsgp_matern(x, nu, alpha, length, ell, m, non_centered)
        )

    rng_key = random.PRNGKey(0)
    approx_trace = trace(seed(model, rng_key)).get_trace(
        x, nu, alpha, length, ell, m, non_centered
    )

    if x.ndim == 1:
        x_ = x[..., None]
    else:
        x_ = x
    if isinstance(m, int):
        m_ = [m] * x_.shape[-1]
    else:
        m_ = m

    assert approx_trace["f"]["value"].shape == x_.shape[:-1]
    assert approx_trace["beta"]["value"].shape == (reduce(mul, m_),)
    assert approx_trace["basis"]["value"].shape == (reduce(mul, m_),)


@pytest.mark.parametrize(
    argnames="ell, m, non_centered, num_dim",
    argvalues=[
        (1.2, 1, True, 1),
        (1.0, 2, True, 1),
        (2.3, 10, False, 1),
        (0.8, 100, False, 1),
        (1.0, [2, 2], True, 2),
        (1.0, 2, True, 2),
    ],
    ids=["m=1-nc", "m=2-nc", "m=10-c", "m=100-c", "m=[2,2]-nc", "m=2,dim=2-nc"],
)
def test_squared_exponential_gp_model(
    synthetic_one_dim_data,
    synthetic_two_dim_data,
    ell: Union[float, int, list[Union[float, int]]],
    m: Union[int, list[int]],
    non_centered: bool,
    num_dim: Literal[1, 2],
):
    def latent_gp(x, alpha, length, ell, m, non_centered):
        return numpyro.deterministic(
            "f",
            hsgp_squared_exponential(
                x=x, alpha=alpha, length=length, ell=ell, m=m, non_centered=non_centered
            ),
        )

    def model(x, ell, m, non_centered, y=None):
        alpha = numpyro.sample("alpha", dist.LogNormal(0.0, 1.0))
        length = numpyro.sample("length", dist.LogNormal(0.0, 1.0))
        noise = numpyro.sample("noise", dist.LogNormal(0.0, 1.0))
        f = scope(latent_gp, prefix="se", divider="::")(
            x=x, alpha=alpha, length=length, ell=ell, m=m, non_centered=non_centered
        )
        with numpyro.plate("data", x.shape[0]):
            numpyro.sample("likelihood", dist.Normal(loc=f, scale=noise), obs=y)

    x, y_obs = synthetic_one_dim_data if num_dim == 1 else synthetic_two_dim_data
    model_trace = trace(seed(model, random.PRNGKey(0))).get_trace(
        x, ell, m, non_centered, y_obs
    )

    if x.ndim == 1:
        x_ = x[..., None]
    else:
        x_ = x
    if isinstance(m, int):
        m_ = [m] * x_.shape[-1]
    else:
        m_ = m

    assert model_trace["se::f"]["value"].shape == x_.shape[:-1]
    assert model_trace["se::beta"]["value"].shape == (reduce(mul, m_),)
    assert model_trace["se::basis"]["value"].shape == (reduce(mul, m_),)


@pytest.mark.parametrize(
    argnames="nu, ell, m, non_centered, num_dim",
    argvalues=[
        (3 / 2, 1.2, 1, True, 1),
        (5 / 2, 1.0, 2, True, 1),
        (4.0, 2.3, 10, False, 1),
        (7 / 2, 0.8, 100, False, 1),
        (5 / 2, 1.0, [2, 2], True, 2),
        (5 / 2, 1.0, 2, True, 2),
    ],
    ids=["m=1-nc", "m=2-nc", "m=10-c", "m=100-c", "m=[2,2]-nc", "m=2,dim=2-nc"],
)
def test_matern_gp_model(
    synthetic_one_dim_data,
    synthetic_two_dim_data,
    nu: float,
    ell: Union[int, float, list[Union[float, int]]],
    m: Union[int, list[int]],
    non_centered: bool,
    num_dim: Literal[1, 2],
):
    def latent_gp(x, nu, alpha, length, ell, m, non_centered):
        return numpyro.deterministic(
            "f",
            hsgp_matern(
                x=x,
                nu=nu,
                alpha=alpha,
                length=length,
                ell=ell,
                m=m,
                non_centered=non_centered,
            ),
        )

    def model(x, nu, ell, m, non_centered, y=None):
        alpha = numpyro.sample("alpha", dist.LogNormal(0.0, 1.0))
        length = numpyro.sample("length", dist.LogNormal(0.0, 1.0))
        noise = numpyro.sample("noise", dist.LogNormal(0.0, 1.0))
        f = scope(latent_gp, prefix="matern", divider="::")(
            x=x,
            nu=nu,
            alpha=alpha,
            length=length,
            ell=ell,
            m=m,
            non_centered=non_centered,
        )
        with numpyro.plate("data", x.shape[0]):
            numpyro.sample("likelihood", dist.Normal(loc=f, scale=noise), obs=y)

    x, y_obs = synthetic_one_dim_data if num_dim == 1 else synthetic_two_dim_data
    model_trace = trace(seed(model, random.PRNGKey(0))).get_trace(
        x, nu, ell, m, non_centered, y_obs
    )

    if x.ndim == 1:
        x_ = x[..., None]
    else:
        x_ = x
    if isinstance(m, int):
        m_ = [m] * x_.shape[-1]
    else:
        m_ = m

    assert model_trace["matern::f"]["value"].shape == x_.shape[:-1]
    assert model_trace["matern::beta"]["value"].shape == (reduce(mul, m_),)
    assert model_trace["matern::basis"]["value"].shape == (reduce(mul, m_),)


@pytest.mark.parametrize(
    argnames="w0, m",
    argvalues=[
        (2 * np.pi / 7, 2),
        (2 * np.pi / 10, 3),
        (2 * np.pi / 5, 10),
    ],
    ids=["m=2", "m=3", "m=10"],
)
def test_periodic_gp_one_dim_model(synthetic_one_dim_data, w0, m):
    def latent_gp(x, alpha, length, w0, m):
        return numpyro.deterministic(
            "f",
            hsgp_periodic_non_centered(
                x=x,
                alpha=alpha,
                length=length,
                w0=w0,
                m=m,
            ),
        )

    def model(x, w0, m, y=None):
        alpha = numpyro.sample("alpha", dist.LogNormal(0.0, 1.0))
        length = numpyro.sample("length", dist.LogNormal(0.0, 1.0))
        noise = numpyro.sample("noise", dist.LogNormal(0.0, 1.0))
        f = scope(latent_gp, prefix="periodic", divider="::")(
            x=x,
            alpha=alpha,
            length=length,
            w0=w0,
            m=m,
        )
        with numpyro.plate("data", x.shape[0]):
            numpyro.sample("likelihood", dist.Normal(loc=f, scale=noise), obs=y)

    x, y_obs = synthetic_one_dim_data
    model_trace = trace(seed(model, random.PRNGKey(0))).get_trace(x, w0, m, y_obs)

    assert model_trace["periodic::f"]["value"].shape == x.shape
    assert model_trace["periodic::cos_basis"]["value"].shape == (m,)
    assert model_trace["periodic::sin_basis"]["value"].shape == (m - 1,)
