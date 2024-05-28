# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import reduce
from operator import mul
from typing import Literal

import pytest

from jax import random
from jax._src.array import ArrayImpl
import jax.numpy as jnp

import numpyro
from numpyro.contrib.hsgp.approximation import (
    hsgp_matern,
    hsgp_periodic_non_centered,
    hsgp_squared_exponential,
)
import numpyro.distributions as dist
from numpyro.handlers import scope, seed, trace


def generate_synthetic_one_dim_data(
    rng_key, start, stop, num, scale
) -> tuple[ArrayImpl, ArrayImpl]:
    x = jnp.linspace(start=start, stop=stop, num=num)
    y = jnp.sin(4 * jnp.pi * x) + jnp.sin(7 * jnp.pi * x)
    y_obs = y + scale * random.normal(rng_key, shape=(num,))
    return x, y_obs


@pytest.fixture
def synthetic_one_dim_data() -> tuple[ArrayImpl, ArrayImpl]:
    kwargs = {
        "rng_key": random.PRNGKey(0),
        "start": -0.2,
        "stop": 1.2,
        "num": 80,
        "scale": 0.3,
    }
    return generate_synthetic_one_dim_data(**kwargs)


def generate_synthetic_two_dim_data(
    rng_key, start, stop, num, scale
) -> tuple[ArrayImpl, ArrayImpl]:
    x = random.uniform(rng_key, shape=(num, 2), minval=start, maxval=stop)
    y = jnp.sin(4 * jnp.pi * x[:, 0]) + jnp.sin(7 * jnp.pi * x[:, 1])
    y_obs = y + scale * random.normal(rng_key, shape=(num, num))
    return x, y_obs


@pytest.fixture
def synthetic_two_dim_data() -> tuple[ArrayImpl, ArrayImpl]:
    kwargs = {
        "rng_key": random.PRNGKey(0),
        "start": -0.2,
        "stop": 1.2,
        "num": 80,
        "scale": 0.3,
    }
    return generate_synthetic_two_dim_data(**kwargs)


@pytest.mark.parametrize(
    argnames="x, alpha, length, ell, m, non_centered",
    argvalues=[
        (jnp.linspace(0, 1, 10), 1.0, 0.2, 12, 10, True),
        (jnp.linspace(0, 1, 10), 1.0, 0.2, 12, 10, False),
        (jnp.linspace(0, 10, 100), 3.0, 0.5, 120, 100, True),
        (jnp.linspace(jnp.zeros(2), jnp.ones(2), 10), 1.0, 0.2, 12, [3, 3], True),
    ],
    ids=["non_centered", "centered", "non_centered-large-domain", "non_centered-2d"],
)
def test_approximation_squared_exponential(
    x: ArrayImpl,
    alpha: float,
    length: float,
    ell: int | float | list[int | float],
    m: int | list[int],
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
        (jnp.linspace(0, 1, 10), 3 / 2, 1.0, 0.2, 12, 10, True),
        (jnp.linspace(0, 1, 10), 5 / 2, 1.0, 0.2, 12, 10, False),
        (jnp.linspace(0, 10, 100), 7 / 2, 3.0, 0.5, 120, 100, True),
        (
            jnp.linspace(jnp.zeros(2), jnp.ones(2), 10),
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
    x: ArrayImpl,
    nu: float,
    alpha: float,
    length: float,
    ell: int | float | list[int | float],
    m: int | list[int],
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
    ell: float | int | list[float | int],
    m: int | list[int],
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
    ell: int | float | list[float | int],
    m: int | list[int],
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
        (2 * jnp.pi / 7, 2),
        (2 * jnp.pi / 10, 3),
        (2 * jnp.pi / 5, 10),
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
