import pytest

from jax import random
import jax.numpy as jnp

import numpyro
from numpyro.contrib.hsgp.approximation import (
    hsgp_approximation_matern,
    hsgp_approximation_periodic_non_centered,
    hsgp_approximation_squared_exponential,
)
import numpyro.distributions as dist
from numpyro.handlers import scope, seed, trace


def generate_synthetic_one_dim_data(rng_key, start, stop, num, scale):
    x = jnp.linspace(start=start, stop=stop, num=num)
    y = jnp.sin(4 * jnp.pi * x) + jnp.sin(7 * jnp.pi * x)
    y_obs = y + scale * random.normal(rng_key, shape=(num,))
    return x, y_obs


@pytest.fixture
def synthetic_one_dim_data():
    kwargs = {
        "rng_key": random.PRNGKey(0),
        "start": -0.2,
        "stop": 1.2,
        "num": 80,
        "scale": 0.3,
    }
    return generate_synthetic_one_dim_data(**kwargs)


@pytest.mark.parametrize(
    argnames="x, alpha, length, ell, m, non_centered",
    argvalues=[
        (jnp.linspace(0, 1, 10), 1.0, 0.2, 12, 10, True),
        (jnp.linspace(0, 1, 10), 1.0, 0.2, 12, 10, False),
        (jnp.linspace(0, 10, 100), 3.0, 0.5, 120, 100, True),
    ],
    ids=["non_centered", "centered", "non_centered-large-domain"],
)
def test_approximation_squared_exponential(x, alpha, length, ell, m, non_centered):
    def model(x, alpha, length, ell, m, non_centered):
        numpyro.deterministic(
            "f",
            hsgp_approximation_squared_exponential(
                x, alpha, length, ell, m, non_centered
            ),
        )

    rng_key = random.PRNGKey(0)
    approx_trace = trace(seed(model, rng_key)).get_trace(
        x, alpha, length, ell, m, non_centered
    )
    assert approx_trace["f"]["value"].shape == x.shape
    assert approx_trace["beta"]["value"].shape == (m,)
    assert approx_trace["basis"]["value"].shape == (m,)


@pytest.mark.parametrize(
    argnames="x, nu, alpha, length, ell, m, non_centered",
    argvalues=[
        (jnp.linspace(0, 1, 10), 3 / 2, 1.0, 0.2, 12, 10, True),
        (jnp.linspace(0, 1, 10), 5 / 2, 1.0, 0.2, 12, 10, False),
        (jnp.linspace(0, 10, 100), 7 / 2, 3.0, 0.5, 120, 100, True),
    ],
    ids=["non_centered", "centered", "non_centered-large-domain"],
)
def test_approximation_matern(x, nu, alpha, length, ell, m, non_centered):
    def model(x, nu, alpha, length, ell, m, non_centered):
        numpyro.deterministic(
            "f", hsgp_approximation_matern(x, nu, alpha, length, ell, m, non_centered)
        )

    rng_key = random.PRNGKey(0)
    approx_trace = trace(seed(model, rng_key)).get_trace(
        x, nu, alpha, length, ell, m, non_centered
    )
    assert approx_trace["f"]["value"].shape == x.shape
    assert approx_trace["beta"]["value"].shape == (m,)
    assert approx_trace["basis"]["value"].shape == (m,)


@pytest.mark.parametrize(
    argnames="ell, m, non_centered",
    argvalues=[
        (1.2, 1, True),
        (1.0, 2, True),
        (2.3, 10, False),
        (0.8, 100, False),
    ],
    ids=["m=1-nc", "m=2-nc", "m=10-c", "m=100-c"],
)
def test_squared_exponential_gp_one_dim_model(
    synthetic_one_dim_data, ell, m, non_centered
):
    def latent_gp(x, alpha, length, ell, m, non_centered):
        return numpyro.deterministic(
            "f",
            hsgp_approximation_squared_exponential(
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

    x, y_obs = synthetic_one_dim_data
    model_trace = trace(seed(model, random.PRNGKey(0))).get_trace(
        x, ell, m, non_centered, y_obs
    )

    assert model_trace["se::f"]["value"].shape == x.shape
    assert model_trace["se::beta"]["value"].shape == (m,)
    assert model_trace["se::basis"]["value"].shape == (m,)


@pytest.mark.parametrize(
    argnames="nu, ell, m, non_centered",
    argvalues=[
        (3 / 2, 1.2, 1, True),
        (5 / 2, 1.0, 2, True),
        (4.0, 2.3, 10, False),
        (7 / 2, 0.8, 100, False),
    ],
    ids=["m=1-nc", "m=2-nc", "m=10-c", "m=100-c"],
)
def test_matern_gp_one_dim_model(synthetic_one_dim_data, nu, ell, m, non_centered):
    def latent_gp(x, nu, alpha, length, ell, m, non_centered):
        return numpyro.deterministic(
            "f",
            hsgp_approximation_matern(
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

    x, y_obs = synthetic_one_dim_data
    model_trace = trace(seed(model, random.PRNGKey(0))).get_trace(
        x, nu, ell, m, non_centered, y_obs
    )

    assert model_trace["matern::f"]["value"].shape == x.shape
    assert model_trace["matern::beta"]["value"].shape == (m,)
    assert model_trace["matern::basis"]["value"].shape == (m,)


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
            hsgp_approximation_periodic_non_centered(
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
