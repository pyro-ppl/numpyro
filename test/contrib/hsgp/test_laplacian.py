import pytest

import jax.numpy as jnp

from numpyro.contrib.hsgp.laplacian import eigenfunctions, sqrt_eigenvalues


@pytest.mark.parametrize(
    argnames="ell, m",
    argvalues=[
        (0.1, 1),
        (0.2, 2),
        (0.3, 10),
        (0.1, 100),
    ],
    ids=["m=1", "m=2", "m=10", "m=100"],
)
def test_sqrt_eigenvalues(ell, m):
    sqrt_eigenvalues_ = sqrt_eigenvalues(ell=ell, m=m)
    diff_sqrt_eigenvalues = jnp.diff(sqrt_eigenvalues_)
    assert sqrt_eigenvalues_.shape == (m,)
    assert jnp.all(sqrt_eigenvalues_ > 0.0)
    assert jnp.all(diff_sqrt_eigenvalues > 0.0)


@pytest.mark.parametrize(
    argnames="x, ell, m",
    argvalues=[
        (jnp.linspace(0, 1, 10), 1, 1),
        (jnp.linspace(-1, 1, 10), 1, 21),
        (jnp.linspace(-2, -1, 10), 2, 10),
        (jnp.linspace(0, 100, 500), 120, 100),
    ],
    ids=["x_pos", "x_contains_zero", "x_neg2", "x_pos2-large"],
)
def test_eigenfunctions(x, m, ell):
    phi = eigenfunctions(x=x, ell=ell, m=m)
    assert phi.shape == (len(x), m)
    assert phi.max() <= 1.0
    assert phi.min() >= -1.0
