# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple

import pytest

from jax import jit, vmap
import jax.numpy as jnp

from numpyro.distributions.constraints import (
    boolean,
    circular,
    corr_cholesky,
    corr_matrix,
    greater_than,
    independent,
    integer_greater_than,
    integer_interval,
    interval,
    l1_ball,
    less_than,
    lower_cholesky,
    multinomial,
    nonnegative_integer,
    open_interval,
    ordered_vector,
    positive,
    positive_definite,
    positive_integer,
    positive_ordered_vector,
    real,
    real_matrix,
    real_vector,
    scaled_unit_lower_cholesky,
    simplex,
    softplus_lower_cholesky,
    softplus_positive,
    sphere,
    unit_interval,
)


class T(namedtuple("TestCase", ["constraint_cls", "params"])):
    pass


SINGLETON_CONSTRAINTS = (
    boolean,
    circular,
    corr_cholesky,
    corr_matrix,
    l1_ball,
    lower_cholesky,
    scaled_unit_lower_cholesky,
    nonnegative_integer,
    ordered_vector,
    positive,
    positive_definite,
    positive_integer,
    positive_ordered_vector,
    real,
    real_vector,
    real_matrix,
    simplex,
    softplus_lower_cholesky,
    softplus_positive,
    sphere,
    unit_interval,
)

PARAMETRIZED_CONSTRAINTS = (
    greater_than,
    less_than,
    independent,
    integer_interval,
    integer_greater_than,
    interval,
    multinomial,
    open_interval,
)


@pytest.mark.parametrize("constraint", SINGLETON_CONSTRAINTS)
def test_singleton_constrains_pytree(constraint):
    # test that singleton constraints objects can be used as pytrees
    def in_cst(constraint, x):
        return x**2

    def out_cst(constraint, x):
        return constraint

    jitted_in_cst = jit(in_cst)
    jitted_out_cst = jit(out_cst)

    assert jitted_in_cst(constraint, 1.0) == 1.0
    assert jitted_out_cst(constraint, 1.0) == constraint

    assert jnp.allclose(
        vmap(in_cst, in_axes=(None, 0), out_axes=0)(constraint, jnp.ones(3)),
        jnp.ones(3),
    )

    assert (
        vmap(out_cst, in_axes=(None, 0), out_axes=None)(constraint, jnp.ones(3))
        is constraint
    )
