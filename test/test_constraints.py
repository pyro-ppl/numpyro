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


SINGLETON_CONSTRAINTS = {
    "boolean": boolean,
    "circular": circular,
    "corr_cholesky": corr_cholesky,
    "corr_matrix": corr_matrix,
    "l1_ball": l1_ball,
    "lower_cholesky": lower_cholesky,
    "scaled_unit_lower_cholesky": scaled_unit_lower_cholesky,
    "nonnegative_integer": nonnegative_integer,
    "ordered_vector": ordered_vector,
    "positive": positive,
    "positive_definite": positive_definite,
    "positive_integer": positive_integer,
    "positive_ordered_vector": positive_ordered_vector,
    "real": real,
    "real_vector": real_vector,
    "real_matrix": real_matrix,
    "simplex": simplex,
    "softplus_lower_cholesky": softplus_lower_cholesky,
    "softplus_positive": softplus_positive,
    "sphere": sphere,
    "unit_interval": unit_interval,
}

_a = jnp.asarray


class T(namedtuple("TestCase", ["constraint_cls", "params"])):
    pass


PARAMETRIZED_CONSTRAINTS = {
    "greater_than": T(greater_than, (_a(0.0),)),
    "less_than": T(less_than, (_a(-1.0),)),
    "independent": T(independent, (greater_than(jnp.zeros((2,))), 1)),
    "integer_interval": T(integer_interval, (_a(-1), _a(1))),
    "integer_greater_than": T(integer_greater_than, (_a(1),)),
    "interval": T(interval, (_a(-1.0), _a(1.0))),
    "multinomial": T(multinomial, (_a(1.0),),),
    "open_interval": T(open_interval, (_a(-1.0), _a(1.0))),
}

# TODO: BijectorConstraint


@pytest.mark.parametrize(
    "constraint", SINGLETON_CONSTRAINTS.values(), ids=SINGLETON_CONSTRAINTS.keys()
)
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


@pytest.mark.parametrize(
    "cls, params",
    PARAMETRIZED_CONSTRAINTS.values(),
    ids=PARAMETRIZED_CONSTRAINTS.keys(),
)
def test_parametrized_constrains_pytree(cls, params):
    constraint = cls(*params)

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
        == constraint
    )
