# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple

import pytest

from jax import jit, vmap
import jax.numpy as jnp

from numpyro.distributions import constraints

SINGLETON_CONSTRAINTS = {
    "boolean": constraints.boolean,
    "circular": constraints.circular,
    "corr_cholesky": constraints.corr_cholesky,
    "corr_matrix": constraints.corr_matrix,
    "l1_ball": constraints.l1_ball,
    "lower_cholesky": constraints.lower_cholesky,
    "scaled_unit_lower_cholesky": constraints.scaled_unit_lower_cholesky,
    "nonnegative_integer": constraints.nonnegative_integer,
    "ordered_vector": constraints.ordered_vector,
    "positive": constraints.positive,
    "positive_definite": constraints.positive_definite,
    "positive_integer": constraints.positive_integer,
    "positive_ordered_vector": constraints.positive_ordered_vector,
    "real": constraints.real,
    "real_vector": constraints.real_vector,
    "real_matrix": constraints.real_matrix,
    "simplex": constraints.simplex,
    "softplus_lower_cholesky": constraints.softplus_lower_cholesky,
    "softplus_positive": constraints.softplus_positive,
    "sphere": constraints.sphere,
    "unit_interval": constraints.unit_interval,
}

_a = jnp.asarray


class T(namedtuple("TestCase", ["constraint_cls", "params"])):
    pass


PARAMETRIZED_CONSTRAINTS = {
    "greater_than": T(constraints.greater_than, (_a(0.0),)),
    "less_than": T(constraints.less_than, (_a(-1.0),)),
    "independent": T(
        constraints.independent, (constraints.greater_than(jnp.zeros((2,))), 1)
    ),
    "integer_interval": T(constraints.integer_interval, (_a(-1), _a(1))),
    "integer_greater_than": T(constraints.integer_greater_than, (_a(1),)),
    "interval": T(constraints.interval, (_a(-1.0), _a(1.0))),
    "multinomial": T(
        constraints.multinomial,
        (_a(1.0),),
    ),
    "open_interval": T(constraints.open_interval, (_a(-1.0), _a(1.0))),
}

# TODO: BijectorConstraint


@pytest.mark.parametrize(
    "constraint", SINGLETON_CONSTRAINTS.values(), ids=SINGLETON_CONSTRAINTS.keys()
)
def test_singleton_constraint_pytree(constraint):
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
def test_parametrized_constraint_pytree(cls, params):
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
