# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple

import pytest

import jax
from jax import jit, vmap
import jax.numpy as jnp

from numpyro.distributions import constraints

SINGLETON_CONSTRAINTS = {
    "boolean": constraints.boolean,
    "circular": constraints.circular,
    "complex": constraints.complex,
    "corr_cholesky": constraints.corr_cholesky,
    "corr_matrix": constraints.corr_matrix,
    "l1_ball": constraints.l1_ball,
    "lower_cholesky": constraints.lower_cholesky,
    "scaled_unit_lower_cholesky": constraints.scaled_unit_lower_cholesky,
    "nonnegative": constraints.nonnegative,
    "nonnegative_integer": constraints.nonnegative_integer,
    "ordered_vector": constraints.ordered_vector,
    "positive": constraints.positive,
    "positive_definite": constraints.positive_definite,
    "positive_semidefinite": constraints.positive_semidefinite,
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


class T(namedtuple("TestCase", ["constraint_cls", "params", "kwargs"])):
    pass


PARAMETRIZED_CONSTRAINTS = {
    "dependent": T(
        type(constraints.dependent), (), dict(is_discrete=True, event_dim=2)
    ),
    "greater_than": T(constraints.greater_than, (_a(0.0),), dict()),
    "greater_than_eq": T(constraints.greater_than_eq, (_a(0.0),), dict()),
    "less_than": T(constraints.less_than, (_a(-1.0),), dict()),
    "less_than_eq": T(constraints.less_than_eq, (_a(-1.0),), dict()),
    "independent": T(
        constraints.independent,
        (constraints.greater_than(jnp.zeros((2,))),),
        dict(reinterpreted_batch_ndims=1),
    ),
    "integer_interval": T(constraints.integer_interval, (_a(-1), _a(1)), dict()),
    "integer_greater_than": T(constraints.integer_greater_than, (_a(1),), dict()),
    "interval": T(constraints.interval, (_a(-1.0), _a(1.0)), dict()),
    "multinomial": T(
        constraints.multinomial,
        (_a(1.0),),
        dict(),
    ),
    "open_interval": T(constraints.open_interval, (_a(-1.0), _a(1.0)), dict()),
    "zero_sum": T(constraints.zero_sum, (), dict(event_dim=1)),
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
    "cls, cst_args, cst_kwargs",
    PARAMETRIZED_CONSTRAINTS.values(),
    ids=PARAMETRIZED_CONSTRAINTS.keys(),
)
def test_parametrized_constraint_pytree(cls, cst_args, cst_kwargs):
    constraint = cls(*cst_args, **cst_kwargs)

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

    if len(cst_args) > 0:
        # test creating and manipulating vmapped constraints
        vmapped_cst_args = jax.tree.map(lambda x: x[None], cst_args)

        vmapped_csts = jit(vmap(lambda args: cls(*args, **cst_kwargs), in_axes=(0,)))(
            vmapped_cst_args
        )
        assert vmap(lambda x: x == constraint, in_axes=0)(vmapped_csts).all()

        twice_vmapped_cst_args = jax.tree.map(lambda x: x[None], vmapped_cst_args)

        vmapped_csts = jit(
            vmap(
                vmap(lambda args: cls(*args, **cst_kwargs), in_axes=(0,)),
                in_axes=(0,),
            ),
        )(twice_vmapped_cst_args)
        assert vmap(vmap(lambda x: x == constraint, in_axes=0), in_axes=0)(
            vmapped_csts
        ).all()


@pytest.mark.parametrize(
    "cls, cst_args, cst_kwargs",
    PARAMETRIZED_CONSTRAINTS.values(),
    ids=PARAMETRIZED_CONSTRAINTS.keys(),
)
def test_parametrized_constraint_eq(cls, cst_args, cst_kwargs):
    constraint = cls(*cst_args, **cst_kwargs)
    constraint2 = cls(*cst_args, **cst_kwargs)
    assert constraint == constraint2
    assert constraint != 1

    # check that equality checks are robust to constraints parametrized
    # by abstract values
    @jit
    def check_constraints(c1, c2):
        return c1 == c2

    assert check_constraints(constraint, constraint2)


@pytest.mark.parametrize(
    "constraint", SINGLETON_CONSTRAINTS.values(), ids=SINGLETON_CONSTRAINTS.keys()
)
def test_singleton_constraint_eq(constraint):
    assert constraint == constraint
    assert constraint != 1

    # check that equality checks are robust to constraints parametrized
    # by abstract values
    @jit
    def check_constraints(c1, c2):
        return c1 == c2

    assert check_constraints(constraint, constraint)
