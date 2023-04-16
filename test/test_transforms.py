# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple

import pytest

from jax import jit, vmap
import jax.numpy as jnp

from numpyro.distributions.flows import (
    BlockNeuralAutoregressiveTransform,
    InverseAutoregressiveTransform,
)
from numpyro.distributions.transforms import (
    AbsTransform,
    AffineTransform,
    CholeskyTransform,
    ComposeTransform,
    CorrCholeskyTransform,
    CorrMatrixCholeskyTransform,
    ExpTransform,
    IdentityTransform,
    IndependentTransform,
    L1BallTransform,
    LowerCholeskyAffine,
    LowerCholeskyTransform,
    OrderedTransform,
    PermuteTransform,
    PowerTransform,
    ScaledUnitLowerCholeskyTransform,
    SigmoidTransform,
    SimplexToOrderedTransform,
    SoftplusLowerCholeskyTransform,
    SoftplusTransform,
    StickBreakingTransform,
    UnpackTransform,
)


def _unpack(x):
    return (x,)


_a = jnp.asarray


def _smoke_neural_network():
    return None, None


class T(namedtuple("TestCase", ["transform_cls", "params"])):
    pass


TRANSFORMS = {
    "affine": T(AffineTransform, (jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))),
    "compose": T(ComposeTransform, ([ExpTransform(), ExpTransform()],)),
    "independent": T(
        IndependentTransform,
        (AffineTransform(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])), 1),
    ),
    "lower_cholesky_affine": T(
        LowerCholeskyAffine, (jnp.array([1.0, 2.0]), jnp.eye(2))
    ),
    "permute": T(PermuteTransform, (jnp.array([1, 0]),)),
    "power": T(PowerTransform, (_a(2.0),),),  # fmt: skip
    "simplex_to_ordered": T(SimplexToOrderedTransform, (_a(1.0),),),  # fmt: skip
    "unpack": T(UnpackTransform, (_unpack,)),
    # unparametrized transforms
    "abs": T(AbsTransform, ()),
    "cholesky": T(CholeskyTransform, ()),
    "corr_chol": T(CorrCholeskyTransform, ()),
    "corr_matrix_chol": T(CorrMatrixCholeskyTransform, ()),
    "exp": T(ExpTransform, ()),
    "identity": T(IdentityTransform, ()),
    "l1_ball": T(L1BallTransform, ()),
    "lower_cholesky": T(LowerCholeskyTransform, ()),
    "ordered": T(OrderedTransform, ()),
    "scaled_unit_lower_cholesky": T(ScaledUnitLowerCholeskyTransform, ()),
    "sigmoid": T(SigmoidTransform, ()),
    "softplus": T(SoftplusTransform, ()),
    "softplus_lower_cholesky": T(SoftplusLowerCholeskyTransform, ()),
    "stick_breaking": T(StickBreakingTransform, ()),
    # neural transforms
    "iaf": T(
        InverseAutoregressiveTransform,
        (_smoke_neural_network, -1.0, 1.0),
    ),
    "bna": T(
        BlockNeuralAutoregressiveTransform,
        (_smoke_neural_network,),
    ),
}


@pytest.mark.parametrize(
    "cls, params",
    TRANSFORMS.values(),
    ids=TRANSFORMS.keys(),
)
def test_parametrized_transform_pytree(cls, params):
    transform = cls(*params)

    # test that singleton transforms objects can be used as pytrees
    def in_t(transform, x):
        return x**2

    def out_t(transform, x):
        return transform

    jitted_in_t = jit(in_t)
    jitted_out_t = jit(out_t)

    assert jitted_in_t(transform, 1.0) == 1.0
    assert jitted_out_t(transform, 1.0) == transform

    assert jnp.allclose(
        vmap(in_t, in_axes=(None, 0), out_axes=0)(transform, jnp.ones(3)),
        jnp.ones(3),
    )

    assert (
        vmap(out_t, in_axes=(None, 0), out_axes=None)(transform, jnp.ones(3))
        == transform
    )
