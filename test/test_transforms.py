# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from functools import partial

import pytest

from jax import jit, tree_map, vmap
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


class T(namedtuple("TestCase", ["transform_cls", "params", "kwargs"])):
    pass


TRANSFORMS = {
    "affine": T(
        AffineTransform, (jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])), dict()
    ),
    "compose": T(
        ComposeTransform,
        (
            [
                AffineTransform(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])),
                ExpTransform(),
            ],
        ),
        dict(),
    ),
    "independent": T(
        IndependentTransform,
        (AffineTransform(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])),),
        dict(reinterpreted_batch_ndims=1),
    ),
    "lower_cholesky_affine": T(
        LowerCholeskyAffine, (jnp.array([1.0, 2.0]), jnp.eye(2)), dict()
    ),
    "permute": T(PermuteTransform, (jnp.array([1, 0]),), dict()),
    "power": T(
        PowerTransform,
        (_a(2.0),),
        dict(),
    ),
    "simplex_to_ordered": T(
        SimplexToOrderedTransform,
        (_a(1.0),),
        dict(),
    ),
    "unpack": T(UnpackTransform, (), dict(unpack_fn=_unpack)),
    # unparametrized transforms
    "abs": T(AbsTransform, (), dict()),
    "cholesky": T(CholeskyTransform, (), dict()),
    "corr_chol": T(CorrCholeskyTransform, (), dict()),
    "corr_matrix_chol": T(CorrMatrixCholeskyTransform, (), dict()),
    "exp": T(ExpTransform, (), dict()),
    "identity": T(IdentityTransform, (), dict()),
    "l1_ball": T(L1BallTransform, (), dict()),
    "lower_cholesky": T(LowerCholeskyTransform, (), dict()),
    "ordered": T(OrderedTransform, (), dict()),
    "scaled_unit_lower_cholesky": T(ScaledUnitLowerCholeskyTransform, (), dict()),
    "sigmoid": T(SigmoidTransform, (), dict()),
    "softplus": T(SoftplusTransform, (), dict()),
    "softplus_lower_cholesky": T(SoftplusLowerCholeskyTransform, (), dict()),
    "stick_breaking": T(StickBreakingTransform, (), dict()),
    # neural transforms
    "iaf": T(
        # autoregressive_nn is a non-jittable arg, which does not fit well with
        # the current test pipeline, which assumes jittable args, and non-jittable kwargs
        partial(InverseAutoregressiveTransform, _smoke_neural_network),
        (_a(-1.0), _a(1.0)),
        dict(),
    ),
    "bna": T(
        partial(BlockNeuralAutoregressiveTransform, _smoke_neural_network),
        (),
        dict(),
    ),
}


@pytest.mark.parametrize(
    "cls, transform_args, transform_kwargs",
    TRANSFORMS.values(),
    ids=TRANSFORMS.keys(),
)
def test_parametrized_transform_pytree(cls, transform_args, transform_kwargs):
    transform = cls(*transform_args, **transform_kwargs)

    # test that singleton transforms objects can be used as pytrees
    def in_t(transform, x):
        return x**2

    def out_t(transform, x):
        return transform

    jitted_in_t = jit(in_t)
    jitted_out_t = jit(out_t)

    assert jitted_in_t(transform, 1.0) == 1.0
    assert jitted_out_t(transform, 1.0) == transform

    assert jitted_out_t(transform.inv, 1.0) == transform.inv

    assert jnp.allclose(
        vmap(in_t, in_axes=(None, 0), out_axes=0)(transform, jnp.ones(3)),
        jnp.ones(3),
    )

    assert (
        vmap(out_t, in_axes=(None, 0), out_axes=None)(transform, jnp.ones(3))
        == transform
    )

    if len(transform_args) > 0:
        # test creating and manipulating vmapped constraints
        # this test assumes jittable args, and non-jittable kwargs, which is
        # not suited for all transforms, see InverseAutoregressiveTransform.
        # TODO: split among jittable and non-jittable args/kwargs instead.
        vmapped_transform_args = tree_map(lambda x: x[None], transform_args)

        vmapped_transform = jit(
            vmap(lambda args: cls(*args, **transform_kwargs), in_axes=(0,))
        )(vmapped_transform_args)
        assert vmap(lambda x: x == transform, in_axes=0)(vmapped_transform).all()

        twice_vmapped_transform_args = tree_map(
            lambda x: x[None], vmapped_transform_args
        )

        vmapped_transform = jit(
            vmap(
                vmap(lambda args: cls(*args, **transform_kwargs), in_axes=(0,)),
                in_axes=(0,),
            )
        )(twice_vmapped_transform_args)
        assert vmap(vmap(lambda x: x == transform, in_axes=0), in_axes=0)(
            vmapped_transform
        ).all()


@pytest.mark.parametrize(
    "cls, transform_args, transform_kwargs",
    TRANSFORMS.values(),
    ids=TRANSFORMS.keys(),
)
def test_parametrized_transform_eq(cls, transform_args, transform_kwargs):
    transform = cls(*transform_args, **transform_kwargs)
    transform2 = cls(*transform_args, **transform_kwargs)
    assert transform == transform2
    assert transform != 1.0

    # check that equality checks are robust to transforms parametrized
    # by abstract values
    @jit
    def check_transforms(t1, t2):
        return t1 == t2

    assert check_transforms(transform, transform2)
