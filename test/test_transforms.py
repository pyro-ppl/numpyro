# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from functools import partial
import math

import numpy as np
import pytest

import jax
from jax import jacfwd, jit, random, vmap
import jax.numpy as jnp

from numpyro.distributions import constraints
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
    RealFastFourierTransform,
    RecursiveLinearTransform,
    ReshapeTransform,
    ScaledUnitLowerCholeskyTransform,
    SigmoidTransform,
    SimplexToOrderedTransform,
    SoftplusLowerCholeskyTransform,
    SoftplusTransform,
    StickBreakingTransform,
    UnpackTransform,
    ZeroSumTransform,
    biject_to,
)


def _unpack(x):
    return (x,)


def _smoke_neural_network():
    return None, None


class T(namedtuple("TestCase", ["transform_cls", "params", "kwargs"])):
    pass


TRANSFORMS = {
    "affine": T(AffineTransform, (np.array([1.0, 2.0]), np.array([3.0, 4.0])), dict()),
    "compose": T(
        ComposeTransform,
        (
            [
                AffineTransform(np.array([1.0, 2.0]), np.array([3.0, 4.0])),
                ExpTransform(),
            ],
        ),
        dict(),
    ),
    "independent": T(
        IndependentTransform,
        (AffineTransform(np.array([1.0, 2.0]), np.array([3.0, 4.0])),),
        dict(reinterpreted_batch_ndims=1),
    ),
    "lower_cholesky_affine": T(
        LowerCholeskyAffine, (np.array([1.0, 2.0]), np.eye(2)), dict()
    ),
    "permute": T(PermuteTransform, (np.array([1, 0]),), dict()),
    "power": T(
        PowerTransform,
        (np.array(2.0),),
        dict(),
    ),
    "rfft": T(
        RealFastFourierTransform,
        (),
        dict(transform_shape=(3, 4, 5), transform_ndims=3),
    ),
    "recursive_linear": T(
        RecursiveLinearTransform,
        (np.eye(5),),
        dict(),
    ),
    "simplex_to_ordered": T(
        SimplexToOrderedTransform,
        (np.array(1.0),),
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
        (np.array(-1.0), np.array(1.0)),
        dict(),
    ),
    "bna": T(
        partial(BlockNeuralAutoregressiveTransform, _smoke_neural_network),
        (),
        dict(),
    ),
    "reshape": T(
        ReshapeTransform, (), {"forward_shape": (3, 4), "inverse_shape": (4, 3)}
    ),
    "zero_sum": T(ZeroSumTransform, (), dict(transform_ndims=1)),
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
        vmapped_transform_args = jax.tree.map(lambda x: x[None], transform_args)

        vmapped_transform = jit(
            vmap(lambda args: cls(*args, **transform_kwargs), in_axes=(0,))
        )(vmapped_transform_args)
        assert vmap(lambda x: x == transform, in_axes=0)(vmapped_transform).all()

        twice_vmapped_transform_args = jax.tree.map(
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


@pytest.mark.parametrize(
    "forward_shape, inverse_shape, batch_shape",
    [
        ((3, 4), (4, 3), ()),
        ((7,), (7, 1), ()),
        ((3, 5), (15,), ()),
        ((2, 4), (2, 2, 2), (17,)),
    ],
)
def test_reshape_transform(forward_shape, inverse_shape, batch_shape):
    x = random.normal(random.key(29), batch_shape + inverse_shape)
    transform = ReshapeTransform(forward_shape, inverse_shape)
    y = transform(x)
    assert y.shape == batch_shape + forward_shape
    x2 = transform.inv(y)
    assert x2.shape == batch_shape + inverse_shape
    assert jnp.allclose(x, x2).all()


def test_reshape_transform_invalid():
    with pytest.raises(ValueError, match="are not compatible"):
        ReshapeTransform((3,), (4,))

    with pytest.raises(TypeError, match="cannot reshape array"):
        ReshapeTransform((2, 3), (6,))(jnp.arange(2))


@pytest.mark.parametrize(
    "input_shape, shape, ndims",
    [
        ((10,), None, 1),
        ((11,), 11, 1),
        ((10, 18), None, 2),
        ((10, 19), (7, 8), 2),
    ],
)
def test_real_fast_fourier_transform(input_shape, shape, ndims):
    x1 = random.normal(random.key(17), input_shape)
    transform = RealFastFourierTransform(shape, ndims)
    y = transform(x1)
    assert transform.codomain(y).all()
    assert y.shape == transform.forward_shape(x1.shape)
    x2 = transform.inv(y)
    assert transform.domain(x2).all()
    if x1.shape == x2.shape:
        assert jnp.allclose(x2, x1, atol=1e-6)


@pytest.mark.parametrize(
    "transform, shape",
    [
        (AffineTransform(3, 2.5), ()),
        (CholeskyTransform(), (10,)),
        (ComposeTransform([SoftplusTransform(), SigmoidTransform()]), ()),
        (CorrCholeskyTransform(), (15,)),
        (CorrMatrixCholeskyTransform(), (15,)),
        (ExpTransform(), ()),
        (IdentityTransform(), ()),
        (IndependentTransform(ExpTransform(), 2), (3, 4)),
        (L1BallTransform(), (9,)),
        (LowerCholeskyAffine(np.ones(3), np.eye(3)), (3,)),
        (LowerCholeskyTransform(), (10,)),
        (OrderedTransform(), (5,)),
        (PermuteTransform(np.roll(np.arange(7), 2)), (7,)),
        (PowerTransform(2.5), ()),
        (RealFastFourierTransform(7), (7,)),
        (RealFastFourierTransform((8, 9), 2), (8, 9)),
        (
            RecursiveLinearTransform(random.normal(random.key(17), (4, 4))),
            (7, 4),
        ),
        (ReshapeTransform((5, 2), (10,)), (10,)),
        (ReshapeTransform((15,), (3, 5)), (3, 5)),
        (ScaledUnitLowerCholeskyTransform(), (6,)),
        (SigmoidTransform(), ()),
        (SimplexToOrderedTransform(), (5,)),
        (SoftplusLowerCholeskyTransform(), (10,)),
        (SoftplusTransform(), ()),
        (StickBreakingTransform(), (11,)),
        (ZeroSumTransform(1), (5,)),
    ],
)
def test_bijective_transforms(transform, shape):
    if isinstance(transform, type):
        pytest.skip()
    # Get a sample from the support of the distribution.
    batch_shape = (13,)
    unconstrained = random.normal(random.key(17), batch_shape + shape)
    x1 = biject_to(transform.domain)(unconstrained)

    # Transform forward and backward, checking shapes, values, and Jacobian shape.
    y = transform(x1)
    assert y.shape == transform.forward_shape(x1.shape)

    x2 = transform.inv(y)
    assert x2.shape == transform.inverse_shape(y.shape)
    # Some transforms are a bit less stable; we give them larger tolerances.
    atol = 1e-6
    less_stable_transforms = (
        CorrCholeskyTransform,
        L1BallTransform,
        StickBreakingTransform,
    )
    if isinstance(transform, less_stable_transforms):
        atol = 1e-2
    assert jnp.allclose(x1, x2, atol=atol)

    log_abs_det_jacobian = transform.log_abs_det_jacobian(x1, y)
    assert log_abs_det_jacobian.shape == batch_shape

    # Also check the Jacobian numerically for transforms with the same input and output
    # size, unless they are explicitly excluded. E.g., the upper triangular of the
    # CholeskyTransform is zero, giving rise to a singular Jacobian.
    skip_jacobian_check = (CholeskyTransform,)
    size_x = int(x1.size / math.prod(batch_shape))
    size_y = int(y.size / math.prod(batch_shape))
    if size_x == size_y and not isinstance(transform, skip_jacobian_check):
        jac = (
            vmap(jacfwd(transform))(x1)
            .reshape((-1,) + x1.shape[len(batch_shape) :])
            .reshape(batch_shape + (size_y, size_x))
        )
        slogdet = jnp.linalg.slogdet(jac)
        assert jnp.allclose(log_abs_det_jacobian, slogdet.logabsdet, atol=atol)
        assert transform.domain.event_dim or jnp.allclose(
            jnp.sign(jnp.diagonal(jac, axis1=-1, axis2=-2)), transform.sign
        )


def test_batched_recursive_linear_transform():
    batch_shape = (4, 17)
    x = random.normal(random.key(8), batch_shape + (10, 3))
    # Get a batch of matrices with eigenvalues that don't blow up the sequence.
    A = CorrCholeskyTransform()(random.normal(random.key(7), batch_shape + (3,)))
    transform = RecursiveLinearTransform(A)
    y = transform(x)
    assert y.shape == x.shape
    assert jnp.allclose(x, transform.inv(y), atol=1e-6)


@pytest.mark.parametrize(
    "constraint, shape",
    [
        (constraints.circular, (3,)),
        (constraints.complex, (3,)),
        (constraints.corr_cholesky, (10, 10)),
        (constraints.corr_matrix, (15,)),
        (constraints.greater_than(3), ()),
        (constraints.greater_than_eq(3), ()),
        (constraints.interval(8, 13), (17,)),
        (constraints.l1_ball, (4,)),
        (constraints.less_than(-1), ()),
        (constraints.less_than_eq(-1), ()),
        (constraints.lower_cholesky, (15,)),
        (constraints.open_interval(3, 4), ()),
        (constraints.ordered_vector, (5,)),
        (constraints.positive_definite, (6,)),
        (constraints.positive_semidefinite, (6,)),
        (constraints.positive_ordered_vector, (7,)),
        (constraints.positive, (7,)),
        (constraints.real_matrix, (17,)),
        (constraints.real_vector, (18,)),
        (constraints.real, (3,)),
        (constraints.scaled_unit_lower_cholesky, (15,)),
        (constraints.simplex, (3,)),
        (constraints.softplus_lower_cholesky, (15,)),
        (constraints.softplus_positive, (2,)),
        (constraints.unit_interval, (4,)),
        (constraints.nonnegative, (7,)),
    ],
    ids=str,
)
def test_biject_to(constraint, shape):
    batch_shape = (13, 19)
    unconstrained = random.normal(random.key(93), batch_shape + shape)
    constrained = biject_to(constraint)(unconstrained)
    passed = constraint.check(constrained)
    expected_shape = constrained.shape[: constrained.ndim - constraint.event_dim]
    assert passed.shape == expected_shape
    assert jnp.all(passed)
