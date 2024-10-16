# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
import warnings
import weakref

import numpy as np

import jax
from jax import lax, vmap
from jax.nn import log_sigmoid, softplus
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import expit, logit
from jax.tree_util import register_pytree_node

from numpyro.distributions import constraints
from numpyro.distributions.util import (
    add_diag,
    matrix_to_tril_vec,
    signed_stick_breaking_tril,
    sum_rightmost,
    vec_to_tril_matrix,
)
from numpyro.util import find_stack_level, not_jax_tracer

__all__ = [
    "biject_to",
    "AbsTransform",
    "AffineTransform",
    "CholeskyTransform",
    "ComposeTransform",
    "CorrCholeskyTransform",
    "CorrMatrixCholeskyTransform",
    "ExpTransform",
    "IdentityTransform",
    "L1BallTransform",
    "LowerCholeskyTransform",
    "ScaledUnitLowerCholeskyTransform",
    "LowerCholeskyAffine",
    "PermuteTransform",
    "PowerTransform",
    "RealFastFourierTransform",
    "ReshapeTransform",
    "SigmoidTransform",
    "SimplexToOrderedTransform",
    "SoftplusTransform",
    "SoftplusLowerCholeskyTransform",
    "StickBreakingTransform",
    "Transform",
    "UnpackTransform",
    "ZeroSumTransform",
]


def _clipped_expit(x):
    finfo = jnp.finfo(jnp.result_type(x))
    return jnp.clip(expit(x), finfo.tiny, 1.0 - finfo.eps)


class Transform(object):
    domain = constraints.real
    codomain = constraints.real
    _inv = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_pytree_node(cls, cls.tree_flatten, cls.tree_unflatten)

    @property
    def inv(self):
        inv = None
        if self._inv is not None:
            inv = self._inv()
        if inv is None:
            inv = _InverseTransform(self)
            self._inv = weakref.ref(inv)
        return inv

    def __call__(self, x):
        raise NotImplementedError

    def _inverse(self, y):
        raise NotImplementedError

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        raise NotImplementedError

    def call_with_intermediates(self, x):
        return self(x), None

    def forward_shape(self, shape):
        """
        Infers the shape of the forward computation, given the input shape.
        Defaults to preserving shape.
        """
        return shape

    def inverse_shape(self, shape):
        """
        Infers the shapes of the inverse computation, given the output shape.
        Defaults to preserving shape.
        """
        return shape

    @property
    def sign(self):
        """
        Sign of the derivative of the transform if it is bijective.
        """
        raise NotImplementedError(
            f"Transform `{self.__class__.__name__}` does not implement `sign`."
        )

    # Allow for pickle serialization of transforms.
    def __getstate__(self):
        attrs = {}
        for k, v in self.__dict__.items():
            if isinstance(v, weakref.ref):
                attrs[k] = None
            else:
                attrs[k] = v
        return attrs

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        params_keys, aux_data = aux_data
        self = cls.__new__(cls)
        for k, v in zip(params_keys, params):
            setattr(self, k, v)

        for k, v in aux_data.items():
            setattr(self, k, v)
        return self


class ParameterFreeTransform(Transform):
    def tree_flatten(self):
        return (), ((), dict())

    def __eq__(self, other):
        return isinstance(other, type(self))


class _InverseTransform(Transform):
    def __init__(self, transform):
        super().__init__()
        self._inv = transform

    @property
    def domain(self):
        return self._inv.codomain

    @property
    def codomain(self):
        return self._inv.domain

    @property
    def sign(self):
        return self._inv.sign

    @property
    def inv(self):
        return self._inv

    def __call__(self, x):
        return self._inv._inverse(x)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        # NB: we don't use intermediates for inverse transform
        return -self._inv.log_abs_det_jacobian(y, x, None)

    def forward_shape(self, shape):
        return self._inv.inverse_shape(shape)

    def inverse_shape(self, shape):
        return self._inv.forward_shape(shape)

    def tree_flatten(self):
        return (self._inv,), (("_inv",), dict())

    def __eq__(self, other):
        if not isinstance(other, _InverseTransform):
            return False
        return self._inv == other._inv


class AbsTransform(ParameterFreeTransform):
    domain = constraints.real
    codomain = constraints.positive

    def __eq__(self, other):
        return isinstance(other, AbsTransform)

    def __call__(self, x):
        return jnp.abs(x)

    def _inverse(self, y):
        warnings.warn(
            "AbsTransform is not a bijective transform."
            " The inverse of `y` will be `y`.",
            stacklevel=find_stack_level(),
        )
        return y


class AffineTransform(Transform):
    """
    .. note:: When `scale` is a JAX tracer, we always assume that `scale > 0`
        when calculating `codomain`.
    """

    def __init__(self, loc, scale, domain=constraints.real):
        self.loc = loc
        self.scale = scale
        self.domain = domain

    @property
    def codomain(self):
        if self.domain is constraints.real:
            return constraints.real
        elif isinstance(self.domain, constraints.greater_than):
            if not_jax_tracer(self.scale) and np.all(np.less(self.scale, 0)):
                return constraints.less_than(self(self.domain.lower_bound))
            # we suppose scale > 0 for any tracer
            else:
                return constraints.greater_than(self(self.domain.lower_bound))
        elif isinstance(self.domain, constraints.less_than):
            if not_jax_tracer(self.scale) and np.all(np.less(self.scale, 0)):
                return constraints.greater_than(self(self.domain.upper_bound))
            # we suppose scale > 0 for any tracer
            else:
                return constraints.less_than(self(self.domain.upper_bound))
        elif isinstance(self.domain, constraints.interval):
            if not_jax_tracer(self.scale) and np.all(np.less(self.scale, 0)):
                return constraints.interval(
                    self(self.domain.upper_bound), self(self.domain.lower_bound)
                )
            else:
                return constraints.interval(
                    self(self.domain.lower_bound), self(self.domain.upper_bound)
                )
        else:
            raise NotImplementedError

    @property
    def sign(self):
        return jnp.sign(self.scale)

    def __call__(self, x):
        return self.loc + self.scale * x

    def _inverse(self, y):
        return (y - self.loc) / self.scale

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return jnp.broadcast_to(jnp.log(jnp.abs(self.scale)), jnp.shape(x))

    def forward_shape(self, shape):
        return lax.broadcast_shapes(
            shape, getattr(self.loc, "shape", ()), getattr(self.scale, "shape", ())
        )

    def inverse_shape(self, shape):
        return lax.broadcast_shapes(
            shape, getattr(self.loc, "shape", ()), getattr(self.scale, "shape", ())
        )

    def tree_flatten(self):
        return (self.loc, self.scale, self.domain), (("loc", "scale", "domain"), dict())

    def __eq__(self, other):
        if not isinstance(other, AffineTransform):
            return False
        return (
            jnp.array_equal(self.loc, other.loc)
            & jnp.array_equal(self.scale, other.scale)
            & (self.domain == other.domain)
        )


def _get_compose_transform_input_event_dim(parts):
    input_event_dim = parts[-1].domain.event_dim
    for part in parts[len(parts) - 1 :: -1]:
        input_event_dim = part.domain.event_dim + max(
            input_event_dim - part.codomain.event_dim, 0
        )
    return input_event_dim


def _get_compose_transform_output_event_dim(parts):
    output_event_dim = parts[0].codomain.event_dim
    for part in parts[1:]:
        output_event_dim = part.codomain.event_dim + max(
            output_event_dim - part.domain.event_dim, 0
        )
    return output_event_dim


class ComposeTransform(Transform):
    def __init__(self, parts):
        self.parts = parts

    @property
    def domain(self):
        input_event_dim = _get_compose_transform_input_event_dim(self.parts)
        first_input_event_dim = self.parts[0].domain.event_dim
        assert input_event_dim >= first_input_event_dim
        if input_event_dim == first_input_event_dim:
            return self.parts[0].domain
        else:
            return constraints.independent(
                self.parts[0].domain, input_event_dim - first_input_event_dim
            )

    @property
    def codomain(self):
        output_event_dim = _get_compose_transform_output_event_dim(self.parts)
        last_output_event_dim = self.parts[-1].codomain.event_dim
        assert output_event_dim >= last_output_event_dim
        if output_event_dim == last_output_event_dim:
            return self.parts[-1].codomain
        else:
            return constraints.independent(
                self.parts[-1].codomain, output_event_dim - last_output_event_dim
            )

    @property
    def sign(self):
        sign = 1
        for transform in self.parts:
            sign *= transform.sign
        return sign

    def __call__(self, x):
        for part in self.parts:
            x = part(x)
        return x

    def _inverse(self, y):
        for part in self.parts[::-1]:
            y = part.inv(y)
        return y

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        if intermediates is not None:
            if len(intermediates) != len(self.parts):
                raise ValueError(
                    "Intermediates array has length = {}. Expected = {}.".format(
                        len(intermediates), len(self.parts)
                    )
                )

        result = 0.0
        input_event_dim = self.domain.event_dim
        for i, part in enumerate(self.parts[:-1]):
            y_tmp = part(x) if intermediates is None else intermediates[i][0]
            inter = None if intermediates is None else intermediates[i][1]
            logdet = part.log_abs_det_jacobian(x, y_tmp, intermediates=inter)
            batch_ndim = input_event_dim - part.domain.event_dim
            result = result + sum_rightmost(logdet, batch_ndim)
            input_event_dim = part.codomain.event_dim + batch_ndim
            x = y_tmp
        # account the the last transform, where y is available
        inter = None if intermediates is None else intermediates[-1]
        part = self.parts[-1]
        logdet = part.log_abs_det_jacobian(x, y, intermediates=inter)
        result = result + sum_rightmost(logdet, input_event_dim - part.domain.event_dim)
        return result

    def call_with_intermediates(self, x):
        intermediates = []
        for part in self.parts[:-1]:
            x, inter = part.call_with_intermediates(x)
            intermediates.append([x, inter])
        # NB: we don't need to hold the last output value in `intermediates`
        x, inter = self.parts[-1].call_with_intermediates(x)
        intermediates.append(inter)
        return x, intermediates

    def forward_shape(self, shape):
        for part in self.parts:
            shape = part.forward_shape(shape)
        return shape

    def inverse_shape(self, shape):
        for part in reversed(self.parts):
            shape = part.inverse_shape(shape)
        return shape

    def tree_flatten(self):
        return (self.parts,), (("parts",), {})

    def __eq__(self, other):
        if not isinstance(other, ComposeTransform):
            return False
        return jnp.logical_and(*(p1 == p2 for p1, p2 in zip(self.parts, other.parts)))


def _matrix_forward_shape(shape, offset=0):
    # Reshape from (..., N) to (..., D, D).
    if len(shape) < 1:
        raise ValueError("Too few dimensions in input")
    N = shape[-1]
    D = round((0.25 + 2 * N) ** 0.5 - 0.5)
    if D * (D + 1) // 2 != N:
        raise ValueError("Input is not a flattened lower-diagonal number")
    D = D - offset
    return shape[:-1] + (D, D)


def _matrix_inverse_shape(shape, offset=0):
    # Reshape from (..., D, D) to (..., N).
    if len(shape) < 2:
        raise ValueError("Too few dimensions on input")
    if shape[-2] != shape[-1]:
        raise ValueError("Input is not square")
    D = shape[-1] + offset
    N = D * (D + 1) // 2
    return shape[:-2] + (N,)


class CholeskyTransform(ParameterFreeTransform):
    r"""
    Transform via the mapping :math:`y = cholesky(x)`, where `x` is a
    positive definite matrix.
    """

    domain = constraints.positive_definite
    codomain = constraints.lower_cholesky

    def __call__(self, x):
        return jnp.linalg.cholesky(x)

    def _inverse(self, y):
        return jnp.matmul(y, jnp.swapaxes(y, -2, -1))

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        # Ref: http://web.mit.edu/18.325/www/handouts/handout2.pdf page 13
        n = jnp.shape(x)[-1]
        order = -jnp.arange(n, 0, -1)
        return -n * jnp.log(2) + jnp.sum(
            order * jnp.log(jnp.diagonal(y, axis1=-2, axis2=-1)), axis=-1
        )


class CorrCholeskyTransform(ParameterFreeTransform):
    r"""
    Transforms a unconstrained real vector :math:`x` with length :math:`D*(D-1)/2` into the
    Cholesky factor of a D-dimension correlation matrix. This Cholesky factor is a lower
    triangular matrix with positive diagonals and unit Euclidean norm for each row.
    The transform is processed as follows:

        1. First we convert :math:`x` into a lower triangular matrix with the following order:

        .. math::
            \begin{bmatrix}
                1   & 0 & 0 & 0 \\
                x_0 & 1 & 0 & 0 \\
                x_1 & x_2 & 1 & 0 \\
                x_3 & x_4 & x_5 & 1
            \end{bmatrix}

        2. For each row :math:`X_i` of the lower triangular part, we apply a *signed* version of
        class :class:`StickBreakingTransform` to transform :math:`X_i` into a
        unit Euclidean length vector using the following steps:

            a. Scales into the interval :math:`(-1, 1)` domain: :math:`r_i = \tanh(X_i)`.
            b. Transforms into an unsigned domain: :math:`z_i = r_i^2`.
            c. Applies :math:`s_i = StickBreakingTransform(z_i)`.
            d. Transforms back into signed domain: :math:`y_i = (sign(r_i), 1) * \sqrt{s_i}`.
    """

    domain = constraints.real_vector
    codomain = constraints.corr_cholesky

    def __call__(self, x):
        # we interchange step 1 and step 2.a for a better performance
        t = jnp.tanh(x)
        return signed_stick_breaking_tril(t)

    def _inverse(self, y):
        # inverse stick-breaking
        z1m_cumprod = 1 - jnp.cumsum(y * y, axis=-1)
        pad_width = [(0, 0)] * y.ndim
        pad_width[-1] = (1, 0)
        z1m_cumprod_shifted = jnp.pad(
            z1m_cumprod[..., :-1], pad_width, mode="constant", constant_values=1.0
        )
        t = matrix_to_tril_vec(y, diagonal=-1) / jnp.sqrt(
            matrix_to_tril_vec(z1m_cumprod_shifted, diagonal=-1)
        )
        # inverse of tanh
        return jnp.arctanh(t)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        # NB: because domain and codomain are two spaces with different dimensions, determinant of
        # Jacobian is not well-defined. Here we return `log_abs_det_jacobian` of `x` and the
        # flatten lower triangular part of `y`.

        # stick_breaking_logdet = log(y / r) = log(z_cumprod)  (modulo right shifted)
        z1m_cumprod = 1 - jnp.cumsum(y * y, axis=-1)
        # by taking diagonal=-2, we don't need to shift z_cumprod to the right
        # NB: diagonal=-2 works fine for (2 x 2) matrix, where we get an empty array
        z1m_cumprod_tril = matrix_to_tril_vec(z1m_cumprod, diagonal=-2)
        stick_breaking_logdet = 0.5 * jnp.sum(jnp.log(z1m_cumprod_tril), axis=-1)

        tanh_logdet = -2 * jnp.sum(x + softplus(-2 * x) - jnp.log(2.0), axis=-1)
        return stick_breaking_logdet + tanh_logdet

    def forward_shape(self, shape):
        return _matrix_forward_shape(shape, offset=-1)

    def inverse_shape(self, shape):
        return _matrix_inverse_shape(shape, offset=-1)


class CorrMatrixCholeskyTransform(CholeskyTransform):
    r"""
    Transform via the mapping :math:`y = cholesky(x)`, where `x` is a
    correlation matrix.
    """

    domain = constraints.corr_matrix
    codomain = constraints.corr_cholesky

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        # NB: see derivation in LKJCholesky implementation
        n = jnp.shape(x)[-1]
        order = -jnp.arange(n - 1, -1, -1)
        return jnp.sum(order * jnp.log(jnp.diagonal(y, axis1=-2, axis2=-1)), axis=-1)


class ExpTransform(Transform):
    sign = 1

    # TODO: refine domain/codomain logic through setters, especially when
    # transforms for inverses are supported
    def __init__(self, domain=constraints.real):
        self.domain = domain

    @property
    def codomain(self):
        if self.domain is constraints.ordered_vector:
            return constraints.positive_ordered_vector
        elif self.domain is constraints.real:
            return constraints.positive
        elif isinstance(self.domain, constraints.greater_than):
            return constraints.greater_than(self.__call__(self.domain.lower_bound))
        elif isinstance(self.domain, constraints.interval):
            return constraints.interval(
                self.__call__(self.domain.lower_bound),
                self.__call__(self.domain.upper_bound),
            )
        else:
            raise NotImplementedError

    def __call__(self, x):
        # XXX consider to clamp from below for stability if necessary
        return jnp.exp(x)

    def _inverse(self, y):
        return jnp.log(y)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return x

    def tree_flatten(self):
        return (self.domain,), (("domain",), dict())

    def __eq__(self, other):
        if not isinstance(other, ExpTransform):
            return False
        return self.domain == other.domain


class IdentityTransform(ParameterFreeTransform):
    sign = 1

    def __call__(self, x):
        return x

    def _inverse(self, y):
        return y

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return jnp.zeros_like(x)


class IndependentTransform(Transform):
    """
    Wraps a transform by aggregating over ``reinterpreted_batch_ndims``-many
    dims in :meth:`check`, so that an event is valid only if all its
    independent entries are valid.
    """

    def __init__(self, base_transform, reinterpreted_batch_ndims):
        assert isinstance(base_transform, Transform)
        assert isinstance(reinterpreted_batch_ndims, int)
        assert reinterpreted_batch_ndims >= 0
        self.base_transform = base_transform
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        super().__init__()

    @property
    def domain(self):
        return constraints.independent(
            self.base_transform.domain, self.reinterpreted_batch_ndims
        )

    @property
    def codomain(self):
        return constraints.independent(
            self.base_transform.codomain, self.reinterpreted_batch_ndims
        )

    def __call__(self, x):
        return self.base_transform(x)

    def _inverse(self, y):
        return self.base_transform._inverse(y)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        result = self.base_transform.log_abs_det_jacobian(
            x, y, intermediates=intermediates
        )
        if jnp.ndim(result) < self.reinterpreted_batch_ndims:
            expected = self.domain.event_dim
            raise ValueError(f"Expected x.dim() >= {expected} but got {jnp.ndim(x)}")
        return sum_rightmost(result, self.reinterpreted_batch_ndims)

    def call_with_intermediates(self, x):
        return self.base_transform.call_with_intermediates(x)

    def forward_shape(self, shape):
        return self.base_transform.forward_shape(shape)

    def inverse_shape(self, shape):
        return self.base_transform.inverse_shape(shape)

    def tree_flatten(self):
        return (self.base_transform, self.reinterpreted_batch_ndims), (
            ("base_transform", "reinterpreted_batch_ndims"),
            dict(),
        )

    def __eq__(self, other):
        if not isinstance(other, IndependentTransform):
            return False
        return (self.base_transform == other.base_transform) & (
            self.reinterpreted_batch_ndims == other.reinterpreted_batch_ndims
        )


class L1BallTransform(ParameterFreeTransform):
    r"""
    Transforms a unconstrained real vector :math:`x` into the unit L1 ball.
    """

    domain = constraints.real_vector
    codomain = constraints.l1_ball

    def __call__(self, x):
        # transform to (-1, 1) interval
        t = jnp.tanh(x)

        # apply stick-breaking transform
        remainder = jnp.cumprod(1 - jnp.abs(t[..., :-1]), axis=-1)
        pad_width = [(0, 0)] * (t.ndim - 1) + [(1, 0)]
        remainder = jnp.pad(remainder, pad_width, mode="constant", constant_values=1.0)
        return t * remainder

    def _inverse(self, y):
        # inverse stick-breaking
        remainder = 1 - jnp.cumsum(jnp.abs(y[..., :-1]), axis=-1)
        pad_width = [(0, 0)] * (y.ndim - 1) + [(1, 0)]
        remainder = jnp.pad(remainder, pad_width, mode="constant", constant_values=1.0)
        finfo = jnp.finfo(y.dtype)
        remainder = jnp.clip(remainder, finfo.tiny)
        t = y / remainder

        # inverse of tanh
        t = jnp.clip(t, -1 + finfo.eps, 1 - finfo.eps)
        return jnp.arctanh(t)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        # compute stick-breaking logdet
        #   t1 -> t1
        #   t2 -> t2 * (1 - abs(t1))
        #   t3 -> t3 * (1 - abs(t1)) * (1 - abs(t2))
        # hence jacobian is triangular and logdet is the sum of the log
        # of the diagonal part of the jacobian
        one_minus_remainder = jnp.cumsum(jnp.abs(y[..., :-1]), axis=-1)
        eps = jnp.finfo(y.dtype).eps
        one_minus_remainder = jnp.clip(one_minus_remainder, None, 1 - eps)
        # log(remainder) = log1p(remainder - 1)
        stick_breaking_logdet = jnp.sum(jnp.log1p(-one_minus_remainder), axis=-1)

        tanh_logdet = -2 * jnp.sum(x + softplus(-2 * x) - jnp.log(2.0), axis=-1)
        return stick_breaking_logdet + tanh_logdet


class LowerCholeskyAffine(Transform):
    r"""
    Transform via the mapping :math:`y = loc + scale\_tril\ @\ x`.

    :param loc: a real vector.
    :param scale_tril: a lower triangular matrix with positive diagonal.

    **Example**

    .. doctest::

       >>> import jax.numpy as jnp
       >>> from numpyro.distributions.transforms import LowerCholeskyAffine
       >>> base = jnp.ones(2)
       >>> loc = jnp.zeros(2)
       >>> scale_tril = jnp.array([[0.3, 0.0], [1.0, 0.5]])
       >>> affine = LowerCholeskyAffine(loc=loc, scale_tril=scale_tril)
       >>> affine(base)
       Array([0.3, 1.5], dtype=float32)
    """

    domain = constraints.real_vector
    codomain = constraints.real_vector

    def __init__(self, loc, scale_tril):
        if jnp.ndim(scale_tril) != 2:
            raise ValueError(
                "Only support 2-dimensional scale_tril matrix. "
                "Please make a feature request if you need to "
                "use this transform with batched scale_tril."
            )
        self.loc = loc
        self.scale_tril = scale_tril

    def __call__(self, x):
        return self.loc + jnp.squeeze(
            jnp.matmul(self.scale_tril, x[..., jnp.newaxis]), axis=-1
        )

    def _inverse(self, y):
        y = y - self.loc
        original_shape = jnp.shape(y)
        yt = jnp.reshape(y, (-1, original_shape[-1])).T
        xt = solve_triangular(self.scale_tril, yt, lower=True)
        return jnp.reshape(xt.T, original_shape)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return jnp.broadcast_to(
            jnp.log(jnp.diagonal(self.scale_tril, axis1=-2, axis2=-1)).sum(-1),
            jnp.shape(x)[:-1],
        )

    def forward_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return lax.broadcast_shapes(shape, self.loc.shape, self.scale_tril.shape[:-1])

    def inverse_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return lax.broadcast_shapes(shape, self.loc.shape, self.scale_tril.shape[:-1])

    def tree_flatten(self):
        return (self.loc, self.scale_tril), (("loc", "scale_tril"), dict())

    def __eq__(self, other):
        if not isinstance(other, LowerCholeskyAffine):
            return False
        return jnp.array_equal(self.loc, other.loc) & jnp.array_equal(
            self.scale_tril, other.scale_tril
        )


class LowerCholeskyTransform(ParameterFreeTransform):
    """
    Transform a real vector to a lower triangular cholesky
    factor, where the strictly lower triangular submatrix is
    unconstrained and the diagonal is parameterized with an
    exponential transform.
    """

    domain = constraints.real_vector
    codomain = constraints.lower_cholesky

    def __call__(self, x):
        n = round((math.sqrt(1 + 8 * x.shape[-1]) - 1) / 2)
        z = vec_to_tril_matrix(x[..., :-n], diagonal=-1)
        diag = jnp.exp(x[..., -n:])
        return add_diag(z, diag)

    def _inverse(self, y):
        z = matrix_to_tril_vec(y, diagonal=-1)
        return jnp.concatenate(
            [z, jnp.log(jnp.diagonal(y, axis1=-2, axis2=-1))], axis=-1
        )

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        # the jacobian is diagonal, so logdet is the sum of diagonal `exp` transform
        n = round((math.sqrt(1 + 8 * x.shape[-1]) - 1) / 2)
        return x[..., -n:].sum(-1)

    def forward_shape(self, shape):
        return _matrix_forward_shape(shape)

    def inverse_shape(self, shape):
        return _matrix_inverse_shape(shape)


class ScaledUnitLowerCholeskyTransform(LowerCholeskyTransform):
    r"""
    Like `LowerCholeskyTransform` this `Transform` transforms
    a real vector to a lower triangular cholesky factor. However
    it does so via a decomposition

    :math:`y = loc + unit\_scale\_tril\ @\ scale\_diag\ @\ x`.

    where :math:`unit\_scale\_tril` has ones along the diagonal
    and :math:`scale\_diag` is a diagonal matrix with all positive
    entries that is parameterized with a softplus transform.
    """

    domain = constraints.real_vector
    codomain = constraints.scaled_unit_lower_cholesky

    def __call__(self, x):
        n = round((math.sqrt(1 + 8 * x.shape[-1]) - 1) / 2)
        z = vec_to_tril_matrix(x[..., :-n], diagonal=-1)
        diag = softplus(x[..., -n:])
        return add_diag(z, 1) * diag[..., None]

    def _inverse(self, y):
        diag = jnp.diagonal(y, axis1=-2, axis2=-1)
        z = matrix_to_tril_vec(y / diag[..., None], diagonal=-1)
        return jnp.concatenate([z, _softplus_inv(diag)], axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        n = round((math.sqrt(1 + 8 * x.shape[-1]) - 1) / 2)
        diag = x[..., -n:]
        diag_softplus = jnp.diagonal(y, axis1=-2, axis2=-1)
        return (jnp.log(diag_softplus) * jnp.arange(n) - softplus(-diag)).sum(-1)


class OrderedTransform(ParameterFreeTransform):
    """
    Transform a real vector to an ordered vector.

    **References:**

    1. *Stan Reference Manual v2.20, section 10.6*,
       Stan Development Team

    **Example**

    .. doctest::

       >>> import jax.numpy as jnp
       >>> from numpyro.distributions.transforms import OrderedTransform
       >>> base = jnp.ones(3)
       >>> transform = OrderedTransform()
       >>> assert jnp.allclose(transform(base), jnp.array([1., 3.7182817, 6.4365635]), rtol=1e-3, atol=1e-3)

    """

    domain = constraints.real_vector
    codomain = constraints.ordered_vector

    def __call__(self, x):
        z = jnp.concatenate([x[..., :1], jnp.exp(x[..., 1:])], axis=-1)
        return jnp.cumsum(z, axis=-1)

    def _inverse(self, y):
        x = jnp.log(y[..., 1:] - y[..., :-1])
        return jnp.concatenate([y[..., :1], x], axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return jnp.sum(x[..., 1:], -1)


class PermuteTransform(Transform):
    domain = constraints.real_vector
    codomain = constraints.real_vector

    def __init__(self, permutation):
        self.permutation = permutation

    def __call__(self, x):
        return x[..., self.permutation]

    def _inverse(self, y):
        size = self.permutation.size
        permutation_inv = (
            jnp.zeros(size, dtype=jnp.result_type(int))
            .at[self.permutation]
            .set(jnp.arange(size))
        )
        return y[..., permutation_inv]

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return jnp.full(jnp.shape(x)[:-1], 0.0)

    def tree_flatten(self):
        return (self.permutation,), (("permutation",), dict())

    def __eq__(self, other):
        if not isinstance(other, PermuteTransform):
            return False
        return jnp.array_equal(self.permutation, other.permutation)


class PowerTransform(Transform):
    domain = constraints.positive
    codomain = constraints.positive

    def __init__(self, exponent):
        self.exponent = exponent

    def __call__(self, x):
        return jnp.power(x, self.exponent)

    def _inverse(self, y):
        return jnp.power(y, 1 / self.exponent)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return jnp.log(jnp.abs(self.exponent * y / x))

    def forward_shape(self, shape):
        return lax.broadcast_shapes(shape, getattr(self.exponent, "shape", ()))

    def inverse_shape(self, shape):
        return lax.broadcast_shapes(shape, getattr(self.exponent, "shape", ()))

    def tree_flatten(self):
        return (self.exponent,), (("exponent",), dict())

    def __eq__(self, other):
        if not isinstance(other, PowerTransform):
            return False
        return jnp.array_equal(self.exponent, other.exponent)

    @property
    def sign(self):
        return jnp.sign(self.exponent)


class SigmoidTransform(ParameterFreeTransform):
    codomain = constraints.unit_interval
    sign = 1

    def __call__(self, x):
        return _clipped_expit(x)

    def _inverse(self, y):
        return logit(y)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return -softplus(x) - softplus(-x)


class SimplexToOrderedTransform(Transform):
    """
    Transform a simplex into an ordered vector (via difference in Logistic CDF between cutpoints)
    Used in [1] to induce a prior on latent cutpoints via transforming ordered category probabilities.

    :param anchor_point: Anchor point is a nuisance parameter to improve the identifiability of the transform.
        For simplicity, we assume it is a scalar value, but it is broadcastable x.shape[:-1].
        For more details please refer to Section 2.2 in [1]

    **References:**

    1. *Ordinal Regression Case Study, section 2.2*,
       M. Betancourt, https://betanalpha.github.io/assets/case_studies/ordinal_regression.html

    **Example**

    .. doctest::

       >>> import jax.numpy as jnp
       >>> from numpyro.distributions.transforms import SimplexToOrderedTransform
       >>> base = jnp.array([0.3, 0.1, 0.4, 0.2])
       >>> transform = SimplexToOrderedTransform()
       >>> assert jnp.allclose(transform(base), jnp.array([-0.8472978, -0.40546507, 1.3862944]), rtol=1e-3, atol=1e-3)

    """

    domain = constraints.simplex
    codomain = constraints.ordered_vector

    def __init__(self, anchor_point=0.0):
        self.anchor_point = anchor_point

    def __call__(self, x):
        s = jnp.cumsum(x[..., :-1], axis=-1)
        y = logit(s) + jnp.expand_dims(self.anchor_point, -1)
        return y

    def _inverse(self, y):
        y = y - jnp.expand_dims(self.anchor_point, -1)
        s = expit(y)
        # x0 = s0, x1 = s1 - s0, x2 = s2 - s1,..., xn = 1 - s[n-1]
        # add two boundary points 0 and 1
        pad_width = [(0, 0)] * (jnp.ndim(s) - 1) + [(1, 1)]
        s = jnp.pad(s, pad_width, constant_values=(0, 1))
        x = s[..., 1:] - s[..., :-1]
        return x

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        # |dp/dc| = |dx/dy| = prod(ds/dy) = prod(expit'(y))
        # we know log derivative of expit(y) is `-softplus(y) - softplus(-y)`
        J_logdet = (softplus(y) + softplus(-y)).sum(-1)
        return J_logdet

    def tree_flatten(self):
        return (self.anchor_point,), (("anchor_point",), dict())

    def __eq__(self, other):
        if not isinstance(other, SimplexToOrderedTransform):
            return False
        return jnp.array_equal(self.anchor_point, other.anchor_point)

    def forward_shape(self, shape):
        return shape[:-1] + (shape[-1] - 1,)

    def inverse_shape(self, shape):
        return shape[:-1] + (shape[-1] + 1,)


def _softplus_inv(y):
    return jnp.log(-jnp.expm1(-y)) + y


class SoftplusTransform(ParameterFreeTransform):
    r"""
    Transform from unconstrained space to positive domain via softplus :math:`y = \log(1 + \exp(x))`.
    The inverse is computed as :math:`x = \log(\exp(y) - 1)`.
    """

    domain = constraints.real
    codomain = constraints.softplus_positive
    sign = 1

    def __call__(self, x):
        return softplus(x)

    def _inverse(self, y):
        return _softplus_inv(y)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return -softplus(-x)


class SoftplusLowerCholeskyTransform(ParameterFreeTransform):
    """
    Transform from unconstrained vector to lower-triangular matrices with
    nonnegative diagonal entries. This is useful for parameterizing positive
    definite matrices in terms of their Cholesky factorization.
    """

    domain = constraints.real_vector
    codomain = constraints.softplus_lower_cholesky

    def __call__(self, x):
        n = round((math.sqrt(1 + 8 * x.shape[-1]) - 1) / 2)
        z = vec_to_tril_matrix(x[..., :-n], diagonal=-1)
        diag = softplus(x[..., -n:])
        return z + jnp.expand_dims(diag, axis=-1) * jnp.identity(n)

    def _inverse(self, y):
        z = matrix_to_tril_vec(y, diagonal=-1)
        diag = _softplus_inv(jnp.diagonal(y, axis1=-2, axis2=-1))
        return jnp.concatenate([z, diag], axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        # the jacobian is diagonal, so logdet is the sum of diagonal
        # `softplus` transform
        n = round((math.sqrt(1 + 8 * x.shape[-1]) - 1) / 2)
        return -softplus(-x[..., -n:]).sum(-1)

    def forward_shape(self, shape):
        return _matrix_forward_shape(shape)

    def inverse_shape(self, shape):
        return _matrix_inverse_shape(shape)


class StickBreakingTransform(ParameterFreeTransform):
    domain = constraints.real_vector
    codomain = constraints.simplex

    def __call__(self, x):
        # we shift x to obtain a balanced mapping (0, 0, ..., 0) -> (1/K, 1/K, ..., 1/K)
        x = x - jnp.log(x.shape[-1] - jnp.arange(x.shape[-1]))
        # convert to probabilities (relative to the remaining) of each fraction of the stick
        z = _clipped_expit(x)
        z1m_cumprod = jnp.cumprod(1 - z, axis=-1)
        pad_width = [(0, 0)] * x.ndim
        pad_width[-1] = (0, 1)
        z_padded = jnp.pad(z, pad_width, mode="constant", constant_values=1.0)
        pad_width = [(0, 0)] * x.ndim
        pad_width[-1] = (1, 0)
        z1m_cumprod_shifted = jnp.pad(
            z1m_cumprod, pad_width, mode="constant", constant_values=1.0
        )
        return z_padded * z1m_cumprod_shifted

    def _inverse(self, y):
        y_crop = y[..., :-1]
        z1m_cumprod = jnp.clip(1 - jnp.cumsum(y_crop, axis=-1), jnp.finfo(y.dtype).tiny)
        # hence x = logit(z) = log(z / (1 - z)) = y[::-1] / z1m_cumprod
        x = jnp.log(y_crop / z1m_cumprod)
        return x + jnp.log(x.shape[-1] - jnp.arange(x.shape[-1]))

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        # Ref: https://mc-stan.org/docs/2_19/reference-manual/simplex-transform-section.html
        # |det|(J) = Product(y * (1 - sigmoid(x)))
        #          = Product(y * sigmoid(x) * exp(-x))
        x = x - jnp.log(x.shape[-1] - jnp.arange(x.shape[-1]))
        return jnp.sum(jnp.log(y[..., :-1]) + (log_sigmoid(x) - x), axis=-1)

    def forward_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return shape[:-1] + (shape[-1] + 1,)

    def inverse_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return shape[:-1] + (shape[-1] - 1,)


class UnpackTransform(Transform):
    """
    Transforms a contiguous array to a pytree of subarrays.

    :param unpack_fn: callable used to unpack a contiguous array.
    :param pack_fn: callable used to pack a pytree into a contiguous array.
    """

    domain = constraints.real_vector
    codomain = constraints.dependent

    def __init__(self, unpack_fn, pack_fn=None):
        self.unpack_fn = unpack_fn
        self.pack_fn = pack_fn

    def __call__(self, x):
        batch_shape = x.shape[:-1]
        if batch_shape:
            unpacked = vmap(self.unpack_fn)(x.reshape((-1,) + x.shape[-1:]))
            return jax.tree.map(
                lambda z: jnp.reshape(z, batch_shape + z.shape[1:]), unpacked
            )
        else:
            return self.unpack_fn(x)

    def _inverse(self, y):
        if self.pack_fn is None:
            raise NotImplementedError(
                "pack_fn needs to be provided to perform UnpackTransform.inv."
            )
        leading_dims = [
            v.shape[0] if jnp.ndim(v) > 0 else 0 for v in jax.tree.flatten(y)[0]
        ]
        if not leading_dims:
            return jnp.array([])
        d0 = leading_dims[0]
        not_scalar = d0 > 0 or len(leading_dims) > 1
        if not_scalar and all(d == d0 for d in leading_dims[1:]):
            warnings.warn(
                "UnpackTransform.inv might lead to an unexpected behavior because it"
                " cannot transform a batch of unpacked arrays.",
                stacklevel=find_stack_level(),
            )
        return self.pack_fn(y)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return jnp.zeros(jnp.shape(x)[:-1])

    def forward_shape(self, shape):
        raise NotImplementedError

    def inverse_shape(self, shape):
        raise NotImplementedError

    def tree_flatten(self):
        # XXX: what if unpack_fn is a parametrized callable pytree?
        return (), ((), {"unpack_fn": self.unpack_fn, "pack_fn": self.pack_fn})

    def __eq__(self, other):
        return (
            isinstance(other, UnpackTransform)
            and (self.unpack_fn is other.unpack_fn)
            and (self.pack_fn is other.pack_fn)
        )


def _get_target_shape(shape, forward_shape, inverse_shape):
    batch_ndims = len(shape) - len(inverse_shape)
    return shape[:batch_ndims] + forward_shape


class ReshapeTransform(Transform):
    """
    Reshape a sample, leaving batch dimensions unchanged.

    :param forward_shape: Shape to transform the sample to.
    :param inverse_shape: Shape of the sample for the inverse transform.
    """

    domain = constraints.real
    codomain = constraints.real
    sign = 1

    def __init__(self, forward_shape, inverse_shape) -> None:
        forward_size = math.prod(forward_shape)
        inverse_size = math.prod(inverse_shape)
        if forward_size != inverse_size:
            raise ValueError(
                f"forward shape {forward_shape} (size {forward_size}) and inverse "
                f"shape {inverse_shape} (size {inverse_size}) are not compatible"
            )
        self._forward_shape = forward_shape
        self._inverse_shape = inverse_shape

    def forward_shape(self, shape):
        return _get_target_shape(shape, self._forward_shape, self._inverse_shape)

    def inverse_shape(self, shape):
        return _get_target_shape(shape, self._inverse_shape, self._forward_shape)

    def __call__(self, x):
        return jnp.reshape(x, self.forward_shape(jnp.shape(x)))

    def _inverse(self, y):
        return jnp.reshape(y, self.inverse_shape(jnp.shape(y)))

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return jnp.zeros_like(x, shape=x.shape[: x.ndim - len(self._inverse_shape)])

    def tree_flatten(self):
        aux_data = {
            "_forward_shape": self._forward_shape,
            "_inverse_shape": self._inverse_shape,
        }
        return (), ((), aux_data)

    def __eq__(self, other):
        return (
            isinstance(other, ReshapeTransform)
            and self._forward_shape == other._forward_shape
            and self._inverse_shape == other._inverse_shape
        )


def _normalize_rfft_shape(input_shape, shape):
    if shape is None:
        return input_shape
    return input_shape[: len(input_shape) - len(shape)] + shape


class RealFastFourierTransform(Transform):
    """
    N-dimensional discrete fast Fourier transform for real input.

    :param transform_shape: Length of each transformed axis to use from the input,
        defaults to the input size.
    :param transform_ndims: Number of trailing dimensions to transform.
    """

    def __init__(
        self,
        transform_shape=None,
        transform_ndims=1,
    ) -> None:
        if isinstance(transform_shape, int):
            transform_shape = (transform_shape,)
        if transform_shape is not None and len(transform_shape) != transform_ndims:
            raise ValueError(
                f"Length of transform shape ({transform_shape}) does not match number "
                f"of dimensions to transform ({transform_ndims})."
            )
        self.transform_shape = transform_shape
        self.transform_ndims = transform_ndims

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        axes = tuple(range(-self.transform_ndims, 0))
        return jnp.fft.rfftn(x, self.transform_shape, axes)

    def _inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        axes = tuple(range(-self.transform_ndims, 0))
        return jnp.fft.irfftn(y, self.transform_shape, axes)

    def forward_shape(self, shape: tuple) -> tuple:
        # Dimensions remain unchanged except the last transformed dimension.
        shape = _normalize_rfft_shape(shape, self.transform_shape)
        return shape[:-1] + (shape[-1] // 2 + 1,)

    def inverse_shape(self, shape: tuple) -> tuple:
        if self.transform_shape:
            return _normalize_rfft_shape(shape, self.transform_shape)
        size = 2 * (shape[-1] - 1)
        return shape[:-1] + (size,)

    def log_abs_det_jacobian(
        self, x: jnp.ndarray, y: jnp.ndarray, intermediates: None = None
    ) -> jnp.ndarray:
        shape = jnp.broadcast_shapes(
            x.shape[: -self.transform_ndims], y.shape[: -self.transform_ndims]
        )
        return jnp.zeros_like(x, shape=shape)

    def tree_flatten(self):
        aux_data = {
            "transform_shape": self.transform_shape,
            "transform_ndims": self.transform_ndims,
        }
        return (), ((), aux_data)

    @property
    def domain(self) -> constraints.Constraint:
        return constraints.independent(constraints.real, self.transform_ndims)

    @property
    def codomain(self) -> constraints.Constraint:
        return constraints.independent(constraints.complex, self.transform_ndims)

    def __eq__(self, other):
        return (
            isinstance(other, RealFastFourierTransform)
            and self.transform_ndims == other.transform_ndims
            and self.transform_shape == other.transform_shape
        )


class RecursiveLinearTransform(Transform):
    """
    Apply a linear transformation recursively such that
    :math:`y_t = A y_{t - 1} + x_t` for :math:`t > 0`, where :math:`x_t` and :math:`y_t`
    are vectors and :math:`A` is a square transition matrix. The series is initialized
    by :math:`y_0 = 0`.

    :param transition_matrix: Squared transition matrix :math:`A` for successive states
        or a batch of transition matrices.

    **Example:**

    .. doctest::

        >>> from jax import random
        >>> from jax import numpy as jnp
        >>> import numpyro
        >>> from numpyro import distributions as dist
        >>>
        >>> def cauchy_random_walk():
        ...     return numpyro.sample(
        ...         "x",
        ...         dist.TransformedDistribution(
        ...             dist.Cauchy(0, 1).expand([10, 1]).to_event(1),
        ...             dist.transforms.RecursiveLinearTransform(jnp.eye(1)),
        ...         ),
        ...     )
        >>>
        >>> numpyro.handlers.seed(cauchy_random_walk, 0)().shape
        (10, 1)
        >>>
        >>> def rocket_trajectory():
        ...     scale = numpyro.sample(
        ...         "scale",
        ...         dist.HalfCauchy(1).expand([2]).to_event(1),
        ...     )
        ...     transition_matrix = jnp.array([[1, 1], [0, 1]])
        ...     return numpyro.sample(
        ...         "x",
        ...         dist.TransformedDistribution(
        ...             dist.Normal(0, scale).expand([10, 2]).to_event(1),
        ...             dist.transforms.RecursiveLinearTransform(transition_matrix),
        ...         ),
        ...     )
        >>>
        >>> numpyro.handlers.seed(rocket_trajectory, 0)().shape
        (10, 2)
    """

    domain = constraints.real_matrix
    codomain = constraints.real_matrix

    def __init__(self, transition_matrix: jnp.ndarray) -> None:
        self.transition_matrix = transition_matrix

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Move the time axis to the first position so we can scan over it.
        x = jnp.moveaxis(x, -2, 0)

        def f(y, x):
            y = jnp.einsum("...ij,...j->...i", self.transition_matrix, y) + x
            return y, y

        _, y = lax.scan(f, jnp.zeros_like(x, shape=x.shape[1:]), x)
        return jnp.moveaxis(y, 0, -2)

    def _inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        # Move the time axis to the first position so we can scan over it in reverse.
        y = jnp.moveaxis(y, -2, 0)

        def f(y, prev):
            x = y - jnp.einsum("...ij,...j->...i", self.transition_matrix, prev)
            return prev, x

        _, x = lax.scan(f, y[-1], jnp.roll(y, 1, axis=0).at[0].set(0), reverse=True)
        return jnp.moveaxis(x, 0, -2)

    def log_abs_det_jacobian(self, x: jnp.ndarray, y: jnp.ndarray, intermediates=None):
        return jnp.zeros_like(x, shape=x.shape[:-2])

    def tree_flatten(self):
        return (self.transition_matrix,), (
            ("transition_matrix",),
            {},
        )

    def __eq__(self, other):
        if not isinstance(other, RecursiveLinearTransform):
            return False
        return jnp.array_equal(self.transition_matrix, other.transition_matrix)


class ZeroSumTransform(Transform):
    """A transform that constrains an array to sum to zero, adapted from PyMC [1] as described in [2,3]

    :param transform_ndims: Number of trailing dimensions to transform.

    **References**
    [1] https://github.com/pymc-devs/pymc/blob/244fb97b01ad0f3dadf5c3837b65839e2a59a0e8/pymc/distributions/transforms.py#L266
    [2] https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.ZeroSumNormal.html
    [3] https://learnbayesstats.com/episode/74-optimizing-nuts-developing-zerosumnormal-distribution-adrian-seyboldt/
    """

    def __init__(self, transform_ndims: int = 1) -> None:
        self.transform_ndims = transform_ndims

    @property
    def domain(self) -> constraints.Constraint:
        return constraints.independent(constraints.real, self.transform_ndims)

    @property
    def codomain(self) -> constraints.Constraint:
        return constraints.zero_sum(self.transform_ndims)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        zero_sum_axes = tuple(range(-self.transform_ndims, 0))
        for axis in zero_sum_axes:
            x = self.extend_axis(x, axis=axis)
        return x

    def _inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        zero_sum_axes = tuple(range(-self.transform_ndims, 0))
        for axis in zero_sum_axes:
            y = self.extend_axis_rev(y, axis=axis)
        return y

    def extend_axis_rev(self, array: jnp.ndarray, axis: int) -> jnp.ndarray:
        normalized_axis = axis if axis >= 0 else jnp.ndim(array) + axis

        n = array.shape[normalized_axis]
        last = jnp.take(array, jnp.array([-1]), axis=normalized_axis)

        sum_vals = -last * jnp.sqrt(n)
        norm = sum_vals / (jnp.sqrt(n) + n)
        slice_before = (slice(None, None),) * normalized_axis
        return array[(*slice_before, slice(None, -1))] + norm

    def extend_axis(self, array: jnp.ndarray, axis: int) -> jnp.ndarray:
        n = array.shape[axis] + 1

        sum_vals = array.sum(axis, keepdims=True)
        norm = sum_vals / (jnp.sqrt(n) + n)
        fill_val = norm - sum_vals / jnp.sqrt(n)

        out = jnp.concatenate([array, fill_val], axis=axis)
        return out - norm

    def log_abs_det_jacobian(
        self, x: jnp.ndarray, y: jnp.ndarray, intermediates: None = None
    ) -> jnp.ndarray:
        shape = jnp.broadcast_shapes(
            x.shape[: -self.transform_ndims], y.shape[: -self.transform_ndims]
        )
        return jnp.zeros_like(x, shape=shape)

    def forward_shape(self, shape: tuple) -> tuple:
        return shape[: -self.transform_ndims] + tuple(
            s + 1 for s in shape[-self.transform_ndims :]
        )

    def inverse_shape(self, shape: tuple) -> tuple:
        return shape[: -self.transform_ndims] + tuple(
            s - 1 for s in shape[-self.transform_ndims :]
        )

    def tree_flatten(self):
        aux_data = {
            "transform_ndims": self.transform_ndims,
        }
        return (), ((), aux_data)

    def __eq__(self, other):
        return (
            isinstance(other, ZeroSumTransform)
            and self.transform_ndims == other.transform_ndims
        )


##########################################################
# CONSTRAINT_REGISTRY
##########################################################


class ConstraintRegistry(object):
    def __init__(self):
        self._registry = {}

    def register(self, constraint, factory=None):
        if factory is None:
            return lambda factory: self.register(constraint, factory)

        if isinstance(constraint, constraints.Constraint):
            constraint = type(constraint)

        self._registry[constraint] = factory
        return factory

    def __call__(self, constraint):
        try:
            factory = self._registry[type(constraint)]
        except KeyError as e:
            raise NotImplementedError from e

        return factory(constraint)


biject_to = ConstraintRegistry()


@biject_to.register(constraints.corr_cholesky)
def _transform_to_corr_cholesky(constraint):
    return CorrCholeskyTransform()


@biject_to.register(constraints.corr_matrix)
def _transform_to_corr_matrix(constraint):
    return ComposeTransform(
        [CorrCholeskyTransform(), CorrMatrixCholeskyTransform().inv]
    )


@biject_to.register(type(constraints.positive))
@biject_to.register(type(constraints.nonnegative))
def _transform_to_positive(constraint):
    return ExpTransform()


@biject_to.register(constraints.greater_than)
@biject_to.register(constraints.greater_than_eq)
def _transform_to_greater_than(constraint):
    return ComposeTransform(
        [
            ExpTransform(),
            AffineTransform(constraint.lower_bound, 1, domain=constraints.positive),
        ]
    )


@biject_to.register(constraints.less_than)
@biject_to.register(constraints.less_than_eq)
def _transform_to_less_than(constraint):
    return ComposeTransform(
        [
            ExpTransform(),
            AffineTransform(constraint.upper_bound, -1, domain=constraints.positive),
        ]
    )


@biject_to.register(type(constraints.real_matrix))
@biject_to.register(type(constraints.real_vector))
@biject_to.register(constraints.independent)
def _biject_to_independent(constraint):
    return IndependentTransform(
        biject_to(constraint.base_constraint), constraint.reinterpreted_batch_ndims
    )


@biject_to.register(type(constraints.unit_interval))
def _transform_to_unit_interval(constraint):
    return SigmoidTransform()


@biject_to.register(type(constraints.circular))
@biject_to.register(constraints.open_interval)
@biject_to.register(constraints.interval)
def _transform_to_interval(constraint):
    scale = constraint.upper_bound - constraint.lower_bound
    return ComposeTransform(
        [
            SigmoidTransform(),
            AffineTransform(
                constraint.lower_bound, scale, domain=constraints.unit_interval
            ),
        ]
    )


@biject_to.register(constraints.l1_ball)
def _transform_to_l1_ball(constraint):
    return L1BallTransform()


@biject_to.register(constraints.lower_cholesky)
def _transform_to_lower_cholesky(constraint):
    return LowerCholeskyTransform()


@biject_to.register(constraints.scaled_unit_lower_cholesky)
def _transform_to_scaled_unit_lower_cholesky(constraint):
    return ScaledUnitLowerCholeskyTransform()


@biject_to.register(constraints.ordered_vector)
def _transform_to_ordered_vector(constraint):
    return OrderedTransform()


@biject_to.register(constraints.positive_definite)
@biject_to.register(constraints.positive_semidefinite)
def _transform_to_positive_definite(constraint):
    return ComposeTransform([LowerCholeskyTransform(), CholeskyTransform().inv])


@biject_to.register(constraints.positive_ordered_vector)
def _transform_to_positive_ordered_vector(constraint):
    return ComposeTransform([OrderedTransform(), ExpTransform()])


@biject_to.register(constraints.complex)
def _transform_to_complex(constraint):
    return IdentityTransform()


@biject_to.register(constraints.real)
def _transform_to_real(constraint):
    return IdentityTransform()


@biject_to.register(constraints.softplus_positive)
def _transform_to_softplus_positive(constraint):
    return SoftplusTransform()


@biject_to.register(constraints.softplus_lower_cholesky)
def _transform_to_softplus_lower_cholesky(constraint):
    return SoftplusLowerCholeskyTransform()


@biject_to.register(constraints.simplex)
def _transform_to_simplex(constraint):
    return StickBreakingTransform()


@biject_to.register(constraints.zero_sum)
def _transform_to_zero_sum(constraint):
    return ZeroSumTransform(constraint.event_dim)
