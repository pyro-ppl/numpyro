import math

from jax import ops
from jax.flatten_util import ravel_pytree
from jax.lib.xla_bridge import canonicalize_dtype
from jax.nn import softplus
import jax.numpy as np
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import expit, logit

from numpyro.distributions import constraints
from numpyro.distributions.util import (
    cumprod,
    cumsum,
    get_dtype,
    matrix_to_tril_vec,
    signed_stick_breaking_tril,
    sum_rightmost,
    vec_to_tril_matrix
)

__all__ = [
    'biject_to',
    'AbsTransform',
    'AffineTransform',
    'ComposeTransform',
    'CorrCholeskyTransform',
    'ExpTransform',
    'IdentityTransform',
    'InvCholeskyTransform',
    'LowerCholeskyTransform',
    'MultivariateAffineTransform',
    'PermuteTransform',
    'PowerTransform',
    'SigmoidTransform',
    'StickBreakingTransform',
    'Transform',
    'UnpackTransform',
]


def _clipped_expit(x):
    finfo = np.finfo(get_dtype(x))
    return np.clip(expit(x), a_min=finfo.tiny, a_max=1. - finfo.eps)


class Transform(object):
    domain = constraints.real
    codomain = constraints.real
    event_dim = 0

    def __call__(self, x):
        return NotImplementedError

    def inv(self, y):
        raise NotImplementedError

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        raise NotImplementedError

    def call_with_intermediates(self, x):
        return self(x), None


class AbsTransform(Transform):
    domain = constraints.real
    codomain = constraints.positive

    def __eq__(self, other):
        return isinstance(other, AbsTransform)

    def __call__(self, x):
        return np.abs(x)

    def inv(self, y):
        return y


class AffineTransform(Transform):
    # TODO: currently, just support scale > 0
    def __init__(self, loc, scale, domain=constraints.real):
        self.loc = loc
        self.scale = scale
        self.domain = domain

    @property
    def codomain(self):
        if self.domain is constraints.real:
            return constraints.real
        elif self.domain is constraints.real_vector:
            return constraints.real_vector
        elif isinstance(self.domain, constraints.greater_than):
            return constraints.greater_than(self.__call__(self.domain.lower_bound))
        elif isinstance(self.domain, constraints.interval):
            return constraints.interval(self.__call__(self.domain.lower_bound),
                                        self.__call__(self.domain.upper_bound))
        else:
            raise NotImplementedError

    @property
    def event_dim(self):
        return 1 if self.domain is constraints.real_vector else 0

    def __call__(self, x):
        return self.loc + self.scale * x

    def inv(self, y):
        return (y - self.loc) / self.scale

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return sum_rightmost(np.broadcast_to(np.log(np.abs(self.scale)), np.shape(x)), self.event_dim)


class ComposeTransform(Transform):
    def __init__(self, parts):
        self.parts = parts

    @property
    def domain(self):
        return self.parts[0].domain

    @property
    def codomain(self):
        return self.parts[-1].codomain

    @property
    def event_dim(self):
        return max(p.event_dim for p in self.parts)

    def __call__(self, x):
        for part in self.parts:
            x = part(x)
        return x

    def inv(self, y):
        for part in self.parts[::-1]:
            y = part.inv(y)
        return y

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        if intermediates is not None:
            if len(intermediates) != len(self.parts):
                raise ValueError('Intermediates array has length = {}. Expected = {}.'
                                 .format(len(intermediates), len(self.parts)))

        result = 0.
        for i, part in enumerate(self.parts[:-1]):
            y_tmp = part(x) if intermediates is None else intermediates[i][0]
            inter = None if intermediates is None else intermediates[i][1]
            logdet = part.log_abs_det_jacobian(x, y_tmp, intermediates=inter)
            result = result + sum_rightmost(logdet, self.event_dim - part.event_dim)
            x = y_tmp
        # account the the last transform, where y is available
        inter = None if intermediates is None else intermediates[-1]
        logdet = self.parts[-1].log_abs_det_jacobian(x, y, intermediates=inter)
        result = result + sum_rightmost(logdet, self.event_dim - self.parts[-1].event_dim)
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


class CorrCholeskyTransform(Transform):
    r"""
    Transforms a uncontrained real vector :math:`x` with length :math:`D*(D-1)/2` into the
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
    event_dim = 2

    def __call__(self, x):
        # we interchange step 1 and step 2.a for a better performance
        t = np.tanh(x)
        return signed_stick_breaking_tril(t)

    def inv(self, y):
        # inverse stick-breaking
        z1m_cumprod = 1 - cumsum(y * y)
        pad_width = [(0, 0)] * y.ndim
        pad_width[-1] = (1, 0)
        z1m_cumprod_shifted = np.pad(z1m_cumprod[..., :-1], pad_width,
                                     mode="constant", constant_values=1.)
        t = matrix_to_tril_vec(y, diagonal=-1) / np.sqrt(
            matrix_to_tril_vec(z1m_cumprod_shifted, diagonal=-1))
        # inverse of tanh
        x = np.log((1 + t) / (1 - t)) / 2
        return x

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        # NB: because domain and codomain are two spaces with different dimensions, determinant of
        # Jacobian is not well-defined. Here we return `log_abs_det_jacobian` of `x` and the
        # flatten lower triangular part of `y`.

        # stick_breaking_logdet = log(y / r) = log(z_cumprod)  (modulo right shifted)
        z1m_cumprod = 1 - cumsum(y * y)
        # by taking diagonal=-2, we don't need to shift z_cumprod to the right
        # NB: diagonal=-2 works fine for (2 x 2) matrix, where we get an empty array
        z1m_cumprod_tril = matrix_to_tril_vec(z1m_cumprod, diagonal=-2)
        stick_breaking_logdet = 0.5 * np.sum(np.log(z1m_cumprod_tril), axis=-1)

        tanh_logdet = -2 * np.sum(x + softplus(-2 * x) - np.log(2.), axis=-1)
        return stick_breaking_logdet + tanh_logdet


class ExpTransform(Transform):
    # TODO: refine domain/codomain logic through setters, especially when
    # transforms for inverses are supported
    def __init__(self, domain=constraints.real):
        self.domain = domain

    @property
    def codomain(self):
        if self.domain is constraints.real:
            return constraints.positive
        elif isinstance(self.domain, constraints.greater_than):
            return constraints.greater_than(self.__call__(self.domain.lower_bound))
        elif isinstance(self.domain, constraints.interval):
            return constraints.interval(self.__call__(self.domain.lower_bound),
                                        self.__call__(self.domain.upper_bound))
        else:
            raise NotImplementedError

    def __call__(self, x):
        # XXX consider to clamp from below for stability if necessary
        return np.exp(x)

    def inv(self, y):
        return np.log(y)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return x


class IdentityTransform(Transform):

    def __init__(self, event_dim=0):
        self.event_dim = event_dim

    def __call__(self, x):
        return x

    def inv(self, y):
        return y

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return np.full(np.shape(x) if self.event_dim == 0 else np.shape(x)[:-1], 0.)


class InvCholeskyTransform(Transform):
    r"""
    Transform via the mapping :math:`y = x @ x.T`, where `x` is a lower
    triangular matrix with positive diagonal.
    """
    event_dim = 2

    def __init__(self, domain=constraints.lower_cholesky):
        assert domain in [constraints.lower_cholesky, constraints.corr_cholesky]
        self.domain = domain

    @property
    def codomain(self):
        if self.domain is constraints.lower_cholesky:
            return constraints.positive_definite
        elif self.domain:
            return constraints.corr_matrix

    def __call__(self, x):
        return np.matmul(x, np.swapaxes(x, -2, -1))

    def inv(self, y):
        return np.linalg.cholesky(y)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        if self.domain is constraints.lower_cholesky:
            # Ref: http://web.mit.edu/18.325/www/handouts/handout2.pdf page 13
            n = np.shape(x)[-1]
            order = np.arange(n, 0, -1)
            return n * np.log(2) + np.sum(order * np.log(np.diagonal(x, axis1=-2, axis2=-1)), axis=-1)
        else:
            # NB: see derivation in LKJCholesky implementation
            n = np.shape(x)[-1]
            order = np.arange(n - 1, -1, -1)
            return np.sum(order * np.log(np.diagonal(x, axis1=-2, axis2=-1)), axis=-1)


class LowerCholeskyTransform(Transform):
    domain = constraints.real_vector
    codomain = constraints.lower_cholesky
    event_dim = 2

    def __call__(self, x):
        n = round((math.sqrt(1 + 8 * x.shape[-1]) - 1) / 2)
        z = vec_to_tril_matrix(x[..., :-n], diagonal=-1)
        diag = np.exp(x[..., -n:])
        return z + np.expand_dims(diag, axis=-1) * np.identity(n)

    def inv(self, y):
        z = matrix_to_tril_vec(y, diagonal=-1)
        return np.concatenate([z, np.log(np.diagonal(y, axis1=-2, axis2=-1))], axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        # the jacobian is diagonal, so logdet is the sum of diagonal `exp` transform
        n = round((math.sqrt(1 + 8 * x.shape[-1]) - 1) / 2)
        return x[..., -n:].sum(-1)


class MultivariateAffineTransform(Transform):
    r"""
    Transform via the mapping :math:`y = loc + scale\_tril\ @\ x`.

    :param loc: a real vector.
    :param scale_tril: a lower triangular matrix with positive diagonal.
    """
    domain = constraints.real_vector
    codomain = constraints.real_vector
    event_dim = 1

    def __init__(self, loc, scale_tril):
        # TODO: relax this condition per user request
        if np.ndim(scale_tril) != 2:
            raise ValueError("Only support 2-dimensional scale_tril matrix.")
        self.loc = loc
        self.scale_tril = scale_tril

    def __call__(self, x):
        return self.loc + np.squeeze(np.matmul(self.scale_tril, x[..., np.newaxis]), axis=-1)

    def inv(self, y):
        y = y - self.loc
        original_shape = np.shape(y)
        yt = np.reshape(y, (-1, original_shape[-1])).T
        xt = solve_triangular(self.scale_tril, yt, lower=True)
        return np.reshape(xt.T, original_shape)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return np.broadcast_to(np.log(np.diagonal(self.scale_tril, axis1=-2, axis2=-1)).sum(-1),
                               np.shape(x)[:-1])


class PermuteTransform(Transform):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    event_dim = 1

    def __init__(self, permutation):
        self.permutation = permutation

    def __call__(self, x):
        return x[..., self.permutation]

    def inv(self, y):
        size = self.permutation.size
        permutation_inv = ops.index_update(np.zeros(size, dtype=canonicalize_dtype(np.int64)),
                                           self.permutation,
                                           np.arange(size))
        return y[..., permutation_inv]

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return np.full(np.shape(x)[:-1], 0.)


class PowerTransform(Transform):
    domain = constraints.positive
    codomain = constraints.positive

    def __init__(self, exponent):
        self.exponent = exponent

    def __call__(self, x):
        return np.power(x, self.exponent)

    def inv(self, y):
        return np.power(y, 1 / self.exponent)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return np.log(np.abs(self.exponent * y / x))


class SigmoidTransform(Transform):
    codomain = constraints.unit_interval

    def __call__(self, x):
        return _clipped_expit(x)

    def inv(self, y):
        return logit(y)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        x_abs = np.abs(x)
        return -x_abs - 2 * np.log1p(np.exp(-x_abs))


class StickBreakingTransform(Transform):
    domain = constraints.real_vector
    codomain = constraints.simplex
    event_dim = 1

    def __call__(self, x):
        # we shift x to obtain a balanced mapping (0, 0, ..., 0) -> (1/K, 1/K, ..., 1/K)
        x = x - np.log(x.shape[-1] - np.arange(x.shape[-1]))
        # convert to probabilities (relative to the remaining) of each fraction of the stick
        z = _clipped_expit(x)
        z1m_cumprod = cumprod(1 - z)
        pad_width = [(0, 0)] * x.ndim
        pad_width[-1] = (0, 1)
        z_padded = np.pad(z, pad_width, mode="constant", constant_values=1.)
        pad_width = [(0, 0)] * x.ndim
        pad_width[-1] = (1, 0)
        z1m_cumprod_shifted = np.pad(z1m_cumprod, pad_width, mode="constant", constant_values=1.)
        return z_padded * z1m_cumprod_shifted

    def inv(self, y):
        y_crop = y[..., :-1]
        z1m_cumprod = np.clip(1 - cumsum(y_crop), a_min=np.finfo(y.dtype).tiny)
        # hence x = logit(z) = log(z / (1 - z)) = y[::-1] / z1m_cumprod
        x = np.log(y_crop / z1m_cumprod)
        return x + np.log(x.shape[-1] - np.arange(x.shape[-1]))

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        # Ref: https://mc-stan.org/docs/2_19/reference-manual/simplex-transform-section.html
        # |det|(J) = Product(y * (1 - z))
        x = x - np.log(x.shape[-1] - np.arange(x.shape[-1]))
        z = np.clip(expit(x), a_min=np.finfo(x.dtype).tiny)
        # XXX we use the identity 1 - z = z * exp(-x) to not worry about
        # the case z ~ 1
        return np.sum(np.log(y[..., :-1] * z) - x, axis=-1)


class UnpackTransform(Transform):
    """
    Transforms a contiguous array to a pytree of subarrays.

    :param unpack_fn: callable used to unpack a contiguous array.
    """
    domain = constraints.real_vector
    event_dim = 1

    def __init__(self, unpack_fn):
        self.unpack_fn = unpack_fn

    def __call__(self, x):
        return self.unpack_fn(x)

    def inv(self, y):
        return ravel_pytree(y)[0]

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return np.zeros(np.shape(x)[:-1])


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

    def __call__(self, constraint):
        try:
            factory = self._registry[type(constraint)]
        except KeyError:
            raise NotImplementedError

        return factory(constraint)


biject_to = ConstraintRegistry()


@biject_to.register(constraints.corr_cholesky)
def _transform_to_corr_cholesky(constraint):
    return CorrCholeskyTransform()


@biject_to.register(constraints.corr_matrix)
def _transform_to_corr_matrix(constraint):
    return ComposeTransform([CorrCholeskyTransform(), InvCholeskyTransform(domain=constraints.corr_cholesky)])


@biject_to.register(constraints.greater_than)
def _transform_to_greater_than(constraint):
    if constraint is constraints.positive:
        return ExpTransform()
    return ComposeTransform([ExpTransform(),
                             AffineTransform(constraint.lower_bound, 1,
                                             domain=constraints.positive)])


@biject_to.register(constraints.interval)
def _transform_to_interval(constraint):
    if constraint is constraints.unit_interval:
        return SigmoidTransform()
    scale = constraint.upper_bound - constraint.lower_bound
    return ComposeTransform([SigmoidTransform(),
                             AffineTransform(constraint.lower_bound, scale,
                                             domain=constraints.unit_interval)])


@biject_to.register(constraints.lower_cholesky)
def _transform_to_lower_cholesky(constraint):
    return LowerCholeskyTransform()


@biject_to.register(constraints.positive_definite)
def _transform_to_positive_definite(constraint):
    return ComposeTransform([LowerCholeskyTransform(), InvCholeskyTransform()])


@biject_to.register(constraints.real)
def _transform_to_real(constraint):
    return IdentityTransform()


@biject_to.register(constraints.real_vector)
def _transform_to_real_vector(constraint):
    return IdentityTransform(event_dim=1)


@biject_to.register(constraints.simplex)
def _transform_to_simplex(constraint):
    return StickBreakingTransform()
