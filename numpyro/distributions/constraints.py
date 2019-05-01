# The implementation follows the design in PyTorch: torch.distributions.constraints.py
#
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import jax.numpy as np
from jax.scipy.special import expit, logit

from numpyro.distributions.util import cumprod, cumsum, matrix_to_tril_vec, sum_rightmost, vec_to_tril_matrix

##########################################################
# CONSTRAINTS
##########################################################


class Constraint(object):
    def __call__(self, x):
        raise NotImplementedError


class _Boolean(Constraint):
    def __call__(self, value):
        return (value == 0) | (value == 1)


class _CorrCholesky(Constraint):
    def __call__(self, x):
        tril = np.tril(x)
        lower_triangular = np.all(np.reshape(tril == x, x.shape[:-2] + (-1,)), axis=-1)
        positive_diagonal = np.all(np.diagonal(x, axis1=-2, axis2=-1) > 0, axis=-1)
        x_norm = np.linalg.norm(x, axis=-1)
        unit_norm_row = np.all((x_norm <= 1) & (x_norm > 1 - 1e-6), axis=-1)
        return lower_triangular & positive_diagonal & unit_norm_row


class _Dependent(Constraint):
    def __call__(self, x):
        raise ValueError('Cannot determine validity of dependent constraint')


def is_dependent(constraint):
    return isinstance(constraint, _Dependent)


class _GreaterThan(Constraint):
    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def __call__(self, x):
        return x > self.lower_bound


class _IntegerInterval(Constraint):
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, x):
        return (x >= self.lower_bound) & (x <= self.upper_bound) & (x == np.floor(x))


class _IntegerGreaterThan(Constraint):
    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def __call__(self, value):
        return (value % 1 == 0) & (value >= self.lower_bound)


class _Interval(Constraint):
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, value):
        return (value > self.lower_bound) & (value < self.upper_bound)


class _Multinomial(Constraint):
    def __init__(self, upper_bound):
        self.upper_bound = upper_bound

    def __call__(self, value):
        return np.all(value >= 0, axis=-1) & (np.sum(value, -1) == self.upper_bound)


class _Real(Constraint):
    def __call__(self, x):
        return np.isfinite(x)


class _Simplex(Constraint):
    def __call__(self, x):
        x_sum = np.sum(x, axis=-1)
        return np.all(x > 0, axis=-1) & (x_sum <= 1) & (x_sum > 1 - 1e-6)


# TODO: Make types consistent

boolean = _Boolean()
corr_cholesky = _CorrCholesky()
dependent = _Dependent()
greater_than = _GreaterThan
integer_interval = _IntegerInterval
integer_greater_than = _IntegerGreaterThan
interval = _Interval
multinomial = _Multinomial
nonnegative_integer = _IntegerGreaterThan(0)
positive_integer = _IntegerGreaterThan(1)
positive = _GreaterThan(0)
real = _Real()
simplex = _Simplex()
unit_interval = _Interval(0, 1)


##########################################################
# TRANSFORMS
##########################################################

def _clipped_expit(x):
    return np.clip(expit(x), a_min=np.finfo(x.dtype).tiny, a_max=1.-np.finfo(x.dtype).eps)


class Transform(object):
    domain = real
    codomain = real
    event_dim = 0

    def __call__(self, x):
        return NotImplementedError

    def inv(self, y):
        raise NotImplementedError

    def log_abs_det_jacobian(self, x, y):
        raise NotImplementedError


class AbsTransform(Transform):
    domain = real
    codomain = positive

    def __eq__(self, other):
        return isinstance(other, AbsTransform)

    def __call__(self, x):
        return np.abs(x)

    def inv(self, y):
        return y


class AffineTransform(Transform):
    # TODO: currently, just support scale > 0
    def __init__(self, loc, scale, domain=real):
        self.loc = loc
        self.scale = scale
        self.domain = domain

    @property
    def codomain(self):
        if self.domain is real:
            return real
        elif isinstance(self.domain, greater_than):
            return greater_than(self.loc + self.scale * self.domain.lower_bound)
        elif isinstance(self.domain, interval):
            return interval(self.loc + self.scale * self.domain.lower_bound,
                            self.loc + self.scale * self.domain.upper_bound)
        else:
            raise NotImplementedError

    def __call__(self, x):
        return self.loc + self.scale * x

    def inv(self, y):
        return (y - self.loc) / self.scale

    def log_abs_det_jacobian(self, x, y):
        return np.broadcast_to(np.log(np.abs(self.scale)), x.shape)


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

    def log_abs_det_jacobian(self, x, y):
        result = 0.
        for part in self.parts[:-1]:
            y_tmp = part(x)
            result = result + sum_rightmost(part.log_abs_det_jacobian(x, y_tmp),
                                            self.event_dim - part.event_dim)
            x = y_tmp
        result = result + sum_rightmost(self.parts[-1].log_abs_det_jacobian(x, y),
                                        self.event_dim - self.parts[-1].event_dim)
        return result


def _signed_stick_breaking_tril(t):
    # transform t to tril matrix with identity diagonal
    r = vec_to_tril_matrix(t, diagonal=-1)
    r = r + np.identity(r.shape[-1])

    # apply stick-breaking on the squared values;
    # we omit the step of computing s = z * z_cumprod by using the fact:
    #     y = sign(r) * s = sign(r) * sqrt(z * z_cumprod) = r * sqrt(z_cumprod)
    z = r ** 2
    z1m_cumprod = cumprod(1 - z)

    # to workaround the issue: NaN propagated through backward pass even when not accessed
    # at https://github.com/pytorch/pytorch/issues/15506 (which also happens in JAX),
    # here we only take sqrt at tril part
    z1m_cumprod_tril_sqrt = np.sqrt(matrix_to_tril_vec(z1m_cumprod, diagonal=-1))
    z1m_cumprod_sqrt = vec_to_tril_matrix(z1m_cumprod_tril_sqrt, diagonal=-1)

    pad_width = [(0, 0)] * z.ndim
    pad_width[-1] = (1, 0)
    z1m_cumprod_sqrt_shifted = np.pad(z1m_cumprod_sqrt[..., :-1], pad_width,
                                      mode="constant", constant_values=1.)
    y = r * z1m_cumprod_sqrt_shifted
    return y


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
    codomain = corr_cholesky
    event_dim = 1

    def __call__(self, x):
        # we interchange step 1 and step 2.a for a better performance
        eps = np.finfo(x.dtype).eps
        t = np.clip(np.tanh(x), a_min=(-1 + eps), a_max=(1 - eps))
        return _signed_stick_breaking_tril(t)

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

    def log_abs_det_jacobian(self, x, y):
        # NB: because domain and codomain are two spaces with different dimensions, determinant of
        # Jacobian is not well-defined. Here we return `log_abs_det_jacobian` of `x` and the
        # flatten lower triangular part of `y`.

        # stick_breaking_logdet = log(y / r) = log(z_cumprod)  (modulo right shifted)
        z1m_cumprod = 1 - cumsum(y * y)
        # by taking diagonal=-2, we don't need to shift z_cumprod to the right
        # NB: diagonal=-2 works fine for (2 x 2) matrix, where we get an empty array
        z1m_cumprod_tril = matrix_to_tril_vec(z1m_cumprod, diagonal=-2)
        stick_breaking_logdet = 0.5 * np.sum(np.log(z1m_cumprod_tril), axis=-1)

        tanh_logdet = -2 * np.sum(np.log(np.cosh(x)), axis=-1)
        return stick_breaking_logdet + tanh_logdet


class ExpTransform(Transform):
    codomain = positive

    def __call__(self, x):
        # XXX consider to clamp from below for stability if necessary
        return np.exp(x)

    def inv(self, y):
        return np.log(y)

    def log_abs_det_jacobian(self, x, y):
        return x


class IdentityTransform(Transform):

    def __call__(self, x):
        return x

    def inv(self, y):
        return y

    def log_abs_det_jacobian(self, x, y):
        return np.full(np.shape(x), 0.)


class SigmoidTransform(Transform):
    codomain = unit_interval

    def __call__(self, x):
        return _clipped_expit(x)

    def inv(self, y):
        return logit(y)

    def log_abs_det_jacobian(self, x, y):
        return np.log(y * (1 - y))


class StickBreakingTransform(Transform):
    codomain = simplex
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
        pad_width[-1] = (1, 0)
        z1m_cumprod_shifted = np.pad(z1m_cumprod, pad_width, mode="constant", constant_values=1.)
        return z_padded * z1m_cumprod_shifted

    def inv(self, y):
        y_crop = y[..., :-1]
        z1m_cumprod = np.clip(1 - cumsum(y_crop), a_min=np.finfo(y.dtype).tiny)
        # hence x = logit(z) = log(z / (1 - z)) = y[::-1] / z1m_cumprod
        x = np.log(y_crop / z1m_cumprod)
        return x + np.log(x.shape[-1] - np.arange(x.shape[-1]))

    def log_abs_det_jacobian(self, x, y):
        # Ref: https://mc-stan.org/docs/2_19/reference-manual/simplex-transform-section.html
        # |det|(J) = Product(y * (1 - z))
        z = _clipped_expit(x - np.log(x.shape[-1] - np.arange(x.shape[-1])))
        return np.sum(np.log(y[..., :-1] * (1 - z)), axis=-1)


##########################################################
# CONSTRAINT_REGISTRY
##########################################################

class ConstraintRegistry(object):
    def __init__(self):
        self._registry = {}

    def register(self, constraint, factory=None):
        if factory is None:
            return lambda factory: self.register(constraint, factory)

        if isinstance(constraint, Constraint):
            constraint = type(constraint)

        self._registry[constraint] = factory

    def __call__(self, constraint):
        try:
            factory = self._registry[type(constraint)]
        except KeyError:
            raise NotImplementedError

        return factory(constraint)


biject_to = ConstraintRegistry()


@biject_to.register(corr_cholesky)
def _transform_to_corr_cholesky(constraint):
    return CorrCholeskyTransform()


@biject_to.register(greater_than)
def _transform_to_greater_than(constraint):
    return ComposeTransform([ExpTransform(),
                             AffineTransform(constraint.lower_bound, 1,
                                             domain=positive)])


@biject_to.register(interval)
def _transform_to_interval(constraint):
    return ComposeTransform([SigmoidTransform(),
                             AffineTransform(constraint.lower_bound, constraint.upper_bound,
                                             domain=unit_interval)])


@biject_to.register(real)
def _transform_to_real(constraint):
    return IdentityTransform()


@biject_to.register(simplex)
def _transform_to_simplex(constraint):
    return StickBreakingTransform()
