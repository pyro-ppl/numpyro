# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

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


__all__ = [
    "boolean",
    "corr_cholesky",
    "corr_matrix",
    "dependent",
    "greater_than",
    "integer_interval",
    "integer_greater_than",
    "interval",
    "is_dependent",
    "less_than",
    "lower_cholesky",
    "multinomial",
    "nonnegative_integer",
    "positive",
    "positive_definite",
    "positive_integer",
    "real",
    "real_vector",
    "simplex",
    "sphere",
    "softplus_lower_cholesky",
    "softplus_positive",
    "unit_interval",
    "Constraint",
]

import numpy as np

import jax.numpy


class Constraint(object):
    """
    Abstract base class for constraints.

    A constraint object represents a region over which a variable is valid,
    e.g. within which a variable can be optimized.
    """

    is_discrete = False
    event_dim = 0

    def __call__(self, x):
        raise NotImplementedError

    def check(self, value):
        """
        Returns a byte tensor of `sample_shape + batch_shape` indicating
        whether each event in value satisfies this constraint.
        """
        return self(value)

    def feasible_like(self, prototype):
        """
        Get a feasible value which has the same shape as dtype as `prototype`.
        """
        raise NotImplementedError


class _Boolean(Constraint):
    is_discrete = True

    def __call__(self, x):
        return (x == 0) | (x == 1)

    def feasible_like(self, prototype):
        return jax.numpy.zeros_like(prototype)


class _CorrCholesky(Constraint):
    event_dim = 2

    def __call__(self, x):
        jnp = np if isinstance(x, (np.ndarray, np.generic)) else jax.numpy
        tril = jnp.tril(x)
        lower_triangular = jnp.all(
            jnp.reshape(tril == x, x.shape[:-2] + (-1,)), axis=-1
        )
        positive_diagonal = jnp.all(jnp.diagonal(x, axis1=-2, axis2=-1) > 0, axis=-1)
        x_norm = jnp.linalg.norm(x, axis=-1)
        unit_norm_row = jnp.all((x_norm <= 1) & (x_norm > 1 - 1e-6), axis=-1)
        return lower_triangular & positive_diagonal & unit_norm_row

    def feasible_like(self, prototype):
        return jax.numpy.broadcast_to(
            jax.numpy.eye(prototype.shape[-1]), prototype.shape
        )


class _CorrMatrix(Constraint):
    event_dim = 2

    def __call__(self, x):
        jnp = np if isinstance(x, (np.ndarray, np.generic)) else jax.numpy
        # check for symmetric
        symmetric = jnp.all(jnp.all(x == jnp.swapaxes(x, -2, -1), axis=-1), axis=-1)
        # check for the smallest eigenvalue is positive
        positive = jnp.linalg.eigh(x)[0][..., 0] > 0
        # check for diagonal equal to 1
        unit_variance = jnp.all(
            jnp.abs(jnp.diagonal(x, axis1=-2, axis2=-1) - 1) < 1e-6, axis=-1
        )
        return symmetric & positive & unit_variance

    def feasible_like(self, prototype):
        return jax.numpy.broadcast_to(
            jax.numpy.eye(prototype.shape[-1]), prototype.shape
        )


class _Dependent(Constraint):
    """
    Placeholder for variables whose support depends on other variables.
    These variables obey no simple coordinate-wise constraints.

    :param bool is_discrete: Optional value of ``.is_discrete`` in case this
        can be computed statically. If not provided, access to the
        ``.is_discrete`` attribute will raise a NotImplementedError.
    :param int event_dim: Optional value of ``.event_dim`` in case this can be
        computed statically. If not provided, access to the ``.event_dim``
        attribute will raise a NotImplementedError.
    """

    def __init__(self, *, is_discrete=NotImplemented, event_dim=NotImplemented):
        self._is_discrete = is_discrete
        self._event_dim = event_dim
        super().__init__()

    @property
    def is_discrete(self):
        if self._is_discrete is NotImplemented:
            raise NotImplementedError(".is_discrete cannot be determined statically")
        return self._is_discrete

    @property
    def event_dim(self):
        if self._event_dim is NotImplemented:
            raise NotImplementedError(".event_dim cannot be determined statically")
        return self._event_dim

    def __call__(self, x=None, *, is_discrete=NotImplemented, event_dim=NotImplemented):
        if x is not None:
            raise ValueError("Cannot determine validity of dependent constraint")

        # Support for syntax to customize static attributes::
        #     constraints.dependent(is_discrete=True, event_dim=1)
        if is_discrete is NotImplemented:
            is_discrete = self._is_discrete
        if event_dim is NotImplemented:
            event_dim = self._event_dim
        return _Dependent(is_discrete=is_discrete, event_dim=event_dim)


class dependent_property(property, _Dependent):
    def __init__(
        self, fn=None, *, is_discrete=NotImplemented, event_dim=NotImplemented
    ):
        super().__init__(fn)
        self._is_discrete = is_discrete
        self._event_dim = event_dim

    def __call__(self, x):
        if not callable(x):
            return super().__call__(x)

        # Support for syntax to customize static attributes::
        #     @constraints.dependent_property(is_discrete=True, event_dim=1)
        #     def support(self):
        #         ...
        return dependent_property(
            x, is_discrete=self._is_discrete, event_dim=self._event_dim
        )


def is_dependent(constraint):
    return isinstance(constraint, _Dependent)


class _GreaterThan(Constraint):
    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def __call__(self, x):
        return x > self.lower_bound

    def feasible_like(self, prototype):
        return jax.numpy.broadcast_to(self.lower_bound + 1, jax.numpy.shape(prototype))


class _IndependentConstraint(Constraint):
    """
    Wraps a constraint by aggregating over ``reinterpreted_batch_ndims``-many
    dims in :meth:`check`, so that an event is valid only if all its
    independent entries are valid.
    """

    def __init__(self, base_constraint, reinterpreted_batch_ndims):
        assert isinstance(base_constraint, Constraint)
        assert isinstance(reinterpreted_batch_ndims, int)
        assert reinterpreted_batch_ndims >= 0
        if isinstance(base_constraint, _IndependentConstraint):
            reinterpreted_batch_ndims = (
                reinterpreted_batch_ndims + base_constraint.reinterpreted_batch_ndims
            )
            base_constraint = base_constraint.base_constraint
        self.base_constraint = base_constraint
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        super().__init__()

    @property
    def event_dim(self):
        return self.base_constraint.event_dim + self.reinterpreted_batch_ndims

    def __call__(self, value):
        result = self.base_constraint(value)
        if self.reinterpreted_batch_ndims == 0:
            return result
        elif jax.numpy.ndim(result) < self.reinterpreted_batch_ndims:
            expected = self.event_dim
            raise ValueError(
                f"Expected value.dim() >= {expected} but got {jax.numpy.ndim(value)}"
            )
        result = result.reshape(
            jax.numpy.shape(result)[
                : jax.numpy.ndim(result) - self.reinterpreted_batch_ndims
            ]
            + (-1,)
        )
        result = result.all(-1)
        return result

    def feasible_like(self, prototype):
        return self.base_constraint.feasible_like(prototype)


class _LessThan(Constraint):
    def __init__(self, upper_bound):
        self.upper_bound = upper_bound

    def __call__(self, x):
        return x < self.upper_bound

    def feasible_like(self, prototype):
        return jax.numpy.broadcast_to(self.upper_bound - 1, jax.numpy.shape(prototype))


class _IntegerInterval(Constraint):
    is_discrete = True

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, x):
        return (x >= self.lower_bound) & (x <= self.upper_bound) & (x % 1 == 0)

    def feasible_like(self, prototype):
        return jax.numpy.broadcast_to(self.lower_bound, jax.numpy.shape(prototype))


class _IntegerGreaterThan(Constraint):
    is_discrete = True

    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def __call__(self, x):
        return (x % 1 == 0) & (x >= self.lower_bound)

    def feasible_like(self, prototype):
        return jax.numpy.broadcast_to(self.lower_bound, jax.numpy.shape(prototype))


class _Interval(Constraint):
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, x):
        return (x >= self.lower_bound) & (x <= self.upper_bound)

    def feasible_like(self, prototype):
        return jax.numpy.broadcast_to(
            (self.lower_bound + self.upper_bound) / 2, jax.numpy.shape(prototype)
        )


class _LowerCholesky(Constraint):
    event_dim = 2

    def __call__(self, x):
        jnp = np if isinstance(x, (np.ndarray, np.generic)) else jax.numpy
        tril = jnp.tril(x)
        lower_triangular = jnp.all(
            jnp.reshape(tril == x, x.shape[:-2] + (-1,)), axis=-1
        )
        positive_diagonal = jnp.all(jnp.diagonal(x, axis1=-2, axis2=-1) > 0, axis=-1)
        return lower_triangular & positive_diagonal

    def feasible_like(self, prototype):
        return jax.numpy.broadcast_to(
            jax.numpy.eye(prototype.shape[-1]), prototype.shape
        )


class _Multinomial(Constraint):
    is_discrete = True
    event_dim = 1

    def __init__(self, upper_bound):
        self.upper_bound = upper_bound

    def __call__(self, x):
        return (x >= 0).all(axis=-1) & (x.sum(axis=-1) == self.upper_bound)

    def feasible_like(self, prototype):
        pad_width = ((0, 0),) * jax.numpy.ndim(self.upper_bound) + (
            (0, prototype.shape[-1] - 1),
        )
        value = jax.numpy.pad(jax.numpy.expand_dims(self.upper_bound, -1), pad_width)
        return jax.numpy.broadcast_to(value, prototype.shape)


class _OrderedVector(Constraint):
    event_dim = 1

    def __call__(self, x):
        return (x[..., 1:] > x[..., :-1]).all(axis=-1)

    def feasible_like(self, prototype):
        return jax.numpy.broadcast_to(
            jax.numpy.arange(float(prototype.shape[-1])), prototype.shape
        )


class _PositiveDefinite(Constraint):
    event_dim = 2

    def __call__(self, x):
        jnp = np if isinstance(x, (np.ndarray, np.generic)) else jax.numpy
        # check for symmetric
        symmetric = jnp.all(jnp.all(x == jnp.swapaxes(x, -2, -1), axis=-1), axis=-1)
        # check for the smallest eigenvalue is positive
        positive = jnp.linalg.eigh(x)[0][..., 0] > 0
        return symmetric & positive

    def feasible_like(self, prototype):
        return jax.numpy.broadcast_to(
            jax.numpy.eye(prototype.shape[-1]), prototype.shape
        )


class _PositiveOrderedVector(Constraint):
    """
    Constrains to a positive real-valued tensor where the elements are monotonically
    increasing along the `event_shape` dimension.
    """

    event_dim = 1

    def __call__(self, x):
        return ordered_vector.check(x) & independent(positive, 1).check(x)

    def feasible_like(self, prototype):
        return jax.numpy.broadcast_to(
            jax.numpy.exp(jax.numpy.arange(float(prototype.shape[-1]))), prototype.shape
        )


class _Real(Constraint):
    def __call__(self, x):
        # XXX: consider to relax this condition to [-inf, inf] interval
        return (x == x) & (x != float("inf")) & (x != float("-inf"))

    def feasible_like(self, prototype):
        return jax.numpy.zeros_like(prototype)


class _Simplex(Constraint):
    event_dim = 1

    def __call__(self, x):
        x_sum = x.sum(axis=-1)
        return (x >= 0).all(axis=-1) & (x_sum < 1 + 1e-6) & (x_sum > 1 - 1e-6)

    def feasible_like(self, prototype):
        return jax.numpy.full_like(prototype, 1 / prototype.shape[-1])


class _SoftplusPositive(_GreaterThan):
    def __init__(self):
        super().__init__(lower_bound=0.0)

    def feasible_like(self, prototype):
        return jax.numpy.full(jax.numpy.shape(prototype), np.log(2))


class _SoftplusLowerCholesky(_LowerCholesky):
    def feasible_like(self, prototype):
        return jax.numpy.broadcast_to(
            jax.numpy.eye(prototype.shape[-1]) * np.log(2), prototype.shape
        )


class _Sphere(Constraint):
    """
    Constrain to the Euclidean sphere of any dimension.
    """

    event_dim = 1
    reltol = 10.0  # Relative to finfo.eps.

    def __call__(self, x):
        jnp = np if isinstance(x, (np.ndarray, np.generic)) else jax.numpy
        eps = jnp.finfo(x.dtype).eps
        norm = jnp.linalg.norm(x, axis=-1)
        error = jnp.abs(norm - 1)
        return error < self.reltol * eps * x.shape[-1] ** 0.5

    def feasible_like(self, prototype):
        return jax.numpy.full_like(prototype, prototype.shape[-1] ** (-0.5))


# TODO: Make types consistent
# See https://github.com/pytorch/pytorch/issues/50616

boolean = _Boolean()
corr_cholesky = _CorrCholesky()
corr_matrix = _CorrMatrix()
dependent = _Dependent()
greater_than = _GreaterThan
less_than = _LessThan
independent = _IndependentConstraint
integer_interval = _IntegerInterval
integer_greater_than = _IntegerGreaterThan
interval = _Interval
lower_cholesky = _LowerCholesky()
multinomial = _Multinomial
nonnegative_integer = _IntegerGreaterThan(0)
ordered_vector = _OrderedVector()
positive = _GreaterThan(0.0)
positive_definite = _PositiveDefinite()
positive_integer = _IntegerGreaterThan(1)
positive_ordered_vector = _PositiveOrderedVector()
real = _Real()
real_vector = independent(real, 1)
simplex = _Simplex()
softplus_lower_cholesky = _SoftplusLowerCholesky()
softplus_positive = _SoftplusPositive()
sphere = _Sphere()
unit_interval = _Interval(0.0, 1.0)
