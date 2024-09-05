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
    "circular",
    "complex",
    "corr_cholesky",
    "corr_matrix",
    "dependent",
    "greater_than",
    "greater_than_eq",
    "integer_interval",
    "integer_greater_than",
    "interval",
    "is_dependent",
    "l1_ball",
    "less_than",
    "lower_cholesky",
    "multinomial",
    "nonnegative",
    "nonnegative_integer",
    "positive",
    "positive_definite",
    "positive_semidefinite",
    "positive_integer",
    "real",
    "real_vector",
    "real_matrix",
    "scaled_unit_lower_cholesky",
    "simplex",
    "sphere",
    "softplus_lower_cholesky",
    "softplus_positive",
    "unit_interval",
    "zero_sum",
    "Constraint",
]

import math

import numpy as np

import jax.numpy
import jax.numpy as jnp
from jax.tree_util import register_pytree_node


class Constraint(object):
    """
    Abstract base class for constraints.

    A constraint object represents a region over which a variable is valid,
    e.g. within which a variable can be optimized.
    """

    is_discrete = False
    event_dim = 0

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_pytree_node(cls, cls.tree_flatten, cls.tree_unflatten)

    def __call__(self, x):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__[1:] + "()"

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

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        params_keys, aux_data = aux_data
        self = cls.__new__(cls)
        for k, v in zip(params_keys, params):
            setattr(self, k, v)

        for k, v in aux_data.items():
            setattr(self, k, v)
        return self


class ParameterFreeConstraint(Constraint):
    def tree_flatten(self):
        return (), ((), dict())


class _SingletonConstraint(ParameterFreeConstraint):
    """
    A constraint type which has only one canonical instance, like constraints.real,
    and unlike constraints.interval.
    """

    def __new__(cls):
        if (not hasattr(cls, "_instance")) or (type(cls._instance) is not cls):
            # Do not use the singleton instance of a superclass of cls.
            cls._instance = super(_SingletonConstraint, cls).__new__(cls)
        return cls._instance


class _Boolean(_SingletonConstraint):
    is_discrete = True

    def __call__(self, x):
        return (x == 0) | (x == 1)

    def feasible_like(self, prototype):
        return jax.numpy.zeros_like(prototype)


class _CorrCholesky(_SingletonConstraint):
    event_dim = 2

    def __call__(self, x):
        jnp = np if isinstance(x, (np.ndarray, np.generic)) else jax.numpy
        tril = jnp.tril(x)
        lower_triangular = jnp.all(
            jnp.reshape(tril == x, x.shape[:-2] + (-1,)), axis=-1
        )
        positive_diagonal = jnp.all(jnp.diagonal(x, axis1=-2, axis2=-1) > 0, axis=-1)
        x_norm = jnp.linalg.norm(x, axis=-1)
        tol = jnp.finfo(x.dtype).eps * x.shape[-1] * 10
        unit_norm_row = jnp.all(jnp.abs(x_norm - 1) <= tol, axis=-1)
        return lower_triangular & positive_diagonal & unit_norm_row

    def feasible_like(self, prototype):
        return jax.numpy.broadcast_to(
            jax.numpy.eye(prototype.shape[-1]), prototype.shape
        )


class _CorrMatrix(_SingletonConstraint):
    event_dim = 2

    def __call__(self, x):
        jnp = np if isinstance(x, (np.ndarray, np.generic)) else jax.numpy
        # check for symmetric
        symmetric = jnp.all(jnp.isclose(x, jnp.swapaxes(x, -2, -1)), axis=(-2, -1))
        # check for the smallest eigenvalue is positive
        positive = jnp.linalg.eigvalsh(x)[..., 0] > 0
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

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self._is_discrete == other._is_discrete
            and self._event_dim == other._event_dim
        )

    def tree_flatten(self):
        return (), (
            (),
            dict(_is_discrete=self._is_discrete, _event_dim=self._event_dim),
        )


class dependent_property(property, _Dependent):
    # XXX: this should not need to be pytree-able since it simply wraps a method
    # and thus is automatically present once the method's object is created
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

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += "(lower_bound={})".format(self.lower_bound)
        return fmt_string

    def feasible_like(self, prototype):
        return jax.numpy.broadcast_to(self.lower_bound + 1, jax.numpy.shape(prototype))

    def tree_flatten(self):
        return (self.lower_bound,), (("lower_bound",), dict())

    def __eq__(self, other):
        if not isinstance(other, _GreaterThan):
            return False
        return jnp.array_equal(self.lower_bound, other.lower_bound)


class _GreaterThanEq(_GreaterThan):
    def __call__(self, x):
        return x >= self.lower_bound

    def __eq__(self, other):
        if not isinstance(other, _GreaterThanEq):
            return False
        return jnp.array_equal(self.lower_bound, other.lower_bound)


class _Positive(_SingletonConstraint, _GreaterThan):
    def __init__(self):
        super().__init__(0.0)


class _Nonnegative(_SingletonConstraint, _GreaterThanEq):
    def __init__(self):
        super().__init__(0.0)


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

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__[1:],
            repr(self.base_constraint),
            self.reinterpreted_batch_ndims,
        )

    def feasible_like(self, prototype):
        return self.base_constraint.feasible_like(prototype)

    def tree_flatten(self):
        return (self.base_constraint,), (
            ("base_constraint",),
            {"reinterpreted_batch_ndims": self.reinterpreted_batch_ndims},
        )

    def __eq__(self, other):
        if not isinstance(other, _IndependentConstraint):
            return False

        return (self.base_constraint == other.base_constraint) & (
            self.reinterpreted_batch_ndims == other.reinterpreted_batch_ndims
        )


class _RealVector(_IndependentConstraint, _SingletonConstraint):
    def __init__(self):
        super().__init__(_Real(), 1)


class _RealMatrix(_IndependentConstraint, _SingletonConstraint):
    def __init__(self):
        super().__init__(_Real(), 2)


class _LessThan(Constraint):
    def __init__(self, upper_bound):
        self.upper_bound = upper_bound

    def __call__(self, x):
        return x < self.upper_bound

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += "(upper_bound={})".format(self.upper_bound)
        return fmt_string

    def feasible_like(self, prototype):
        return jax.numpy.broadcast_to(self.upper_bound - 1, jax.numpy.shape(prototype))

    def tree_flatten(self):
        return (self.upper_bound,), (("upper_bound",), dict())

    def __eq__(self, other):
        if not isinstance(other, _LessThan):
            return False
        return jnp.array_equal(self.upper_bound, other.upper_bound)


class _LessThanEq(_LessThan):
    def __call__(self, x):
        return x <= self.upper_bound

    def __eq__(self, other):
        if not isinstance(other, _LessThanEq):
            return False
        return jnp.array_equal(self.upper_bound, other.upper_bound)


class _IntegerInterval(Constraint):
    is_discrete = True

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, x):
        return (x >= self.lower_bound) & (x <= self.upper_bound) & (x % 1 == 0)

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += "(lower_bound={}, upper_bound={})".format(
            self.lower_bound, self.upper_bound
        )
        return fmt_string

    def feasible_like(self, prototype):
        return jax.numpy.broadcast_to(self.lower_bound, jax.numpy.shape(prototype))

    def tree_flatten(self):
        return (self.lower_bound, self.upper_bound), (
            ("lower_bound", "upper_bound"),
            dict(),
        )

    def __eq__(self, other):
        if not isinstance(other, _IntegerInterval):
            return False

        return jnp.array_equal(self.lower_bound, other.lower_bound) & jnp.array_equal(
            self.upper_bound, other.upper_bound
        )


class _IntegerGreaterThan(Constraint):
    is_discrete = True

    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def __call__(self, x):
        return (x % 1 == 0) & (x >= self.lower_bound)

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += "(lower_bound={})".format(self.lower_bound)
        return fmt_string

    def feasible_like(self, prototype):
        return jax.numpy.broadcast_to(self.lower_bound, jax.numpy.shape(prototype))

    def tree_flatten(self):
        return (self.lower_bound,), (("lower_bound",), dict())

    def __eq__(self, other):
        if not isinstance(other, _IntegerGreaterThan):
            return False
        return jnp.array_equal(self.lower_bound, other.lower_bound)


class _IntegerPositive(_SingletonConstraint, _IntegerGreaterThan):
    def __init__(self):
        super().__init__(1)


class _IntegerNonnegative(_SingletonConstraint, _IntegerGreaterThan):
    def __init__(self):
        super().__init__(0)


class _Interval(Constraint):
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, x):
        return (x >= self.lower_bound) & (x <= self.upper_bound)

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += "(lower_bound={}, upper_bound={})".format(
            self.lower_bound, self.upper_bound
        )
        return fmt_string

    def feasible_like(self, prototype):
        return jax.numpy.broadcast_to(
            (self.lower_bound + self.upper_bound) / 2, jax.numpy.shape(prototype)
        )

    def __eq__(self, other):
        if not isinstance(other, _Interval):
            return False
        return jnp.array_equal(self.lower_bound, other.lower_bound) & jnp.array_equal(
            self.upper_bound, other.upper_bound
        )

    def tree_flatten(self):
        return (self.lower_bound, self.upper_bound), (
            ("lower_bound", "upper_bound"),
            dict(),
        )


class _Circular(_SingletonConstraint, _Interval):
    def __init__(self):
        super().__init__(-math.pi, math.pi)


class _UnitInterval(_SingletonConstraint, _Interval):
    def __init__(self):
        super().__init__(0.0, 1.0)


class _OpenInterval(_Interval):
    def __call__(self, x):
        return (x > self.lower_bound) & (x < self.upper_bound)

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += "(lower_bound={}, upper_bound={})".format(
            self.lower_bound, self.upper_bound
        )
        return fmt_string


class _LowerCholesky(_SingletonConstraint):
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

    def tree_flatten(self):
        return (self.upper_bound,), (("upper_bound",), dict())

    def __eq__(self, other):
        if not isinstance(other, _Multinomial):
            return False
        return jnp.array_equal(self.upper_bound, other.upper_bound)


class _L1Ball(_SingletonConstraint):
    """
    Constrain to the L1 ball of any dimension.
    """

    event_dim = 1
    reltol = 10.0  # Relative to finfo.eps.

    def __call__(self, x):
        jnp = np if isinstance(x, (np.ndarray, np.generic)) else jax.numpy
        eps = jnp.finfo(x.dtype).eps
        return jnp.abs(x).sum(axis=-1) < 1 + self.reltol * eps

    def feasible_like(self, prototype):
        return jax.numpy.zeros_like(prototype)


class _OrderedVector(_SingletonConstraint):
    event_dim = 1

    def __call__(self, x):
        return (x[..., 1:] > x[..., :-1]).all(axis=-1)

    def feasible_like(self, prototype):
        return jax.numpy.broadcast_to(
            jax.numpy.arange(float(prototype.shape[-1])), prototype.shape
        )


class _PositiveDefinite(_SingletonConstraint):
    event_dim = 2

    def __call__(self, x):
        jnp = np if isinstance(x, (np.ndarray, np.generic)) else jax.numpy
        # check for symmetric
        symmetric = jnp.all(jnp.isclose(x, jnp.swapaxes(x, -2, -1)), axis=(-2, -1))
        # check for the smallest eigenvalue is positive
        positive = jnp.linalg.eigh(x)[0][..., 0] > 0
        return symmetric & positive

    def feasible_like(self, prototype):
        return jax.numpy.broadcast_to(
            jax.numpy.eye(prototype.shape[-1]), prototype.shape
        )


class _PositiveSemiDefinite(_SingletonConstraint):
    event_dim = 2

    def __call__(self, x):
        jnp = np if isinstance(x, (np.ndarray, np.generic)) else jax.numpy
        # check for symmetric
        symmetric = jnp.all(jnp.isclose(x, jnp.swapaxes(x, -2, -1)), axis=(-2, -1))
        # check for the smallest eigenvalue is nonnegative
        nonnegative = jnp.linalg.eigh(x)[0][..., 0] >= 0
        return symmetric & nonnegative

    def feasible_like(self, prototype):
        return jax.numpy.broadcast_to(
            jax.numpy.eye(prototype.shape[-1]), prototype.shape
        )


class _PositiveOrderedVector(_SingletonConstraint):
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


class _Complex(_SingletonConstraint):
    def __call__(self, x):
        # XXX: consider to relax this condition to [-inf, inf] interval
        return (x == x) & (x != float("inf")) & (x != float("-inf"))

    def feasible_like(self, prototype):
        return jax.numpy.zeros_like(prototype)


class _Real(_SingletonConstraint):
    def __call__(self, x):
        # XXX: consider to relax this condition to [-inf, inf] interval
        return (x == x) & (x != float("inf")) & (x != float("-inf"))

    def feasible_like(self, prototype):
        return jax.numpy.zeros_like(prototype)


class _Simplex(_SingletonConstraint):
    event_dim = 1

    def __call__(self, x):
        x_sum = x.sum(axis=-1)
        return (x >= 0).all(axis=-1) & (x_sum < 1 + 1e-6) & (x_sum > 1 - 1e-6)

    def feasible_like(self, prototype):
        return jax.numpy.full_like(prototype, 1 / prototype.shape[-1])


class _SoftplusPositive(_SingletonConstraint, _GreaterThan):
    def __init__(self):
        super().__init__(lower_bound=0.0)

    def feasible_like(self, prototype):
        return jax.numpy.full(jax.numpy.shape(prototype), np.log(2))


class _SoftplusLowerCholesky(_LowerCholesky):
    def feasible_like(self, prototype):
        return jax.numpy.broadcast_to(
            jax.numpy.eye(prototype.shape[-1]) * np.log(2), prototype.shape
        )


class _ScaledUnitLowerCholesky(_LowerCholesky):
    pass


class _Sphere(_SingletonConstraint):
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


class _ZeroSum(Constraint):
    def __init__(self, event_dim=1):
        self.event_dim = event_dim
        super().__init__()

    def __call__(self, x):
        jnp = np if isinstance(x, (np.ndarray, np.generic)) else jax.numpy
        tol = jnp.finfo(x.dtype).eps * x.shape[-1] * 10
        zerosum_true = True
        for dim in range(-self.event_dim, 0):
            zerosum_true = zerosum_true & jnp.allclose(x.sum(dim), 0, atol=tol)
        return zerosum_true

    def __eq__(self, other):
        return type(self) is type(other) and self.event_dim == other.event_dim

    def feasible_like(self, prototype):
        return jax.numpy.zeros_like(prototype)

    def tree_flatten(self):
        return (self.event_dim,), (("event_dim",), dict())


# TODO: Make types consistent
# See https://github.com/pytorch/pytorch/issues/50616

boolean = _Boolean()
circular = _Circular()
complex = _Complex()
corr_cholesky = _CorrCholesky()
corr_matrix = _CorrMatrix()
dependent = _Dependent()
greater_than = _GreaterThan
greater_than_eq = _GreaterThanEq
less_than = _LessThan
less_than_eq = _LessThanEq
independent = _IndependentConstraint
integer_interval = _IntegerInterval
integer_greater_than = _IntegerGreaterThan
interval = _Interval
l1_ball = _L1Ball()
lower_cholesky = _LowerCholesky()
scaled_unit_lower_cholesky = _ScaledUnitLowerCholesky()
multinomial = _Multinomial
nonnegative = _Nonnegative()
nonnegative_integer = _IntegerNonnegative()
ordered_vector = _OrderedVector()
positive = _Positive()
positive_definite = _PositiveDefinite()
positive_semidefinite = _PositiveSemiDefinite()
positive_integer = _IntegerPositive()
positive_ordered_vector = _PositiveOrderedVector()
real = _Real()
real_vector = _RealVector()
real_matrix = _RealMatrix()
simplex = _Simplex()
softplus_lower_cholesky = _SoftplusLowerCholesky()
softplus_positive = _SoftplusPositive()
sphere = _Sphere()
unit_interval = _UnitInterval()
open_interval = _OpenInterval
zero_sum = _ZeroSum
