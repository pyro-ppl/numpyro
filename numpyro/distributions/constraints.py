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
    'boolean',
    'corr_cholesky',
    'corr_matrix',
    'dependent',
    'greater_than',
    'integer_interval',
    'integer_greater_than',
    'interval',
    'is_dependent',
    'less_than',
    'lower_cholesky',
    'multinomial',
    'nonnegative_integer',
    'positive',
    'positive_definite',
    'positive_integer',
    'real',
    'real_vector',
    'simplex',
    'unit_interval',
    'Constraint',
]

import jax.numpy as jnp


class Constraint(object):
    """
    Abstract base class for constraints.

    A constraint object represents a region over which a variable is valid,
    e.g. within which a variable can be optimized.
    """

    def __call__(self, x):
        raise NotImplementedError

    def check(self, value):
        """
        Returns a byte tensor of `sample_shape + batch_shape` indicating
        whether each event in value satisfies this constraint.
        """
        return self(value)


class _Boolean(Constraint):
    def __call__(self, x):
        return (x == 0) | (x == 1)


class _CorrCholesky(Constraint):
    def __call__(self, x):
        tril = jnp.tril(x)
        lower_triangular = jnp.all(jnp.reshape(tril == x, x.shape[:-2] + (-1,)), axis=-1)
        positive_diagonal = jnp.all(jnp.diagonal(x, axis1=-2, axis2=-1) > 0, axis=-1)
        x_norm = jnp.linalg.norm(x, axis=-1)
        unit_norm_row = jnp.all((x_norm <= 1) & (x_norm > 1 - 1e-6), axis=-1)
        return lower_triangular & positive_diagonal & unit_norm_row


class _CorrMatrix(Constraint):
    def __call__(self, x):
        # check for symmetric
        symmetric = jnp.all(jnp.all(x == jnp.swapaxes(x, -2, -1), axis=-1), axis=-1)
        # check for the smallest eigenvalue is positive
        positive = jnp.linalg.eigh(x)[0][..., 0] > 0
        # check for diagonal equal to 1
        unit_variance = jnp.all(jnp.abs(jnp.diagonal(x, axis1=-2, axis2=-1) - 1) < 1e-6, axis=-1)
        return symmetric & positive & unit_variance


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


class _LessThan(Constraint):
    def __init__(self, upper_bound):
        self.upper_bound = upper_bound

    def __call__(self, x):
        return x < self.upper_bound


class _IntegerInterval(Constraint):
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, x):
        return (x >= self.lower_bound) & (x <= self.upper_bound) & (x == jnp.floor(x))


class _IntegerGreaterThan(Constraint):
    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def __call__(self, x):
        return (x % 1 == 0) & (x >= self.lower_bound)


class _Interval(Constraint):
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, x):
        return (x >= self.lower_bound) & (x <= self.upper_bound)


class _LowerCholesky(Constraint):
    def __call__(self, x):
        tril = jnp.tril(x)
        lower_triangular = jnp.all(jnp.reshape(tril == x, x.shape[:-2] + (-1,)), axis=-1)
        positive_diagonal = jnp.all(jnp.diagonal(x, axis1=-2, axis2=-1) > 0, axis=-1)
        return lower_triangular & positive_diagonal


class _Multinomial(Constraint):
    def __init__(self, upper_bound):
        self.upper_bound = upper_bound

    def __call__(self, x):
        return jnp.all(x >= 0, axis=-1) & (jnp.sum(x, -1) == self.upper_bound)


class _OrderedVector(Constraint):
    def __call__(self, x):
        return jnp.all(x[..., 1:] > x[..., :-1], axis=-1)


class _PositiveDefinite(Constraint):
    def __call__(self, x):
        # check for symmetric
        symmetric = jnp.all(jnp.all(x == jnp.swapaxes(x, -2, -1), axis=-1), axis=-1)
        # check for the smallest eigenvalue is positive
        positive = jnp.linalg.eigh(x)[0][..., 0] > 0
        return symmetric & positive


class _Real(Constraint):
    def __call__(self, x):
        # XXX: consider to relax this condition to [-inf, inf] interval
        return jnp.isfinite(x)


class _RealVector(Constraint):
    def __call__(self, x):
        return jnp.all(jnp.isfinite(x), axis=-1)


class _Simplex(Constraint):
    def __call__(self, x):
        x_sum = jnp.sum(x, axis=-1)
        return jnp.all(x >= 0, axis=-1) & (x_sum < 1 + 1e-6) & (x_sum > 1 - 1e-6)


# TODO: Make types consistent

boolean = _Boolean()
corr_cholesky = _CorrCholesky()
corr_matrix = _CorrMatrix()
dependent = _Dependent()
greater_than = _GreaterThan
less_than = _LessThan
integer_interval = _IntegerInterval
integer_greater_than = _IntegerGreaterThan
interval = _Interval
lower_cholesky = _LowerCholesky()
multinomial = _Multinomial
nonnegative_integer = _IntegerGreaterThan(0)
ordered_vector = _OrderedVector()
positive = _GreaterThan(0.)
positive_definite = _PositiveDefinite()
positive_integer = _IntegerGreaterThan(1)
real = _Real()
real_vector = _RealVector()
simplex = _Simplex()
unit_interval = _Interval(0., 1.)
