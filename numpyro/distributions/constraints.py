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


class Constraint(object):
    def __call__(self, x):
        raise NotImplementedError


class _Boolean(Constraint):
    def __call__(self, value):
        return (value == 0) | (value == 1)


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


class _Real(Constraint):
    def __call__(self, x):
        return np.isfinite(x)


class _Simplex(Constraint):
    def __call__(self, x):
        x_sum = np.sum(x, axis=-1)
        return np.all(x > 0, axis=-1) & (x_sum <= 1) & (x_sum > 1 - 1e-6)


boolean = _Boolean()
dependent = _Dependent
greater_than = _GreaterThan
integer_interval = _IntegerInterval
integer_greater_than = _IntegerGreaterThan
interval = _Interval
nonnegative_integer = _IntegerGreaterThan(0)
positive_integer = _IntegerGreaterThan(1)
positive = _GreaterThan(0)
real = _Real()
simplex = _Simplex()
unit_interval = _Interval(0, 1)
