# The implementation follows the design in PyTorch: torch.distributions.constraints.py

import jax.numpy as np


class Constraint(object):
    def __call__(self, x):
        raise NotImplementedError


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


greater_than = _GreaterThan
integer_interval = _IntegerInterval
interval = _Interval
positive = _GreaterThan(0)
real = _Real()
simplex = _Simplex()
unit_interval = _Interval(0, 1)
