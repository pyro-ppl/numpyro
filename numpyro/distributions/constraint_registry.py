# The implementation follows the design in PyTorch: torch.distributions.constraint_registry.py

from numpyro.distributions import constraints
from numpyro.distributions.transforms import (
    AffineTransform,
    ComposeTransform,
    ExpTransform,
    IdentityTransform,
    SigmoidTransform,
    StickBreakingTransform
)


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


@biject_to.register(constraints.greater_than)
def _transform_to_greater_than(constraint):
    return ComposeTransform([ExpTransform(),
                             AffineTransform(constraint.lower_bound, 1,
                                             domain=constraints.positive)])


@biject_to.register(constraints.interval)
def _transform_to_interval(constraint):
    return ComposeTransform([SigmoidTransform(),
                             AffineTransform(constraint.lower_bound, constraint.upper_bound,
                                             domain=constraints.unit_interval)])


@biject_to.register(constraints.real)
def _transform_to_real(constraint):
    return IdentityTransform()


@biject_to.register(constraints.simplex)
def _transform_to_simplex(constraint):
    return StickBreakingTransform()
