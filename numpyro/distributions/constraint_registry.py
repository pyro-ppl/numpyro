# The implementation follows the design in PyTorch: torch.distributions.constraint_registry.py


class ConstraintRegistry(object):
    def __init__(self):
        self._registry = {}

    def register(self, constraint, factory):
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
    return transforms.ComposeTransform([transforms.ExpTransform(),
                                        transforms.AffineTransform(constraint.lower_bound, 1)])


@biject_to.register(constraints.interval)
def _transform_to_interval(constraint):
    return transforms.ComposeTransform([transforms.SigmoidTransform(),
                                        transforms.AffineTransform(constraint.lower_bound,
                                                                   constraint.upper_bound)])


@transform_to.register(constraints.real)
def _transform_to_real(constraint):
    return transforms.IdentityTransform


@biject_to.register(constraints.simplex)
def _transform_to_simplex(constraint):
    return transforms.StickBreakingTransform()
