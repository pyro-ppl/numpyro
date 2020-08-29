# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd

import numpyro.distributions as numpyro_dist
from numpyro.distributions import Distribution as NumPyroDistribution
from numpyro.distributions import constraints
from numpyro.distributions.transforms import Transform, biject_to
from numpyro.util import not_jax_tracer


class _CallableTuple(tuple):
    """
    A tuple that upon calling returns itself.
    """
    def __call__(self):
        return self


def _bijector_to_constraint(bijector):
    if bijector.__class__.__name__ == "Sigmoid":
        return constraints.interval(bijector.low, bijector.high)
    elif bijector.__class__.__name__ == "Identity":
        return constraints.real
    elif bijector.__class__.__name__ in ["Exp", "SoftPlus"]:
        return constraints.positive
    elif bijector.__class__.__name__ == "GeneralizedPareto":
        loc, scale, concentration = bijector.loc, bijector.scale, bijector.concentration
        if not_jax_tracer(concentration) and jnp.all(concentration < 0):
            return constraints.interval(loc, loc + scale / jnp.abs(concentration))
        # XXX: here we suppose concentration > 0
        # which is not true in general, but should cover enough usage cases
        else:
            return constraints.greater_than(loc)
    elif bijector.__class__.__name__ == "SoftmaxCentered":
        return constraints.simplex
    elif bijector.__class__.__name__ == "Chain":
        return _bijector_to_constraint(bijector.bijectors[-1])
    else:
        return constraints.real


class BijectorConstraint(constraints.Constraint):
    def __init__(self, bijector):
        self.bijector = bijector

    def __call__(self, x):
        return self.constraint(x)

    @property
    def constraint(self):
        return _bijector_to_constraint(self.bijector)


class BijectorTransform(Transform):
    def __init__(self, bijector):
        self.bijector = bijector

    @property
    def event_dim(self):
        return self.bijector.forward_min_event_ndims

    def __call__(self, x):
        return self.bijector.forward(x)

    def inv(self, y):
        return self.bijector.inverse(y)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return self.bijector.forward_log_det_jacobian(x, self.event_dim)


@biject_to.register(BijectorConstraint)
def _transform_to_bijector_constraint(constraint):
    return BijectorTransform(constraint.bijector)


class TFPDistributionMixin(NumPyroDistribution):
    """
    A mixin layer to make Tensorflow Probability (TFP) distribution compatible
    with NumPyro internal.
    """
    def __call__(self, *args, **kwargs):
        key = kwargs.pop('rng_key')
        return self.sample(*args, seed=key, **kwargs)

    # In TFP, batch_shape, event_shape are methods, so we need this workaround
    @property
    def batch_shape(self):
        return _CallableTuple(super().batch_shape())

    @property
    def event_shape(self):
        return _CallableTuple(super().event_shape())

    @property
    def support(self):
        bijector = self._default_event_space_bijector()
        if bijector is not None:
            return BijectorConstraint(bijector)
        else:
            return None

    @property
    def is_discrete(self):
        # XXX: this should cover most cases
        return self.support is None


__all__ = []
for _name, _Dist in tfd.__dict__.items():
    if not isinstance(_Dist, type):
        continue
    if not issubclass(_Dist, tfd.Distribution):
        continue
    if _Dist is tfd.Distribution:
        continue

    print(_name)
    try:
        _PyroDist = locals()[_name]
    except KeyError:
        _PyroDist = type(_name, (_Dist, TFPDistributionMixin), {})
        _PyroDist.__module__ = __name__
        if hasattr(numpyro_dist, _name):
            # TODO: write tests to check for consistency
            numpyro_dist_class = getattr(numpyro_dist, _name)
            # resolve FooProbs/FooLogits namespaces
            if type(numpyro_dist_class).__name__ == "function":
                if not hasattr(numpyro_dist, _name + "Logits"):
                    continue
                numpyro_dist_class = getattr(numpyro_dist, _name + "Logits")
            _PyroDist.arg_constraints = numpyro_dist_class.arg_constraints
            _PyroDist.has_enumerate_support = numpyro_dist_class.has_enumerate_support
            _PyroDist.enumerate_support = numpyro_dist_class.enumerate_support
        locals()[_name] = _PyroDist

    _PyroDist.__doc__ = '''
    Wraps :class:`{}.{}` with
    :class:`~numpyro.contrib.tfp.distributions.TFPDistributionMixin`.
    '''.format(_Dist.__module__, _Dist.__name__)

    __all__.append(_name)


# Create sphinx documentation.
__doc__ = '\n\n'.join([

    '''
    {0}
    ----------------------------------------------------------------
    .. autoclass:: numpyro.contrib.tfp.distributions.{0}
    '''.format(_name)
    for _name in sorted(__all__)
])
