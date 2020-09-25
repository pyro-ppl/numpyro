# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from jax.dtypes import canonicalize_dtype
import jax.numpy as jnp
from tensorflow_probability.substrates.jax import bijectors as tfb
from tensorflow_probability.substrates.jax import distributions as tfd

import numpyro.distributions as numpyro_dist
from numpyro.distributions import Distribution as NumPyroDistribution
from numpyro.distributions import constraints
from numpyro.distributions.transforms import Transform, biject_to
from numpyro.util import not_jax_tracer


def _get_codomain(bijector):
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
        return _get_codomain(bijector.bijectors[-1])
    else:
        return constraints.real


class BijectorConstraint(constraints.Constraint):
    """
    A constraint which is codomain of a TensorFlow bijector.

    :param ~tensorflow_probability.substrates.jax.bijectors.Bijector bijector: a TensorFlow bijector
    """
    def __init__(self, bijector):
        self.bijector = bijector

    def __call__(self, x):
        return self.codomain(x)

    # a convenient property to inspect the actual support of a TFP distribution
    @property
    def codomain(self):
        return _get_codomain(self.bijector)


class BijectorTransform(Transform):
    """
    A wrapper for TensorFlow bijectors to make them compatible with NumPyro's transforms.

    :param ~tensorflow_probability.substrates.jax.bijectors.Bijector bijector: a TensorFlow bijector
    """
    def __init__(self, bijector):
        self.bijector = bijector

    @property
    def event_dim(self):
        return self.bijector.forward_min_event_ndims

    @property
    def domain(self):
        return BijectorConstraint(tfb.Invert(self.bijector))

    @property
    def codomain(self):
        return BijectorConstraint(self.bijector)

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
    A mixin layer to make TensorFlow Probability (TFP) distribution compatible
    with NumPyro internal.
    """
    def __init_subclass__(cls, **kwargs):
        # skip register pytree because TFP distributions are already pytrees
        super(object, cls).__init_subclass__(**kwargs)

    def __call__(self, *args, **kwargs):
        key = kwargs.pop('rng_key')
        kwargs.pop('sample_intermediates', False)
        return self.sample(*args, seed=key, **kwargs)

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


class InverseGamma(tfd.InverseGamma):
    arg_constraints = {"concentration": constraints.positive, "scale": constraints.positive}


class OneHotCategorical(tfd.OneHotCategorical):
    arg_constraints = {"logits": constraints.real_vector}
    has_enumerate_support = True
    support = constraints.simplex
    is_discrete = True

    def enumerate_support(self, expand=True):
        n = self.event_shape[-1]
        values = jnp.identity(n, dtype=canonicalize_dtype(self.dtype))
        values = values.reshape((n,) + (1,) * len(self.batch_shape) + (n,))
        if expand:
            values = jnp.broadcast_to(values, (n,) + self.batch_shape + (n,))
        return values


class OrderedLogistic(tfd.OrderedLogistic):
    arg_constraints = {"cutpoints": constraints.ordered_vector, "loc": constraints.real}


class Pareto(tfd.Pareto):
    arg_constraints = {"concentration": constraints.positive, "scale": constraints.positive}


__all__ = ['BijectorConstraint', 'BijectorTransform', 'TFPDistributionMixin']
_len_all = len(__all__)
for _name, _Dist in tfd.__dict__.items():
    if not isinstance(_Dist, type):
        continue
    if not issubclass(_Dist, tfd.Distribution):
        continue
    if _Dist is tfd.Distribution:
        continue

    try:
        _PyroDist = locals()[_name]
    except KeyError:
        _PyroDist = type(_name, (_Dist, TFPDistributionMixin), {})
        _PyroDist.__module__ = __name__
        if hasattr(numpyro_dist, _name):
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
    Wraps `{}.{} <https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/distributions/{}>`_
    with :class:`~numpyro.contrib.tfp.distributions.TFPDistributionMixin`.
    '''.format(_Dist.__module__, _Dist.__name__, _Dist.__name__)

    __all__.append(_name)


# Create sphinx documentation.
__doc__ = '\n\n'.join([

    '''
    {0}
    ----------------------------------------------------------------
    .. autoclass:: numpyro.contrib.tfp.distributions.{0}
    '''.format(_name)
    for _name in __all__[:_len_all] + sorted(__all__[_len_all:])
])
