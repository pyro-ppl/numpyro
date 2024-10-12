# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache
import warnings

from multipledispatch import dispatch
import numpy as np

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax import bijectors as tfb, distributions as tfd

import numpyro.distributions as numpyro_dist
from numpyro.distributions import (
    Distribution as NumPyroDistribution,
    constraints,
    kl_divergence,
)
from numpyro.distributions.transforms import Transform, biject_to
from numpyro.util import find_stack_level, not_jax_tracer


def _get_codomain(bijector):
    if bijector.__class__.__name__ == "Sigmoid":
        return constraints.interval(bijector.low, bijector.high)
    elif bijector.__class__.__name__ == "Identity":
        return constraints.real
    elif bijector.__class__.__name__ in ["Exp", "SoftPlus"]:
        return constraints.positive
    elif bijector.__class__.__name__ == "GeneralizedPareto":
        loc, scale, concentration = bijector.loc, bijector.scale, bijector.concentration
        if not_jax_tracer(concentration) and np.all(np.less(concentration, 0)):
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

    @property
    def event_dim(self):
        return self.bijector.forward_min_event_ndims

    def __call__(self, x):
        return self.codomain(x)

    # a convenient property to inspect the actual support of a TFP distribution
    @property
    def codomain(self):
        return _get_codomain(self.bijector)

    def tree_flatten(self):
        return self.bijector, ()

    @classmethod
    def tree_unflatten(cls, _, bijector):
        return cls(bijector)


class BijectorTransform(Transform):
    """
    A wrapper for TensorFlow bijectors to make them compatible with NumPyro's transforms.

    :param ~tensorflow_probability.substrates.jax.bijectors.Bijector bijector: a TensorFlow bijector
    """

    def __init__(self, bijector):
        self.bijector = bijector

    @property
    def domain(self):
        return BijectorConstraint(tfb.Invert(self.bijector))

    @property
    def codomain(self):
        return BijectorConstraint(self.bijector)

    def __call__(self, x):
        return self.bijector.forward(x)

    def _inverse(self, y):
        return self.bijector.inverse(y)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return self.bijector.forward_log_det_jacobian(x, self.domain.event_dim)

    def forward_shape(self, shape):
        out_shape = self.bijector.forward_event_shape(shape)
        in_event_shape = self.bijector.inverse_event_shape(out_shape)
        batch_shape = shape[: len(shape) - len(in_event_shape)]
        return batch_shape + out_shape

    def inverse_shape(self, shape):
        in_shape = self.bijector.inverse_event_shape(shape)
        out_event_shape = self.bijector.forward_event_shape(in_shape)
        batch_shape = shape[: len(shape) - len(out_event_shape)]
        return batch_shape + in_shape

    def tree_flatten(self):
        return self.bijector, ()

    @classmethod
    def tree_unflatten(cls, _, bijector):
        return cls(bijector)


@biject_to.register(BijectorConstraint)
def _transform_to_bijector_constraint(constraint):
    return BijectorTransform(constraint.bijector)


def _onehot_enumerate_support(self, expand=True):
    n = self.event_shape[-1]
    values = jnp.identity(n, dtype=jnp.result_type(self.dtype))
    values = values.reshape((n,) + (1,) * len(self.batch_shape) + (n,))
    if expand:
        values = jnp.broadcast_to(values, (n,) + self.batch_shape + (n,))
    return values


class _TFPDistributionMeta(type(NumPyroDistribution)):
    @lru_cache(maxsize=None)
    def __getitem__(cls, tfd_class):
        assert issubclass(tfd_class, tfd.Distribution)

        tfd_class_name = tfd_class.__name__

        def init(self, *args, **kwargs):
            warnings.warn(
                "Importing distributions from numpyro.contrib.tfp.distributions is "
                "deprecated. You should import distributions directly from "
                "tensorflow_probability.substrates.jax.distributions instead.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
            self.tfp_dist = tfd_class(*args, **kwargs)

        _PyroDist = type(tfd_class_name, (TFPDistribution,), {})
        _PyroDist.__init__ = init

        if tfd_class is tfd.InverseGamma:
            _PyroDist.arg_constraints = {
                "concentration": constraints.positive,
                "scale": constraints.positive,
            }
        elif tfd_class is tfd.OneHotCategorical:
            _PyroDist.arg_constraints = {"logits": constraints.real_vector}
            _PyroDist.has_enumerate_support = True
            _PyroDist.support = constraints.simplex
            _PyroDist.is_discrete = True
            _PyroDist.enumerate_support = _onehot_enumerate_support
        elif tfd_class is tfd.OrderedLogistic:
            _PyroDist.arg_constraints = {
                "cutpoints": constraints.ordered_vector,
                "loc": constraints.real,
            }
        elif tfd_class is tfd.Pareto:
            _PyroDist.arg_constraints = {
                "concentration": constraints.positive,
                "scale": constraints.positive,
            }
        elif tfd_class is tfd.TruncatedNormal:
            _PyroDist.arg_constraints = {
                "low": constraints.real,
                "high": constraints.real,
                "loc": constraints.real,
                "scale": constraints.positive,
            }
        elif tfd_class is tfd.TruncatedCauchy:
            _PyroDist.arg_constraints = {
                "low": constraints.real,
                "high": constraints.real,
                "loc": constraints.real,
                "scale": constraints.positive,
            }
        else:
            # Mixture distributions in tfp and numpyro are different.
            if hasattr(numpyro_dist, tfd_class_name) and tfd_class_name != "Mixture":
                numpyro_dist_class = getattr(numpyro_dist, tfd_class_name)
                # resolve FooProbs/FooLogits namespaces
                numpyro_dist_class = getattr(
                    numpyro_dist, f"{tfd_class_name}Logits", numpyro_dist_class
                )
                _PyroDist.arg_constraints = numpyro_dist_class.arg_constraints
                _PyroDist.has_enumerate_support = (
                    numpyro_dist_class.has_enumerate_support
                )
                _PyroDist.enumerate_support = numpyro_dist_class.enumerate_support

        return _PyroDist


class TFPDistribution(NumPyroDistribution, metaclass=_TFPDistributionMeta):
    """
    A thin wrapper for TensorFlow Probability (TFP) distributions. The constructor
    has the same signature as the corresponding TFP distribution.

    This class can be used to convert a TFP distribution to a NumPyro-compatible one
    as follows::

        d = TFPDistribution[tfd.Normal](0, 1)

    Note that typical use cases do not require explicitly invoking this wrapper, since
    NumPyro wraps TFP distributions automatically under the hood in model code, e.g.::

        from tensorflow_probability.substrates.jax import distributions as tfd

        def model():
            numpyro.sample("x", tfd.Normal(0, 1))
    """

    def __getattr__(self, name):
        # return parameters from the constructor
        if name in self.tfp_dist.parameters:
            return self.tfp_dist.parameters[name]
        elif name in ["dtype", "reparameterization_type"]:
            return getattr(self.tfp_dist, name)
        raise AttributeError(name)

    @property
    def batch_shape(self):
        # TFP shapes are special tuples that can not be used directly
        # with lax.broadcast_shapes. So we convert them to tuple.
        return tuple(self.tfp_dist.batch_shape)

    @property
    def event_shape(self):
        return tuple(self.tfp_dist.event_shape)

    @property
    def has_rsample(self):
        return self.tfp_dist.reparameterization_type is tfd.FULLY_REPARAMETERIZED

    def sample(self, key, sample_shape=()):
        return self.tfp_dist.sample(sample_shape=sample_shape, seed=key)

    def log_prob(self, value):
        return self.tfp_dist.log_prob(value)

    @property
    def mean(self):
        return self.tfp_dist.mean()

    @property
    def variance(self):
        return self.tfp_dist.variance()

    def cdf(self, value):
        return self.tfp_dist.cdf(value)

    def icdf(self, q):
        return self.tfp_dist.quantile(q)

    @property
    def support(self):
        bijector = self.tfp_dist._default_event_space_bijector()
        if bijector is not None:
            return BijectorConstraint(bijector)
        else:
            return None

    @property
    def is_discrete(self):
        # XXX: this should cover most cases
        return self.support is None

    def tree_flatten(self):
        return jax.tree.flatten(self.tfp_dist)

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        fn = jax.tree.unflatten(aux_data, params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            return TFPDistribution[fn.__class__](**fn.parameters)


@dispatch(TFPDistribution, TFPDistribution)
def kl_divergence(p, q):  # noqa: F811
    return tfd.kl_divergence(p.tfp_dist, q.tfp_dist)


__all__ = ["BijectorConstraint", "BijectorTransform", "TFPDistribution"]
_len_all = len(__all__)
for _name, _Dist in tfd.__dict__.items():
    if not isinstance(_Dist, type):
        continue
    if not issubclass(_Dist, tfd.Distribution):
        continue
    if _Dist is tfd.Distribution:
        continue

    _PyroDist = TFPDistribution[_Dist]
    _PyroDist.__module__ = __name__
    locals()[_name] = _PyroDist

    _PyroDist.__doc__ = """
    Wraps `{}.{} <https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/distributions/{}>`_
    with :class:`~numpyro.contrib.tfp.distributions.TFPDistribution`.
    """.format(_Dist.__module__, _Dist.__name__, _Dist.__name__)

    __all__.append(_name)


# Create sphinx documentation.
__doc__ = "\n\n".join(
    [
        """
    {0}
    ----------------------------------------------------------------
    .. autoclass:: numpyro.contrib.tfp.distributions.{0}
    """.format(_name)
        for _name in __all__[:_len_all]
    ]
)
