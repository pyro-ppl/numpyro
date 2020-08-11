# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from tensorflow_probability.substrates.jax import distributions as tfd

import numpyro.distributions as numpyro_dist
from numpyro.distributions import Distribution as NumPyroDistribution
from numpyro.distributions import constraints


class TFPDistributionMixin(NumPyroDistribution):
    support = constraints.real

    def __call__(self, *args, **kwargs):
        key = kwargs.pop('rng_key')
        return self.sample(*args, seed=key, **kwargs)


__all__ = []
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
            # TODO: write tests to check for consistency
            numpyro_dist_class = getattr(numpyro_dist, _name)
            # resolve FooProbs/FooLogits namespaces
            if type(numpyro_dist_class).__name__ == "function":
                if not hasattr(numpyro_dist, _name + "Logits"):
                    continue
                numpyro_dist_class = getattr(numpyro_dist, _name + "Logits")
            _PyroDist.support = numpyro_dist_class.support
            _PyroDist.enumerate_support = numpyro_dist_class.enumerate_support
            _PyroDist.arg_constraints = numpyro_dist_class.arg_constraints
            _PyroDist.has_enumerate_support = numpyro_dist_class.has_enumerate_support
            _PyroDist.is_discrete = numpyro_dist_class.is_discrete
        # TODO: for those not appear in numpyro_dist (or there are some inconsistency
        # such as having different paramterization), make custom TFPDistributionMixin
        # for them
        locals()[_name] = _PyroDist

    _PyroDist.__doc__ = '''
    Wraps :class:`{}.{}` with
    :class:`~numpyro.contrib.tfp.distributions.TFPDistributionMixin`.
    '''.format(_Dist.__module__, _Dist.__name__)

    __all__.append(_name)
