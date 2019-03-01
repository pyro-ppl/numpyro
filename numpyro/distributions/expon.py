import jax.numpy as np
import jax.random as random

from numpyro.distributions.distribution import jax_continuous

# Copyright (c) 2001, 2002 Enthought, Inc.
# All rights reserved.
#
# Copyright (c) 2003-2019 SciPy Developers.
# All rights reserved.


class expon_gen(jax_continuous):
    r"""An exponential continuous random variable.
    %(before_notes)s
    Notes
    -----
    The probability density function for `expon` is:
    .. math::
        f(x) = \exp(-x)
    for :math:`x \ge 0`.
    %(after_notes)s
    A common parameterization for `expon` is in terms of the rate parameter
    ``lambda``, such that ``pdf = lambda * exp(-lambda * x)``. This
    parameterization corresponds to using ``scale = 1 / lambda``.
    %(example)s
    """
    def _rvs(self):
        u = random.uniform(self._random_state, self._size)
        return -np.log1p(-u)

    def _pdf(self, x):
        # expon.pdf(x) = exp(-x)
        return np.exp(-x)

    def _logpdf(self, x):
        return -x

    def _cdf(self, x):
        return -np.expm1(-x)

    def _ppf(self, q):
        return -np.log1p(-q)

    def _sf(self, x):
        return np.exp(-x)

    def _logsf(self, x):
        return -x

    def _isf(self, q):
        return -np.log(q)

    def _stats(self):
        return 1.0, 1.0, 2.0, 6.0

    def _entropy(self):
        return 1.0


expon = expon_gen(name='expon')
