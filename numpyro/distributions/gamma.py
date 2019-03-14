# Source code modified from scipy.stats._continous_distns.py
#
# Copyright (c) 2001, 2002 Enthought, Inc.
# All rights reserved.
#
# Copyright (c) 2003-2019 SciPy Developers.
# All rights reserved.

import jax.numpy as np
from jax.scipy import special

from numpyro.distributions.distribution import jax_continuous
from numpyro.distributions.util import standard_gamma


class gamma_gen(jax_continuous):
    def _rvs(self, a):
        return standard_gamma(self._random_state, a, self._size)

    # TODO: add _cdf/_sf methods when incomplete gamma is available
    # https://github.com/google/jax/issues/479
    def _cdf(self, x, a):
        raise NotImplementedError

    def _stats(self, a):
        return a, a, 2.0 / np.sqrt(a), 6.0 / a

    def _entropy(self, a):
        return special.digamma(a) * (1 - a) + a + special.gammaln(a)


gamma = gamma_gen(a=0.0, name='gamma')
