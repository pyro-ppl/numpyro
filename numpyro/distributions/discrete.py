# Source code modified from scipy.stats._discrete_distns.py
#
# Copyright (c) 2001, 2002 Enthought, Inc.
# All rights reserved.
#
# Copyright (c) 2003-2019 SciPy Developers.
# All rights reserved.

import jax.numpy as np
from jax.lax import lgamma
from jax.numpy.lax_numpy import _promote_dtypes
from scipy.stats._discrete_distns import binom_gen, bernoulli_gen

from numpyro.distributions.distribution import jax_discrete
from numpyro.distributions.util import entr, xlogy, xlog1py


class _binom_gen(jax_discrete, binom_gen):
    def _logpmf(self, x, n, p):
        k = np.floor(x)
        n, p = _promote_dtypes(n, p)
        combiln = (lgamma(n + 1) - (lgamma(k + 1) + lgamma(n - k + 1)))
        return combiln + xlogy(k, p) + xlog1py(n - k, -p)

    def _entropy(self, n, p):
        k = np.arange(n + 1)
        vals = self._pmf(k, n, p)
        return np.sum(entr(vals), axis=0)


class _bernoulli_gen(jax_discrete, bernoulli_gen):
    def _entropy(self, p):
        return entr(p) + entr(1 - p)


bernoulli = _bernoulli_gen(b=1, name='bernoulli')
binom = _binom_gen(name='binom')
