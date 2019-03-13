# Source code modified from scipy.stats._discrete_distns.py
#
# Copyright (c) 2001, 2002 Enthought, Inc.
# All rights reserved.
#
# Copyright (c) 2003-2019 SciPy Developers.
# All rights reserved.

import jax.numpy as np
from jax import device_put, lax
from jax.lax import lgamma
from jax.numpy.lax_numpy import _promote_dtypes
import numpy as onp
from scipy.stats._discrete_distns import binom_gen, bernoulli_gen
from scipy.stats._multivariate import multinomial_gen

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


class _multinomial_gen(multinomial_gen):
    def __init__(self, seed=None, name='multinomial'):
        self.name = name
        super(multinomial_gen, self).__init__(seed)

    def _checkresult(self, result, cond, bad_value):
        if cond.ndim != 0:
            result = np.where(cond, bad_value, result)
        elif cond:
            if result.ndim == 0:
                return bad_value
            result = lax.full_like(result, bad_value)
        return device_put(result)

    def _process_quantiles(self, x, n, p):
        xx = x.astype(n.dtype)

        if xx.ndim == 0:
            raise ValueError("x must be an array.")

        if xx.size != 0 and not x.shape[-1] == p.shape[-1]:
            raise ValueError("Size of each quantile should be size of p: "
                             "received %d, but expected %d." %
                             (x.shape[-1], p.shape[-1]))

        # true for x out of the domain
        cond = np.any(xx != x, axis=-1)
        cond |= np.any(xx < 0, axis=-1)
        cond = cond | (np.sum(xx, axis=-1) != n)
        return x, cond

    def _logpmf(self, x, n, p):
        n, p, x = _promote_dtypes(n, p, x)
        return lgamma(n+1) + np.sum(xlogy(x, p) - lgamma(x+1), axis=-1)

    def logpmf(self, x, n, p):
        n, p, npcond = self._process_parameters(n, p)
        x, xcond = self._process_quantiles(x, n, p)

        result = self._logpmf(x, n, p)

        # replace values for which x was out of the domain; broadcast
        # xcond to the right shape
        xcond_ = xcond | np.zeros(npcond.shape, dtype=np.bool_)
        result = self._checkresult(result, xcond_, np.NINF)

        # replace values bad for n or p; broadcast npcond to the right shape
        npcond_ = npcond | np.zeros(xcond.shape, dtype=np.bool_)
        return self._checkresult(result, npcond_, np.nan)

    def entropy(self, n, p):
        n, p, npcond = self._process_parameters(n, p)

        x = np.arange(1, np.max(n)+1)

        term1 = n*np.sum(entr(p), axis=-1)
        term1 -= lgamma(n+1)

        n = n[..., np.newaxis]
        new_axes_needed = max(p.ndim, n.ndim) - x.ndim + 1
        x.shape += (1,)*new_axes_needed

        term2 = np.sum(binom.pmf(x, n, p)*lgamma(x+1),
                       axis=(-1, -1-new_axes_needed))

        return self._checkresult(term1 + term2, npcond, np.nan)

    def rvs(self, n, p, size=None, random_state=None):
        n, p, npcond = self._process_parameters(n, p)
        random_state = self._get_random_state(onp.random.RandomState(random_state))
        return device_put(random_state.multinomial(n, p, size))


bernoulli = _bernoulli_gen(b=1, name='bernoulli')
binom = _binom_gen(name='binom')
multinomial = _multinomial_gen()
