# Source code modified from scipy.stats._discrete_distns.py
#
# Copyright (c) 2001, 2002 Enthought, Inc.
# All rights reserved.
#
# Copyright (c) 2003-2019 SciPy Developers.
# All rights reserved.

import numpy as onp
from scipy.stats._multivariate import multinomial_gen

import jax.numpy as np
from jax import device_put, lax
from jax.numpy.lax_numpy import _promote_dtypes
from jax.scipy.special import expit, gammaln

from numpyro.distributions.distribution import jax_discrete
from numpyro.distributions.util import binary_cross_entropy_with_logits, entr, promote_shapes, xlog1py, xlogy


class binom_gen(jax_discrete):
    _support_mask = constraints.

    def _rvs(self, n, p):
        # use scipy samplers directly and put the samples on device later.
        random_state = onp.random.RandomState(self._random_state)
        sample = random_state.binomial(n, p, self._size)
        return device_put(sample)

    def _logpmf(self, x, n, p):
        k = np.floor(x)
        n, p = _promote_dtypes(n, p)
        combiln = (gammaln(n + 1) - (gammaln(k + 1) + gammaln(n - k + 1)))
        return combiln + xlogy(k, p) + xlog1py(n - k, -p)

    def _entropy(self, n, p):
        k = np.arange(n + 1)
        vals = self._pmf(k, n, p)
        return np.sum(entr(vals), axis=0)


class bernoulli_gen(jax_discrete):
    _support_mask = constraints.integer_interval(0, 1)

    def __new__(cls, *args, **kwargs):
        return super(_bernoulli_gen, cls).__new__(cls)

    def __init__(self, *args, **kwargs):
        self.is_logits = kwargs.pop("is_logits", False)
        super(_bernoulli_gen, self).__init__(*args, **kwargs)

    def freeze(self, *args, **kwargs):
        self._ctor_param.update(is_logits=kwargs.pop("is_logits", False))
        return super(_bernoulli_gen, self).freeze(*args, **kwargs)

    def rvs(self, *args, **kwargs):
        if self.is_logits:
            # convert logits to probs
            if args:
                args = list(args)
                args[0] = expit(args[0])
            else:
                kwargs['p'] = expit(kwargs['p'])
        return super(_bernoulli_gen, self).rvs(*args, **kwargs)

    def _argcheck(self, p):
        if self.is_logits:
            return np.isfinite(p)
        else:
            return (p >= 0) & (p <= 1)

    def _logpmf(self, x, p):
        if self.is_logits:
            return -binary_cross_entropy_with_logits(p, x)
        else:
            # TODO: consider always clamp and convert probs to logits
            return xlogy(x, p) + xlog1py(1 - x, -p)

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

    def _process_parameters(self, n, p):
        p_ = 1. - p[..., :-1].sum(axis=-1)
        p, p_ = promote_shapes(p, p_)
        lax.dynamic_update_slice_in_dim(p, p_, 0, axis=-1)

        # true for bad p
        pcond = np.any(p < 0, axis=-1) | np.any(p > 1, axis=-1)

        # true for bad n
        n = np.array(n, dtype=np.int32)
        ncond = n <= 0

        return n, p, ncond | pcond

    def _process_quantiles(self, x, n, p):
        xx = x.astype(n.dtype)

        if xx.ndim == 0:
            raise ValueError("x must be an array.")

        if xx.size != 0 and not x.shape[-1] == p.shape[-1]:
            raise ValueError("Size of each quantile should be size of p: "
                             "received %d, but expected %d." %
                             (x.shape[-1], p.shape[-1]))

        # true for x out of the domain
        cond = np.any(xx != x, axis=-1) | np.any(xx < 0, axis=-1)
        cond = cond | (np.sum(xx, axis=-1) != n)
        return x, cond

    def _logpmf(self, x, n, p):
        n, p, x = _promote_dtypes(n, p, x)
        return gammaln(n + 1) + np.sum(xlogy(x, p) - gammaln(x + 1), axis=-1)

    def logpmf(self, x, n, p, args_check=True):
        n, p, npcond = self._process_parameters(n, p)
        x, xcond = self._process_quantiles(x, n, p)
        if args_check:
            if np.any(npcond):
                raise ValueError('Invalid distribution arguments provided to {}.logpmf'.format(self))
            if np.any(xcond):
                raise ValueError('Invalid values provided to {}.logpmf'.format(self))

        return device_put(self._logpmf(x, n, p))

    def entropy(self, n, p):
        n, p, npcond = self._process_parameters(n, p)

        x = np.arange(1, np.max(n) + 1)

        term1 = n * np.sum(entr(p), axis=-1) - gammaln(n + 1)

        n = n[..., np.newaxis]
        new_axes_needed = max(p.ndim, n.ndim) - x.ndim + 1
        x.shape += (1,) * new_axes_needed

        term2 = np.sum(binom.pmf(x, n, p) * gammaln(x + 1),
                       axis=(-1, -1 - new_axes_needed))

        return self._checkresult(term1 + term2, npcond, np.nan)

    def rvs(self, n, p, size=None, random_state=None):
        n, p, npcond = self._process_parameters(n, p)
        random_state = self._get_random_state(onp.random.RandomState(random_state))
        return device_put(random_state.multinomial(n, p, size))


bernoulli = _bernoulli_gen(b=1, name='bernoulli')
binom = _binom_gen(name='binom')
multinomial = _multinomial_gen()
