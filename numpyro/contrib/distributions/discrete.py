# Source code modified from scipy.stats._discrete_distns.py
#
# Copyright (c) 2001, 2002 Enthought, Inc.
# All rights reserved.
#
# Copyright (c) 2003-2019 SciPy Developers.
# All rights reserved.

import numpy as onp

from jax import device_put, random
import jax.numpy as np
from jax.numpy.lax_numpy import _promote_dtypes
from jax.scipy.special import entr, expit, gammaln, xlog1py, xlogy

from numpyro.contrib.distributions.distribution import jax_discrete
from numpyro.distributions import constraints
from numpyro.distributions.util import binary_cross_entropy_with_logits


class bernoulli_gen(jax_discrete):
    _support_mask = constraints.integer_interval(0, 1)

    @property
    def arg_constraints(self):
        if self.is_logits:
            return {'p': constraints.real}
        else:
            return {'p': constraints.unit_interval}

    def _rvs(self, p):
        if self.is_logits:
            p = expit(p)
        return random.bernoulli(self._random_state, p, self._size)

    def _logpmf(self, x, p):
        if self.is_logits:
            return -binary_cross_entropy_with_logits(p, x)
        else:
            # TODO: consider always clamp and convert probs to logits
            return xlogy(x, p) + xlog1py(1 - x, -p)

    def _pmf(self, x, p):
        return np.exp(self._logpmf(x, p))

    def _cdf(self, x, p):
        return binom._cdf(x, 1, p)

    def _sf(self, x, p):
        return binom._sf(x, 1, p)

    def _ppf(self, q, p):
        return binom._ppf(q, 1, p)

    def _stats(self, p):
        return binom._stats(1, p)

    def _entropy(self, p):
        # TODO: use logits and binary_cross_entropy_with_logits for more stable
        if self.is_logits:
            p = expit(p)
        return entr(p) + entr(1 - p)


class binom_gen(jax_discrete):
    @property
    def arg_constraints(self):
        if self.is_logits:
            return {'n': constraints.nonnegative_integer,
                    'p': constraints.real}
        else:
            return {'n': constraints.nonnegative_integer,
                    'p': constraints.unit_interval}

    def _support(self, *args, **kwargs):
        (n, p), loc, _ = self._parse_args(*args, **kwargs)
        return constraints.integer_interval(loc, loc + n)

    def _rvs(self, n, p):
        if self.is_logits:
            p = expit(p)
        # use scipy samplers directly and put the samples on device later.
        # TODO: use util.binomial instead
        random_state = onp.random.RandomState(self._random_state)
        sample = random_state.binomial(n, p, self._size)
        return device_put(sample)

    def _logpmf(self, x, n, p):
        x, n, p = _promote_dtypes(x, n, p)
        combiln = gammaln(n + 1) - (gammaln(x + 1) + gammaln(n - x + 1))
        if self.is_logits:
            # TODO: move this implementation to PyTorch if it does not get non-continuous problem
            # In PyTorch, k * logit - n * log1p(e^logit) get overflow when logit is a large
            # positive number. In that case, we can reformulate into
            # k * logit - n * log1p(e^logit) = k * logit - n * (log1p(e^-logit) + logit)
            #                                = k * logit - n * logit - n * log1p(e^-logit)
            # More context: https://github.com/pytorch/pytorch/pull/15962/
            return combiln + x * p - (n * np.clip(p, 0) + xlog1py(n, np.exp(-np.abs(p))))
        else:
            return combiln + xlogy(x, p) + xlog1py(n - x, -p)

    def _pmf(self, x, n, p):
        return np.exp(self._logpmf(x, n, p))

    def _cdf(self, x, n, p):
        raise NotImplementedError('Missing jax.scipy.special.bdtr')

    def _sf(self, x, n, p):
        raise NotImplementedError('Missing jax.scipy.special.bdtrc')

    def _ppf(self, q, n, p):
        raise NotImplementedError('Missing jax.scipy.special.bdtrk')

    def _stats(self, n, p, moments='mv'):
        q = 1.0 - p
        mu = n * p
        var = n * p * q
        g1, g2 = None, None
        if 's' in moments:
            g1 = (q - p) / np.sqrt(var)
        if 'k' in moments:
            g2 = (1.0 - 6 * p * q) / var
        return mu, var, g1, g2

    def _entropy(self, n, p):
        if self.is_logits:
            p = expit(p)
        k = np.arange(n + 1)
        vals = self._pmf(k, n, p)
        return np.sum(entr(vals), axis=0)


class poisson_gen(jax_discrete):
    arg_constraints = {'mu': constraints.positive}
    _support_mask = constraints.nonnegative_integer

    def _rvs(self, mu):
        random_state = onp.random.RandomState(self._random_state)
        sample = random_state.poisson(mu, self._size)
        return device_put(sample)

    def _logpmf(self, x, mu):
        x, mu = _promote_dtypes(x, mu)
        Pk = xlogy(x, mu) - gammaln(x + 1) - mu
        return Pk

    def _pmf(self, x, mu):
        # poisson.pmf(k) = exp(-mu) * mu**k / k!
        return np.exp(self._logpmf(x, mu))

    def _cdf(self, x, mu):
        raise NotImplementedError('Missing jax.scipy.special.pdtr')

    def _sf(self, x, mu):
        raise NotImplementedError('Missing jax.scipy.special.pdtrc')

    def _ppf(self, q, mu):
        raise NotImplementedError('Missing jax.scipy.special.pdtrk')

    def _stats(self, mu):
        var = mu
        tmp = np.asarray(mu)
        mu_nonzero = tmp > 0
        g1 = np.where(mu_nonzero, np.sqrt(1.0 / tmp), np.inf)
        g2 = np.where(mu_nonzero, 1.0 / tmp, np.inf)
        return mu, var, g1, g2


bernoulli = bernoulli_gen(b=1, name='bernoulli')
binom = binom_gen(name='binom')
poisson = poisson_gen(name='poisson')
