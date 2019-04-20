# Source code modified from scipy.stats._continous_distns.py
#
# Copyright (c) 2001, 2002 Enthought, Inc.
# All rights reserved.
#
# Copyright (c) 2003-2019 SciPy Developers.
# All rights reserved.


import jax.numpy as np
import jax.random as random
import jax.scipy.stats as lsp_stats
from jax.scipy.special import digamma, gammaln

from numpyro.distributions import constraints
from numpyro.distributions.distribution import jax_continuous
from numpyro.distributions.util import standard_gamma


class beta_gen(jax_continuous):
    arg_constraints = {"a": constraints.positive, "b": constraints.positive}
    _support_mask = constraints.unit_interval

    def _rvs(self, a, b):
        # TODO: use upstream implementation when available
        # XXX the implementation is different from PyTorch's one
        # in PyTorch, a sample is generated from dirichlet distribution
        key_a, key_b = random.split(self._random_state)
        gamma_a = standard_gamma(key_a, a, shape=self._size)
        gamma_b = standard_gamma(key_b, b, shape=self._size)
        return gamma_a / (gamma_a + gamma_b)

    def _cdf(self, x, a, b):
        raise NotImplementedError

    def _stats(self, a, b):
        mn = a * 1.0 / (a + b)
        var = (a * b * 1.0) / (a + b + 1.0) / (a + b) ** 2.0
        g1 = 2.0 * (b - a) * np.sqrt((1.0 + a + b) / (a * b)) / (2 + a + b)
        g2 = 6.0 * (a ** 3 + a ** 2 * (1 - 2 * b) + b ** 2 * (1 + b) - 2 * a * b * (2 + b))
        g2 = g2 / (a * b * (a + b + 2) * (a + b + 3))
        return mn, var, g1, g2


class cauchy_gen(jax_continuous):
    _support_mask = constraints.real

    def _rvs(self):
        return random.cauchy(self._random_state, shape=self._size)

    def _cdf(self, x):
        return 0.5 + 1.0 / np.pi * np.arctan(x)

    def _ppf(self, q):
        return np.tan(np.pi * q - np.pi / 2.0)

    def _sf(self, x):
        return 0.5 - 1.0 / np.pi * np.arctan(x)

    def _isf(self, q):
        return np.tan(np.pi / 2.0 - np.pi * q)

    def _stats(self):
        return np.nan, np.nan, np.nan, np.nan

    def _entropy(self):
        return np.log(4 * np.pi)


class expon_gen(jax_continuous):
    _support_mask = constraints.positive

    def _rvs(self):
        return random.exponential(self._random_state, shape=self._size)

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


class gamma_gen(jax_continuous):
    arg_constraints = {"a": constraints.positive}
    _support_mask = constraints.positive

    def _rvs(self, a):
        return standard_gamma(self._random_state, a, shape=self._size)

    # TODO: add _cdf/_sf methods when incomplete gamma is available
    # https://github.com/google/jax/issues/479
    def _cdf(self, x, a):
        raise NotImplementedError

    def _stats(self, a):
        return a, a, 2.0 / np.sqrt(a), 6.0 / a

    def _entropy(self, a):
        return digamma(a) * (1 - a) + a + gammaln(a)


class lognorm_gen(jax_continuous):
    arg_constraints = {"s": constraints.positive}
    _support_mask = constraints.positive

    def _rvs(self, s):
        # TODO: use upstream implementation when available
        return np.exp(s * random.normal(self._random_state, shape=self._size))

    def _pdf(self, x, s):
        # lognorm.pdf(x, s) = 1 / (s*x*sqrt(2*pi)) * exp(-1/2*(log(x)/s)**2)
        return np.exp(self._logpdf(x, s))

    def _logpdf(self, x, s):
        return np.where(x != 0,
                        -np.log(x) ** 2 / (2 * s ** 2) - np.log(s * x * np.sqrt(2 * np.pi)),
                        -np.inf)

    def _stats(self, s):
        p = np.exp(s * s)
        mu = np.sqrt(p)
        mu2 = p * (p - 1)
        g1 = np.sqrt(p - 1) * (2 + p)
        g2 = np.polyval([1, 2, 3, 0, -6.0], p)
        return mu, mu2, g1, g2

    def _entropy(self, s):
        return 0.5 * (1 + np.log(2 * np.pi) + 2 * np.log(s))


class norm_gen(jax_continuous):
    _support_mask = constraints.real

    def _rvs(self):
        return random.normal(self._random_state, shape=self._size)

    def _stats(self):
        return 0.0, 1.0, 0.0, 0.0

    def _entropy(self):
        return 0.5 * (np.log(2 * np.pi) + 1)

    def cdf(x, loc=0, scale=1):
        return lsp_stats.norm.cdf(x, loc, scale)

    def logcdf(x, loc=0, scale=1):
        return lsp_stats.norm.logcdf(x, loc, scale)


class t_gen(jax_continuous):
    arg_constraints = {"df": constraints.positive}
    _support_mask = constraints.real

    def _rvs(self, df):
        # TODO: use upstream implementation when available
        key_n, key_g = random.split(self._random_state)
        normal = random.normal(key_n, shape=self._size)
        half_df = df / 2.0
        gamma = standard_gamma(key_n, half_df, shape=self._size)
        return normal * np.sqrt(half_df / gamma)

    def _cdf(self, x, df):
        raise NotImplementedError

    def _stats(self, df):
        mu = np.where(df > 1, 0.0, np.inf)
        mu2 = np.where(df > 2, df / (df - 2.0), np.inf)
        mu2 = np.where(df <= 1, np.nan, mu2)
        g1 = np.where(df > 3, 0.0, np.nan)
        g2 = np.where(df > 4, 6.0 / (df - 4.0), np.inf)
        g2 = np.where(df <= 2, np.nan, g2)
        return mu, mu2, g1, g2


class uniform_gen(jax_continuous):
    _support_mask = constraints.unit_interval

    def _rvs(self):
        return random.uniform(self._random_state, shape=self._size)

    def _cdf(self, x):
        return x

    def _ppf(self, q):
        return q

    def _stats(self):
        return 0.5, 1.0 / 12, 0, -1.2

    def _entropy(self):
        return 0.0


beta = beta_gen(a=0.0, b=1.0, name='beta')
cauchy = cauchy_gen(name='cauchy')
expon = expon_gen(a=0.0, name='expon')
gamma = gamma_gen(a=0.0, name='gamma')
lognorm = lognorm_gen(a=0.0, name='lognorm')
norm = norm_gen(name='norm')
t = t_gen(name='t')
uniform = uniform_gen(a=0.0, b=1.0, name='uniform')
