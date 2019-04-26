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
from jax.scipy.special import digamma, gammaln, log_ndtr, ndtr, ndtri

from numpyro.distributions import constraints
from numpyro.distributions.distribution import jax_continuous
from numpyro.distributions.util import standard_gamma


class beta_gen(jax_continuous):
    arg_constraints = {'a': constraints.positive, 'b': constraints.positive}
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
        raise NotImplementedError('Missing jax.scipy.special.btdtr')

    def _ppf(self, q, a, b):
        raise NotImplementedError('Missing jax.scipy.special.btdtri')

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
    arg_constraints = {'a': constraints.positive}
    _support_mask = constraints.positive

    def _rvs(self, a):
        return standard_gamma(self._random_state, a, shape=self._size)

    def _cdf(self, x, a):
        raise NotImplementedError('Missing jax.scipy.special.gammainc')

    def _sf(self, x, a):
        raise NotImplementedError('Missing jax.scipy.special.gammainc')

    def _ppf(self, q, a):
        raise NotImplementedError('Missing jax.scipy.special.gammaincinv')

    def _stats(self, a):
        return a, a, 2.0 / np.sqrt(a), 6.0 / a

    def _entropy(self, a):
        return digamma(a) * (1 - a) + a + gammaln(a)


class halfcauchy_gen(jax_continuous):
    _support_mask = constraints.positive

    def _rvs(self):
        return np.abs(random.cauchy(self._random_state, shape=self._size))

    def _pdf(self, x):
        return 2.0 / np.pi / (1.0 + x * x)

    def _logpdf(self, x):
        return np.log(2.0 / np.pi) - np.log1p(x * x)

    def _cdf(self, x):
        return 2.0 / np.pi * np.arctan(x)

    def _ppf(self, q):
        return np.tan(np.pi / 2 * q)

    def _stats(self):
        return np.inf, np.inf, np.nan, np.nan

    def _entropy(self):
        return np.log(2 * np.pi)


class halfnorm_gen(rv_continuous):
    _support_mask = constraints.positive

    def _rvs(self):
        return np.abs(random.normal(self._random_state, shape=self._size))

    def _pdf(self, x):
        # halfnorm.pdf(x) = sqrt(2/pi) * exp(-x**2/2)
        return np.sqrt(2.0 / np.pi) * np.exp(-x * x / 2.0)

    def _logpdf(self, x):
        return 0.5 * np.log(2.0 / np.pi) - x * x / 2.0

    def _cdf(self, x):
        return norm._cdf(x) * 2 - 1.0

    def _ppf(self, q):
        return norm._ppf((1 + q) / 2.0)

    def _stats(self):
        return (np.sqrt(2.0 / np.pi), 1 - 2.0 / np.pi,
                np.sqrt(2) * (4 - np.pi) / (np.pi - 2) ** 1.5, 8 * (np.pi - 3) / (np.pi - 2) **2)

    def _entropy(self):
        return 0.5 * np.log( np.pi / 2.0) + 0.5


class lognorm_gen(jax_continuous):
    arg_constraints = {'s': constraints.positive}
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

    def _cdf(self, x, s):
        return norm._cdf(np.log(x) / s)

    def _logcdf(self, x, s):
        return norm._logcdf(np.log(x) / s)

    def _ppf(self, q, s):
        return np.exp(s * norm._ppf(q))

    def _sf(self, x, s):
        return norm._sf(np.log(x) / s)

    def _logsf(self, x, s):
        return norm._logsf(-np.log(x) / s)

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

    def _pdf(self, x):
        # norm.pdf(x) = exp(-x**2/2)/sqrt(2*pi)
        return np.exp(-x**2 / 2.0) / np.sqrt(2 * np.pi)

    def _logpdf(self, x):
        return -(x ** 2 + np.log(2 * np.pi)) / 2.0

    def _cdf(self, x):
        return ndtr(x)

    def _logcdf(self, x):
        return log_ndtr(x)

    def _sf(self, x):
        return ndtr(-x)

    def _logsf(self, x):
        return log_ndtr(-x)

    def _ppf(self, q):
        return ndtri(q)

    def _isf(self, q):
        return -ndtri(q)

    def _stats(self):
        return 0.0, 1.0, 0.0, 0.0

    def _entropy(self):
        return 0.5 * (np.log(2 * np.pi) + 1)


class pareto_gen(jax_continuous):
    arg_constraints = {'b': constraints.positive}
    _support_mask = constraints.greater_than(1)

    def _rvs(self, b):
        return random.pareto(self._random_state, b, shape=self._size)

    def _cdf(self, x, b):
        return 1 - x ** (-b)

    def _ppf(self, q, b):
        return np.pow(1 - q, -1.0 / b)

    def _sf(self, x, b):
        return x ** (-b)

    def _stats(self, b, moments='mv'):
        mu, mu2, g1, g2 = None, None, None, None
        if 'm' in moments:
            mask = b > 1
            bt = np.extract(mask, b)
            mu = np.where(mask, bt / (bt - 1.0), np.inf)
        if 'v' in moments:
            mask = b > 2
            bt = np.extract(mask, b)
            mu2 = np.where(mask, bt / (bt - 2.0) / (bt - 1.0) ** 2, np.inf)
        if 's' in moments:
            mask = b > 3
            bt = np.extract(mask, b)
            vals = 2 * (bt + 1.0) * np.sqrt(bt - 2.0) / ((bt - 3.0) * np.sqrt(bt))
            g1 = np.where(mask, vals, np.nan)
        if 'k' in moments:
            mask = b > 4
            bt = np.extract(mask, b)
            vals = (6.0 * np.polyval([1.0, 1.0, -6, -2], bt)
                    / np.polyval([1.0, -7.0, 12.0, 0.0], bt))
            g2 = np.where(mask, vals, np.nan)
        return mu, mu2, g1, g2

    def _entropy(self, c):
        return 1 + 1.0 / c - np.log(c)


pareto = pareto_gen(a=1.0, name="pareto")


class pareto_gen(jax_continuous):
    arg_constraints = {'b': constraints.positive}
    _support_mask = constraints.greater_than(1)

    def _rvs(self, b):
        return random.pareto(self._random_state, b, shape=self._size)

    def _cdf(self, x, b):
        return 1 - x ** (-b)

    def _ppf(self, q, b):
        return np.pow(1 - q, -1.0 / b)

    def _sf(self, x, b):
        return x ** (-b)

    def _stats(self, b, moments='mv'):
        mu, mu2, g1, g2 = None, None, None, None
        if 'm' in moments:
            mask = b > 1
            bt = np.extract(mask, b)
            mu = np.where(mask, bt / (bt - 1.0), np.inf)
        if 'v' in moments:
            mask = b > 2
            bt = np.extract(mask, b)
            mu2 = np.where(mask, bt / (bt - 2.0) / (bt - 1.0) ** 2, np.inf)
        if 's' in moments:
            mask = b > 3
            bt = np.extract(mask, b)
            vals = 2 * (bt + 1.0) * np.sqrt(bt - 2.0) / ((bt - 3.0) * np.sqrt(bt))
            g1 = np.where(mask, vals, np.nan)
        if 'k' in moments:
            mask = b > 4
            bt = np.extract(mask, b)
            vals = (6.0 * np.polyval([1.0, 1.0, -6, -2], bt)
                    / np.polyval([1.0, -7.0, 12.0, 0.0], bt))
            g2 = np.where(mask, vals, np.nan)
        return mu, mu2, g1, g2

    def _entropy(self, c):
        return 1 + 1.0 / c - np.log(c)


pareto = pareto_gen(a=1.0, name="pareto")


class t_gen(jax_continuous):
    arg_constraints = {'df': constraints.positive}
    _support_mask = constraints.real

    def _rvs(self, df):
        # TODO: use upstream implementation when available
        key_n, key_g = random.split(self._random_state)
        normal = random.normal(key_n, shape=self._size)
        half_df = df / 2.0
        gamma = standard_gamma(key_n, half_df, shape=self._size)
        return normal * np.sqrt(half_df / gamma)

    def _cdf(self, x, df):
        raise NotImplementedError('Missing jax.scipy.special.stdtr')

    def _sf(self, x, df):
        raise NotImplementedError('Missing jax.scipy.special.stdtr')

    def _ppf(self, q, df):
        raise NotImplementedError('Missing jax.scipy.special.stdtrit')

    def _isf(self, q, df):
        raise NotImplementedError('Missing jax.scipy.special.stdtrit')

    def _stats(self, df):
        mu = np.where(df > 1, 0.0, np.inf)
        mu2 = np.where(df > 2, df / (df - 2.0), np.inf)
        mu2 = np.where(df <= 1, np.nan, mu2)
        g1 = np.where(df > 3, 0.0, np.nan)
        g2 = np.where(df > 4, 6.0 / (df - 4.0), np.inf)
        g2 = np.where(df <= 2, np.nan, g2)
        return mu, mu2, g1, g2


class trunccauchy_gen(jax_continuous):
    # TODO: override _argcheck with the constraint that a < b

    def _support(self, *args, **kwargs):
        (a, b), loc, scale = self._parse_args(*args, **kwargs)
        # TODO: make constraints.less_than and support a == -np.inf
        if b == np.inf:
            return constraints.greater_than((a - loc) * scale)
        else:
            return constraints.interval((a - loc) * scale, (b - loc) * scale)

    def _rvs(self, a, b):
        # We use inverse transform method:
        # z ~ ppf(U), where U ~ Uniform(cdf(a), cdf(b)).
        #                     ~ Uniform(arctan(a), arctan(b)) / pi + 1/2
        u = random.uniform(self._random_state, shape=self._size,
                           minval=np.arctan(a), maxval=np.arctan(b))
        return np.tan(u)

    def _pdf(self, x, a, b):
        return np.reciprocal((1 + x * x) * (np.arctan(b) - np.arctan(a)))

    def _logpdf(self, x, a, b):
        # trunc_pdf(x) = pdf(x) / (cdf(b) - cdf(a))
        #              = 1 / (1 + x^2) / (arctan(b) - arctan(a))
        normalizer = np.log(np.arctan(b) - np.arctan(a))
        return -(np.log(1 + x * x) + normalizer)


class truncnorm_gen(jax_continuous):
    # TODO: override _argcheck with the constraint that a < b

    def _support(self, *args, **kwargs):
        (a, b), loc, scale = self._parse_args(*args, **kwargs)
        # TODO: make constraints.less_than and support a == -np.inf
        if b == np.inf:
            return constraints.greater_than((a - loc) * scale)
        else:
            return constraints.interval((a - loc) * scale, (b - loc) * scale)

    def _rvs(self, a, b):
        # We use inverse transform method:
        # z ~ ppf(U), where U ~ Uniform(0, 1).
        u = random.uniform(self._random_state, shape=self._size)
        return self._ppf(u, a, b)

    def _pdf(self, x, a, b):
        delta = np.where(a > 0,
                         norm._sf(a) - norm._sf(b),
                         norm._cdf(b) - norm._cdf(a))
        return norm._pdf(x) / self._delta

    def _logpdf(self, x, a, b):
        delta = np.where(a > 0,
                         norm._sf(a) - norm._sf(b),
                         norm._cdf(b) - norm._cdf(a))
        return norm._logpdf(x) - np.log(delta)

    def _cdf(self, x, a, b):
        delta = np.where(a > 0,
                         norm._sf(a) - norm._sf(b),
                         norm._cdf(b) - norm._cdf(a))
        return (norm._cdf(x) - norm._cdf(a)) / delta

    def _ppf(self, q, a, b):
        ppf = np.where(a > 0,
                       norm._isf(q * norm._sf(b) + norm._sf(a) * (1.0 - q)),
                       norm._ppf(q * norm._cdf(b) + norm._cdf(a) * (1.0 - q)))
        return ppf

    def _stats(self, a, b):
        nA, nB = norm._cdf(a), norm._cdf(b)
        d = nB - nA
        pA, pB = norm._pdf(a), norm._pdf(b)
        mu = (pA - pB) / d   # correction sign
        mu2 = 1 + (a * pA - b * pB) / d - mu * mu
        return mu, mu2, None, None


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
halfcauchy = halfcauchy_gen(a=0.0, name='halfcauchy')
<<<<<<< HEAD
halfnorm = halfnorm_gen(a=0.0, name='halfnorm')
=======
>>>>>>> upstream/master
lognorm = lognorm_gen(a=0.0, name='lognorm')
norm = norm_gen(name='norm')
t = t_gen(name='t')
trunccauchy = trunccauchy_gen(name='trunccauchy')
<<<<<<< HEAD
truncnorm = truncnorm_gen(name='truncnorm')
=======
>>>>>>> upstream/master
uniform = uniform_gen(a=0.0, b=1.0, name='uniform')
