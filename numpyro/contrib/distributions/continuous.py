# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

# Source code modified from scipy.stats._continous_distns.py
#
# Copyright (c) 2001, 2002 Enthought, Inc.
# All rights reserved.
#
# Copyright (c) 2003-2019 SciPy Developers.
# All rights reserved.

import jax.numpy as jnp
from jax.numpy.lax_numpy import _promote_dtypes
import jax.random as random
from jax.scipy.special import digamma, gammaln, log_ndtr, ndtr, ndtri

from numpyro.contrib.distributions.distribution import jax_continuous
from numpyro.distributions import constraints


class beta_gen(jax_continuous):
    arg_constraints = {'a': constraints.positive, 'b': constraints.positive}
    _support_mask = constraints.unit_interval

    def _rvs(self, a, b):
        # TODO: use upstream implementation when available
        # XXX the implementation is different from PyTorch's one
        # in PyTorch, a sample is generated from dirichlet distribution
        key_a, key_b = random.split(self._random_state)
        gamma_a = random.gamma(key_a, a, shape=self._size)
        gamma_b = random.gamma(key_b, b, shape=self._size)
        return gamma_a / (gamma_a + gamma_b)

    def _cdf(self, x, a, b):
        raise NotImplementedError('Missing jax.scipy.special.btdtr')

    def _ppf(self, q, a, b):
        raise NotImplementedError('Missing jax.scipy.special.btdtri')

    def _stats(self, a, b):
        mn = a * 1.0 / (a + b)
        var = (a * b * 1.0) / (a + b + 1.0) / (a + b) ** 2.0
        g1 = 2.0 * (b - a) * jnp.sqrt((1.0 + a + b) / (a * b)) / (2 + a + b)
        g2 = 6.0 * (a ** 3 + a ** 2 * (1 - 2 * b) + b ** 2 * (1 + b) - 2 * a * b * (2 + b))
        g2 = g2 / (a * b * (a + b + 2) * (a + b + 3))
        return mn, var, g1, g2


class cauchy_gen(jax_continuous):
    _support_mask = constraints.real

    def _rvs(self):
        return random.cauchy(self._random_state, shape=self._size)

    def _cdf(self, x):
        return 0.5 + 1.0 / jnp.pi * jnp.arctan(x)

    def _ppf(self, q):
        return jnp.tan(jnp.pi * q - jnp.pi / 2.0)

    def _sf(self, x):
        return 0.5 - 1.0 / jnp.pi * jnp.arctan(x)

    def _isf(self, q):
        return jnp.tan(jnp.pi / 2.0 - jnp.pi * q)

    def _stats(self):
        return jnp.nan, jnp.nan, jnp.nan, jnp.nan

    def _entropy(self):
        return jnp.log(4 * jnp.pi)


class expon_gen(jax_continuous):
    _support_mask = constraints.positive

    def _rvs(self):
        return random.exponential(self._random_state, shape=self._size)

    def _cdf(self, x):
        return -jnp.expm1(-x)

    def _ppf(self, q):
        return -jnp.log1p(-q)

    def _sf(self, x):
        return jnp.exp(-x)

    def _logsf(self, x):
        return -x

    def _isf(self, q):
        return -jnp.log(q)

    def _stats(self):
        return 1.0, 1.0, 2.0, 6.0

    def _entropy(self):
        return 1.0


class gamma_gen(jax_continuous):
    arg_constraints = {'a': constraints.positive}
    _support_mask = constraints.positive

    def _rvs(self, a):
        return random.gamma(self._random_state, a, shape=self._size)

    def _cdf(self, x, a):
        raise NotImplementedError('Missing jax.scipy.special.gammainc')

    def _sf(self, x, a):
        raise NotImplementedError('Missing jax.scipy.special.gammainc')

    def _ppf(self, q, a):
        raise NotImplementedError('Missing jax.scipy.special.gammaincinv')

    def _stats(self, a):
        return a, a, 2.0 / jnp.sqrt(a), 6.0 / a

    def _entropy(self, a):
        return digamma(a) * (1 - a) + a + gammaln(a)


class halfcauchy_gen(jax_continuous):
    _support_mask = constraints.positive

    def _rvs(self):
        return jnp.abs(random.cauchy(self._random_state, shape=self._size))

    def _pdf(self, x):
        return 2.0 / jnp.pi / (1.0 + x * x)

    def _logpdf(self, x):
        return jnp.log(2.0 / jnp.pi) - jnp.log1p(x * x)

    def _cdf(self, x):
        return 2.0 / jnp.pi * jnp.arctan(x)

    def _ppf(self, q):
        return jnp.tan(jnp.pi / 2 * q)

    def _stats(self):
        return jnp.inf, jnp.inf, jnp.nan, jnp.nan

    def _entropy(self):
        return jnp.log(2 * jnp.pi)


class halfnorm_gen(jax_continuous):
    _support_mask = constraints.positive

    def _rvs(self):
        return jnp.abs(random.normal(self._random_state, shape=self._size))

    def _pdf(self, x):
        # halfnorm.pdf(x) = sqrt(2/pi) * exp(-x**2/2)
        return jnp.sqrt(2.0 / jnp.pi) * jnp.exp(-x * x / 2.0)

    def _logpdf(self, x):
        return 0.5 * jnp.log(2.0 / jnp.pi) - x * x / 2.0

    def _cdf(self, x):
        return norm._cdf(x) * 2 - 1.0

    def _ppf(self, q):
        return norm._ppf((1 + q) / 2.0)

    def _stats(self):
        return (jnp.sqrt(2.0 / jnp.pi), 1 - 2.0 / jnp.pi,
                jnp.sqrt(2) * (4 - jnp.pi) / (jnp.pi - 2) ** 1.5, 8 * (jnp.pi - 3) / (jnp.pi - 2) ** 2)

    def _entropy(self):
        return 0.5 * jnp.log(jnp.pi / 2.0) + 0.5


class lognorm_gen(jax_continuous):
    arg_constraints = {'s': constraints.positive}
    _support_mask = constraints.positive

    def _rvs(self, s):
        # TODO: use upstream implementation when available
        return jnp.exp(s * random.normal(self._random_state, shape=self._size))

    def _pdf(self, x, s):
        # lognorm.pdf(x, s) = 1 / (s*x*sqrt(2*pi)) * exp(-1/2*(log(x)/s)**2)
        return jnp.exp(self._logpdf(x, s))

    def _logpdf(self, x, s):
        return jnp.where(x != 0,
                         -jnp.log(x) ** 2 / (2 * s ** 2) - jnp.log(s * x * jnp.sqrt(2 * jnp.pi)),
                         -jnp.inf)

    def _cdf(self, x, s):
        return norm._cdf(jnp.log(x) / s)

    def _logcdf(self, x, s):
        return norm._logcdf(jnp.log(x) / s)

    def _ppf(self, q, s):
        return jnp.exp(s * norm._ppf(q))

    def _sf(self, x, s):
        return norm._sf(jnp.log(x) / s)

    def _logsf(self, x, s):
        return norm._logsf(-jnp.log(x) / s)

    def _stats(self, s):
        p = jnp.exp(s * s)
        mu = jnp.sqrt(p)
        mu2 = p * (p - 1)
        g1 = jnp.sqrt(p - 1) * (2 + p)
        g2 = jnp.polyval([1, 2, 3, 0, -6.0], p)
        return mu, mu2, g1, g2

    def _entropy(self, s):
        return 0.5 * (1 + jnp.log(2 * jnp.pi) + 2 * jnp.log(s))


class norm_gen(jax_continuous):
    _support_mask = constraints.real

    def _rvs(self):
        return random.normal(self._random_state, shape=self._size)

    def _pdf(self, x):
        # norm.pdf(x) = exp(-x**2/2)/sqrt(2*pi)
        return jnp.exp(-x**2 / 2.0) / jnp.sqrt(2 * jnp.pi)

    def _logpdf(self, x):
        return -(x ** 2 + jnp.log(2 * jnp.pi)) / 2.0

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
        return 0.5 * (jnp.log(2 * jnp.pi) + 1)


class pareto_gen(jax_continuous):
    arg_constraints = {'b': constraints.positive}
    _support_mask = constraints.greater_than(1)

    def _rvs(self, b):
        return random.pareto(self._random_state, b, shape=self._size)

    def _cdf(self, x, b):
        return 1 - x ** (-b)

    def _ppf(self, q, b):
        return jnp.pow(1 - q, -1.0 / b)

    def _sf(self, x, b):
        return x ** (-b)

    def _stats(self, b, moments='mv'):
        mu, mu2, g1, g2 = None, None, None, None
        if 'm' in moments:
            mask = b > 1
            bt = jnp.extract(mask, b)
            mu = jnp.where(mask, bt / (bt - 1.0), jnp.inf)
        if 'v' in moments:
            mask = b > 2
            bt = jnp.extract(mask, b)
            mu2 = jnp.where(mask, bt / (bt - 2.0) / (bt - 1.0) ** 2, jnp.inf)
        if 's' in moments:
            mask = b > 3
            bt = jnp.extract(mask, b)
            vals = 2 * (bt + 1.0) * jnp.sqrt(bt - 2.0) / ((bt - 3.0) * jnp.sqrt(bt))
            g1 = jnp.where(mask, vals, jnp.nan)
        if 'k' in moments:
            mask = b > 4
            bt = jnp.extract(mask, b)
            vals = (6.0 * jnp.polyval([1.0, 1.0, -6, -2], bt)
                    / jnp.polyval([1.0, -7.0, 12.0, 0.0], bt))
            g2 = jnp.where(mask, vals, jnp.nan)
        return mu, mu2, g1, g2

    def _entropy(self, c):
        return 1 + 1.0 / c - jnp.log(c)


class t_gen(jax_continuous):
    arg_constraints = {'df': constraints.positive}
    _support_mask = constraints.real

    def _rvs(self, df):
        # TODO: use upstream implementation when available
        key_n, key_g = random.split(self._random_state)
        normal = random.normal(key_n, shape=self._size)
        half_df = df / 2.0
        gamma = random.gamma(key_n, half_df, shape=self._size)
        return normal * jnp.sqrt(half_df / gamma)

    def _cdf(self, x, df):
        raise NotImplementedError('Missing jax.scipy.special.stdtr')

    def _sf(self, x, df):
        raise NotImplementedError('Missing jax.scipy.special.stdtr')

    def _ppf(self, q, df):
        raise NotImplementedError('Missing jax.scipy.special.stdtrit')

    def _isf(self, q, df):
        raise NotImplementedError('Missing jax.scipy.special.stdtrit')

    def _stats(self, df):
        mu = jnp.where(df > 1, 0.0, jnp.inf)
        mu2 = jnp.where(df > 2, df / (df - 2.0), jnp.inf)
        mu2 = jnp.where(df <= 1, jnp.nan, mu2)
        g1 = jnp.where(df > 3, 0.0, jnp.nan)
        g2 = jnp.where(df > 4, 6.0 / (df - 4.0), jnp.inf)
        g2 = jnp.where(df <= 2, jnp.nan, g2)
        return mu, mu2, g1, g2


class trunccauchy_gen(jax_continuous):
    # TODO: override _argcheck with the constraint that a < b

    def _support(self, *args, **kwargs):
        (a, b), loc, scale = self._parse_args(*args, **kwargs)
        # TODO: make constraints.less_than and support a == -jnp.inf
        if b == jnp.inf:
            return constraints.greater_than((a - loc) * scale)
        else:
            return constraints.interval((a - loc) * scale, (b - loc) * scale)

    def _rvs(self, a, b):
        # We use inverse transform method:
        # z ~ ppf(U), where U ~ Uniform(cdf(a), cdf(b)).
        #                     ~ Uniform(arctan(a), arctan(b)) / pi + 1/2
        u = random.uniform(self._random_state, shape=self._size,
                           minval=jnp.arctan(a), maxval=jnp.arctan(b))
        return jnp.tan(u)

    def _pdf(self, x, a, b):
        return jnp.reciprocal((1 + x * x) * (jnp.arctan(b) - jnp.arctan(a)))

    def _logpdf(self, x, a, b):
        # trunc_pdf(x) = pdf(x) / (cdf(b) - cdf(a))
        #              = 1 / (1 + x^2) / (arctan(b) - arctan(a))
        normalizer = jnp.log(jnp.arctan(b) - jnp.arctan(a))
        return -(jnp.log(1 + x * x) + normalizer)


class truncnorm_gen(jax_continuous):
    # TODO: override _argcheck with the constraint that a < b

    def _support(self, *args, **kwargs):
        (a, b), loc, scale = self._parse_args(*args, **kwargs)
        # TODO: make constraints.less_than and support a == -jnp.inf
        if b == jnp.inf:
            return constraints.greater_than((a - loc) * scale)
        else:
            return constraints.interval((a - loc) * scale, (b - loc) * scale)

    def _rvs(self, a, b):
        # We use inverse transform method:
        # z ~ ppf(U), where U ~ Uniform(0, 1).
        u = random.uniform(self._random_state, shape=self._size)
        return self._ppf(u, a, b)

    def _pdf(self, x, a, b):
        delta = norm._sf(a) - norm._sf(b)
        return norm._pdf(x) / delta

    def _logpdf(self, x, a, b):
        x, a, b = _promote_dtypes(x, a, b)
        # XXX: consider to use norm._cdf(b) - norm._cdf(a) when a, b < 0
        delta = norm._sf(a) - norm._sf(b)
        return norm._logpdf(x) - jnp.log(delta)

    def _cdf(self, x, a, b):
        delta = norm._sf(a) - norm._sf(b)
        return (norm._cdf(x) - norm._cdf(a)) / delta

    def _ppf(self, q, a, b):
        q, a, b = _promote_dtypes(q, a, b)
        # XXX: consider to use norm._ppf(q * norm._cdf(b) + norm._cdf(a) * (1.0 - q))
        # when a, b < 0
        ppf = norm._isf(q * norm._sf(b) + norm._sf(a) * (1.0 - q))
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
halfnorm = halfnorm_gen(a=0.0, name='halfnorm')
lognorm = lognorm_gen(a=0.0, name='lognorm')
norm = norm_gen(name='norm')
pareto = pareto_gen(a=1.0, name="pareto")
t = t_gen(name='t')
trunccauchy = trunccauchy_gen(name='trunccauchy')
truncnorm = truncnorm_gen(name='truncnorm')
uniform = uniform_gen(a=0.0, b=1.0, name='uniform')
