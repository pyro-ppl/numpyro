# Source code modified from scipy.stats._discrete_distns.py
#
# Copyright (c) 2001, 2002 Enthought, Inc.
# All rights reserved.
#
# Copyright (c) 2003-2019 SciPy Developers.
# All rights reserved.

import numpy as onp

import jax.numpy as np
from jax import device_put, lax, random
from jax.scipy.special import expit, gammaln

from numpyro.distributions import constraints
from numpyro.distributions.distribution import jax_discrete
from numpyro.distributions.util import binary_cross_entropy_with_logits, entr, xlog1py, xlogy


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
        random_state = onp.random.RandomState(self._random_state)
        sample = random_state.binomial(n, p, self._size)
        return device_put(sample)

    def _logpmf(self, x, n, p):
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

    def _entropy(self, n, p):
        if self.is_logits:
            p = expit(p)
        k = np.arange(n + 1)
        vals = self._pmf(k, n, p)
        return np.sum(entr(vals), axis=0)


bernoulli = bernoulli_gen(b=1, name='bernoulli')
binom = binom_gen(name='binom')
