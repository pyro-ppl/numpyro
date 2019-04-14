# Source code modified from scipy.stats._multivariate.py
#
# Copyright (c) 2001, 2002 Enthought, Inc.
# All rights reserved.
#
# Copyright (c) 2003-2019 SciPy Developers.
# All rights reserved.

import jax.numpy as np
from jax.scipy.special import digamma, gammaln

from numpyro.distributions import constraints
from numpyro.distributions.distribution import jax_continuous, jax_discrete, jax_multivariate
from numpyro.distributions.util import standard_gamma, xlogy


def _lnB(alpha):
    return np.sum(gammaln(alpha), axis=-1) - gammaln(np.sum(alpha, axis=-1))


# TODO: either use multivariate docs instead of the default one of
# rv_continuous/rv_discrete

class categorical_gen(jax_multivariate, jax_discrete):
    def _support(self, *args, **kwargs):
        (p,), _, _ = self._parse_args(*args, **kwargs)
        return constraints.integer_interval(0, p.shape[-1] - 1)

    @property
    def arg_constraints(self):
        if self.is_logits:
            return {'p': constraints.real}
        else:
            return {'p': constraints.simplex}

    def _rvs(self, n, p):
        random_state = onp.random.RandomState(self._random_state)
        sample = random_state.multinomial(1, p, self._size)
        return device_put(random_state.multinomial(n, p, size))


class dirichlet_gen(jax_multivariate, jax_continuous):
    arg_constraints = {"alpha": constraints.positive}
    _support_mask = constraints.simplex

    def _batch_shape(self, alpha):
        return alpha.shape[:-1]

    def logpdf(self, x, alpha):
        lnB = _lnB(alpha)
        return -lnB + np.sum(xlogy(alpha - 1, x), axis=-1)

    def pdf(self, x, alpha):
        return np.exp(self.logpdf(x, alpha))

    def mean(self, alpha):
        return alpha / np.sum(alpha, axis=-1, keepdims=True)

    def var(self, alpha):
        alpha0 = np.sum(alpha, axis=-1, keepdims=True)
        return (alpha * (alpha0 - alpha)) / ((alpha0 * alpha0) * (alpha0 + 1))

    def entropy(self, alpha):
        alpha0 = np.sum(alpha, axis=-1)
        lnB = _lnB(alpha)
        K = alpha.shape[-1]
        return lnB + (alpha0 - K) * digamma(alpha0) - np.inner((alpha - 1) * digamma(alpha))

    def _rvs(self, alpha):
        K = alpha.shape[-1]
        gamma_samples = standard_gamma(self._random_state, alpha, shape=self._size + (K,))
        return gamma_samples / np.sum(gamma_samples, axis=-1, keepdims=True)


class multinomial_gen(jax_multivariate, jax_discrete):
    def _support(self, *args, **kwargs):
        (n, p), _, _ = self._parse_args(*args, **kwargs)
        return constraints.integer_interval(0, n)

    @property
    def arg_constraints(self):
        if self.is_logits:
            return {'n': constraints.nonnegative_integer,
                    'p': constraints.real}
        else:
            return {'n': constraints.nonnegative_integer,
                    'p': constraints.simplex}

    def _batch_shape(self, n, p):
        return p.shape[:-1]

    def logpmf(self, x, n, p):
        n, p, x = _promote_dtypes(n, p, x)
        if self.is_logits:
            
        else:
            return gammaln(n + 1) + np.sum(xlogy(x, p) - gammaln(x + 1), axis=-1)

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

    def _rvs(self, n, p):
        random_state = onp.random.RandomState(self._random_state)
        sample = random_state.multinomial(n, p, self._size)
        return device_put(random_state.multinomial(n, p, size))


categorical = categorical_gen(name='categorical')
dirichlet = dirichlet_gen(name='dirichlet')
multinomial = multinomial_gen(name='multinomial')
