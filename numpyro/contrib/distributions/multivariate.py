# Source code modified from scipy.stats._multivariate.py
#
# Copyright (c) 2001, 2002 Enthought, Inc.
# All rights reserved.
#
# Copyright (c) 2003-2019 SciPy Developers.
# All rights reserved.

from jax import lax
from jax.experimental.stax import softmax
import jax.numpy as np
from jax.numpy.lax_numpy import _promote_dtypes
from jax.scipy.special import digamma, entr, gammaln, logsumexp

from numpyro.contrib.distributions.discrete import binom
from numpyro.contrib.distributions.distribution import jax_continuous, jax_discrete, jax_multivariate
from numpyro.distributions import constraints
from numpyro.distributions.util import categorical as categorical_rvs
from numpyro.distributions.util import multinomial as multinomial_rvs
from numpyro.distributions.util import standard_gamma, xlogy


def _lnB(alpha):
    return np.sum(gammaln(alpha), axis=-1) - gammaln(np.sum(alpha, axis=-1))


# TODO: either use multivariate docs instead of the default one of
# rv_continuous/rv_discrete

class categorical_gen(jax_multivariate, jax_discrete):
    @property
    def arg_constraints(self):
        if self.is_logits:
            return {'p': constraints.real}
        else:
            return {'p': constraints.simplex}

    def _support(self, *args, **kwargs):
        (p,), _, _ = self._parse_args(*args, **kwargs)
        return constraints.integer_interval(0, p.shape[-1] - 1)

    def _batch_shape(self, p):
        return p.shape[:-1]

    def _event_shape(self, p):
        return ()

    def logpmf(self, x, p):
        batch_shape = lax.broadcast_shapes(x.shape, p.shape[:-1])
        # append a dimension to x
        # TODO: consider to convert x.dtype to int
        x = np.expand_dims(x, axis=-1)
        x = np.broadcast_to(x, batch_shape + (1,))
        p = np.broadcast_to(p, batch_shape + p.shape[-1:])
        if self.is_logits:
            # normalize log prob
            p = p - logsumexp(p, axis=-1, keepdims=True)
            # gather and remove the trailing dimension
            return np.take_along_axis(p, x, axis=-1)[..., 0]
        else:
            return np.take_along_axis(np.log(p), x, axis=-1)[..., 0]

    def pmf(self, x, p):
        return np.exp(self.logpmf(x, p))

    def _rvs(self, p):
        if self.is_logits:
            p = softmax(p)
        return categorical_rvs(self._random_state, p, self._size)


class dirichlet_gen(jax_multivariate, jax_continuous):
    arg_constraints = {"alpha": constraints.positive}
    _support_mask = constraints.simplex

    def _batch_shape(self, alpha):
        return alpha.shape[:-1]

    def _event_shape(self, alpha):
        return alpha.shape[-1:]

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

    def _event_shape(self, n, p):
        return p.shape[-1:]

    def logpmf(self, x, n, p):
        x, n, p = _promote_dtypes(x, n, p)
        if self.is_logits:
            return gammaln(n + 1) + np.sum(x * p - gammaln(x + 1), axis=-1) - n * logsumexp(p, axis=-1)
        else:
            return gammaln(n + 1) + np.sum(xlogy(x, p) - gammaln(x + 1), axis=-1)

    def pmf(self, x, n, p):
        return np.exp(self.logpmf(x, n, p))

    def entropy(self, n, p):
        x = np.arange(1, np.max(n) + 1)

        term1 = n * np.sum(entr(p), axis=-1) - gammaln(n + 1)

        n = n[..., np.newaxis]
        new_axes_needed = max(p.ndim, n.ndim) - x.ndim + 1
        x.shape += (1,) * new_axes_needed

        term2 = np.sum(binom.pmf(x, n, p) * gammaln(x + 1),
                       axis=(-1, -1 - new_axes_needed))

        return term1 + term2

    def _rvs(self, n, p):
        if self.is_logits:
            p = softmax(p)
        return multinomial_rvs(self._random_state, p, n, self._size)


categorical = categorical_gen(name='categorical')
dirichlet = dirichlet_gen(name='dirichlet')
multinomial = multinomial_gen(name='multinomial')
