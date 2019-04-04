# Source code modified from scipy.stats._multivariate.py
#
# Copyright (c) 2001, 2002 Enthought, Inc.
# All rights reserved.
#
# Copyright (c) 2003-2019 SciPy Developers.
# All rights reserved.

import jax.numpy as np
from jax.scipy.special import digamma, gammaln

from numpyro.distributions.util import xlogy, standard_gamma
from numpyro.distributions.distribution import jax_mvcontinuous


def _lnB(alpha):
    return np.sum(gammaln(alpha), axis=-1) - gammaln(np.sum(alpha, axis=-1))


class dirichlet_gen(jax_mvcontinuous):
    # TODO: use dirichlet doc instead of the default one of rv_continuous
    # TODO: add _argcheck, _support_mask with simplex
    def _logpdf(self, x, alpha):
        lnB = _lnB(alpha)
        return -lnB + np.sum(xlogy(alpha - 1, x), axis=-1)

    def _pdf(self, x, alpha):
        return np.exp(self._logpdf(x, alpha))

    def mean(self, alpha):
        return alpha / np.sum(alpha, axis=-1, keepdims=True)

    def var(self, alpha):
        alpha0 = np.sum(alpha, axis=-1, keepdims=True)
        return (alpha * (alpha0 - alpha)) / ((alpha0 * alpha0) * (alpha0 + 1))

    def _entropy(self, alpha):
        alpha0 = np.sum(alpha, axis=-1)
        lnB = _lnB(alpha)
        K = alpha.shape[-1]
        return lnB + (alpha0 - K) * digamma(alpha0) - np.inner((alpha - 1) * digamma(alpha))

    def _rvs(self, alpha):
        K = alpha.shape[-1]
        gamma_samples = standard_gamma(self._random_state, alpha, self._size + (K,))
        return gamma_samples / np.sum(gamma_samples, axis=-1, keepdims=True)


dirichlet = dirichlet_gen(a=0.0, b=1.0, name='dirichlet')
