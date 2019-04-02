# Source code modified from scipy.stats._multivariate.py
#
# Copyright (c) 2001, 2002 Enthought, Inc.
# All rights reserved.
#
# Copyright (c) 2003-2019 SciPy Developers.
# All rights reserved.

import jax.numpy as np
from jax import random
from jax.scipy.special import digamma, gammaln

from numpyro.distributions.distribution import jax_multivariate
from numpyro.distributions.util import xlogy


def _lnB(alpha):
    return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))


class dirichlet_gen(jax_multivariate):
    def _logpdf(self, x, alpha):
        lnB = _lnB(alpha)
        return -lnB + np.sum((xlogy(alpha - 1, x.T)).T, 0)

    def logpdf(self, x, alpha):
        alpha = _dirichlet_check_parameters(alpha)
        x = _dirichlet_check_input(alpha, x)

        out = self._logpdf(x, alpha)
        return _squeeze_output(out)

    def pdf(self, x, alpha):
        alpha = _dirichlet_check_parameters(alpha)
        x = _dirichlet_check_input(alpha, x)

        out = np.exp(self._logpdf(x, alpha))
        return _squeeze_output(out)

    def mean(self, alpha):
        alpha = _dirichlet_check_parameters(alpha)

        out = alpha / (np.sum(alpha))
        return _squeeze_output(out)

    def var(self, alpha):
        alpha = _dirichlet_check_parameters(alpha)

        alpha0 = np.sum(alpha)
        out = (alpha * (alpha0 - alpha)) / ((alpha0 * alpha0) * (alpha0 + 1))
        return _squeeze_output(out)

    def entropy(self, alpha):
        alpha = _dirichlet_check_parameters(alpha)

        alpha0 = np.sum(alpha)
        lnB = _lnB(alpha)
        K = alpha.shape[0]

        out = lnB + (alpha0 - K) * scipy.special.psi(alpha0) - np.sum(
            (alpha - 1) * scipy.special.psi(alpha))
        return _squeeze_output(out)

    def rvs(self, alpha, size=1, random_state=None):
        """
        Draw random samples from a Dirichlet distribution.
        Parameters
        ----------
        %(_dirichlet_doc_default_callparams)s
        size : int, optional
            Number of samples to draw (default 1).
        %(_doc_random_state)s
        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `N`), where `N` is the
            dimension of the random variable.
        """
        alpha = _dirichlet_check_parameters(alpha)
        random_state = self._get_random_state(random_state)
        return random_state.dirichlet(alpha, size=size)
