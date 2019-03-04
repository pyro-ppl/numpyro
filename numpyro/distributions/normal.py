# Source code modified from scipy.stats._continuous_distns.py
#
# Copyright (c) 2001, 2002 Enthought, Inc.
# All rights reserved.
#
# Copyright (c) 2003-2019 SciPy Developers.
# All rights reserved.

import jax.numpy as np
import jax.random as random

from numpyro.distributions.distribution import jax_continuous


_norm_pdf_C = np.sqrt(2 * np.pi)
_norm_pdf_logC = np.log(_norm_pdf_C)


def _norm_pdf(x):
    return np.exp(-x ** 2 / 2.0) / _norm_pdf_C


def _norm_logpdf(x):
    return -x ** 2 / 2.0 - _norm_pdf_logC


class norm_gen(jax_continuous):
    r"""A normal continuous random variable.
    The location (``loc``) keyword specifies the mean.
    The scale (``scale``) keyword specifies the standard deviation.
    %(before_notes)s
    Notes
    -----
    The probability density function for `norm` is:
    .. math::
        f(x) = \frac{\exp(-x^2/2)}{\sqrt{2\pi}}
    for a real number :math:`x`.
    %(after_notes)s
    %(example)s
    """

    def _rvs(self):
        return random.normal(self._random_state, self._size)

    def _pdf(self, x):
        # norm.pdf(x) = exp(-x**2/2)/sqrt(2*pi)
        return _norm_pdf(x)

    def _logpdf(self, x):
        return _norm_logpdf(x)

    def _stats(self):
        return 0.0, 1.0, 0.0, 0.0

    def _entropy(self):
        return 0.5 * (np.log(2 * np.pi) + 1)


norm = norm_gen(name='norm')
