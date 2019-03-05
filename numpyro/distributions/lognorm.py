# Source code modified from scipy.stats._continous_distns.py
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


def _lognorm_logpdf(x, s):
    return np.where(x != 0,
                    -np.log(x)**2 / (2 * s**2) - np.log(s * x * _norm_pdf_C),
                    -np.inf)


class lognorm_gen(jax_continuous):
    # TODO: check if this is fine
    _support_mask = jax_continuous._open_support_mask

    def _rvs(self, s):
        return np.exp(s * random.normal(self._random_state, self._size))

    def _pdf(self, x, s):
        # lognorm.pdf(x, s) = 1 / (s*x*sqrt(2*pi)) * exp(-1/2*(log(x)/s)**2)
        return np.exp(self._logpdf(x, s))

    def _logpdf(self, x, s):
        return _lognorm_logpdf(x, s)

    def _stats(self, s):
        p = np.exp(s * s)
        mu = np.sqrt(p)
        mu2 = p * (p - 1)
        g1 = np.sqrt(p - 1) * (2 + p)
        g2 = np.polyval([1, 2, 3, 0, -6.0], p)
        return mu, mu2, g1, g2

    def _entropy(self, s):
        return 0.5 * (1 + np.log(2 * np.pi) + 2 * np.log(s))


lognorm = lognorm_gen(a=0.0, name='lognorm')
