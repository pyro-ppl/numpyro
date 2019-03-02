import jax.numpy as np
import jax.random as random

from numpyro.distributions.distribution import jax_continuous

# Copyright (c) 2001, 2002 Enthought, Inc.
# All rights reserved.
#
# Copyright (c) 2003-2019 SciPy Developers.
# All rights reserved.


_norm_pdf_C = np.sqrt(2 * np.pi)


def _lognorm_logpdf(x, s):
    return np.where(x != 0,
                    -np.log(x)**2 / (2 * s**2) - np.log(s * x * _norm_pdf_C),
                    -np.inf)


class lognorm_gen(jax_continuous):
    r"""A lognormal continuous random variable.
    %(before_notes)s
    Notes
    -----
    The probability density function for `lognorm` is:
    .. math::
        f(x, s) = \frac{1}{s x \sqrt{2\pi}}
                  \exp\left(-\frac{\log^2(x)}{2s^2}\right)
    for :math:`x > 0`, :math:`s > 0`.
    `lognorm` takes ``s`` as a shape parameter for :math:`s`.
    %(after_notes)s
    A common parametrization for a lognormal random variable ``Y`` is in
    terms of the mean, ``mu``, and standard deviation, ``sigma``, of the
    unique normally distributed random variable ``X`` such that exp(X) = Y.
    This parametrization corresponds to setting ``s = sigma`` and ``scale =
    exp(mu)``.
    %(example)s
    """
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
