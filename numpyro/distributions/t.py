# Source code modified from scipy.stats._continous_distns.py
#
# Copyright (c) 2001, 2002 Enthought, Inc.
# All rights reserved.
#
# Copyright (c) 2003-2019 SciPy Developers.
# All rights reserved.

import jax.numpy as np
import jax.random as random
from jax.scipy import special

from numpyro.distributions.distribution import jax_continuous
from numpyro.distributions.util import standard_gamma


class t_gen(jax_continuous):
    def _rvs(self, df):
        key_n, key_g = random.split(self._random_state)
        normal = random.normal(key_n, self._size)
        half_df = df / 2.0
        gamma = standard_gamma(key_n, half_df, self._size)
        return normal * np.sqrt(half_df / gamma)

    def _pdf(self, x, df):
        #                                gamma((df+1)/2)
        # t.pdf(x, df) = ---------------------------------------------------
        #                sqrt(pi*df) * gamma(df/2) * (1+x**2/df)**((df+1)/2)
        r = np.asarray(df * 1.0)
        Px = np.exp(special.gammaln((r + 1) / 2) - special.gammaln(r / 2))
        Px = Px / np.sqrt(r * np.pi) * (1 + (x ** 2) / r) ** ((r + 1) / 2)
        return Px

    def _logpdf(self, x, df):
        r = df*1.0
        lPx = special.gammaln((r+1)/2) - special.gammaln(r/2)
        lPx = lPx - (0.5 * np.log(r * np.pi) + (r + 1) / 2 * np.log(1 + (x ** 2) / r))
        return lPx

    def _stats(self, df):
        mu = np.where(df > 1, 0.0, np.inf)
        mu2 = np.where(df > 2, df / (df - 2.0), np.inf)
        mu2 = np.where(df <= 1, np.nan, mu2)
        g1 = np.where(df > 3, 0.0, np.nan)
        g2 = np.where(df > 4, 6.0 / (df - 4.0), np.inf)
        g2 = np.where(df <= 2, np.nan, g2)
        return mu, mu2, g1, g2


t = t_gen(name='t')
