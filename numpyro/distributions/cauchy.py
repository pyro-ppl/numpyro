# Code modified from scipy.distributions._continous_distns.py
#
# Copyright (c) 2001, 2002 Enthought, Inc.
# All rights reserved.
#
# Copyright (c) 2003-2019 SciPy Developers.
# All rights reserved.

import jax.numpy as np
import jax.random as random

from numpyro.distributions.distribution import jax_continuous


class cauchy_gen(jax_continuous):
    def _rvs(self):
        # TODO: move this implementation upstream to jax.random.standard_cauchy
        # Another way is to generate X, Y ~ Normal(0, 1) and return X / Y
        u = random.uniform(self._random_state, self._size)
        return np.tan(np.pi * (u - 0.5))

    def _pdf(self, x):
        # cauchy.pdf(x) = 1 / (pi * (1 + x**2))
        return 1.0 / np.pi / (1.0 + x*x)

    def _cdf(self, x):
        return 0.5 + 1.0 / np.pi * np.arctan(x)

    def _ppf(self, q):
        return np.tan(np.pi * q - np.pi / 2.0)

    def _sf(self, x):
        return 0.5 - 1.0 / np.pi * np.arctan(x)

    def _isf(self, q):
        return np.tan(np.pi / 2.0 - np.pi * q)

    def _stats(self):
        return np.nan, np.nan, np.nan, np.nan

    def _entropy(self):
        return np.log(4 * np.pi)


cauchy = cauchy_gen(name='cauchy')
