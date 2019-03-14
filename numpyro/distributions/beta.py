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
from numpyro.distributions.util import standard_gamma


class beta_gen(jax_continuous):
    def _rvs(self, a, b):
        # XXX the implementation is different from PyTorch's one
        # in PyTorch, a sample is generated from dirichlet distribution
        key_a, key_b = random.split(self._random_state)
        gamma_a = standard_gamma(key_a, a, self._size)
        gamma_b = standard_gamma(key_b, b, self._size)
        return gamma_a / (gamma_a + gamma_b)

    def _cdf(self, x, a, b):
        raise NotImplementedError

    def _stats(self, a, b):
        mn = a * 1.0 / (a + b)
        var = (a * b * 1.0) / (a + b + 1.0) / (a + b) ** 2.0
        g1 = 2.0 * (b - a) * np.sqrt((1.0 + a + b) / (a * b)) / (2 + a + b)
        g2 = 6.0 * (a ** 3 + a ** 2 * (1 - 2 * b) + b ** 2 * (1 + b) - 2 * a * b * (2 + b))
        g2 = g2 / (a * b * (a + b + 2) * (a + b + 3))
        return mn, var, g1, g2


beta = beta_gen(a=0.0, b=1.0, name='beta')
