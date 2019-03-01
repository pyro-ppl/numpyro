import jax.random as random

from numpyro.distributions.distribution import jax_continuous

# Copyright (c) 2001, 2002 Enthought, Inc.
# All rights reserved.
#
# Copyright (c) 2003-2019 SciPy Developers.
# All rights reserved.


class uniform_gen(jax_continuous):
    r"""A uniform continuous random variable.
    In the standard form, the distribution is uniform on ``[0, 1]``. Using
    the parameters ``loc`` and ``scale``, one obtains the uniform distribution
    on ``[loc, loc + scale]``.
    %(before_notes)s
    %(example)s
    """
    def _rvs(self):
        return random.uniform(self._random_state, self._size, minval=0.0, maxval=1.0)

    def _pdf(self, x):
        return 1.0 * (x == x)

    def _cdf(self, x):
        return x

    def _ppf(self, q):
        return q

    def _stats(self):
        return 0.5, 1.0 / 12, 0, -1.2

    def _entropy(self):
        return 0.0


uniform = uniform_gen(name='uniform')
