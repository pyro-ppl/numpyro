import jax.numpy as np
from jax import lax

from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import validate_sample
from numpyro.util import copy_docs_from


def _eval_poly(y, coef):
    "TODO: rewrite as vector"
    coef = list(coef)
    result = coef.pop()
    while coef:
        result = coef.pop() + y * result
    return result


def log_modfied_bessel_0(x):
    """
    Returns ``log(I0(x))`` for ``x > 0``.
    """

    _COEF_SMALL = [1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.360768e-1, 0.45813e-2]
    _COEF_LARGE = [0.39894228, 0.1328592e-1, 0.225319e-2, -0.157565e-2, 0.916281e-2,
                   -0.2057706e-1, 0.2635537e-1, -0.1647633e-1, 0.392377e-2]

    # compute small solution
    y = (x / 3.75).pow(2)
    small = _eval_poly(y, _COEF_SMALL).log()

    # compute large solution
    y = 3.75 / x
    large = x - 0.5 * x.log() + _eval_poly(y, _COEF_LARGE).log()

    mask = (x < 3.75)
    result = large
    if mask.any():
        result[mask] = small[mask]
    return result


@copy_docs_from(Distribution)
class VonMises(Distribution):
    arg_constraints = {'location': constraints.real, 'concentration': constraints.positive}
    support = constraints.real

    def __init__(self, location, concentration, validate_args=None):

        self.conc = concentration
        self.loc = location

        batch_shape = lax.broadcast_shapes(np.shape(concentration), np.shape(location))

        super(VonMises, self).__init__(batch_shape=batch_shape,
                                       validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        return self.loc + self.scale

    @validate_sample
    def log_prob(self, value):
        return -(np.log(2 * np.pi) + log_modfied_bessel_0(self.conc)) + (self.conc * np.cos(value - self.loc))

    @property
    def mean(self):
        return np.full(self.batch_shape, np.nan)

    @property
    def variance(self):
        return np.full(self.batch_shape, np.nan)
