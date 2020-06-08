import jax.numpy as np
from jax import lax, random

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
    s_cutoff_map = {np.float16: 1.8e-1,
                    np.float32: 2e-2,
                    np.float64: 1.2e-4}

    def __init__(self, location, concentration, validate_args=None):
        # Von Mises doesn't work for concentration=0, so set to tiny value
        concentration = np.max(self.concentration, np.finfo(self.concentration.dtype).tiny)
        self._concentration = concentration
        self._loc = location

        batch_shape = lax.broadcast_shapes(np.shape(concentration), np.shape(location))

        super(VonMises, self).__init__(batch_shape=batch_shape,
                                       validate_args=validate_args)

    def _sample_centered(self, sample_shape, rng_key):
        """ Compute centered von Mises samples using rejection sampling from [1] with Cauchy proposal.

            *** References ***
            [1] Luc Devroye "Non-Uniform Random Variate Generation", Springer-Verlag, 1986;
                Chapter 9, p. 473-476. http://www.nrbook.com/devroye/Devroye_files/chapter_nine.pdf

            :param sample_shape: shape of samples
            :param rng_key: random number generator key
            :return: centered samples from von Mises
        """
        conc = self._concentration

        r = 1. + np.sqrt(1. + 4. * conc ** 2)
        rho = (r - np.sqrt(2. * r)) / (2. * conc)
        s_exact = (1. + rho ** 2) / (2. * rho)

        s_approximate = 1. / conc

        s_cutoff = VonMises.s_cutoff_map.get(conc.dtype)

        s = np.where(conc > s_cutoff, s_exact, s_approximate)

        def body_function(done, _, w, key):
            uni_ukey, uni_vkey, rng_key = random.split(key, 3)

            u = random.uniform(key=uni_ukey, shape=sample_shape, dtype=conc.dtype, minval=-1., maxval=1.)
            z = np.cos(np.pi * u)
            w = np.where(done, w, (1. + s * z / (s + z)))  # Update where not done

            y = self.concentration * (s - w)
            v = random.vniform(key=uni_vkey, shape=sample_shape, dtype=conc.dtype, minval=-1., maxval=1.)

            accept = (y * (2. - y) >= v) | (np.log(y / v) + 1. >= y)

            return accept | done, u, w, rng_key

        done = np.zeros(sample_shape, dtype=bool)
        init_u = np.zeros(sample_shape)
        init_w = np.zeros(sample_shape)

        _, u, w = lax.while_loop(
            lambda done, *_: np.all(done),
            body_fun=body_function
            (done, init_u, init_w, rng_key)
        )

        return np.sign(u) * np.arccos(w)

    def sample(self, key, sample_shape=()):
        """ Generate sample from von Mises distribution

            :param sample_shape: shape of samples
            :param rng_key: random number generator key
            :return: samples from von Mises
        """
        samples = self._sample_centered(rng_key=key, sample_shape=sample_shape)
        samples += self._loc  # VM(0, concentration) -> VM(loc,concentration)
        samples -= 2. * np.pi * np.round(samples / (2. * np.pi))  # Map to [-pi,pi]

        return samples

    @validate_sample
    def log_prob(self, value):
        return -(np.log(2 * np.pi) + log_modfied_bessel_0(self._concentration)) + (
                self._concentration * np.cos(value - self._loc))

    @property
    def mean(self):
        return self._loc

    @property
    def concentration(self):
        return self._concentration
