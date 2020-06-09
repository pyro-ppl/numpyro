import jax.numpy as np
from jax import lax, random

from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import validate_sample
from numpyro.util import copy_docs_from


@copy_docs_from(Distribution)
class VonMises(Distribution):
    arg_constraints = {'location': constraints.real, 'concentration': constraints.positive}
    support = constraints.real
    s_cutoff_map = {np.dtype(np.float16): 1.8e-1,
                    np.dtype(np.float32): 2e-2,
                    np.dtype(np.float64): 1.2e-4}

    def __init__(self, location, concentration, validate_args=None):
        # von Mises doesn't work for concentration=0, so set to tiny value
        try:
            dtype = np.dtype(concentration.dtype)
        except AttributeError:
            dtype = np.dtype(type(concentration))
        concentration = np.maximum(concentration, np.finfo(dtype).tiny)
        self.dtype = dtype
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

        s_cutoff = VonMises.s_cutoff_map.get(self.dtype)

        s = np.where(conc > s_cutoff, s_exact, s_approximate)

        def body_function(i, *args):
            done, _, w = args[0]
            nonlocal rng_key
            uni_ukey, uni_vkey, rng_key = random.split(rng_key, 3)

            u = random.uniform(key=uni_ukey, shape=sample_shape, dtype=conc.dtype, minval=-1., maxval=1.)
            z = np.cos(np.pi * u)
            w = np.where(done, w, (1. + s * z) / (s + z))  # Update where not done

            y = self.concentration * (s - w)
            v = random.uniform(key=uni_vkey, shape=sample_shape, dtype=conc.dtype, minval=-1., maxval=1.)

            accept = (y * (2. - y) >= v) | (np.log(y / v) + 1. >= y)

            return accept | done, u, w

        init_done = np.zeros(sample_shape, dtype=bool)
        init_u = np.zeros(sample_shape)
        init_w = np.zeros(sample_shape)

        done, u, w = lax.fori_loop(
            lower=0,
            upper=100,
            body_fun=body_function,
            init_val=(init_done, init_u, init_w)
        )

        # cond_fun=lambda done, *_: np.logical_not(np.all(done)),
        return np.sign(u) * np.arccos(w)

    def sample(self, key, sample_shape=()):
        """ Generate sample from von Mises distribution

            :param sample_shape: shape of samples
            :param rng_key: random number generator key
            :return: samples from von Mises
        """
        samples = self._sample_centered(rng_key=key, sample_shape=sample_shape)
        samples -= 2. * np.pi * np.round(samples / (2. * np.pi))  # Map to [-pi,pi]
        samples += self._loc  # VM(0, concentration) -> VM(loc,concentration)

        return samples

    @validate_sample
    def log_prob(self, value):
        return -(np.log(2 * np.pi) + lax.bessel_i0e(self._concentration)) + (
                self._concentration * np.cos(value - self._loc))

    @property
    def mean(self):
        return self._loc

    @property
    def location(self):
        return self._loc

    @property
    def concentration(self):
        return self._concentration

    @property
    def variance(self):
        return 1 - lax.bessel_i1e(self._concentration) / lax.bessel_i0e(self._concentration)

if __name__ == '__main__':

    vm = VonMises(2., 10.)
    print(vm.variance)
