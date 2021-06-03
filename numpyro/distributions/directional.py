# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

from jax import lax
import jax.numpy as jnp
import jax.random as random
from jax.scipy.special import erf, i0e, i1e

from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import (
    is_prng_key,
    promote_shapes,
    safe_normalize,
    validate_sample,
    von_mises_centered,
)


class SineSkewed(Distribution):
    """Sine Skewing [1] is a procedure for producing a distribution that breaks pointwise symmetry on a torus
    distribution. The new distribution is called the Sine Skewed X distribution, where X is the name of the (symmetric)
    base distribution.
    Torus distributions are distributions with support on products of circles
    (i.e., ⨂^d S^1 where S^1=[-pi,pi) ). So, a 0-torus is a point, the 1-torus is a circle,
    and the 2-torus is commonly associated with the donut shape.
    The Sine Skewed X distribution is parameterized by a weight parameter for each dimension of the event of X.
    For example with a von Mises distribution over a circle (1-torus), the Sine Skewed von Mises Distribution has one
    skew parameter. The skewness parameters can be inferred using :class:`~numpyro.infer.HMC` or
    :class:`~numpyro.infer.NUTS`. For example, the following will produce a uniform prior over
    skewness for the 2-torus,::

        def model(obs):
            # Sine priors
            phi_loc = numpyro.sample('phi_loc', VonMises(pi, 2.))
            psi_loc = numpyro.sample('psi_loc', VonMises(-pi / 2, 2.))
            phi_conc = numpyro.sample('phi_conc', Beta(halpha_phi, beta_prec_phi - halpha_phi))
            psi_conc = numpyro.sample('psi_conc', Beta(halpha_psi, beta_prec_psi - halpha_psi))
            corr_scale = numpyro.sample('corr_scale', Beta(2., 5.))
            # SS prior
            skew_phi = numpyro.sample('skew_phi', Uniform(-1., 1.))
            psi_bound = 1 - skew_phi.abs()
            skew_psi = numpyro.sample('skew_psi', Uniform(-1., 1.))
            skewness = torch.stack((skew_phi, psi_bound * skew_psi), dim=-1)
            assert skewness.shape == (num_mix_comp, 2)
            with numpyro.plate('obs_plate'):
                sine = SineBivariateVonMises(phi_loc=phi_loc, psi_loc=psi_loc,
                                            phi_concentration=1000 * phi_conc,
                                            psi_concentration=1000 * psi_conc,
                                            weighted_correlation=corr_scale)
                        return numpyro.sample('phi_psi', SineSkewed(sine, skewness), obs=obs)
    To ensure the skewing does not alter the normalization constant of the (Sine Bivaraite von Mises) base
    distribution the skewness parameters are constraint. The constraint requires the sum of the absolute values of
    skewness to be less than or equal to one.
    So for the above snippet it must hold that::
        skew_phi.abs()+skew_psi.abs() <= 1
    We handle this in the prior by computing psi_bound and use it to scale skew_psi.
    We do **not** use psi_bound as::
        skew_psi = pyro.sample('skew_psi', Uniform(-psi_bound, psi_bound))
    as it would make the support for the Uniform distribution dynamic.
    In the context of :class:`~pyro.infer.SVI`, this distribution can freely be used as a likelihood, but use as
    latent variables it will lead to slow inference for 2 and higher dim toruses. This is because the base_dist
    cannot be reparameterized.
    .. note:: An event in the base distribution must be on a d-torus, so the event_shape must be (d,).
    .. note:: For the skewness parameter, it must hold that the sum of the absolute value of its weights for an event
        must be less than or equal to one. See eq. 2.1 in [1].
    ** References: **
      1. Sine-skewed toroidal distributions and their application in protein bioinformatics
         Ameijeiras-Alonso, J., Ley, C. (2019)
    :param torch.distributions.Distribution base_dist: base density on a d-dimensional torus. Supported base
        distributions include: 1D :class:`~numpyro.distributions.VonMises`,
        :class:`~numnumpyro.distributions.SineBivariateVonMises`, 1D :class:`~numpyro.distributions.ProjectedNormal`,
        and :class:`~numpyro.distributions.Uniform` (-pi, pi).
    :param torch.tensor skewness: skewness of the distribution.
    """

    arg_constraints = {
        "skewness": constraints.independent(constraints.interval(-1.0, 1.0), 1)
    }

    support = constraints.independent(constraints.real, 1)

    def __init__(self, base_dist: Distribution, skewness, validate_args=None):
        batch_shape = jnp.broadcast_shapes(base_dist.batch_shape, skewness.shape[:-1])
        event_shape = skewness.shape[-1:]
        self.skewness = jnp.broadcast_to(skewness, batch_shape + event_shape)
        self.base_dist = base_dist.expand(batch_shape)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def __repr__(self):
        args_string = ", ".join(
            [
                "{}: {}".format(
                    p,
                    getattr(self, p)
                    if getattr(self, p).numel() == 1
                    else getattr(self, p).size(),
                )
                for p in self.arg_constraints.keys()
            ]
        )
        return (
            self.__class__.__name__
            + "("
            + f"base_density: {str(self.base_dist)}, "
            + args_string
            + ")"
        )

    def sample(self, key, sample_shape=()):
        base_key, skew_key = random.split(key)
        bd = self.base_dist
        ys = bd.sample(base_key, sample_shape)
        u = random.uniform(skew_key, sample_shape + self.batch_shape)

        # Section 2.3 step 3 in [1]
        mask = u <= 0.5 + 0.5 * (
            self.skewness * jnp.sin((ys - bd.mean) % (2 * jnp.pi))
        ).sum(-1)
        mask = mask[..., None]
        samples = (jnp.where(mask, ys, -ys + 2 * bd.mean) + jnp.pi) % (
            2 * jnp.pi
        ) - jnp.pi
        return samples

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        if self.base_dist._validate_args:
            self.base_dist._validate_sample(value)

        # Eq. 2.1 in [1]
        skew_prob = jnp.log(
            1
            + (
                self.skewness * jnp.sin((value - self.base_dist.mean) % (2 * jnp.pi))
            ).sum(-1)
        )
        return self.base_dist.log_prob(value) + skew_prob

    @property
    def mean(self):
        return self.base_dist.mean


class VonMises(Distribution):
    arg_constraints = {"loc": constraints.real, "concentration": constraints.positive}
    reparametrized_params = ["loc"]
    support = constraints.interval(-math.pi, math.pi)

    def __init__(self, loc, concentration, validate_args=None):
        """von Mises distribution for sampling directions.

        :param loc: center of distribution
        :param concentration: concentration of distribution
        """
        self.loc, self.concentration = promote_shapes(loc, concentration)

        batch_shape = lax.broadcast_shapes(jnp.shape(concentration), jnp.shape(loc))

        super(VonMises, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        """Generate sample from von Mises distribution

        :param key: random number generator key
        :param sample_shape: shape of samples
        :return: samples from von Mises
        """
        assert is_prng_key(key)
        samples = von_mises_centered(
            key, self.concentration, sample_shape + self.shape()
        )
        samples = samples + self.loc  # VM(0, concentration) -> VM(loc,concentration)
        samples = (samples + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

        return samples

    @validate_sample
    def log_prob(self, value):
        return -(
            jnp.log(2 * jnp.pi) + jnp.log(i0e(self.concentration))
        ) + self.concentration * (jnp.cos((value - self.loc) % (2 * jnp.pi)) - 1)

    @property
    def mean(self):
        """Computes circular mean of distribution. NOTE: same as location when mapped to support [-pi, pi]"""
        return jnp.broadcast_to(
            (self.loc + jnp.pi) % (2.0 * jnp.pi) - jnp.pi, self.batch_shape
        )

    @property
    def variance(self):
        """Computes circular variance of distribution"""
        return jnp.broadcast_to(
            1.0 - i1e(self.concentration) / i0e(self.concentration), self.batch_shape
        )


class ProjectedNormal(Distribution):
    """
    Projected isotropic normal distribution of arbitrary dimension.

    This distribution over directional data is qualitatively similar to the von
    Mises and von Mises-Fisher distributions, but permits tractable variational
    inference via reparametrized gradients.

    To use this distribution with autoguides and HMC, use ``handlers.reparam``
    with a :class:`~numpyro.infer.reparam.ProjectedNormalReparam`
    reparametrizer in the model, e.g.::

        @handlers.reparam(config={"direction": ProjectedNormalReparam()})
        def model():
            direction = numpyro.sample("direction",
                                       ProjectedNormal(zeros(3)))
            ...

    .. note:: This implements :meth:`log_prob` only for dimensions {2,3}.

    [1] D. Hernandez-Stumpfhauser, F.J. Breidt, M.J. van der Woerd (2017)
        "The General Projected Normal Distribution of Arbitrary Dimension:
        Modeling and Bayesian Inference"
        https://projecteuclid.org/euclid.ba/1453211962
    """

    arg_constraints = {"concentration": constraints.real_vector}
    reparametrized_params = ["concentration"]
    support = constraints.sphere

    def __init__(self, concentration, *, validate_args=None):
        assert jnp.ndim(concentration) >= 1
        self.concentration = concentration
        batch_shape = concentration.shape[:-1]
        event_shape = concentration.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def mean(self):
        """
        Note this is the mean in the sense of a centroid in the submanifold
        that minimizes expected squared geodesic distance.
        """
        return safe_normalize(self.concentration)

    @property
    def mode(self):
        return safe_normalize(self.concentration)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        eps = random.normal(key, shape=shape)
        return safe_normalize(self.concentration + eps)

    def log_prob(self, value):
        if self._validate_args:
            event_shape = value.shape[-1:]
            if event_shape != self.event_shape:
                raise ValueError(
                    f"Expected event shape {self.event_shape}, "
                    f"but got {event_shape}"
                )
            self._validate_sample(value)
        dim = int(self.concentration.shape[-1])
        if dim == 2:
            return _projected_normal_log_prob_2(self.concentration, value)
        if dim == 3:
            return _projected_normal_log_prob_3(self.concentration, value)
        raise NotImplementedError(
            f"ProjectedNormal.log_prob() is not implemented for dim = {dim}. "
            "Consider using handlers.reparam with ProjectedNormalReparam."
        )

    @staticmethod
    def infer_shapes(concentration):
        batch_shape = concentration[:-1]
        event_shape = concentration[-1:]
        return batch_shape, event_shape


def _projected_normal_log_prob_2(concentration, value):
    def _dot(x, y):
        return (x[..., None, :] @ y[..., None])[..., 0, 0]

    # We integrate along a ray, factorizing the integrand as a product of:
    # a truncated normal distribution over coordinate t parallel to the ray, and
    # a univariate normal distribution over coordinate r perpendicular to the ray.
    t = _dot(concentration, value)
    t2 = t * t
    r2 = _dot(concentration, concentration) - t2
    perp_part = (-0.5) * r2 - 0.5 * math.log(2 * math.pi)

    # This is the log of a definite integral, computed by mathematica:
    # Integrate[x/(E^((x-t)^2/2) Sqrt[2 Pi]), {x, 0, Infinity}]
    # = (t + Sqrt[2/Pi]/E^(t^2/2) + t Erf[t/Sqrt[2]])/2
    para_part = jnp.log(
        (jnp.exp((-0.5) * t2) * ((2 / math.pi) ** 0.5) + t * (1 + erf(t * 0.5 ** 0.5)))
        / 2
    )

    return para_part + perp_part


def _projected_normal_log_prob_3(concentration, value):
    def _dot(x, y):
        return (x[..., None, :] @ y[..., None])[..., 0, 0]

    # We integrate along a ray, factorizing the integrand as a product of:
    # a truncated normal distribution over coordinate t parallel to the ray, and
    # a bivariate normal distribution over coordinate r perpendicular to the ray.
    t = _dot(concentration, value)
    t2 = t * t
    r2 = _dot(concentration, concentration) - t2
    perp_part = (-0.5) * r2 - math.log(2 * math.pi)

    # This is the log of a definite integral, computed by mathematica:
    # Integrate[x^2/(E^((x-t)^2/2) Sqrt[2 Pi]), {x, 0, Infinity}]
    # = t/(E^(t^2/2) Sqrt[2 Pi]) + ((1 + t^2) (1 + Erf[t/Sqrt[2]]))/2
    para_part = jnp.log(
        t * jnp.exp((-0.5) * t2) / (2 * math.pi) ** 0.5
        + (1 + t2) * (1 + erf(t * 0.5 ** 0.5)) / 2
    )

    return para_part + perp_part
