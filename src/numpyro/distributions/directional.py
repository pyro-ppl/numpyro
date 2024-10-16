# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
import functools
import math
from math import pi
import operator

from jax import lax
import jax.numpy as jnp
import jax.random as random
from jax.scipy import special
from jax.scipy.special import erf, i0e, i1e, logsumexp

from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import (
    assert_one_of,
    lazy_property,
    promote_shapes,
    safe_normalize,
    validate_sample,
    von_mises_centered,
)
from numpyro.util import is_prng_key, while_loop


def _numel(shape):
    return functools.reduce(operator.mul, shape, 1)


def log_I1(orders: int, value, terms=250):
    r"""Compute first n log modified bessel function of first kind
    .. math ::
        \log(I_v(z)) = v*\log(z/2) + \log(\sum_{k=0}^\inf \exp\left[2*k*\log(z/2) - \sum_kk^k log(kk)
        - \lgamma(v + k + 1)\right])
    :param orders: orders of the log modified bessel function.
    :param value: values to compute modified bessel function for
    :param terms: truncation of summation
    :return: 0 to orders modified bessel function
    """
    orders = orders + 1
    if value.ndim == 0:
        vshape = jnp.shape([1])
    else:
        vshape = value.shape
    value = value.reshape(-1, 1)
    flat_vshape = _numel(vshape)

    k = jnp.arange(terms)
    lgammas_all = special.gammaln(jnp.arange(1.0, terms + orders + 1))
    assert lgammas_all.shape == (orders + terms,)  # lgamma(0) = inf => start from 1

    lvalues = jnp.log(value / 2) * k.reshape(1, -1)
    assert lvalues.shape == (flat_vshape, terms)

    lfactorials = lgammas_all[:terms]
    assert lfactorials.shape == (terms,)

    lgammas = jnp.tile(lgammas_all, orders).reshape((orders, -1))
    assert lgammas.shape == (orders, terms + orders)  # lgamma(0) = inf => start from 1

    indices = k[:orders].reshape(-1, 1) + k.reshape(1, -1)
    assert indices.shape == (orders, terms)

    seqs = logsumexp(
        2 * lvalues[None, :, :]
        - lfactorials[None, None, :]
        - jnp.take_along_axis(lgammas, indices, axis=1)[:, None, :],
        -1,
    )
    assert seqs.shape == (orders, flat_vshape)

    i1s = lvalues[..., :orders].T + seqs
    assert i1s.shape == (orders, flat_vshape)
    return i1s.reshape(-1, *vshape)


class VonMises(Distribution):
    """
    The von Mises distribution, also known as the circular normal distribution.

    This distribution is supported by a circular constraint from -pi to +pi. By
    default, the circular support behaves like
    ``constraints.interval(-math.pi, math.pi)``. To avoid issues at the
    boundaries of this interval during sampling, you should reparameterize this
    distribution using ``handlers.reparam`` with a
    :class:`~numpyro.infer.reparam.CircularReparam` reparametrizer in
    the model, e.g.::

        @handlers.reparam(config={"direction": CircularReparam()})
        def model():
            direction = numpyro.sample("direction", VonMises(0.0, 4.0))
            ...
    """

    arg_constraints = {"loc": constraints.real, "concentration": constraints.positive}
    reparametrized_params = ["loc"]
    support = constraints.circular

    def __init__(self, loc, concentration, *, validate_args=None):
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


PhiMarginalState = namedtuple("PhiMarginalState", ["i", "done", "phi", "key"])


class SineSkewed(Distribution):
    r"""Sine-skewing [1] is a procedure for producing a distribution that breaks pointwise symmetry on a torus
    distribution. The new distribution is called the Sine Skewed X distribution, where X is the name of the (symmetric)
    base distribution. Torus distributions are distributions with support on products of circles
    (i.e., :math:`\otimes S^1` where :math:`S^1 = [-pi,pi)`).
    So, a 0-torus is a point, the 1-torus is a circle,
    and the 2-torus is commonly associated with the donut shape.

    The sine skewed X distribution is parameterized by a weight parameter for each dimension of the event of X.
    For example with a von Mises distribution over a circle (1-torus), the sine skewed von Mises distribution has one
    skew parameter. The skewness parameters can be inferred using :class:`~numpyro.infer.HMC` or
    :class:`~numpyro.infer.NUTS`. For example, the following will produce a prior over
    skewness for the 2-torus,::

        @numpyro.handlers.reparam(config={'phi_loc': CircularReparam(), 'psi_loc': CircularReparam()})
        def model(obs):
            # Sine priors
            phi_loc = numpyro.sample('phi_loc', VonMises(pi, 2.))
            psi_loc = numpyro.sample('psi_loc', VonMises(-pi / 2, 2.))
            phi_conc = numpyro.sample('phi_conc', Beta(1., 1.))
            psi_conc = numpyro.sample('psi_conc', Beta(1., 1.))
            corr_scale = numpyro.sample('corr_scale', Beta(2., 5.))

            # Skewing prior
            ball_trans = L1BallTransform()
            skewness = numpyro.sample('skew_phi', Normal(0, 0.5).expand((2,)))
            skewness = ball_trans(skewness)  # constraint sum |skewness_i| <= 1

            with numpyro.plate('obs_plate'):
                sine = SineBivariateVonMises(phi_loc=phi_loc, psi_loc=psi_loc,
                                             phi_concentration=70 * phi_conc,
                                             psi_concentration=70 * psi_conc,
                                             weighted_correlation=corr_scale)
                return numpyro.sample('phi_psi', SineSkewed(sine, skewness), obs=obs)

    To ensure the skewing does not alter the normalization constant of the (sine bivariate von Mises) base
    distribution the skewness parameters are constraint. The constraint requires the sum of the absolute values of
    skewness to be less than or equal to one. We can use the :class:`~numpyro.distriubtions.transforms.L1BallTransform`
    to achieve this.

    In the context of :class:`~pyro.infer.SVI`, this distribution can freely be used as a likelihood, but use as
    latent variables it will lead to slow inference for 2 and higher dim toruses. This is because the base_dist
    cannot be reparameterized.

    .. note:: An event in the base distribution must be on a d-torus, so the event_shape must be `(d,)`.

    .. note:: For the skewness parameter, it must hold that the sum of the absolute value of its weights for an event
        must be less than or equal to one. See eq. 2.1 in [1].

    ** References: **
        1. Sine-skewed toroidal distributions and their application in protein bioinformatics
            Ameijeiras-Alonso, J., Ley, C. (2019)

    :param numpyro.distributions.Distribution base_dist: base density on a d-dimensional torus. Supported base
        distributions include: 1D :class:`~numpyro.distributions.VonMises`,
        :class:`~numnumpyro.distributions.SineBivariateVonMises`, 1D :class:`~numpyro.distributions.ProjectedNormal`,
        and :class:`~numpyro.distributions.Uniform` (-pi, pi).
    :param jax.numpy.array skewness: skewness of the distribution.
    """

    arg_constraints = {"skewness": constraints.l1_ball}
    pytree_data_fields = ("base_dist", "skewness")

    support = constraints.independent(constraints.circular, 1)

    def __init__(self, base_dist: Distribution, skewness, *, validate_args=None):
        assert (
            base_dist.event_shape == skewness.shape[-1:]
        ), "Sine Skewing is only valid with a skewness parameter for each dimension of `base_dist.event_shape`."

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
                    if getattr(self, p).size == 1
                    else getattr(self, p).size,
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
        skew_prob = jnp.log1p(
            (self.skewness * jnp.sin((value - self.base_dist.mean) % (2 * jnp.pi))).sum(
                -1
            )
        )
        return self.base_dist.log_prob(value) + skew_prob

    @property
    def mean(self):
        """Mean of the base distribution"""
        return self.base_dist.mean


class SineBivariateVonMises(Distribution):
    r"""Unimodal distribution of two dependent angles on the 2-torus
    (:math:`S^1 \otimes S^1`) given by

    .. math::
        C^{-1}\exp(\kappa_1\cos(x_1-\mu_1) + \kappa_2\cos(x_2 -\mu_2) + \rho\sin(x_1 - \mu_1)\sin(x_2 - \mu_2))

    and

    .. math::
        C = (2\pi)^2 \sum_{i=0} {2i \choose i}
        \left(\frac{\rho^2}{4\kappa_1\kappa_2}\right)^i I_i(\kappa_1)I_i(\kappa_2),

    where :math:`I_i(\cdot)` is the modified bessel function of first kind, mu's are the locations of the distribution,
    kappa's are the concentration and rho gives the correlation between angles :math:`x_1` and :math:`x_2`.
    This distribution is helpful for modeling coupled angles such as torsion angles in peptide chains.

    To infer parameters, use :class:`~numpyro.infer.hmc.NUTS` or :class:`~numpyro.infer.hmc.HMC` with priors that
    avoid parameterizations where the distribution becomes bimodal; see note below.

    .. note:: Sample efficiency drops as

        .. math::
            \frac{\rho}{\kappa_1\kappa_2} \rightarrow 1

        because the distribution becomes increasingly bimodal. To avoid bimodality use the `weighted_correlation`
        parameter with a skew away from one (e.g., Beta(1,3)). The `weighted_correlation` should be in [0,1].

    .. note:: The correlation and weighted_correlation params are mutually exclusive.

    .. note:: In the context of :class:`~numpyro.infer.svi.SVI`, this distribution can be used as a likelihood but not
        for latent variables.

    ** References: **
        1. Probabilistic model for two dependent circular variables Singh, H., Hnizdo, V., and Demchuck, E. (2002)

    :param np.ndarray phi_loc: location of first angle
    :param np.ndarray psi_loc: location of second angle
    :param np.ndarray phi_concentration: concentration of first angle
    :param np.ndarray psi_concentration: concentration of second angle
    :param np.ndarray correlation: correlation between the two angles
    :param np.ndarray weighted_correlation: set correlation to weighted_corr * sqrt(phi_conc*psi_conc)
        to avoid bimodality (see note). The `weighted_correlation` should be in [0,1].
    """

    arg_constraints = {
        "phi_loc": constraints.circular,
        "psi_loc": constraints.circular,
        "phi_concentration": constraints.positive,
        "psi_concentration": constraints.positive,
        "correlation": constraints.real,
    }
    support = constraints.independent(constraints.circular, 1)
    max_sample_iter = 1000

    def __init__(
        self,
        phi_loc,
        psi_loc,
        phi_concentration,
        psi_concentration,
        correlation=None,
        weighted_correlation=None,
        validate_args=None,
    ):
        assert_one_of(
            correlation=correlation, weighted_correlation=weighted_correlation
        )

        if weighted_correlation is not None:
            correlation = weighted_correlation * jnp.sqrt(
                phi_concentration * psi_concentration
            )

        batch_shape = lax.broadcast_shapes(
            jnp.shape(phi_loc),
            jnp.shape(psi_loc),
            jnp.shape(phi_concentration),
            jnp.shape(psi_concentration),
            jnp.shape(correlation),
        )
        (
            self.phi_loc,
            self.psi_loc,
            self.phi_concentration,
            self.psi_concentration,
            self.correlation,
        ) = promote_shapes(
            phi_loc,
            psi_loc,
            phi_concentration,
            psi_concentration,
            correlation,
            shape=batch_shape,
        )

        super().__init__(batch_shape, (2,), validate_args=validate_args)

        self.phi_loc = jnp.broadcast_to(self.phi_loc, batch_shape)
        self.psi_loc = jnp.broadcast_to(self.psi_loc, batch_shape)
        self.phi_concentration = jnp.broadcast_to(self.phi_concentration, batch_shape)
        self.psi_concentration = jnp.broadcast_to(self.psi_concentration, batch_shape)
        self.correlation = jnp.broadcast_to(self.correlation, batch_shape)

    @lazy_property
    def norm_const(self):
        corr = jnp.reshape(self.correlation, (1, -1))
        conc = jnp.stack(
            (self.phi_concentration, self.psi_concentration), axis=-1
        ).reshape(-1, 2)
        m = jnp.arange(50).reshape(-1, 1)
        num = special.gammaln(2 * m + 1.0)
        den = special.gammaln(m + 1.0)
        lbinoms = num - 2 * den

        fs = lbinoms.reshape(-1, 1) + m * (
            jnp.log(jnp.clip(corr**2, jnp.finfo(jnp.result_type(float)).tiny))
            - jnp.log(4 * jnp.prod(conc, axis=-1))
        )
        fs += log_I1(49, conc, terms=51).sum(-1)
        norm_const = 2 * jnp.log(jnp.array(2 * pi)) + logsumexp(fs, 0)
        return norm_const.reshape(jnp.shape(self.phi_loc))

    @validate_sample
    def log_prob(self, value):
        indv = self.phi_concentration * jnp.cos(
            value[..., 0] - self.phi_loc
        ) + self.psi_concentration * jnp.cos(value[..., 1] - self.psi_loc)
        corr = (
            self.correlation
            * jnp.sin(value[..., 0] - self.phi_loc)
            * jnp.sin(value[..., 1] - self.psi_loc)
        )
        return indv + corr - self.norm_const

    def sample(self, key, sample_shape=()):
        """
        ** References: **
            1. A New Unified Approach for the Simulation of a Wide Class of Directional Distributions
               John T. Kent, Asaad M. Ganeiber & Kanti V. Mardia (2018)
        """
        assert is_prng_key(key)
        phi_key, psi_key = random.split(key)

        corr = self.correlation
        conc = jnp.stack((self.phi_concentration, self.psi_concentration))

        eig = 0.5 * (conc[0] - corr**2 / conc[1])
        eig = jnp.stack((jnp.zeros_like(eig), eig))
        eigmin = jnp.where(eig[1] < 0, eig[1], jnp.zeros_like(eig[1], dtype=eig.dtype))
        eig = eig - eigmin
        b0 = self._bfind(eig)

        total = _numel(sample_shape)
        phi_den = log_I1(0, conc[1]).squeeze(0)
        batch_size = _numel(self.batch_shape)
        phi_shape = (total, 2, batch_size)
        phi_state = SineBivariateVonMises._phi_marginal(
            phi_shape,
            phi_key,
            jnp.reshape(conc, (2, batch_size)),
            jnp.reshape(corr, (batch_size,)),
            jnp.reshape(eig, (2, batch_size)),
            jnp.reshape(b0, (batch_size,)),
            jnp.reshape(eigmin, (batch_size,)),
            jnp.reshape(phi_den, (batch_size,)),
        )

        phi = jnp.arctan2(phi_state.phi[:, 1], phi_state.phi[:, 0])

        alpha = jnp.sqrt(
            conc[1].reshape(-1) ** 2 + (corr.reshape(-1) * jnp.sin(phi)) ** 2
        )
        beta = jnp.arctan(corr.reshape(-1) / conc[1].reshape(-1) * jnp.sin(phi))

        psi = VonMises(beta, alpha).sample(psi_key)

        phi_psi = jnp.stack(
            (
                (phi + jnp.reshape(self.phi_loc, -1) + pi) % (2 * pi) - pi,
                (psi + jnp.reshape(self.psi_loc, -1) + pi) % (2 * pi) - pi,
            ),
            axis=-1,
        )

        return phi_psi.reshape(*sample_shape, *self.batch_shape, *self.event_shape)

    @staticmethod
    def _phi_marginal(shape, rng_key, conc, corr, eig, b0, eigmin, phi_den):
        def update_fn(curr):
            i, done, phi, key = curr
            phi_key, key = random.split(key)
            accept_key, acg_key, phi_key = random.split(phi_key, 3)

            x = lax.rsqrt(1 + 2 * eig / b0) * random.normal(acg_key, shape)
            x /= jnp.linalg.norm(
                x, axis=1, keepdims=True
            )  # Angular Central Gaussian distribution

            lf = (
                conc[0] * (x[:, 0] - 1)
                + eigmin
                + log_I1(0, jnp.sqrt(conc[1] ** 2 + (corr * x[:, 1]) ** 2)).squeeze(0)
                - phi_den
            )

            lg_inv = 1.0 - b0 / 2 + jnp.log(b0 / 2 + (eig * x**2).sum(1))
            assert lg_inv.shape == lf.shape

            accepted = random.uniform(accept_key, lf.shape) < jnp.exp(lf + lg_inv)
            accepted = accepted[:, None]

            phi = jnp.where(accepted, x, phi)
            return PhiMarginalState(i + 1, done | accepted, phi, key)

        def cond_fn(curr):
            return jnp.bitwise_and(
                curr.i < SineBivariateVonMises.max_sample_iter,
                jnp.logical_not(jnp.all(curr.done)),
            )

        phi_state = while_loop(
            cond_fn,
            update_fn,
            PhiMarginalState(
                i=jnp.array(0),
                done=jnp.zeros(shape, dtype=bool),
                phi=jnp.empty(shape, dtype=float),
                key=rng_key,
            ),
        )
        return PhiMarginalState(
            phi_state.i, phi_state.done, phi_state.phi, phi_state.key
        )

    @property
    def mean(self):
        """Computes circular mean of distribution. Note: same as location when mapped to support [-pi, pi]"""
        mean = (jnp.stack((self.phi_loc, self.psi_loc), axis=-1) + jnp.pi) % (
            2.0 * jnp.pi
        ) - jnp.pi
        return jnp.broadcast_to(mean, (*self.batch_shape, 2))

    def _bfind(self, eig):
        b = eig.shape[0] / 2 * jnp.ones(self.batch_shape, dtype=eig.dtype)
        g1 = jnp.sum(1 / (b + 2 * eig) ** 2, axis=0)
        g2 = jnp.sum(-2 / (b + 2 * eig) ** 3, axis=0)
        return jnp.where(jnp.linalg.norm(eig, axis=0) != 0, b - g1 / g2, b)


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
        (jnp.exp((-0.5) * t2) * ((2 / math.pi) ** 0.5) + t * (1 + erf(t * 0.5**0.5)))
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
        + (1 + t2) * (1 + erf(t * 0.5**0.5)) / 2
    )

    return para_part + perp_part
