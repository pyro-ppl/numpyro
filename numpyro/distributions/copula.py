# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

import jax
from jax import Array, lax, numpy as jnp
from jax.typing import ArrayLike

import numpyro.distributions.constraints as constraints
from numpyro.distributions.constraints import Constraint
from numpyro.distributions.continuous import Beta, MultivariateNormal, Normal
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import clamp_probs, lazy_property, validate_sample
from numpyro.util import is_prng_key


class GaussianCopula(Distribution):
    r"""
    Couple continuous univariate marginals with a Gaussian copula.

    Let :math:`F_i` and :math:`f_i` denote the CDF and density of the
    :math:`i`-th marginal, and let :math:`\Phi` be the standard normal CDF.
    For a positive definite correlation matrix :math:`R`, samples are constructed as

    .. math::

        Z \sim \mathcal{N}(0, R), \qquad X_i = F_i^{-1}(\Phi(Z_i)).

    For interior points with :math:`0 < F_i(x_i) < 1`, write
    :math:`z_i = \Phi^{-1}(F_i(x_i))`. The joint density is

    .. math::

        p(x) = \frac{\exp\left[-\frac{1}{2}z^\top(R^{-1}-I)z\right]}
                    {\sqrt{\det R}} \prod_{i=1}^D f_i(x_i),

    or equivalently

    .. math::

        \log p(x) = -\frac{1}{2}\log\det R
                    -\frac{1}{2}z^\top(R^{-1}-I)z
                    + \sum_{i=1}^D \log f_i(x_i).

    The support is the Cartesian product of the marginal supports. Each coordinate
    retains its marginal mean and variance, when these moments exist. In particular,
    :math:`R=I` gives independent marginals. The matrix :math:`R` describes correlation
    in the latent normal variables; it generally differs from the Pearson correlation
    of :math:`X` after the nonlinear marginal transformations.

    When present, the final batch axis of ``marginal_dist`` supplies the event coordinates
    and must broadcast to :math:`D`, the dimension of the correlation matrix. The remaining
    batch axes broadcast with those of the correlation parameters. Scalar marginals
    are shared across all :math:`D` event coordinates. The marginal
    distribution must implement ``cdf`` and ``icdf`` as well as ``log_prob``.

    .. note::

        ``log_prob`` clamps marginal CDF values away from zero and one before taking
        normal quantiles, so the formula above describes the mathematical density
        rather than exact numerical behavior at boundaries or in extreme tails.

    **References**

    [1] `An Introduction to Copulas`, second edition,
    Roger B. Nelsen (2006)
    https://doi.org/10.1007/0-387-28678-0

    :param Distribution marginal_dist: Continuous univariate distribution whose final
        batch axis is to be coupled.
    :param array_like correlation_matrix: Latent normal correlation matrix :math:`R`.
        Specify exactly one of this or ``correlation_cholesky``.
    :param array_like correlation_cholesky: Lower triangular correlation Cholesky factor
        :math:`L`, where :math:`R = LL^\top`.
    """

    arg_constraints = {
        "correlation_matrix": constraints.corr_matrix,
        "correlation_cholesky": constraints.corr_cholesky,
    }
    reparametrized_params = [
        "correlation_matrix",
        "correlation_cholesky",
    ]

    pytree_data_fields = ("marginal_dist", "base_dist")

    def __init__(
        self,
        marginal_dist: Distribution,
        correlation_matrix: Optional[ArrayLike] = None,
        correlation_cholesky: Optional[ArrayLike] = None,
        *,
        validate_args: Optional[bool] = None,
    ):
        if len(marginal_dist.event_shape) > 0:
            raise ValueError("`marginal_dist` needs to be a univariate distribution.")

        self.marginal_dist = marginal_dist
        self.base_dist = MultivariateNormal(
            covariance_matrix=correlation_matrix,
            scale_tril=correlation_cholesky,
        )

        event_shape = self.base_dist.event_shape
        batch_shape = lax.broadcast_shapes(
            self.marginal_dist.batch_shape[:-1],
            self.base_dist.batch_shape,
        )

        super(GaussianCopula, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(
        self, key: Optional[jax.Array], sample_shape: tuple[int, ...] = ()
    ) -> Array:
        assert is_prng_key(key)

        shape = sample_shape + self.batch_shape
        normal_samples = self.base_dist.expand(shape).sample(key)
        cdf = Normal().cdf(normal_samples)
        return self.marginal_dist.icdf(cdf)

    @validate_sample
    def log_prob(self, value: ArrayLike) -> Array:
        # Ref: https://en.wikipedia.org/wiki/Copula_(probability_theory)#Gaussian_copula
        # see also https://github.com/pyro-ppl/numpyro/pull/1506#discussion_r1037525015
        marginal_lps = self.marginal_dist.log_prob(value)
        probs = self.marginal_dist.cdf(value)
        quantiles = Normal().icdf(clamp_probs(probs))

        copula_lp = (
            self.base_dist.log_prob(quantiles)
            + 0.5 * (quantiles**2).sum(-1)
            + 0.5 * jnp.log(2 * jnp.pi) * quantiles.shape[-1]
        )
        return copula_lp + jnp.sum(marginal_lps, axis=-1)

    @property
    def mean(self) -> Array:
        return jnp.broadcast_to(self.marginal_dist.mean, self.shape())

    @property
    def variance(self) -> Array:
        return jnp.broadcast_to(self.marginal_dist.variance, self.shape())

    @constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self) -> Constraint:
        assert self.marginal_dist.support is not None
        return constraints.independent(self.marginal_dist.support, 1)

    @lazy_property
    def correlation_matrix(self) -> Array:
        return jnp.asarray(self.base_dist.covariance_matrix)

    @lazy_property
    def correlation_cholesky(self) -> Array:
        return self.base_dist.scale_tril


class GaussianCopulaBeta(GaussianCopula):
    r"""
    Gaussian copula with beta marginals on :math:`[0,1]^D`.

    Write :math:`\alpha_i = \text{concentration1}_i > 0` and
    :math:`\beta_i = \text{concentration0}_i > 0`. Each marginal has density

    .. math::

        f_i(x_i) = \frac{x_i^{\alpha_i-1}(1-x_i)^{\beta_i-1}}
                           {B(\alpha_i,\beta_i)}, \qquad 0 < x_i < 1,

    where :math:`B` is the beta function. Substituting these densities and their CDFs
    into :class:`GaussianCopula` gives the joint density and sampling construction.
    The coordinate-wise moments are

    .. math::

        \mathbb{E}[X_i] = \frac{\alpha_i}{\alpha_i+\beta_i}, \qquad
        \operatorname{Var}(X_i) =
            \frac{\alpha_i\beta_i}
            {(\alpha_i+\beta_i)^2(\alpha_i+\beta_i+1)}.

    The correlation parameters describe the latent normal variables, rather than
    the Pearson correlation of the beta variables. Batch-shape broadcasting and the
    CDF clamping in ``log_prob`` are inherited from :class:`GaussianCopula`.

    **References**

    [1] SciPy beta distribution reference
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html

    :param array_like concentration1: Positive first beta shape parameter :math:`\alpha`.
    :param array_like concentration0: Positive second beta shape parameter :math:`\beta`.
    :param array_like correlation_matrix: Latent normal correlation matrix :math:`R`.
        Specify exactly one of this or ``correlation_cholesky``.
    :param array_like correlation_cholesky: Lower triangular correlation Cholesky factor
        :math:`L`, where :math:`R = LL^\top`.
    """

    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
        "correlation_matrix": constraints.corr_matrix,
        "correlation_cholesky": constraints.corr_cholesky,
    }
    support = constraints.independent(constraints.unit_interval, 1)
    pytree_data_fields = ("concentration1", "concentration0")

    def __init__(
        self,
        concentration1: ArrayLike,
        concentration0: ArrayLike,
        correlation_matrix: Optional[ArrayLike] = None,
        correlation_cholesky: Optional[ArrayLike] = None,
        *,
        validate_args: bool = False,
    ):
        # set initially to allow argument validation
        self.concentration1, self.concentration0 = concentration1, concentration0

        super().__init__(
            Beta(concentration1, concentration0),
            correlation_matrix,
            correlation_cholesky,
            validate_args=validate_args,
        )
