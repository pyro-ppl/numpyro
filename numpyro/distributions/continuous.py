# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

# The implementation largely follows the design in PyTorch's `torch.distributions`
#
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


from jax import lax, ops
import jax.nn as nn
import jax.numpy as jnp
import jax.random as random
from jax.scipy.linalg import cho_solve, solve_triangular
from jax.scipy.special import betainc, expit, gammaln, logit, multigammaln, ndtr, ndtri

from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution, TransformedDistribution
from numpyro.distributions.transforms import (
    AffineTransform,
    CorrMatrixCholeskyTransform,
    ExpTransform,
    PowerTransform,
)
from numpyro.distributions.util import (
    cholesky_of_inverse,
    is_prng_key,
    lazy_property,
    matrix_to_tril_vec,
    promote_shapes,
    signed_stick_breaking_tril,
    validate_sample,
    vec_to_tril_matrix,
)

EULER_MASCHERONI = 0.5772156649015328606065120900824024310421


class Beta(Distribution):
    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
    }
    reparametrized_params = ["concentration1", "concentration0"]
    support = constraints.unit_interval

    def __init__(self, concentration1, concentration0, validate_args=None):
        self.concentration1, self.concentration0 = promote_shapes(
            concentration1, concentration0
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(concentration1), jnp.shape(concentration0)
        )
        concentration1 = jnp.broadcast_to(concentration1, batch_shape)
        concentration0 = jnp.broadcast_to(concentration0, batch_shape)
        self._dirichlet = Dirichlet(
            jnp.stack([concentration1, concentration0], axis=-1)
        )
        super(Beta, self).__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return self._dirichlet.sample(key, sample_shape)[..., 0]

    @validate_sample
    def log_prob(self, value):
        return self._dirichlet.log_prob(jnp.stack([value, 1.0 - value], -1))

    @property
    def mean(self):
        return self.concentration1 / (self.concentration1 + self.concentration0)

    @property
    def variance(self):
        total = self.concentration1 + self.concentration0
        return self.concentration1 * self.concentration0 / (total ** 2 * (total + 1))

    def cdf(self, value):
        return betainc(self.concentration1, self.concentration0, value)


class Cauchy(Distribution):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    reparametrized_params = ["loc", "scale"]

    def __init__(self, loc=0.0, scale=1.0, validate_args=None):
        self.loc, self.scale = promote_shapes(loc, scale)
        batch_shape = lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        super(Cauchy, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        eps = random.cauchy(key, shape=sample_shape + self.batch_shape)
        return self.loc + eps * self.scale

    @validate_sample
    def log_prob(self, value):
        return (
            -jnp.log(jnp.pi)
            - jnp.log(self.scale)
            - jnp.log1p(((value - self.loc) / self.scale) ** 2)
        )

    @property
    def mean(self):
        return jnp.full(self.batch_shape, jnp.nan)

    @property
    def variance(self):
        return jnp.full(self.batch_shape, jnp.nan)

    def cdf(self, value):
        scaled = (value - self.loc) / self.scale
        return jnp.arctan(scaled) / jnp.pi + 0.5

    def icdf(self, q):
        return self.loc + self.scale * jnp.tan(jnp.pi * (q - 0.5))


class Dirichlet(Distribution):
    arg_constraints = {
        "concentration": constraints.independent(constraints.positive, 1)
    }
    reparametrized_params = ["concentration"]
    support = constraints.simplex

    def __init__(self, concentration, validate_args=None):
        if jnp.ndim(concentration) < 1:
            raise ValueError(
                "`concentration` parameter must be at least one-dimensional."
            )
        self.concentration = concentration
        batch_shape, event_shape = concentration.shape[:-1], concentration.shape[-1:]
        super(Dirichlet, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        shape = sample_shape + self.batch_shape + self.event_shape
        key_gamma, key_expon = random.split(key)
        # To improve precision for the cases concentration << 1,
        # we boost concentration to concentration + 1 and get gamma samples according to
        #   Gamma(concentration) ~ Gamma(concentration+1) * Uniform()^(1 / concentration)
        # When concentration << 1, u^(1 / concentration) is very near 0 and lost precision, so
        # we will convert the samples to log space
        #   log(Gamma(concentration)) ~ log(Gamma(concentration + 1)) - Expon() / concentration
        # and apply softmax to get a dirichlet sample
        gamma_samples = random.gamma(key_gamma, self.concentration + 1, shape=shape)
        expon_samples = random.exponential(key_expon, shape=shape)
        samples = nn.softmax(
            jnp.log(gamma_samples) - expon_samples / self.concentration, -1
        )
        return jnp.clip(
            samples, a_min=jnp.finfo(samples).tiny, a_max=1 - jnp.finfo(samples).eps
        )

    @validate_sample
    def log_prob(self, value):
        normalize_term = jnp.sum(gammaln(self.concentration), axis=-1) - gammaln(
            jnp.sum(self.concentration, axis=-1)
        )
        return (
            jnp.sum(jnp.log(value) * (self.concentration - 1.0), axis=-1)
            - normalize_term
        )

    @property
    def mean(self):
        return self.concentration / jnp.sum(self.concentration, axis=-1, keepdims=True)

    @property
    def variance(self):
        con0 = jnp.sum(self.concentration, axis=-1, keepdims=True)
        return (
            self.concentration * (con0 - self.concentration) / (con0 ** 2 * (con0 + 1))
        )

    @staticmethod
    def infer_shapes(concentration):
        batch_shape = concentration[:-1]
        event_shape = concentration[-1:]
        return batch_shape, event_shape


class Exponential(Distribution):
    reparametrized_params = ["rate"]
    arg_constraints = {"rate": constraints.positive}
    support = constraints.positive

    def __init__(self, rate=1.0, validate_args=None):
        self.rate = rate
        super(Exponential, self).__init__(
            batch_shape=jnp.shape(rate), validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return (
            random.exponential(key, shape=sample_shape + self.batch_shape) / self.rate
        )

    @validate_sample
    def log_prob(self, value):
        return jnp.log(self.rate) - self.rate * value

    @property
    def mean(self):
        return jnp.reciprocal(self.rate)

    @property
    def variance(self):
        return jnp.reciprocal(self.rate ** 2)

    def cdf(self, value):
        return -jnp.expm1(-self.rate * value)

    def icdf(self, q):
        return -jnp.log1p(-q) / self.rate


class Gamma(Distribution):
    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
    }
    support = constraints.positive
    reparametrized_params = ["concentration", "rate"]

    def __init__(self, concentration, rate=1.0, validate_args=None):
        self.concentration, self.rate = promote_shapes(concentration, rate)
        batch_shape = lax.broadcast_shapes(jnp.shape(concentration), jnp.shape(rate))
        super(Gamma, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.gamma(key, self.concentration, shape=shape) / self.rate

    @validate_sample
    def log_prob(self, value):
        normalize_term = gammaln(self.concentration) - self.concentration * jnp.log(
            self.rate
        )
        return (
            (self.concentration - 1) * jnp.log(value)
            - self.rate * value
            - normalize_term
        )

    @property
    def mean(self):
        return self.concentration / self.rate

    @property
    def variance(self):
        return self.concentration / jnp.power(self.rate, 2)


class Chi2(Gamma):
    arg_constraints = {"df": constraints.positive}
    reparametrized_params = ["df"]

    def __init__(self, df, validate_args=None):
        self.df = df
        super(Chi2, self).__init__(0.5 * df, 0.5, validate_args=validate_args)


class GaussianRandomWalk(Distribution):
    arg_constraints = {"scale": constraints.positive}
    support = constraints.real_vector
    reparametrized_params = ["scale"]

    def __init__(self, scale=1.0, num_steps=1, validate_args=None):
        assert (
            isinstance(num_steps, int) and num_steps > 0
        ), "`num_steps` argument should be an positive integer."
        self.scale = scale
        self.num_steps = num_steps
        batch_shape, event_shape = jnp.shape(scale), (num_steps,)
        super(GaussianRandomWalk, self).__init__(
            batch_shape, event_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        shape = sample_shape + self.batch_shape + self.event_shape
        walks = random.normal(key, shape=shape)
        return jnp.cumsum(walks, axis=-1) * jnp.expand_dims(self.scale, axis=-1)

    @validate_sample
    def log_prob(self, value):
        init_prob = Normal(0.0, self.scale).log_prob(value[..., 0])
        scale = jnp.expand_dims(self.scale, -1)
        step_probs = Normal(value[..., :-1], scale).log_prob(value[..., 1:])
        return init_prob + jnp.sum(step_probs, axis=-1)

    @property
    def mean(self):
        return jnp.zeros(self.batch_shape + self.event_shape)

    @property
    def variance(self):
        return jnp.broadcast_to(
            jnp.expand_dims(self.scale, -1) ** 2 * jnp.arange(1, self.num_steps + 1),
            self.batch_shape + self.event_shape,
        )

    def tree_flatten(self):
        return (self.scale,), self.num_steps

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        return cls(*params, num_steps=aux_data)


class HalfCauchy(Distribution):
    reparametrized_params = ["scale"]
    support = constraints.positive
    arg_constraints = {"scale": constraints.positive}

    def __init__(self, scale=1.0, validate_args=None):
        self._cauchy = Cauchy(0.0, scale)
        self.scale = scale
        super(HalfCauchy, self).__init__(
            batch_shape=jnp.shape(scale), validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return jnp.abs(self._cauchy.sample(key, sample_shape))

    @validate_sample
    def log_prob(self, value):
        return self._cauchy.log_prob(value) + jnp.log(2)

    def cdf(self, value):
        return self._cauchy.cdf(value) * 2 - 1

    def icdf(self, q):
        return self._cauchy.icdf((q + 1) / 2)

    @property
    def mean(self):
        return jnp.full(self.batch_shape, jnp.inf)

    @property
    def variance(self):
        return jnp.full(self.batch_shape, jnp.inf)


class HalfNormal(Distribution):
    reparametrized_params = ["scale"]
    support = constraints.positive
    arg_constraints = {"scale": constraints.positive}

    def __init__(self, scale=1.0, validate_args=None):
        self._normal = Normal(0.0, scale)
        self.scale = scale
        super(HalfNormal, self).__init__(
            batch_shape=jnp.shape(scale), validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return jnp.abs(self._normal.sample(key, sample_shape))

    @validate_sample
    def log_prob(self, value):
        return self._normal.log_prob(value) + jnp.log(2)

    def cdf(self, value):
        return self._normal.cdf(value) * 2 - 1

    def icdf(self, q):
        return self._normal.icdf((q + 1) / 2)

    @property
    def mean(self):
        return jnp.sqrt(2 / jnp.pi) * self.scale

    @property
    def variance(self):
        return (1 - 2 / jnp.pi) * self.scale ** 2


class InverseGamma(TransformedDistribution):
    """
    .. note:: We keep the same notation `rate` as in Pyro but
        it plays the role of scale parameter of InverseGamma in literatures
        (e.g. wikipedia: https://en.wikipedia.org/wiki/Inverse-gamma_distribution)
    """

    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
    }
    reparametrized_params = ["concentration", "rate"]
    support = constraints.positive

    def __init__(self, concentration, rate=1.0, validate_args=None):
        base_dist = Gamma(concentration, rate)
        self.concentration = base_dist.concentration
        self.rate = base_dist.rate
        super(InverseGamma, self).__init__(
            base_dist, PowerTransform(-1.0), validate_args=validate_args
        )

    @property
    def mean(self):
        # mean is inf for alpha <= 1
        a = self.rate / (self.concentration - 1)
        return jnp.where(self.concentration <= 1, jnp.inf, a)

    @property
    def variance(self):
        # var is inf for alpha <= 2
        a = (self.rate / (self.concentration - 1)) ** 2 / (self.concentration - 2)
        return jnp.where(self.concentration <= 2, jnp.inf, a)

    def tree_flatten(self):
        return super(TransformedDistribution, self).tree_flatten()


class Gumbel(Distribution):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    reparametrized_params = ["loc", "scale"]

    def __init__(self, loc=0.0, scale=1.0, validate_args=None):
        self.loc, self.scale = promote_shapes(loc, scale)
        batch_shape = lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))

        super(Gumbel, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        standard_gumbel_sample = random.gumbel(
            key, shape=sample_shape + self.batch_shape + self.event_shape
        )
        return self.loc + self.scale * standard_gumbel_sample

    @validate_sample
    def log_prob(self, value):
        z = (value - self.loc) / self.scale
        return -(z + jnp.exp(-z)) - jnp.log(self.scale)

    @property
    def mean(self):
        return jnp.broadcast_to(
            self.loc + self.scale * EULER_MASCHERONI, self.batch_shape
        )

    @property
    def variance(self):
        return jnp.broadcast_to(jnp.pi ** 2 / 6.0 * self.scale ** 2, self.batch_shape)

    def cdf(self, value):
        return jnp.exp(-jnp.exp((self.loc - value) / self.scale))

    def icdf(self, q):
        return self.loc - self.scale * jnp.log(-jnp.log(q))


class Laplace(Distribution):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    reparametrized_params = ["loc", "scale"]

    def __init__(self, loc=0.0, scale=1.0, validate_args=None):
        self.loc, self.scale = promote_shapes(loc, scale)
        batch_shape = lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        super(Laplace, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        eps = random.laplace(
            key, shape=sample_shape + self.batch_shape + self.event_shape
        )
        return self.loc + eps * self.scale

    @validate_sample
    def log_prob(self, value):
        normalize_term = jnp.log(2 * self.scale)
        value_scaled = jnp.abs(value - self.loc) / self.scale
        return -value_scaled - normalize_term

    @property
    def mean(self):
        return jnp.broadcast_to(self.loc, self.batch_shape)

    @property
    def variance(self):
        return jnp.broadcast_to(2 * self.scale ** 2, self.batch_shape)

    def cdf(self, value):
        scaled = (value - self.loc) / self.scale
        return 0.5 - 0.5 * jnp.sign(scaled) * jnp.expm1(-jnp.abs(scaled))

    def icdf(self, q):
        a = q - 0.5
        return self.loc - self.scale * jnp.sign(a) * jnp.log1p(-2 * jnp.abs(a))


class LKJ(TransformedDistribution):
    r"""
    LKJ distribution for correlation matrices. The distribution is controlled by ``concentration``
    parameter :math:`\eta` to make the probability of the correlation matrix :math:`M` propotional
    to :math:`\det(M)^{\eta - 1}`. Because of that, when ``concentration == 1``, we have a
    uniform distribution over correlation matrices.

    When ``concentration > 1``, the distribution favors samples with large large determinent. This
    is useful when we know a priori that the underlying variables are not correlated.

    When ``concentration < 1``, the distribution favors samples with small determinent. This is
    useful when we know a priori that some underlying variables are correlated.

    Sample code for using LKJ in the context of multivariate normal sample::

        def model(y):  # y has dimension N x d
            d = y.shape[1]
            N = y.shape[0]
            # Vector of variances for each of the d variables
            theta = numpyro.sample("theta", dist.HalfCauchy(jnp.ones(d)))

            concentration = jnp.ones(1)  # Implies a uniform distribution over correlation matrices
            corr_mat = numpyro.sample("corr_mat", dist.LKJ(d, concentration))
            sigma = jnp.sqrt(theta)
            # we can also use a faster formula `cov_mat = jnp.outer(theta, theta) * corr_mat`
            cov_mat = jnp.matmul(jnp.matmul(jnp.diag(sigma), corr_mat), jnp.diag(sigma))

            # Vector of expectations
            mu = jnp.zeros(d)

            with numpyro.plate("observations", N):
                obs = numpyro.sample("obs", dist.MultivariateNormal(mu, covariance_matrix=cov_mat), obs=y)
            return obs

    :param int dimension: dimension of the matrices
    :param ndarray concentration: concentration/shape parameter of the
        distribution (often referred to as eta)
    :param str sample_method: Either "cvine" or "onion". Both methods are proposed in [1] and
        offer the same distribution over correlation matrices. But they are different in how
        to generate samples. Defaults to "onion".

    **References**

    [1] `Generating random correlation matrices based on vines and extended onion method`,
    Daniel Lewandowski, Dorota Kurowicka, Harry Joe
    """
    arg_constraints = {"concentration": constraints.positive}
    reparametrized_params = ["concentration"]
    support = constraints.corr_matrix

    def __init__(
        self, dimension, concentration=1.0, sample_method="onion", validate_args=None
    ):
        base_dist = LKJCholesky(dimension, concentration, sample_method)
        self.dimension, self.concentration = (
            base_dist.dimension,
            base_dist.concentration,
        )
        self.sample_method = sample_method
        super(LKJ, self).__init__(
            base_dist, CorrMatrixCholeskyTransform().inv, validate_args=validate_args
        )

    @property
    def mean(self):
        return jnp.broadcast_to(
            jnp.identity(self.dimension),
            self.batch_shape + (self.dimension, self.dimension),
        )

    def tree_flatten(self):
        return (self.concentration,), (self.dimension, self.sample_method)

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        dimension, sample_method = aux_data
        return cls(dimension, *params, sample_method=sample_method)


class LKJCholesky(Distribution):
    r"""
    LKJ distribution for lower Cholesky factors of correlation matrices. The distribution is
    controlled by ``concentration`` parameter :math:`\eta` to make the probability of the
    correlation matrix :math:`M` generated from a Cholesky factor propotional to
    :math:`\det(M)^{\eta - 1}`. Because of that, when ``concentration == 1``, we have a
    uniform distribution over Cholesky factors of correlation matrices.

    When ``concentration > 1``, the distribution favors samples with large diagonal entries
    (hence large determinent). This is useful when we know a priori that the underlying
    variables are not correlated.

    When ``concentration < 1``, the distribution favors samples with small diagonal entries
    (hence small determinent). This is useful when we know a priori that some underlying
    variables are correlated.

    Sample code for using LKJCholesky in the context of multivariate normal sample::

        def model(y):  # y has dimension N x d
            d = y.shape[1]
            N = y.shape[0]
            # Vector of variances for each of the d variables
            theta = numpyro.sample("theta", dist.HalfCauchy(jnp.ones(d)))
            # Lower cholesky factor of a correlation matrix
            concentration = jnp.ones(1)  # Implies a uniform distribution over correlation matrices
            L_omega = numpyro.sample("L_omega", dist.LKJCholesky(d, concentration))
            # Lower cholesky factor of the covariance matrix
            sigma = jnp.sqrt(theta)
            # we can also use a faster formula `L_Omega = sigma[..., None] * L_omega`
            L_Omega = jnp.matmul(jnp.diag(sigma), L_omega)

            # Vector of expectations
            mu = jnp.zeros(d)

            with numpyro.plate("observations", N):
                obs = numpyro.sample("obs", dist.MultivariateNormal(mu, scale_tril=L_Omega), obs=y)
            return obs

    :param int dimension: dimension of the matrices
    :param ndarray concentration: concentration/shape parameter of the
        distribution (often referred to as eta)
    :param str sample_method: Either "cvine" or "onion". Both methods are proposed in [1] and
        offer the same distribution over correlation matrices. But they are different in how
        to generate samples. Defaults to "onion".

    **References**

    [1] `Generating random correlation matrices based on vines and extended onion method`,
    Daniel Lewandowski, Dorota Kurowicka, Harry Joe
    """
    arg_constraints = {"concentration": constraints.positive}
    reparametrized_params = ["concentration"]
    support = constraints.corr_cholesky

    def __init__(
        self, dimension, concentration=1.0, sample_method="onion", validate_args=None
    ):
        if dimension < 2:
            raise ValueError("Dimension must be greater than or equal to 2.")
        self.dimension = dimension
        self.concentration = concentration
        batch_shape = jnp.shape(concentration)
        event_shape = (dimension, dimension)

        # We construct base distributions to generate samples for each method.
        # The purpose of this base distribution is to generate a distribution for
        # correlation matrices which is propotional to `det(M)^{\eta - 1}`.
        # (note that this is not a unique way to define base distribution)
        # Both of the following methods have marginal distribution of each off-diagonal
        # element of sampled correlation matrices is Beta(eta + (D-2) / 2, eta + (D-2) / 2)
        # (up to a linear transform: x -> 2x - 1)
        Dm1 = self.dimension - 1
        marginal_concentration = concentration + 0.5 * (self.dimension - 2)
        offset = 0.5 * jnp.arange(Dm1)
        if sample_method == "onion":
            # The following construction follows from the algorithm in Section 3.2 of [1]:
            # NB: in [1], the method for case k > 1 can also work for the case k = 1.
            beta_concentration0 = (
                jnp.expand_dims(marginal_concentration, axis=-1) - offset
            )
            beta_concentration1 = offset + 0.5
            self._beta = Beta(beta_concentration1, beta_concentration0)
        elif sample_method == "cvine":
            # The following construction follows from the algorithm in Section 2.4 of [1]:
            # offset_tril is [0, 1, 1, 2, 2, 2,...] / 2
            offset_tril = matrix_to_tril_vec(jnp.broadcast_to(offset, (Dm1, Dm1)))
            beta_concentration = (
                jnp.expand_dims(marginal_concentration, axis=-1) - offset_tril
            )
            self._beta = Beta(beta_concentration, beta_concentration)
        else:
            raise ValueError("`method` should be one of 'cvine' or 'onion'.")
        self.sample_method = sample_method

        super(LKJCholesky, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def _cvine(self, key, size):
        # C-vine method first uses beta_dist to generate partial correlations,
        # then apply signed stick breaking to transform to cholesky factor.
        # Here is an attempt to prove that using signed stick breaking to
        # generate correlation matrices is the same as the C-vine method in [1]
        # for the entry r_32.
        #
        # With notations follow from [1], we define
        #   p: partial correlation matrix,
        #   c: cholesky factor,
        #   r: correlation matrix.
        # From recursive formula (2) in [1], we have
        #   r_32 = p_32 * sqrt{(1 - p_21^2)*(1 - p_31^2)} + p_21 * p_31 =: I
        # On the other hand, signed stick breaking process gives:
        #   l_21 = p_21, l_31 = p_31, l_22 = sqrt(1 - p_21^2), l_32 = p_32 * sqrt(1 - p_31^2)
        #   r_32 = l_21 * l_31 + l_22 * l_32
        #        = p_21 * p_31 + p_32 * sqrt{(1 - p_21^2)*(1 - p_31^2)} = I
        beta_sample = self._beta.sample(key, size)
        partial_correlation = 2 * beta_sample - 1  # scale to domain to (-1, 1)
        return signed_stick_breaking_tril(partial_correlation)

    def _onion(self, key, size):
        key_beta, key_normal = random.split(key)
        # Now we generate w term in Algorithm 3.2 of [1].
        beta_sample = self._beta.sample(key_beta, size)
        # The following Normal distribution is used to create a uniform distribution on
        # a hypershere (ref: http://mathworld.wolfram.com/HyperspherePointPicking.html)
        normal_sample = random.normal(
            key_normal,
            shape=size
            + self.batch_shape
            + (self.dimension * (self.dimension - 1) // 2,),
        )
        normal_sample = vec_to_tril_matrix(normal_sample, diagonal=0)
        u_hypershere = normal_sample / jnp.linalg.norm(
            normal_sample, axis=-1, keepdims=True
        )
        w = jnp.expand_dims(jnp.sqrt(beta_sample), axis=-1) * u_hypershere

        # put w into the off-diagonal triangular part
        cholesky = ops.index_add(
            jnp.zeros(size + self.batch_shape + self.event_shape),
            ops.index[..., 1:, :-1],
            w,
        )
        # correct the diagonal
        # NB: we clip due to numerical precision
        diag = jnp.sqrt(jnp.clip(1 - jnp.sum(cholesky ** 2, axis=-1), a_min=0.0))
        cholesky = cholesky + jnp.expand_dims(diag, axis=-1) * jnp.identity(
            self.dimension
        )
        return cholesky

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        if self.sample_method == "onion":
            return self._onion(key, sample_shape)
        else:
            return self._cvine(key, sample_shape)

    @validate_sample
    def log_prob(self, value):
        # Note about computing Jacobian of the transformation from Cholesky factor to
        # correlation matrix:
        #
        #   Assume C = L@Lt and L = (1 0 0; a \sqrt(1-a^2) 0; b c \sqrt(1-b^2-c^2)), we have
        #   Then off-diagonal lower triangular vector of L is transformed to the off-diagonal
        #   lower triangular vector of C by the transform:
        #       (a, b, c) -> (a, b, ab + c\sqrt(1-a^2))
        #   Hence, Jacobian = 1 * 1 * \sqrt(1 - a^2) = \sqrt(1 - a^2) = L22, where L22
        #       is the 2th diagonal element of L
        #   Generally, for a D dimensional matrix, we have:
        #       Jacobian = L22^(D-2) * L33^(D-3) * ... * Ldd^0
        #
        # From [1], we know that probability of a correlation matrix is propotional to
        #   determinant ** (concentration - 1) = prod(L_ii ^ 2(concentration - 1))
        # On the other hand, Jabobian of the transformation from Cholesky factor to
        # correlation matrix is:
        #   prod(L_ii ^ (D - i))
        # So the probability of a Cholesky factor is propotional to
        #   prod(L_ii ^ (2 * concentration - 2 + D - i)) =: prod(L_ii ^ order_i)
        # with order_i = 2 * concentration - 2 + D - i,
        # i = 2..D (we omit the element i = 1 because L_11 = 1)

        # Compute `order` vector (note that we need to reindex i -> i-2):
        one_to_D = jnp.arange(1, self.dimension)
        order_offset = (3 - self.dimension) + one_to_D
        order = 2 * jnp.expand_dims(self.concentration, axis=-1) - order_offset

        # Compute unnormalized log_prob:
        value_diag = value[..., one_to_D, one_to_D]
        unnormalized = jnp.sum(order * jnp.log(value_diag), axis=-1)

        # Compute normalization constant (on the first proof of page 1999 of [1])
        Dm1 = self.dimension - 1
        alpha = self.concentration + 0.5 * Dm1
        denominator = gammaln(alpha) * Dm1
        numerator = multigammaln(alpha - 0.5, Dm1)
        # pi_constant in [1] is D * (D - 1) / 4 * log(pi)
        # pi_constant in multigammaln is (D - 1) * (D - 2) / 4 * log(pi)
        # hence, we need to add a pi_constant = (D - 1) * log(pi) / 2
        pi_constant = 0.5 * Dm1 * jnp.log(jnp.pi)
        normalize_term = pi_constant + numerator - denominator
        return unnormalized - normalize_term

    def tree_flatten(self):
        return (self.concentration,), (self.dimension, self.sample_method)

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        dimension, sample_method = aux_data
        return cls(dimension, *params, sample_method=sample_method)


class LogNormal(TransformedDistribution):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.positive
    reparametrized_params = ["loc", "scale"]

    def __init__(self, loc=0.0, scale=1.0, validate_args=None):
        base_dist = Normal(loc, scale)
        self.loc, self.scale = base_dist.loc, base_dist.scale
        super(LogNormal, self).__init__(
            base_dist, ExpTransform(), validate_args=validate_args
        )

    @property
    def mean(self):
        return jnp.exp(self.loc + self.scale ** 2 / 2)

    @property
    def variance(self):
        return (jnp.exp(self.scale ** 2) - 1) * jnp.exp(2 * self.loc + self.scale ** 2)

    def tree_flatten(self):
        return super(TransformedDistribution, self).tree_flatten()


class Logistic(Distribution):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    reparametrized_params = ["loc", "scale"]

    def __init__(self, loc=0.0, scale=1.0, validate_args=None):
        self.loc, self.scale = promote_shapes(loc, scale)
        batch_shape = lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        super(Logistic, self).__init__(batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        z = random.logistic(
            key, shape=sample_shape + self.batch_shape + self.event_shape
        )
        return self.loc + z * self.scale

    @validate_sample
    def log_prob(self, value):
        log_exponent = (self.loc - value) / self.scale
        log_denominator = jnp.log(self.scale) + 2 * nn.softplus(log_exponent)
        return log_exponent - log_denominator

    @property
    def mean(self):
        return jnp.broadcast_to(self.loc, self.batch_shape)

    @property
    def variance(self):
        var = (self.scale ** 2) * (jnp.pi ** 2) / 3
        return jnp.broadcast_to(var, self.batch_shape)

    def cdf(self, value):
        scaled = (value - self.loc) / self.scale
        return expit(scaled)

    def icdf(self, q):
        return self.loc + self.scale * logit(q)


def _batch_mahalanobis(bL, bx):
    if bL.shape[:-1] == bx.shape:
        # no need to use the below optimization procedure
        solve_bL_bx = solve_triangular(bL, bx[..., None], lower=True).squeeze(-1)
        return jnp.sum(jnp.square(solve_bL_bx), -1)

    # NB: The following procedure handles the case: bL.shape = (i, 1, n, n), bx.shape = (i, j, n)
    # because we don't want to broadcast bL to the shape (i, j, n, n).

    # Assume that bL.shape = (i, 1, n, n), bx.shape = (..., i, j, n),
    # we are going to make bx have shape (..., 1, j,  i, 1, n) to apply batched tril_solve
    sample_ndim = bx.ndim - bL.ndim + 1  # size of sample_shape
    out_shape = jnp.shape(bx)[:-1]  # shape of output
    # Reshape bx with the shape (..., 1, i, j, 1, n)
    bx_new_shape = out_shape[:sample_ndim]
    for (sL, sx) in zip(bL.shape[:-2], out_shape[sample_ndim:]):
        bx_new_shape += (sx // sL, sL)
    bx_new_shape += (-1,)
    bx = jnp.reshape(bx, bx_new_shape)
    # Permute bx to make it have shape (..., 1, j, i, 1, n)
    permute_dims = (
        tuple(range(sample_ndim))
        + tuple(range(sample_ndim, bx.ndim - 1, 2))
        + tuple(range(sample_ndim + 1, bx.ndim - 1, 2))
        + (bx.ndim - 1,)
    )
    bx = jnp.transpose(bx, permute_dims)

    # reshape to (-1, i, 1, n)
    xt = jnp.reshape(bx, (-1,) + bL.shape[:-1])
    # permute to (i, 1, n, -1)
    xt = jnp.moveaxis(xt, 0, -1)
    solve_bL_bx = solve_triangular(bL, xt, lower=True)  # shape: (i, 1, n, -1)
    M = jnp.sum(solve_bL_bx ** 2, axis=-2)  # shape: (i, 1, -1)
    # permute back to (-1, i, 1)
    M = jnp.moveaxis(M, -1, 0)
    # reshape back to (..., 1, j, i, 1)
    M = jnp.reshape(M, bx.shape[:-1])
    # permute back to (..., 1, i, j, 1)
    permute_inv_dims = tuple(range(sample_ndim))
    for i in range(bL.ndim - 2):
        permute_inv_dims += (sample_ndim + i, len(out_shape) + i)
    M = jnp.transpose(M, permute_inv_dims)
    return jnp.reshape(M, out_shape)


class MultivariateNormal(Distribution):
    arg_constraints = {
        "loc": constraints.real_vector,
        "covariance_matrix": constraints.positive_definite,
        "precision_matrix": constraints.positive_definite,
        "scale_tril": constraints.lower_cholesky,
    }
    support = constraints.real_vector
    reparametrized_params = [
        "loc",
        "covariance_matrix",
        "precision_matrix",
        "scale_tril",
    ]

    def __init__(
        self,
        loc=0.0,
        covariance_matrix=None,
        precision_matrix=None,
        scale_tril=None,
        validate_args=None,
    ):
        if jnp.ndim(loc) == 0:
            (loc,) = promote_shapes(loc, shape=(1,))
        # temporary append a new axis to loc
        loc = loc[..., jnp.newaxis]
        if covariance_matrix is not None:
            loc, self.covariance_matrix = promote_shapes(loc, covariance_matrix)
            self.scale_tril = jnp.linalg.cholesky(self.covariance_matrix)
        elif precision_matrix is not None:
            loc, self.precision_matrix = promote_shapes(loc, precision_matrix)
            self.scale_tril = cholesky_of_inverse(self.precision_matrix)
        elif scale_tril is not None:
            loc, self.scale_tril = promote_shapes(loc, scale_tril)
        else:
            raise ValueError(
                "One of `covariance_matrix`, `precision_matrix`, `scale_tril`"
                " must be specified."
            )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(loc)[:-2], jnp.shape(self.scale_tril)[:-2]
        )
        event_shape = jnp.shape(self.scale_tril)[-1:]
        self.loc = loc[..., 0]
        super(MultivariateNormal, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        eps = random.normal(
            key, shape=sample_shape + self.batch_shape + self.event_shape
        )
        return self.loc + jnp.squeeze(
            jnp.matmul(self.scale_tril, eps[..., jnp.newaxis]), axis=-1
        )

    @validate_sample
    def log_prob(self, value):
        M = _batch_mahalanobis(self.scale_tril, value - self.loc)
        half_log_det = jnp.log(jnp.diagonal(self.scale_tril, axis1=-2, axis2=-1)).sum(
            -1
        )
        normalize_term = half_log_det + 0.5 * self.scale_tril.shape[-1] * jnp.log(
            2 * jnp.pi
        )
        return -0.5 * M - normalize_term

    @lazy_property
    def covariance_matrix(self):
        return jnp.matmul(self.scale_tril, jnp.swapaxes(self.scale_tril, -1, -2))

    @lazy_property
    def precision_matrix(self):
        identity = jnp.broadcast_to(
            jnp.eye(self.scale_tril.shape[-1]), self.scale_tril.shape
        )
        return cho_solve((self.scale_tril, True), identity)

    @property
    def mean(self):
        return jnp.broadcast_to(self.loc, self.shape())

    @property
    def variance(self):
        return jnp.broadcast_to(
            jnp.sum(self.scale_tril ** 2, axis=-1), self.batch_shape + self.event_shape
        )

    def tree_flatten(self):
        return (self.loc, self.scale_tril), None

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        loc, scale_tril = params
        return cls(loc, scale_tril=scale_tril)

    @staticmethod
    def infer_shapes(
        loc=(), covariance_matrix=None, precision_matrix=None, scale_tril=None
    ):
        batch_shape, event_shape = loc[:-1], loc[-1:]
        for matrix in [covariance_matrix, precision_matrix, scale_tril]:
            if matrix is not None:
                batch_shape = lax.broadcast_shapes(batch_shape, matrix[:-2])
                event_shape = lax.broadcast_shapes(event_shape, matrix[-1:])
        return batch_shape, event_shape


def _batch_mv(bmat, bvec):
    r"""
    Performs a batched matrix-vector product, with compatible but different batch shapes.
    This function takes as input `bmat`, containing :math:`n \times n` matrices, and
    `bvec`, containing length :math:`n` vectors.
    Both `bmat` and `bvec` may have any number of leading dimensions, which correspond
    to a batch shape. They are not necessarily assumed to have the same batch shape,
    just ones which can be broadcasted.
    """
    return jnp.squeeze(jnp.matmul(bmat, jnp.expand_dims(bvec, axis=-1)), axis=-1)


def _batch_capacitance_tril(W, D):
    r"""
    Computes Cholesky of :math:`I + W.T @ inv(D) @ W` for a batch of matrices :math:`W`
    and a batch of vectors :math:`D`.
    """
    Wt_Dinv = jnp.swapaxes(W, -1, -2) / jnp.expand_dims(D, -2)
    K = jnp.matmul(Wt_Dinv, W)
    # could be inefficient
    return jnp.linalg.cholesky(jnp.add(K, jnp.identity(K.shape[-1])))


def _batch_lowrank_logdet(W, D, capacitance_tril):
    r"""
    Uses "matrix determinant lemma"::
        log|W @ W.T + D| = log|C| + log|D|,
    where :math:`C` is the capacitance matrix :math:`I + W.T @ inv(D) @ W`, to compute
    the log determinant.
    """
    return 2 * jnp.sum(
        jnp.log(jnp.diagonal(capacitance_tril, axis1=-2, axis2=-1)), axis=-1
    ) + jnp.log(D).sum(-1)


def _batch_lowrank_mahalanobis(W, D, x, capacitance_tril):
    r"""
    Uses "Woodbury matrix identity"::
        inv(W @ W.T + D) = inv(D) - inv(D) @ W @ inv(C) @ W.T @ inv(D),
    where :math:`C` is the capacitance matrix :math:`I + W.T @ inv(D) @ W`, to compute the squared
    Mahalanobis distance :math:`x.T @ inv(W @ W.T + D) @ x`.
    """
    Wt_Dinv = jnp.swapaxes(W, -1, -2) / jnp.expand_dims(D, -2)
    Wt_Dinv_x = _batch_mv(Wt_Dinv, x)
    mahalanobis_term1 = jnp.sum(jnp.square(x) / D, axis=-1)
    mahalanobis_term2 = _batch_mahalanobis(capacitance_tril, Wt_Dinv_x)
    return mahalanobis_term1 - mahalanobis_term2


class LowRankMultivariateNormal(Distribution):
    arg_constraints = {
        "loc": constraints.real_vector,
        "cov_factor": constraints.independent(constraints.real, 2),
        "cov_diag": constraints.independent(constraints.positive, 1),
    }
    support = constraints.real_vector
    reparametrized_params = ["loc", "cov_factor", "cov_diag"]

    def __init__(self, loc, cov_factor, cov_diag, validate_args=None):
        if jnp.ndim(loc) < 1:
            raise ValueError("`loc` must be at least one-dimensional.")
        event_shape = jnp.shape(loc)[-1:]
        if jnp.ndim(cov_factor) < 2:
            raise ValueError(
                "`cov_factor` must be at least two-dimensional, "
                "with optional leading batch dimensions"
            )
        if jnp.shape(cov_factor)[-2:-1] != event_shape:
            raise ValueError(
                "`cov_factor` must be a batch of matrices with shape {} x m".format(
                    event_shape[0]
                )
            )
        if jnp.shape(cov_diag)[-1:] != event_shape:
            raise ValueError(
                "`cov_diag` must be a batch of vectors with shape {}".format(
                    self.event_shape
                )
            )

        loc, cov_factor, cov_diag = promote_shapes(
            loc[..., jnp.newaxis], cov_factor, cov_diag[..., jnp.newaxis]
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(loc), jnp.shape(cov_factor), jnp.shape(cov_diag)
        )[:-2]
        self.loc = loc[..., 0]
        self.cov_factor = cov_factor
        cov_diag = cov_diag[..., 0]
        self.cov_diag = cov_diag
        self._capacitance_tril = _batch_capacitance_tril(cov_factor, cov_diag)
        super(LowRankMultivariateNormal, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    @property
    def mean(self):
        return self.loc

    @lazy_property
    def variance(self):
        raw_variance = jnp.square(self.cov_factor).sum(-1) + self.cov_diag
        return jnp.broadcast_to(raw_variance, self.batch_shape + self.event_shape)

    @lazy_property
    def scale_tril(self):
        # The following identity is used to increase the numerically computation stability
        # for Cholesky decomposition (see http://www.gaussianprocess.org/gpml/, Section 3.4.3):
        #     W @ W.T + D = D1/2 @ (I + D-1/2 @ W @ W.T @ D-1/2) @ D1/2
        # The matrix "I + D-1/2 @ W @ W.T @ D-1/2" has eigenvalues bounded from below by 1,
        # hence it is well-conditioned and safe to take Cholesky decomposition.
        cov_diag_sqrt_unsqueeze = jnp.expand_dims(jnp.sqrt(self.cov_diag), axis=-1)
        Dinvsqrt_W = self.cov_factor / cov_diag_sqrt_unsqueeze
        K = jnp.matmul(Dinvsqrt_W, jnp.swapaxes(Dinvsqrt_W, -1, -2))
        K = jnp.add(K, jnp.identity(K.shape[-1]))
        scale_tril = cov_diag_sqrt_unsqueeze * jnp.linalg.cholesky(K)
        return scale_tril

    @lazy_property
    def covariance_matrix(self):
        # TODO: find a better solution to create a diagonal matrix
        new_diag = self.cov_diag[..., jnp.newaxis] * jnp.identity(self.loc.shape[-1])
        covariance_matrix = new_diag + jnp.matmul(
            self.cov_factor, jnp.swapaxes(self.cov_factor, -1, -2)
        )
        return covariance_matrix

    @lazy_property
    def precision_matrix(self):
        # We use "Woodbury matrix identity" to take advantage of low rank form::
        #     inv(W @ W.T + D) = inv(D) - inv(D) @ W @ inv(C) @ W.T @ inv(D)
        # where :math:`C` is the capacitance matrix.
        Wt_Dinv = jnp.swapaxes(self.cov_factor, -1, -2) / jnp.expand_dims(
            self.cov_diag, axis=-2
        )
        A = solve_triangular(Wt_Dinv, self._capacitance_tril, lower=True)
        # TODO: find a better solution to create a diagonal matrix
        inverse_cov_diag = jnp.reciprocal(self.cov_diag)
        diag_embed = inverse_cov_diag[..., jnp.newaxis] * jnp.identity(
            self.loc.shape[-1]
        )
        return diag_embed - jnp.matmul(jnp.swapaxes(A, -1, -2), A)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        key_W, key_D = random.split(key)
        batch_shape = sample_shape + self.batch_shape
        W_shape = batch_shape + self.cov_factor.shape[-1:]
        D_shape = batch_shape + self.cov_diag.shape[-1:]
        eps_W = random.normal(key_W, W_shape)
        eps_D = random.normal(key_D, D_shape)
        return (
            self.loc
            + _batch_mv(self.cov_factor, eps_W)
            + jnp.sqrt(self.cov_diag) * eps_D
        )

    @validate_sample
    def log_prob(self, value):
        diff = value - self.loc
        M = _batch_lowrank_mahalanobis(
            self.cov_factor, self.cov_diag, diff, self._capacitance_tril
        )
        log_det = _batch_lowrank_logdet(
            self.cov_factor, self.cov_diag, self._capacitance_tril
        )
        return -0.5 * (self.loc.shape[-1] * jnp.log(2 * jnp.pi) + log_det + M)

    def entropy(self):
        log_det = _batch_lowrank_logdet(
            self.cov_factor, self.cov_diag, self._capacitance_tril
        )
        H = 0.5 * (self.loc.shape[-1] * (1.0 + jnp.log(2 * jnp.pi)) + log_det)
        return jnp.broadcast_to(H, self.batch_shape)

    @staticmethod
    def infer_shapes(loc, cov_factor, cov_diag):
        event_shape = loc[-1:]
        batch_shape = lax.broadcast_shapes(loc[:-1], cov_factor[:-2], cov_diag[:-1])
        return batch_shape, event_shape


class Normal(Distribution):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    reparametrized_params = ["loc", "scale"]

    def __init__(self, loc=0.0, scale=1.0, validate_args=None):
        self.loc, self.scale = promote_shapes(loc, scale)
        batch_shape = lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        super(Normal, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        eps = random.normal(
            key, shape=sample_shape + self.batch_shape + self.event_shape
        )
        return self.loc + eps * self.scale

    @validate_sample
    def log_prob(self, value):
        normalize_term = jnp.log(jnp.sqrt(2 * jnp.pi) * self.scale)
        value_scaled = (value - self.loc) / self.scale
        return -0.5 * value_scaled ** 2 - normalize_term

    def cdf(self, value):
        scaled = (value - self.loc) / self.scale
        return ndtr(scaled)

    def icdf(self, q):
        return self.loc + self.scale * ndtri(q)

    @property
    def mean(self):
        return jnp.broadcast_to(self.loc, self.batch_shape)

    @property
    def variance(self):
        return jnp.broadcast_to(self.scale ** 2, self.batch_shape)


class Pareto(TransformedDistribution):
    arg_constraints = {"scale": constraints.positive, "alpha": constraints.positive}
    reparametrized_params = ["scale", "alpha"]

    def __init__(self, scale, alpha, validate_args=None):
        self.scale, self.alpha = promote_shapes(scale, alpha)
        batch_shape = lax.broadcast_shapes(jnp.shape(scale), jnp.shape(alpha))
        scale, alpha = jnp.broadcast_to(scale, batch_shape), jnp.broadcast_to(
            alpha, batch_shape
        )
        base_dist = Exponential(alpha)
        transforms = [ExpTransform(), AffineTransform(loc=0, scale=scale)]
        super(Pareto, self).__init__(base_dist, transforms, validate_args=validate_args)

    @property
    def mean(self):
        # mean is inf for alpha <= 1
        a = jnp.divide(self.alpha * self.scale, (self.alpha - 1))
        return jnp.where(self.alpha <= 1, jnp.inf, a)

    @property
    def variance(self):
        # var is inf for alpha <= 2
        a = jnp.divide(
            (self.scale ** 2) * self.alpha, (self.alpha - 1) ** 2 * (self.alpha - 2)
        )
        return jnp.where(self.alpha <= 2, jnp.inf, a)

    # override the default behaviour to save computations
    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.greater_than(self.scale)

    def cdf(self, value):
        return 1 - jnp.power(self.scale / value, self.alpha)

    def icdf(self, q):
        return self.scale / jnp.power(1 - q, 1 / self.alpha)

    def tree_flatten(self):
        return super(TransformedDistribution, self).tree_flatten()


class SoftLaplace(Distribution):
    """
    Smooth distribution with Laplace-like tail behavior.

    This distribution corresponds to the log-convex density::

        z = (value - loc) / scale
        log_prob = log(2 / pi) - log(scale) - logaddexp(z, -z)

    Like the Laplace density, this density has the heaviest possible tails
    (asymptotically) while still being log-convex. Unlike the Laplace
    distribution, this distribution is infinitely differentiable everywhere,
    and is thus suitable for HMC and Laplace approximation.

    :param loc: Location parameter.
    :param scale: Scale parameter.
    """

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    reparametrized_params = ["loc", "scale"]

    def __init__(self, loc, scale, *, validate_args=None):
        self.loc, self.scale = promote_shapes(loc, scale)
        batch_shape = lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @validate_sample
    def log_prob(self, value):
        z = (value - self.loc) / self.scale
        return jnp.log(2 / jnp.pi) - jnp.log(self.scale) - jnp.logaddexp(z, -z)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        u = random.uniform(
            key, shape=sample_shape + self.batch_shape + self.event_shape
        )
        return self.icdf(u)

    # TODO: refactor validate_sample to only does validation check and use it here
    def cdf(self, value):
        z = (value - self.loc) / self.scale
        return jnp.arctan(jnp.exp(z)) * (2 / jnp.pi)

    def icdf(self, value):
        return jnp.log(jnp.tan(value * (jnp.pi / 2))) * self.scale + self.loc

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return (jnp.pi / 2 * self.scale) ** 2


class StudentT(Distribution):
    arg_constraints = {
        "df": constraints.positive,
        "loc": constraints.real,
        "scale": constraints.positive,
    }
    support = constraints.real
    reparametrized_params = ["df", "loc", "scale"]

    def __init__(self, df, loc=0.0, scale=1.0, validate_args=None):
        batch_shape = lax.broadcast_shapes(
            jnp.shape(df), jnp.shape(loc), jnp.shape(scale)
        )
        self.df, self.loc, self.scale = promote_shapes(
            df, loc, scale, shape=batch_shape
        )
        df = jnp.broadcast_to(df, batch_shape)
        self._chi2 = Chi2(df)
        super(StudentT, self).__init__(batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        key_normal, key_chi2 = random.split(key)
        std_normal = random.normal(key_normal, shape=sample_shape + self.batch_shape)
        z = self._chi2.sample(key_chi2, sample_shape)
        y = std_normal * jnp.sqrt(self.df / z)
        return self.loc + self.scale * y

    @validate_sample
    def log_prob(self, value):
        y = (value - self.loc) / self.scale
        z = (
            jnp.log(self.scale)
            + 0.5 * jnp.log(self.df)
            + 0.5 * jnp.log(jnp.pi)
            + gammaln(0.5 * self.df)
            - gammaln(0.5 * (self.df + 1.0))
        )
        return -0.5 * (self.df + 1.0) * jnp.log1p(y ** 2.0 / self.df) - z

    @property
    def mean(self):
        # for df <= 1. should be jnp.nan (keeping jnp.inf for consistency with scipy)
        return jnp.broadcast_to(
            jnp.where(self.df <= 1, jnp.inf, self.loc), self.batch_shape
        )

    @property
    def variance(self):
        var = jnp.where(
            self.df > 2, jnp.divide(self.scale ** 2 * self.df, self.df - 2.0), jnp.inf
        )
        var = jnp.where(self.df <= 1, jnp.nan, var)
        return jnp.broadcast_to(var, self.batch_shape)

    def cdf(self, value):
        # Ref: https://en.wikipedia.org/wiki/Student's_t-distribution#Related_distributions
        # X^2 ~ F(1, df) -> df / (df + X^2) ~ Beta(df/2, 0.5)
        scaled = (value - self.loc) / self.scale
        scaled_squared = scaled * scaled
        beta_value = self.df / (self.df + scaled_squared)
        # when scaled < 0, returns 0.5 * Beta(df/2, 0.5).cdf(beta_value)
        # when scaled > 0, returns 1 - 0.5 * Beta(df/2, 0.5).cdf(beta_value)
        scaled_sign_half = 0.5 * jnp.sign(scaled)
        return (
            0.5
            + scaled_sign_half
            - 0.5 * jnp.sign(scaled) * betainc(0.5 * self.df, 0.5, beta_value)
        )

    def icdf(self, q):
        # scipy.special.betaincinv is not avaiable yet in JAX
        # upstream issue: https://github.com/google/jax/issues/2399
        raise NotImplementedError


class Uniform(Distribution):
    arg_constraints = {"low": constraints.dependent, "high": constraints.dependent}
    reparametrized_params = ["low", "high"]

    def __init__(self, low=0.0, high=1.0, validate_args=None):
        self.low, self.high = promote_shapes(low, high)
        batch_shape = lax.broadcast_shapes(jnp.shape(low), jnp.shape(high))
        self._support = constraints.interval(low, high)
        super().__init__(batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape
        return random.uniform(key, shape=shape, minval=self.low, maxval=self.high)

    @validate_sample
    def log_prob(self, value):
        shape = lax.broadcast_shapes(jnp.shape(value), self.batch_shape)
        return -jnp.broadcast_to(jnp.log(self.high - self.low), shape)

    def cdf(self, value):
        cdf = (value - self.low) / (self.high - self.low)
        return jnp.clip(cdf, a_min=0.0, a_max=1.0)

    def icdf(self, value):
        return self.low + value * (self.high - self.low)

    @property
    def mean(self):
        return self.low + (self.high - self.low) / 2.0

    @property
    def variance(self):
        return (self.high - self.low) ** 2 / 12.0

    def tree_flatten(self):
        if isinstance(self._support.lower_bound, (int, float)) and isinstance(
            self._support.upper_bound, (int, float)
        ):
            aux_data = (self._support.lower_bound, self._support.upper_bound)
        else:
            aux_data = None
        return (self.low, self.high), aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        d = cls(*params)
        if aux_data is not None:
            d._support = constraints.interval(*aux_data)
        return d

    @staticmethod
    def infer_shapes(low=(), high=()):
        batch_shape = lax.broadcast_shapes(low, high)
        event_shape = ()
        return batch_shape, event_shape


class Weibull(Distribution):
    arg_constraints = {
        "scale": constraints.positive,
        "concentration": constraints.positive,
    }
    support = constraints.positive
    reparametrized_params = ["scale", "concentration"]

    def __init__(self, scale, concentration, validate_args=None):
        self.concentration, self.scale = promote_shapes(concentration, scale)
        batch_shape = lax.broadcast_shapes(jnp.shape(concentration), jnp.shape(scale))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return random.weibull_min(
            key,
            scale=self.scale,
            concentration=self.concentration,
            shape=sample_shape + self.batch_shape,
        )

    @validate_sample
    def log_prob(self, value):
        ll = -jnp.power(value / self.scale, self.concentration)
        ll += jnp.log(self.concentration)
        ll += (self.concentration - 1.0) * jnp.log(value)
        ll -= self.concentration * jnp.log(self.scale)
        return ll

    def cdf(self, value):
        return 1 - jnp.exp(-((value / self.scale) ** self.concentration))

    @property
    def mean(self):
        return self.scale * jnp.exp(gammaln(1.0 + 1.0 / self.concentration))

    @property
    def variance(self):
        return self.scale ** 2 * (
            jnp.exp(gammaln(1.0 + 2.0 / self.concentration))
            - jnp.exp(gammaln(1.0 + 1.0 / self.concentration)) ** 2
        )


class BetaProportion(Beta):
    """
    The BetaProportion distribution is a reparameterization of the conventional
    Beta distribution in terms of a the variate mean and a
    precision parameter.

    **Reference:**
     `Beta regression for modelling rates and proportion`, Ferrari Silvia, and
      Francisco Cribari-Neto. Journal of Applied Statistics  31.7 (2004): 799-815.
    """

    arg_constraints = {
        "mean": constraints.unit_interval,
        "concentration": constraints.positive,
    }
    reparametrized_params = ["mean", "concentration"]
    support = constraints.unit_interval

    def __init__(self, mean, concentration, validate_args=None):
        self.concentration = jnp.broadcast_to(
            concentration, lax.broadcast_shapes(jnp.shape(concentration))
        )
        super().__init__(
            mean * concentration,
            (1.0 - mean) * concentration,
            validate_args=validate_args,
        )
