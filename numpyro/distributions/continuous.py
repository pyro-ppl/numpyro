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
import jax.numpy as np
import jax.random as random
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import gammaln, log_ndtr, ndtr, ndtri

from numpyro.distributions import constraints
from numpyro.distributions.constraints import AbsTransform, AffineTransform, ExpTransform, PowerTransform
from numpyro.distributions.distribution import Distribution, TransformedDistribution
from numpyro.distributions.util import (
    cholesky_inverse,
    cumsum,
    lazy_property,
    matrix_to_tril_vec,
    multigammaln,
    promote_shapes,
    signed_stick_breaking_tril,
    standard_gamma,
    vec_to_tril_matrix
)
from numpyro.util import copy_docs_from


@copy_docs_from(Distribution)
class Beta(Distribution):
    arg_constraints = {'concentration1': constraints.positive, 'concentration0': constraints.positive}
    support = constraints.unit_interval

    def __init__(self, concentration1, concentration0, validate_args=None):
        batch_shape = lax.broadcast_shapes(np.shape(concentration1), np.shape(concentration0))
        self.concentration1 = np.broadcast_to(concentration1, batch_shape)
        self.concentration0 = np.broadcast_to(concentration0, batch_shape)
        self._dirichlet = Dirichlet(np.stack([self.concentration1, self.concentration0],
                                             axis=-1))
        super(Beta, self).__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        return self._dirichlet.sample(key, sample_shape)[..., 0]

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self._dirichlet.log_prob(np.stack([value, 1. - value], -1))

    @property
    def mean(self):
        return self.concentration1 / (self.concentration1 + self.concentration0)

    @property
    def variance(self):
        total = self.concentration1 + self.concentration0
        return self.concentration1 * self.concentration0 / (total ** 2 * (total + 1))


@copy_docs_from(Distribution)
class Cauchy(Distribution):
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    reparametrized_params = ['loc', 'scale']

    def __init__(self, loc=0., scale=1., validate_args=None):
        self.loc, self.scale = promote_shapes(loc, scale)
        batch_shape = lax.broadcast_shapes(np.shape(loc), np.shape(scale))
        super(Cauchy, self).__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        eps = random.cauchy(key, shape=sample_shape + self.batch_shape)
        return self.loc + eps * self.scale

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return - np.log(np.pi) - np.log(self.scale) - np.log1p(((value - self.loc) / self.scale) ** 2)

    @property
    def mean(self):
        return np.full(self.batch_shape, np.nan)

    @property
    def variance(self):
        return np.full(self.batch_shape, np.nan)


@copy_docs_from(Distribution)
class Dirichlet(Distribution):
    arg_constraints = {'concentration': constraints.positive}
    support = constraints.simplex

    def __init__(self, concentration, validate_args=None):
        if np.ndim(concentration) < 1:
            raise ValueError("`concentration` parameter must be at least one-dimensional.")
        self.concentration = concentration
        batch_shape, event_shape = concentration.shape[:-1], concentration.shape[-1:]
        super(Dirichlet, self).__init__(batch_shape=batch_shape,
                                        event_shape=event_shape,
                                        validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        gamma_samples = standard_gamma(key, self.concentration, shape=shape)
        return gamma_samples / np.sum(gamma_samples, axis=-1, keepdims=True)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        normalize_term = (np.sum(gammaln(self.concentration), axis=-1) -
                          gammaln(np.sum(self.concentration, axis=-1)))
        return np.sum(np.log(value) * (self.concentration - 1.), axis=-1) - normalize_term

    @property
    def mean(self):
        return self.concentration / np.sum(self.concentration, axis=-1, keepdims=True)

    @property
    def variance(self):
        con0 = np.sum(self.concentration, axis=-1, keepdims=True)
        return self.concentration * (con0 - self.concentration) / (con0 ** 2 * (con0 + 1))


@copy_docs_from(Distribution)
class Exponential(Distribution):
    reparametrized_params = ['rate']
    arg_constraints = {'rate': constraints.positive}
    support = constraints.positive

    def __init__(self, rate=1., validate_args=None):
        self.rate = rate
        super(Exponential, self).__init__(batch_shape=np.shape(rate), validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        return random.exponential(key, shape=sample_shape + self.batch_shape) / self.rate

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return np.log(self.rate) - self.rate * value

    @property
    def mean(self):
        return np.reciprocal(self.rate)

    @property
    def variance(self):
        return np.reciprocal(self.rate ** 2)


@copy_docs_from(Distribution)
class Gamma(Distribution):
    arg_constraints = {'concentration': constraints.positive,
                       'rate': constraints.positive}
    support = constraints.positive
    reparametrized_params = ['rate']

    def __init__(self, concentration, rate=1., validate_args=None):
        self.concentration, self.rate = promote_shapes(concentration, rate)
        batch_shape = lax.broadcast_shapes(np.shape(concentration), np.shape(rate))
        super(Gamma, self).__init__(batch_shape=batch_shape,
                                    validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return standard_gamma(key, self.concentration, shape=shape) / self.rate

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        normalize_term = (gammaln(self.concentration) -
                          self.concentration * np.log(self.rate))
        return (self.concentration - 1) * np.log(value) - self.rate * value - normalize_term

    @property
    def mean(self):
        return self.concentration / self.rate

    @property
    def variance(self):
        return self.concentration / np.power(self.rate, 2)


@copy_docs_from(Distribution)
class Chi2(Gamma):
    arg_constraints = {'df': constraints.positive}

    def __init__(self, df, validate_args=None):
        self.df = df
        super(Chi2, self).__init__(0.5 * df, 0.5, validate_args=validate_args)


@copy_docs_from(Distribution)
class GaussianRandomWalk(Distribution):
    arg_constraints = {'num_steps': constraints.positive_integer, 'scale': constraints.positive}
    support = constraints.real
    reparametrized_params = ['scale']

    def __init__(self, scale=1., num_steps=1, validate_args=None):
        assert np.shape(num_steps) == ()
        self.scale = scale
        self.num_steps = num_steps
        batch_shape, event_shape = np.shape(scale), (num_steps,)
        super(GaussianRandomWalk, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        walks = random.normal(key, shape=shape)
        return cumsum(walks) * np.expand_dims(self.scale, axis=-1)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        init_prob = Normal(0., self.scale).log_prob(value[..., 0])
        scale = np.expand_dims(self.scale, -1)
        step_probs = Normal(value[..., :-1], scale).log_prob(value[..., 1:])
        return init_prob + np.sum(step_probs, axis=-1)

    @property
    def mean(self):
        return np.zeros(self.batch_shape + self.event_shape)

    @property
    def variance(self):
        return np.broadcast_to(np.expand_dims(self.scale, -1) ** 2 * np.arange(1, self.num_steps + 1),
                               self.batch_shape + self.event_shape)


@copy_docs_from(Distribution)
class HalfCauchy(TransformedDistribution):
    reparametrized_params = ['scale']
    arg_constraints = {'scale': constraints.positive}

    def __init__(self, scale=1., validate_args=None):
        base_dist = Cauchy(0., scale)
        self.scale = scale
        super(HalfCauchy, self).__init__(base_dist, AbsTransform(),
                                         validate_args=validate_args)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.base_dist.log_prob(value) + np.log(2)

    @property
    def mean(self):
        return np.full(self.batch_shape, np.inf)

    @property
    def variance(self):
        return np.full(self.batch_shape, np.inf)


@copy_docs_from(Distribution)
class HalfNormal(TransformedDistribution):
    reparametrized_params = ['scale']
    arg_constraints = {'scale': constraints.positive}

    def __init__(self, scale=1., validate_args=None):
        base_dist = Normal(0., scale)
        self.scale = scale
        super(HalfNormal, self).__init__(base_dist, AbsTransform(),
                                         validate_args=validate_args)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.base_dist.log_prob(value) + np.log(2)

    @property
    def mean(self):
        return np.sqrt(2 / np.pi) * self.scale

    @property
    def variance(self):
        return (1 - 2 / np.pi) * self.scale ** 2


@copy_docs_from(Distribution)
class InverseGamma(TransformedDistribution):
    arg_constraints = {'concentration': constraints.positive, 'rate': constraints.positive}
    support = constraints.positive
    reparametrized_params = ['rate']

    def __init__(self, concentration, rate=1., validate_args=None):
        # NB: we keep the same notation `rate` as in Pyro and tensorflow but
        # it plays the role of scale parameter of InverseGamma in literatures
        # (e.g. wikipedia: https://en.wikipedia.org/wiki/Inverse-gamma_distribution)
        base_dist = Gamma(concentration, rate)
        self.concentration = concentration
        self.rate = rate
        super(InverseGamma, self).__init__(base_dist, PowerTransform(-1.0),
                                           validate_args=validate_args)

    @property
    def mean(self):
        # mean is inf for alpha <= 1
        a = self.rate / (self.concentration - 1)
        return np.where(self.concentration <= 1, np.inf, a)

    @property
    def variance(self):
        # var is inf for alpha <= 2
        a = (self.rate / (self.concentration - 1)) ** 2 / (self.concentration - 2)
        return np.where(self.concentration <= 2, np.inf, a)


@copy_docs_from(Distribution)
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
    arg_constraints = {'concentration': constraints.positive}
    support = constraints.corr_cholesky

    def __init__(self, dimension, concentration=1., sample_method='onion', validate_args=None):
        if dimension < 2:
            raise ValueError("Dimension must be greater than or equal to 2.")
        self.dimension = dimension
        self.concentration = concentration
        batch_shape = np.shape(concentration)
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
        offset = 0.5 * np.arange(Dm1)
        if sample_method == 'onion':
            # The following construction follows from the algorithm in Section 3.2 of [1]:
            # NB: in [1], the method for case k > 1 can also work for the case k = 1.
            beta_concentration0 = np.expand_dims(marginal_concentration, axis=-1) - offset
            beta_concentration1 = offset + 0.5
            self._beta = Beta(beta_concentration1, beta_concentration0)
        elif sample_method == 'cvine':
            # The following construction follows from the algorithm in Section 2.4 of [1]:
            # offset_tril is [0, 1, 1, 2, 2, 2,...] / 2
            offset_tril = matrix_to_tril_vec(np.broadcast_to(offset, (Dm1, Dm1)))
            beta_concentration = np.expand_dims(marginal_concentration, axis=-1) - offset_tril
            self._beta = Beta(beta_concentration, beta_concentration)
        else:
            raise ValueError("`method` should be one of 'cvine' or 'onion'.")
        self.sample_method = sample_method

        super(LKJCholesky, self).__init__(batch_shape=batch_shape,
                                          event_shape=event_shape,
                                          validate_args=validate_args)

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
            shape=size + self.batch_shape + (self.dimension * (self.dimension - 1) // 2,)
        )
        normal_sample = vec_to_tril_matrix(normal_sample, diagonal=0)
        u_hypershere = normal_sample / np.linalg.norm(normal_sample, axis=-1, keepdims=True)
        w = np.expand_dims(np.sqrt(beta_sample), axis=-1) * u_hypershere

        # put w into the off-diagonal triangular part
        cholesky = ops.index_add(np.zeros(size + self.batch_shape + self.event_shape),
                                 ops.index[..., 1:, :-1], w)
        # correct the diagonal
        # NB: we clip due to numerical precision
        diag = np.sqrt(np.clip(1 - np.sum(cholesky ** 2, axis=-1), a_min=0.))
        cholesky = cholesky + np.expand_dims(diag, axis=-1) * np.identity(self.dimension)
        return cholesky

    def sample(self, key, sample_shape=()):
        if self.sample_method == "onion":
            return self._onion(key, sample_shape)
        else:
            return self._cvine(key, sample_shape)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
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
        one_to_D = np.arange(1, self.dimension)
        order_offset = (3 - self.dimension) + one_to_D
        order = 2 * np.expand_dims(self.concentration, axis=-1) - order_offset

        # Compute unnormalized log_prob:
        value_diag = value[..., one_to_D, one_to_D]
        unnormalized = np.sum(order * np.log(value_diag), axis=-1)

        # Compute normalization constant (on the first proof of page 1999 of [1])
        Dm1 = self.dimension - 1
        alpha = self.concentration + 0.5 * Dm1
        denominator = gammaln(alpha) * Dm1
        numerator = multigammaln(alpha - 0.5, Dm1)
        # pi_constant in [1] is D * (D - 1) / 4 * log(pi)
        # pi_constant in multigammaln is (D - 1) * (D - 2) / 4 * log(pi)
        # hence, we need to add a pi_constant = (D - 1) * log(pi) / 2
        pi_constant = 0.5 * Dm1 * np.log(np.pi)
        normalize_term = pi_constant + numerator - denominator
        return unnormalized - normalize_term


@copy_docs_from(Distribution)
class LogNormal(TransformedDistribution):
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    reparametrized_params = ['loc', 'scale']

    def __init__(self, loc=0., scale=1., validate_args=None):
        base_dist = Normal(loc, scale)
        self.loc, self.scale = base_dist.loc, base_dist.scale
        super(LogNormal, self).__init__(base_dist, ExpTransform(), validate_args=validate_args)

    @property
    def mean(self):
        return np.exp(self.loc + self.scale ** 2 / 2)

    @property
    def variance(self):
        return (np.exp(self.scale ** 2) - 1) * np.exp(2 * self.loc + self.scale ** 2)


def _batch_mahalanobis(bL, bx):
    # NB: The following procedure handles the case: bL.shape = (i, 1, n, n), bx.shape = (i, j, n)
    # because we don't want to broadcast bL to the shape (i, j, n).

    # Assume that bL.shape = (i, 1, n, n), bx.shape = (..., i, j, n),
    # we are going to make bx have shape (..., 1, j,  i, 1, n) to apply batched tril_solve
    sample_ndim = bx.ndim - bL.ndim + 1  # size of sample_shape
    out_shape = np.shape(bx)[:-1]  # shape of output
    # Reshape bx with the shape (..., 1, i, j, 1, n)
    bx_new_shape = out_shape[:sample_ndim]
    for (sL, sx) in zip(bL.shape[:-2], out_shape[sample_ndim:]):
        bx_new_shape += (sx // sL, sL)
    bx_new_shape += (-1,)
    bx = np.reshape(bx, bx_new_shape)
    # Permute bx to make it have shape (..., 1, j, i, 1, n)
    permute_dims = (tuple(range(sample_ndim))
                    + tuple(range(sample_ndim, bx.ndim - 1, 2))
                    + tuple(range(sample_ndim + 1, bx.ndim - 1, 2))
                    + (bx.ndim - 1,))
    bx = np.transpose(bx, permute_dims)

    # reshape to (-1, i, 1, n)
    xt = np.reshape(bx, (-1,) + bL.shape[:-1])
    # permute to (i, 1, n, -1)
    xt = np.moveaxis(xt, 0, -1)
    solve_bL_bx = solve_triangular(bL, xt, lower=True)  # shape: (i, 1, n, -1)
    M = np.sum(solve_bL_bx ** 2, axis=-2)  # shape: (i, 1, -1)
    # permute back to (-1, i, 1)
    M = np.moveaxis(M, -1, 0)
    # reshape back to (..., 1, j, i, 1)
    M = np.reshape(M, bx.shape[:-1])
    # permute back to (..., 1, i, j, 1)
    permute_inv_dims = tuple(range(sample_ndim))
    for i in range(bL.ndim - 2):
        permute_inv_dims += (sample_ndim + i, len(out_shape) + i)
    M = np.transpose(M, permute_inv_dims)
    return np.reshape(M, out_shape)


@copy_docs_from(Distribution)
class MultivariateNormal(Distribution):
    arg_constraints = {'loc': constraints.real,
                       'covariance_matrix': constraints.positive_definite,
                       'precision_matrix': constraints.positive_definite,
                       'scale_tril': constraints.lower_cholesky}
    support = constraints.real
    reparametrized_params = ['loc', 'covariance_matrix', 'precision_matrix', 'scale_tril']

    def __init__(self, loc=0., covariance_matrix=None, precision_matrix=None, scale_tril=None,
                 validate_args=None):
        if np.isscalar(loc):
            loc = np.expand_dims(loc, axis=-1)
        # temporary append a new axis to loc
        loc = loc[..., np.newaxis]
        if covariance_matrix is not None:
            loc, self.covariance_matrix = promote_shapes(loc, covariance_matrix)
            self.scale_tril = np.linalg.cholesky(self.covariance_matrix)
        elif precision_matrix is not None:
            loc, self.precision_matrix = promote_shapes(loc, precision_matrix)
            self.scale_tril = cholesky_inverse(self.precision_matrix)
        elif scale_tril is not None:
            loc, self.scale_tril = promote_shapes(loc, scale_tril)
        else:
            raise ValueError('One of `covariance_matrix`, `precision_matrix`, `scale_tril`'
                             ' must be specified.')
        batch_shape = lax.broadcast_shapes(np.shape(loc)[:-2], np.shape(self.scale_tril)[:-2])
        event_shape = np.shape(self.scale_tril)[-1:]
        self.loc = np.broadcast_to(np.squeeze(loc, axis=-1), batch_shape + event_shape)
        super(MultivariateNormal, self).__init__(batch_shape=batch_shape,
                                                 event_shape=event_shape,
                                                 validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        eps = random.normal(key, shape=sample_shape + self.batch_shape + self.event_shape)
        return self.loc + np.squeeze(np.matmul(self.scale_tril, eps[..., np.newaxis]), axis=-1)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        M = _batch_mahalanobis(self.scale_tril, value - self.loc)
        half_log_det = np.log(np.diagonal(self.scale_tril, axis1=-2, axis2=-1)).sum(-1)
        normalize_term = half_log_det + 0.5 * self.scale_tril.shape[-1] * np.log(2 * np.pi)
        return - 0.5 * M - normalize_term

    @lazy_property
    def covariance_matrix(self):
        return np.dot(self.scale_tril, self.scale_tril.T)

    @lazy_property
    def precision_matrix(self):
        scale_tril_inv = np.linalg.inv(self.scale_tril)
        return np.dot(scale_tril_inv.T, scale_tril_inv)

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return np.broadcast_to(np.sum(self.scale_tril ** 2, axis=-1),
                               self.batch_shape + self.event_shape)


@copy_docs_from(Distribution)
class Normal(Distribution):
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    reparametrized_params = ['loc', 'scale']

    def __init__(self, loc=0., scale=1., validate_args=None):
        self.loc, self.scale = promote_shapes(loc, scale)
        batch_shape = lax.broadcast_shapes(np.shape(loc), np.shape(scale))
        super(Normal, self).__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        eps = random.normal(key, shape=sample_shape + self.batch_shape)
        return self.loc + eps * self.scale

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        normalize_term = np.log(np.sqrt(2 * np.pi) * self.scale)
        value_scaled = (value - self.loc) / self.scale
        return -0.5 * value_scaled ** 2 - normalize_term

    @property
    def mean(self):
        return np.broadcast_to(self.loc, self.batch_shape)

    @property
    def variance(self):
        return np.broadcast_to(self.scale ** 2, self.batch_shape)


@copy_docs_from(Distribution)
class Pareto(TransformedDistribution):
    arg_constraints = {'alpha': constraints.positive, 'scale': constraints.positive}

    def __init__(self, alpha, scale=1., validate_args=None):
        batch_shape = lax.broadcast_shapes(np.shape(scale), np.shape(alpha))
        self.scale, self.alpha = np.broadcast_to(scale, batch_shape), np.broadcast_to(alpha, batch_shape)
        base_dist = Exponential(self.alpha)
        transforms = [ExpTransform(), AffineTransform(loc=0, scale=self.scale)]
        super(Pareto, self).__init__(base_dist, transforms, validate_args=validate_args)

    @property
    def mean(self):
        # mean is inf for alpha <= 1
        a = lax.div(self.alpha * self.scale, (self.alpha - 1))
        return np.where(self.alpha <= 1, np.inf, a)

    @property
    def variance(self):
        # var is inf for alpha <= 2
        a = lax.div((self.scale ** 2) * self.alpha, (self.alpha - 1) ** 2 * (self.alpha - 2))
        return np.where(self.alpha <= 2, np.inf, a)

    # override the default behaviour to save computations
    @property
    def support(self):
        return constraints.greater_than(self.scale)


@copy_docs_from(Distribution)
class StudentT(Distribution):
    arg_constraints = {'df': constraints.positive, 'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    reparametrized_params = ['loc', 'scale']

    def __init__(self, df, loc=0., scale=1., validate_args=None):
        batch_shape = lax.broadcast_shapes(np.shape(df), np.shape(loc), np.shape(scale))
        self.df = np.broadcast_to(df, batch_shape)
        self.loc, self.scale = promote_shapes(loc, scale, shape=batch_shape)
        self._chi2 = Chi2(self.df)
        super(StudentT, self).__init__(batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        key_normal, key_chi2 = random.split(key)
        std_normal = random.normal(key_normal, shape=sample_shape + self.batch_shape)
        z = self._chi2.sample(key_chi2, sample_shape)
        y = std_normal * np.sqrt(self.df / z)
        return self.loc + self.scale * y

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        y = (value - self.loc) / self.scale
        z = (np.log(self.scale) + 0.5 * np.log(self.df) + 0.5 * np.log(np.pi) +
             gammaln(0.5 * self.df) - gammaln(0.5 * (self.df + 1.)))
        return -0.5 * (self.df + 1.) * np.log1p(y ** 2. / self.df) - z

    @property
    def mean(self):
        # for df <= 1. should be np.nan (keeping np.inf for consistency with scipy)
        return np.broadcast_to(np.where(self.df <= 1, np.inf, self.loc), self.batch_shape)

    @property
    def variance(self):
        var = np.where(self.df > 2, self.scale ** 2 * self.df / (self.df - 2.0), np.inf)
        var = np.where(self.df <= 1, np.nan, var)
        return np.broadcast_to(var, self.batch_shape)


@copy_docs_from(Distribution)
class TruncatedCauchy(Distribution):
    arg_constraints = {'low': constraints.real, 'loc': constraints.real,
                       'scale': constraints.positive}
    reparametrized_params = ['low', 'loc', 'scale']

    def __init__(self, low=0., loc=0., scale=1., validate_args=None):
        self.low, self.loc, self.scale = promote_shapes(low, loc, scale)
        batch_shape = lax.broadcast_shapes(np.shape(low), np.shape(loc), np.shape(scale))
        super(TruncatedCauchy, self).__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        # We use inverse transform method:
        # z ~ inv_cdf(U), where U ~ Uniform(cdf(low), cdf(high)).
        #                         ~ Uniform(arctan(low), arctan(high)) / pi + 1/2
        size = sample_shape + self.batch_shape
        low = (self.low - self.loc) / self.scale
        minval = np.arctan(low)
        maxval = np.pi / 2
        u = minval + random.uniform(key, shape=size) * (maxval - minval)
        return self.loc + np.tan(u) * self.scale

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        low = (self.low - self.loc) / self.scale
        # pi / 2 is arctan of self.high when that arg is supported
        normalize_term = np.log(np.pi / 2 - np.arctan(low)) + np.log(self.scale)
        return - np.log1p(((value - self.loc) / self.scale) ** 2) - normalize_term

    # NB: these stats do not apply when arg `high` is supported
    @property
    def mean(self):
        return np.full(self.batch_shape, np.nan)

    @property
    def variance(self):
        return np.full(self.batch_shape, np.nan)

    @property
    def support(self):
        return constraints.greater_than(self.low)


@copy_docs_from(Distribution)
class TruncatedNormal(Distribution):
    arg_constraints = {'low': constraints.real, 'loc': constraints.real,
                       'scale': constraints.positive}
    reparametrized_params = ['low', 'loc', 'scale']

    # TODO: support `high` arg
    def __init__(self, low=0., loc=0., scale=1., validate_args=None):
        self.low, self.loc, self.scale = promote_shapes(low, loc, scale)
        batch_shape = lax.broadcast_shapes(np.shape(low), np.shape(loc), np.shape(scale))
        self._normal = Normal(self.loc, self.scale)
        super(TruncatedNormal, self).__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        size = sample_shape + self.batch_shape
        # We use inverse transform method:
        # z ~ icdf(U), where U ~ Uniform(0, 1).
        u = random.uniform(key, shape=size)
        low = (self.low - self.loc) / self.scale
        # Ref: https://en.wikipedia.org/wiki/Truncated_normal_distribution#Simulating
        # icdf[cdf_a + u * (1 - cdf_a)] = icdf[1 - (1 - cdf_a)(1 - u)]
        #                                 = - icdf[(1 - cdf_a)(1 - u)]
        return self.loc - ndtri(ndtr(-low) * (1 - u)) * self.scale

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # log(cdf(high) - cdf(low)) = log(1 - cdf(low)) = log(cdf(-low))
        low = (self.low - self.loc) / self.scale
        return self._normal.log_prob(value) - log_ndtr(-low)

    @property
    def mean(self):
        low = (self.low - self.loc) / self.scale
        low_prob_scaled = np.exp(self._normal.log_prob(self.low)) * self.scale / ndtr(-low)
        return self.loc + low_prob_scaled * self.scale

    @property
    def variance(self):
        low = (self.low - self.loc) / self.scale
        low_prob_scaled = np.exp(self._normal.log_prob(self.low)) * self.scale / ndtr(-low)
        return self._normal.variance * (1 + low * low_prob_scaled - low_prob_scaled ** 2)

    @property
    def support(self):
        return constraints.greater_than(self.low)


@copy_docs_from(Distribution)
class Uniform(Distribution):
    arg_constraints = {'low': constraints.dependent, 'high': constraints.dependent}
    reparametrized_params = ['low', 'high']

    def __init__(self, low=0., high=1., validate_args=None):
        self.low, self.high = promote_shapes(low, high)
        batch_shape = lax.broadcast_shapes(np.shape(low), np.shape(high))
        super(Uniform, self).__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        size = sample_shape + self.batch_shape
        return self.low + random.uniform(key, shape=size) * (self.high - self.low)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return - np.broadcast_to(np.log(self.high - self.low), np.shape(value))

    @property
    def mean(self):
        return self.low + (self.high - self.low) / 2.

    @property
    def variance(self):
        return (self.high - self.low) ** 2 / 12.

    @property
    def support(self):
        return constraints.interval(self.low, self.high)
