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

import numpy as np

from jax import lax, vmap
from jax.experimental.sparse import BCOO
from jax.lax import scan
import jax.nn as nn
import jax.numpy as jnp
import jax.random as random
from jax.scipy.linalg import cho_solve, solve_triangular
from jax.scipy.special import (
    betaln,
    digamma,
    expi,
    expit,
    gammainc,
    gammaln,
    logit,
    multigammaln,
    ndtr,
    ndtri,
    xlog1py,
    xlogy,
)
from jax.scipy.stats import norm as jax_norm

from numpyro.distributions import constraints
from numpyro.distributions.discrete import _to_logits_bernoulli
from numpyro.distributions.distribution import Distribution, TransformedDistribution
from numpyro.distributions.transforms import (
    AffineTransform,
    CholeskyTransform,
    CorrMatrixCholeskyTransform,
    ExpTransform,
    PowerTransform,
    SigmoidTransform,
    ZeroSumTransform,
)
from numpyro.distributions.util import (
    add_diag,
    assert_one_of,
    betainc,
    betaincinv,
    cholesky_of_inverse,
    gammaincinv,
    lazy_property,
    matrix_to_tril_vec,
    multidigamma,
    promote_shapes,
    signed_stick_breaking_tril,
    tri_logabsdet,
    validate_sample,
    vec_to_tril_matrix,
)
from numpyro.util import is_prng_key


class AsymmetricLaplace(Distribution):
    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "asymmetry": constraints.positive,
    }
    reparametrized_params = ["loc", "scale", "asymmetry"]
    support = constraints.real

    def __init__(self, loc=0.0, scale=1.0, asymmetry=1.0, *, validate_args=None):
        batch_shape = lax.broadcast_shapes(
            jnp.shape(loc), jnp.shape(scale), jnp.shape(asymmetry)
        )
        self.loc, self.scale, self.asymmetry = promote_shapes(
            loc, scale, asymmetry, shape=batch_shape
        )
        super(AsymmetricLaplace, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    @lazy_property
    def left_scale(self):
        return self.scale * self.asymmetry

    @lazy_property
    def right_scale(self):
        return self.scale / self.asymmetry

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = value - self.loc
        z = -jnp.abs(z) / jnp.where(z < 0, self.left_scale, self.right_scale)
        return z - jnp.log(self.left_scale + self.right_scale)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        shape = (2,) + sample_shape + self.batch_shape + self.event_shape
        u, v = random.exponential(key, shape=shape)
        return self.loc - self.left_scale * u + self.right_scale * v

    @property
    def mean(self):
        total_scale = self.left_scale + self.right_scale
        mean = self.loc + (self.right_scale**2 - self.left_scale**2) / total_scale
        return jnp.broadcast_to(mean, self.batch_shape)

    @property
    def variance(self):
        left = self.left_scale
        right = self.right_scale
        total = left + right
        p = left / total
        q = right / total
        variance = p * left**2 + q * right**2 + p * q * total**2
        return jnp.broadcast_to(variance, self.batch_shape)

    def cdf(self, value):
        z = value - self.loc
        k = self.asymmetry
        return jnp.where(
            z >= 0,
            1 - (1 / (1 + k**2)) * jnp.exp(-jnp.abs(z) / self.right_scale),
            k**2 / (1 + k**2) * jnp.exp(-jnp.abs(z) / self.left_scale),
        )

    def icdf(self, value):
        k = self.asymmetry
        temp = k**2 / (1 + k**2)
        return jnp.where(
            value <= temp,
            self.loc + self.left_scale * jnp.log(value / temp),
            self.loc - self.right_scale * jnp.log((1 + k**2) * (1 - value)),
        )


class Beta(Distribution):
    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
    }
    reparametrized_params = ["concentration1", "concentration0"]
    support = constraints.unit_interval
    pytree_data_fields = ("concentration0", "concentration1", "_dirichlet")

    def __init__(self, concentration1, concentration0, *, validate_args=None):
        self.concentration1, self.concentration0 = promote_shapes(
            concentration1, concentration0
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(concentration1), jnp.shape(concentration0)
        )
        concentration1 = jnp.broadcast_to(concentration1, batch_shape)
        concentration0 = jnp.broadcast_to(concentration0, batch_shape)
        super(Beta, self).__init__(batch_shape=batch_shape, validate_args=validate_args)
        self._dirichlet = Dirichlet(
            jnp.stack([concentration1, concentration0], axis=-1)
        )

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
        return self.concentration1 * self.concentration0 / (total**2 * (total + 1))

    def cdf(self, value):
        return betainc(self.concentration1, self.concentration0, value)

    def icdf(self, q):
        return betaincinv(self.concentration1, self.concentration0, q)

    def entropy(self):
        total = self.concentration0 + self.concentration1
        return (
            betaln(self.concentration0, self.concentration1)
            - (self.concentration0 - 1) * digamma(self.concentration0)
            - (self.concentration1 - 1) * digamma(self.concentration1)
            + (total - 2) * digamma(total)
        )


class Cauchy(Distribution):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    reparametrized_params = ["loc", "scale"]

    def __init__(self, loc=0.0, scale=1.0, *, validate_args=None):
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

    def entropy(self):
        return jnp.broadcast_to(jnp.log(4 * np.pi * self.scale), self.batch_shape)


class Dirichlet(Distribution):
    arg_constraints = {
        "concentration": constraints.independent(constraints.positive, 1)
    }
    reparametrized_params = ["concentration"]
    support = constraints.simplex

    def __init__(self, concentration, *, validate_args=None):
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
        shape = sample_shape + self.batch_shape
        samples = random.dirichlet(key, self.concentration, shape=shape)
        return jnp.clip(samples, jnp.finfo(samples).tiny, 1 - jnp.finfo(samples).eps)

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
        return self.concentration * (con0 - self.concentration) / (con0**2 * (con0 + 1))

    @staticmethod
    def infer_shapes(concentration):
        batch_shape = concentration[:-1]
        event_shape = concentration[-1:]
        return batch_shape, event_shape

    def entropy(self):
        (n,) = self.event_shape
        total = self.concentration.sum(axis=-1)
        return (
            gammaln(self.concentration).sum(axis=-1)
            - gammaln(total)
            + (total - n) * digamma(total)
            - ((self.concentration - 1) * digamma(self.concentration)).sum(axis=-1)
        )


class EulerMaruyama(Distribution):
    """
    Euler–Maruyama method is a method for the approximate numerical solution
    of a stochastic differential equation (SDE)

    :param ndarray t: discretized time
    :param callable sde_fn: function returning the drift and diffusion coefficients of SDE
    :param Distribution init_dist: Distribution for initial values.

    **References**

    [1] https://en.wikipedia.org/wiki/Euler-Maruyama_method
    """

    arg_constraints = {"t": constraints.ordered_vector}
    pytree_data_fields = ("t", "init_dist")
    pytree_aux_fields = ("sde_fn",)

    def __init__(self, t, sde_fn, init_dist, *, validate_args=None):
        self.t = t
        self.sde_fn = sde_fn
        self.init_dist = init_dist

        if not isinstance(init_dist, Distribution):
            raise TypeError("Init distribution is expected to be Distribution class.")

        batch_shape_t = jnp.shape(t)[:-1]
        batch_shape = lax.broadcast_shapes(batch_shape_t, init_dist.batch_shape)
        event_shape = (jnp.shape(t)[-1],) + init_dist.event_shape

        super(EulerMaruyama, self).__init__(
            batch_shape, event_shape, validate_args=validate_args
        )

    @constraints.dependent_property(is_discrete=False)
    def support(self):
        return constraints.independent(constraints.real, self.event_dim)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        batch_shape = sample_shape + self.batch_shape

        def step(y_curr, xs):
            noise_curr, t_curr, dt_curr = xs
            f, g = self.sde_fn(y_curr, t_curr)
            mu = y_curr + dt_curr * f
            sigma = jnp.sqrt(dt_curr) * g
            y_next = mu + sigma * noise_curr
            return y_next, y_next

        rng_noise, rng_init = random.split(key)
        noises = random.normal(
            rng_noise,
            shape=batch_shape + (self.event_shape[0] - 1,) + self.event_shape[1:],
        )
        inits = self.init_dist.expand(batch_shape).sample(rng_init)

        def scan_fn(init, noise, tm1, dt):
            return scan(step, init, (noise, tm1, dt))

        batch_dim = len(batch_shape)
        if batch_dim:
            inits = inits.reshape((-1,) + inits.shape[batch_dim:])
            noises = noises.reshape((-1,) + noises.shape[batch_dim:])
            t = jnp.broadcast_to(self.t, batch_shape + (self.event_shape[0],))
            t = t.reshape((-1,) + t.shape[batch_dim:])
            dt = jnp.diff(t, axis=-1)
            _, sde_out = vmap(scan_fn)(inits, noises, t[..., :-1], dt)
            sde_out = jnp.concatenate([inits[:, None], sde_out], axis=1)
            sde_out = jnp.reshape(sde_out, batch_shape + self.event_shape)
        else:
            dt = jnp.diff(self.t, axis=-1)
            _, sde_out = scan_fn(inits, noises, self.t[:-1], dt)
            sde_out = jnp.concatenate([inits[None], sde_out], axis=0)

        return sde_out

    @validate_sample
    def log_prob(self, value):
        sample_shape = lax.broadcast_shapes(
            value.shape[: -self.event_dim], self.batch_shape
        )
        value = jnp.broadcast_to(value, sample_shape + self.event_shape)

        if sample_shape:
            reshaped_value = value.reshape((-1,) + self.event_shape)
            xtm1, xt = reshaped_value[:, :-1], reshaped_value[:, 1:]
            value0 = reshaped_value[:, 0]
            t = jnp.broadcast_to(self.t, sample_shape + (self.event_shape[0],))
            t = t.reshape((-1,) + (self.event_shape[0],))

            f, g = vmap(vmap(self.sde_fn))(xtm1, t[:, :-1])

            f = f.reshape(sample_shape + f.shape[1:])
            g = g.reshape(sample_shape + g.shape[1:])
            xtm1 = xtm1.reshape(sample_shape + xtm1.shape[1:])
            xt = xt.reshape(sample_shape + xt.shape[1:])
            value0 = value0.reshape(sample_shape + value0.shape[1:])

        else:
            xtm1, xt = value[:-1], value[1:]
            value0 = value[0]

            f, g = vmap(self.sde_fn)(xtm1, self.t[:-1])

        # add missing event dimensions
        batch_dim = len(sample_shape)
        f = f.reshape(
            f.shape[: batch_dim + 1]
            + (1,) * (xt.ndim - f.ndim)
            + f.shape[batch_dim + 1 :]
        )
        g = g.reshape(
            g.shape[: batch_dim + 1]
            + (1,) * (xt.ndim - g.ndim)
            + g.shape[batch_dim + 1 :]
        )

        dt = jnp.diff(self.t, axis=-1)
        dt = dt.reshape(dt.shape + (1,) * (self.event_dim - 1))
        mu = xtm1 + dt * f
        sigma = jnp.sqrt(dt) * g

        sde_log_prob = Normal(mu, sigma).to_event(self.event_dim).log_prob(xt)
        init_log_prob = self.init_dist.log_prob(value0)

        return sde_log_prob + init_log_prob


class Exponential(Distribution):
    reparametrized_params = ["rate"]
    arg_constraints = {"rate": constraints.positive}
    support = constraints.positive

    def __init__(self, rate=1.0, *, validate_args=None):
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
        return jnp.reciprocal(self.rate**2)

    def cdf(self, value):
        return -jnp.expm1(-self.rate * value)

    def icdf(self, q):
        return -jnp.log1p(-q) / self.rate

    def entropy(self):
        return 1 - jnp.log(self.rate)


class Gamma(Distribution):
    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
    }
    support = constraints.positive
    reparametrized_params = ["concentration", "rate"]

    def __init__(self, concentration, rate=1.0, *, validate_args=None):
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

    def cdf(self, x):
        return gammainc(self.concentration, self.rate * x)

    def icdf(self, q):
        return gammaincinv(self.concentration, q) / self.rate

    def entropy(self):
        return (
            self.concentration
            - jnp.log(self.rate)
            + gammaln(self.concentration)
            + (1 - self.concentration) * digamma(self.concentration)
        )


class Chi2(Gamma):
    arg_constraints = {"df": constraints.positive}
    reparametrized_params = ["df"]

    def __init__(self, df, *, validate_args=None):
        self.df = df
        super(Chi2, self).__init__(0.5 * df, 0.5, validate_args=validate_args)


class GaussianRandomWalk(Distribution):
    arg_constraints = {"scale": constraints.positive}
    support = constraints.real_vector
    reparametrized_params = ["scale"]
    pytree_aux_fields = ("num_steps",)

    def __init__(self, scale=1.0, num_steps=1, *, validate_args=None):
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


class HalfCauchy(Distribution):
    reparametrized_params = ["scale"]
    support = constraints.positive
    arg_constraints = {"scale": constraints.positive}
    pytree_data_fields = ("_cauchy", "scale")

    def __init__(self, scale=1.0, *, validate_args=None):
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
    pytree_data_fields = ("_normal", "scale")

    def __init__(self, scale=1.0, *, validate_args=None):
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
        return (1 - 2 / jnp.pi) * self.scale**2


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

    def __init__(self, concentration, rate=1.0, *, validate_args=None):
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

    def cdf(self, x):
        return 1 - self.base_dist.cdf(1 / x)

    def entropy(self):
        return (
            self.concentration
            + jnp.log(self.rate)
            + gammaln(self.concentration)
            - (1 + self.concentration) * digamma(self.concentration)
        )


class Gompertz(Distribution):
    r"""Gompertz Distribution.

    The Gompertz distribution is a distribution with support on the positive real line that is closely
    related to the Gumbel distribution. This implementation follows the notation used in the Wikipedia
    entry for the Gompertz distribution. See https://en.wikipedia.org/wiki/Gompertz_distribution.

    However, we call the parameter "eta" a concentration parameter and the parameter
    "b" a rate parameter (as opposed to scale parameter as in wikipedia description.)

    The CDF, in terms of `concentration` (`con`) and `rate`, is

    .. math::
        F(x) = 1 - \exp \left\{ - \text{con} * \left [ \exp\{x * rate \} - 1 \right ] \right\}
    """

    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
    }
    support = constraints.positive
    reparametrized_params = ["concentration", "rate"]

    def __init__(self, concentration, rate=1.0, *, validate_args=None):
        self.concentration, self.rate = promote_shapes(concentration, rate)
        super(Gompertz, self).__init__(
            batch_shape=lax.broadcast_shapes(jnp.shape(concentration), jnp.shape(rate)),
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        random_shape = sample_shape + self.batch_shape + self.event_shape
        unifs = random.uniform(key, shape=random_shape)
        return self.icdf(unifs)

    @validate_sample
    def log_prob(self, value):
        scaled_value = value * self.rate
        return (
            jnp.log(self.concentration)
            + jnp.log(self.rate)
            + scaled_value
            - self.concentration * jnp.expm1(scaled_value)
        )

    def cdf(self, value):
        return -jnp.expm1(-self.concentration * jnp.expm1(value * self.rate))

    def icdf(self, q):
        return jnp.log1p(-jnp.log1p(-q) / self.concentration) / self.rate

    @property
    def mean(self):
        return -jnp.exp(self.concentration) * expi(-self.concentration) / self.rate


class Gumbel(Distribution):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    reparametrized_params = ["loc", "scale"]

    def __init__(self, loc=0.0, scale=1.0, *, validate_args=None):
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
            self.loc + self.scale * jnp.euler_gamma, self.batch_shape
        )

    @property
    def variance(self):
        return jnp.broadcast_to(jnp.pi**2 / 6.0 * self.scale**2, self.batch_shape)

    def cdf(self, value):
        return jnp.exp(-jnp.exp((self.loc - value) / self.scale))

    def icdf(self, q):
        return self.loc - self.scale * jnp.log(-jnp.log(q))


class Kumaraswamy(Distribution):
    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
    }
    reparametrized_params = ["concentration1", "concentration0"]
    support = constraints.unit_interval
    # XXX: This flag is used to approximate the Taylor expansion
    # of KL(Kumaraswamy||Beta) following
    # https://arxiv.org/abs/1605.06197 Formula (12)
    # We follow the paper and set this to 10 but to get more precise KL,
    # we can set this flag to 1000.
    KL_KUMARASWAMY_BETA_TAYLOR_ORDER = 10

    def __init__(self, concentration1, concentration0, *, validate_args=None):
        self.concentration1, self.concentration0 = promote_shapes(
            concentration1, concentration0
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(concentration1), jnp.shape(concentration0)
        )
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        finfo = jnp.finfo(jnp.result_type(float))
        u = random.uniform(
            key, shape=sample_shape + self.batch_shape, minval=finfo.tiny
        )
        u_con0 = jnp.clip(u ** (1 / self.concentration0), None, 1 - finfo.eps)
        log_sample = jnp.log1p(-u_con0) / self.concentration1
        return jnp.clip(jnp.exp(log_sample), finfo.tiny, 1 - finfo.eps)

    @validate_sample
    def log_prob(self, value):
        finfo = jnp.finfo(jnp.result_type(float))
        normalize_term = jnp.log(self.concentration0) + jnp.log(self.concentration1)
        value_con1 = jnp.clip(value**self.concentration1, None, 1 - finfo.eps)
        return (
            xlogy(self.concentration1 - 1, value)
            + xlog1py(self.concentration0 - 1, -value_con1)
            + normalize_term
        )

    @property
    def mean(self):
        log_beta = betaln(1 + 1 / self.concentration1, self.concentration0)
        return self.concentration0 * jnp.exp(log_beta)

    @property
    def variance(self):
        log_beta = betaln(1 + 2 / self.concentration1, self.concentration0)
        return self.concentration0 * jnp.exp(log_beta) - jnp.square(self.mean)


class Laplace(Distribution):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    reparametrized_params = ["loc", "scale"]

    def __init__(self, loc=0.0, scale=1.0, *, validate_args=None):
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
        return jnp.broadcast_to(2 * self.scale**2, self.batch_shape)

    def cdf(self, value):
        scaled = (value - self.loc) / self.scale
        return 0.5 - 0.5 * jnp.sign(scaled) * jnp.expm1(-jnp.abs(scaled))

    def icdf(self, q):
        a = q - 0.5
        return self.loc - self.scale * jnp.sign(a) * jnp.log1p(-2 * jnp.abs(a))

    def entropy(self):
        return jnp.log(2 * self.scale) + 1


class LKJ(TransformedDistribution):
    r"""
    LKJ distribution for correlation matrices. The distribution is controlled by ``concentration``
    parameter :math:`\eta` to make the probability of the correlation matrix :math:`M` proportional
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
            # we can also use a faster formula `cov_mat = jnp.outer(sigma, sigma) * corr_mat`
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
    pytree_aux_fields = ("dimension", "sample_method")

    def __init__(
        self, dimension, concentration=1.0, sample_method="onion", *, validate_args=None
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
    pytree_data_fields = ("_beta", "concentration")
    pytree_aux_fields = ("dimension", "sample_method")

    def __init__(
        self, dimension, concentration=1.0, sample_method="onion", *, validate_args=None
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
        cholesky = jnp.zeros(size + self.batch_shape + self.event_shape)
        cholesky = cholesky.at[..., 1:, :-1].set(w)
        # correct the diagonal
        # NB: beta_sample = sum(w ** 2) because norm 2 of u is 1.
        diag = jnp.ones(cholesky.shape[:-1]).at[..., 1:].set(jnp.sqrt(1 - beta_sample))
        return add_diag(cholesky, diag)

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
        value_diag = jnp.asarray(value)[..., one_to_D, one_to_D]
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


class LogNormal(TransformedDistribution):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.positive
    reparametrized_params = ["loc", "scale"]

    def __init__(self, loc=0.0, scale=1.0, *, validate_args=None):
        base_dist = Normal(loc, scale)
        self.loc, self.scale = base_dist.loc, base_dist.scale
        super(LogNormal, self).__init__(
            base_dist, ExpTransform(), validate_args=validate_args
        )

    @property
    def mean(self):
        return jnp.exp(self.loc + self.scale**2 / 2)

    @property
    def variance(self):
        return (jnp.exp(self.scale**2) - 1) * jnp.exp(2 * self.loc + self.scale**2)

    def cdf(self, x):
        return self.base_dist.cdf(jnp.log(x))

    def entropy(self):
        return (1 + jnp.log(2 * jnp.pi)) / 2 + self.loc + jnp.log(self.scale)


class Logistic(Distribution):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    reparametrized_params = ["loc", "scale"]

    def __init__(self, loc=0.0, scale=1.0, *, validate_args=None):
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
        var = (self.scale**2) * (jnp.pi**2) / 3
        return jnp.broadcast_to(var, self.batch_shape)

    def cdf(self, value):
        scaled = (value - self.loc) / self.scale
        return expit(scaled)

    def icdf(self, q):
        return self.loc + self.scale * logit(q)

    def entropy(self):
        return jnp.broadcast_to(jnp.log(self.scale) + 2, self.batch_shape)


class LogUniform(TransformedDistribution):
    arg_constraints = {"low": constraints.positive, "high": constraints.positive}
    reparametrized_params = ["low", "high"]
    pytree_data_fields = ("low", "high", "_support")

    def __init__(self, low, high, *, validate_args=None):
        base_dist = Uniform(jnp.log(low), jnp.log(high))
        self.low, self.high = promote_shapes(low, high)
        self._support = constraints.interval(self.low, self.high)
        super(LogUniform, self).__init__(
            base_dist, ExpTransform(), validate_args=validate_args
        )

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    @property
    def mean(self):
        return (self.high - self.low) / jnp.log(self.high / self.low)

    @property
    def variance(self):
        return (
            0.5 * (self.high**2 - self.low**2) / jnp.log(self.high / self.low)
            - self.mean**2
        )

    def cdf(self, x):
        return self.base_dist.cdf(jnp.log(x))

    def entropy(self):
        log_low = jnp.log(self.low)
        log_high = jnp.log(self.high)
        return (log_low + log_high) / 2 + jnp.log(log_high - log_low)


def _batch_solve_triangular(A, B):
    """
    Extende solve_triangular for the case that B.ndim > A.ndim.
    This is achived by first flattening the leading B.ndim - A.ndim dimensions of B and then
    moving the first dimension to the end.


    :param jnp.ndarray (...,M,M) A: An array with lower triangular structure in the last two dimensions.
    :param jnp.ndarray (...,M,N) B: Right-hand side matrix in A x = B.

    :return: Solution of A x = B.
    """
    event_shape = B.shape[-2:]
    batch_shape = lax.broadcast_shapes(A.shape[:-2], B.shape[-A.ndim : -2])
    sample_shape = B.shape[: -A.ndim]
    n, p = event_shape

    A = jnp.broadcast_to(A, batch_shape + A.shape[-2:])
    B = jnp.broadcast_to(B, sample_shape + batch_shape + event_shape)

    B_flat = jnp.moveaxis(B.reshape((-1,) + batch_shape + event_shape), 0, -2).reshape(
        batch_shape + (n,) + (-1,)
    )

    X_flat = solve_triangular(A, B_flat, lower=True)

    sample_shape_dim = len(sample_shape)
    src_axes = tuple([-2 - i for i in range(sample_shape_dim)])
    src_axes = src_axes[::-1]
    dest_axes = tuple([i for i in range(sample_shape_dim)])

    X = jnp.moveaxis(
        X_flat.reshape(batch_shape + (n,) + sample_shape + (p,)),
        src_axes,
        dest_axes,
    )
    return X


def _batch_trace_from_cholesky(L):
    """Computes the trace of matrix X given it's Cholesky decomposition matrix L.

    :param jnp.ndarray(..., M, M) L: An array with lower triangular structure in the last two dimensions.

    :return: Trace of X, where X = L L^T
    """
    return jnp.square(L).sum((-1, -2))


class MatrixNormal(Distribution):
    """
    Matrix variate normal distribution as described in [1] but with a lower_triangular parametrization,
    i.e. :math:`U=scale_tril_row @ scale_tril_row^{T}` and :math:`V=scale_tril_column @ scale_tril_column^{T}`.
    The distribution is related to the multivariate normal distribution in the following way.
    If :math:`X ~ MN(loc,U,V)` then :math:`vec(X) ~ MVN(vec(loc), kron(V,U) )`.

    :param array_like loc: Location of the distribution.
    :param array_like scale_tril_row: Lower cholesky of rows correlation matrix.
    :param array_like scale_tril_column: Lower cholesky of columns correlation matrix.

    **References**

    [1] https://en.wikipedia.org/wiki/Matrix_normal_distribution
    """

    arg_constraints = {
        "loc": constraints.real_vector,
        "scale_tril_row": constraints.lower_cholesky,
        "scale_tril_column": constraints.lower_cholesky,
    }
    support = constraints.real_matrix
    reparametrized_params = [
        "loc",
        "scale_tril_row",
        "scale_tril_column",
    ]

    def __init__(self, loc, scale_tril_row, scale_tril_column, validate_args=None):
        event_shape = loc.shape[-2:]
        batch_shape = lax.broadcast_shapes(
            jnp.shape(loc)[:-2],
            jnp.shape(scale_tril_row)[:-2],
            jnp.shape(scale_tril_column)[:-2],
        )
        (self.loc,) = promote_shapes(loc, shape=batch_shape + loc.shape[-2:])
        (self.scale_tril_row,) = promote_shapes(
            scale_tril_row, shape=batch_shape + scale_tril_row.shape[-2:]
        )
        (self.scale_tril_column,) = promote_shapes(
            scale_tril_column, shape=batch_shape + scale_tril_column.shape[-2:]
        )
        super(MatrixNormal, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    @property
    def mean(self):
        return jnp.broadcast_to(self.loc, self.shape())

    def sample(self, key, sample_shape=()):
        eps = random.normal(
            key, shape=sample_shape + self.batch_shape + self.event_shape
        )
        samples = self.loc + self.scale_tril_row @ eps @ jnp.swapaxes(
            self.scale_tril_column, -2, -1
        )

        return samples

    @validate_sample
    def log_prob(self, values):
        n, p = self.event_shape

        row_log_det = tri_logabsdet(self.scale_tril_row)
        col_log_det = tri_logabsdet(self.scale_tril_column)
        log_det_term = (
            p * row_log_det + n * col_log_det + 0.5 * n * p * jnp.log(2 * jnp.pi)
        )

        # compute the trace term
        diff = values - self.loc
        diff_row_solve = _batch_solve_triangular(A=self.scale_tril_row, B=diff)
        diff_col_solve = _batch_solve_triangular(
            A=self.scale_tril_column, B=jnp.swapaxes(diff_row_solve, -2, -1)
        )
        batched_trace_term = _batch_trace_from_cholesky(diff_col_solve)

        log_prob = -0.5 * batched_trace_term - log_det_term

        return log_prob


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
    for sL, sx in zip(bL.shape[:-2], out_shape[sample_ndim:]):
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
    M = jnp.sum(solve_bL_bx**2, axis=-2)  # shape: (i, 1, -1)
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
        assert_one_of(
            covariance_matrix=covariance_matrix,
            precision_matrix=precision_matrix,
            scale_tril=scale_tril,
        )
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
        half_log_det = tri_logabsdet(self.scale_tril)
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
            jnp.sum(self.scale_tril**2, axis=-1), self.batch_shape + self.event_shape
        )

    @staticmethod
    def infer_shapes(
        loc=(), covariance_matrix=None, precision_matrix=None, scale_tril=None
    ):
        assert_one_of(
            covariance_matrix=covariance_matrix,
            precision_matrix=precision_matrix,
            scale_tril=scale_tril,
        )
        batch_shape, event_shape = loc[:-1], loc[-1:]
        for matrix in [covariance_matrix, precision_matrix, scale_tril]:
            if matrix is not None:
                batch_shape = lax.broadcast_shapes(batch_shape, matrix[:-2])
                event_shape = lax.broadcast_shapes(event_shape, matrix[-1:])
                return batch_shape, event_shape

    def entropy(self):
        (n,) = self.event_shape
        half_log_det = tri_logabsdet(self.scale_tril)
        return n * (jnp.log(2 * np.pi) + 1) / 2 + half_log_det


def _is_sparse(A):
    from scipy import sparse

    return sparse.issparse(A)


def _to_sparse(A):
    from scipy import sparse

    return sparse.csr_matrix(A)


class CAR(Distribution):
    r"""
    The Conditional Autoregressive (CAR) distribution is a special case of the multivariate
    normal in which the precision matrix is structured according to the adjacency matrix of
    sites. The amount of autocorrelation between sites is controlled by ``correlation``. The
    distribution is a popular prior for areal spatial data.

    :param float or ndarray loc: mean of the multivariate normal
    :param float correlation: autoregression parameter. For most cases, the value should lie
        between 0 (sites are independent, collapses to an iid multivariate normal) and
        1 (perfect autocorrelation between sites), but the specification allows for negative
        correlations.
    :param float conditional_precision: positive precision for the multivariate normal
    :param ndarray or scipy.sparse.csr_matrix adj_matrix: symmetric adjacency matrix where 1
        indicates adjacency between sites and 0 otherwise. :class:`jax.numpy.ndarray` ``adj_matrix`` is
        supported but is **not** recommended over :class:`numpy.ndarray` or :class:`scipy.sparse.spmatrix`.
    :param bool is_sparse: whether to use a sparse form of ``adj_matrix`` in calculations (must be True if
        ``adj_matrix`` is a :class:`scipy.sparse.spmatrix`)
    """

    arg_constraints = {
        "loc": constraints.real_vector,
        "correlation": constraints.open_interval(-1, 1),
        "conditional_precision": constraints.positive,
        "adj_matrix": constraints.dependent(is_discrete=False, event_dim=2),
    }
    support = constraints.real_vector
    reparametrized_params = [
        "loc",
        "correlation",
        "conditional_precision",
        "adj_matrix",
    ]
    pytree_aux_fields = ("is_sparse", "adj_matrix")

    def __init__(
        self,
        loc,
        correlation,
        conditional_precision,
        adj_matrix,
        *,
        is_sparse=False,
        validate_args=None,
    ):
        if jnp.ndim(loc) == 0:
            (loc,) = promote_shapes(loc, shape=(1,))

        self.is_sparse = is_sparse

        batch_shape = lax.broadcast_shapes(
            jnp.shape(loc)[:-1],
            jnp.shape(correlation),
            jnp.shape(conditional_precision),
            jnp.shape(adj_matrix)[:-2],
        )

        if self.is_sparse:
            if adj_matrix.ndim != 2:
                raise ValueError(
                    "Currently, we only support 2-dimensional adj_matrix. Please make a feature request",
                    " if you need higher dimensional adj_matrix.",
                )
            if not (isinstance(adj_matrix, np.ndarray) or _is_sparse(adj_matrix)):
                raise ValueError(
                    "adj_matrix needs to be a numpy array or a scipy sparse matrix. Please make a feature",
                    " request if you need to support jax ndarrays.",
                )
            # TODO: look into future jax sparse csr functionality and other developments
            self.adj_matrix = _to_sparse(adj_matrix)
        else:
            assert not _is_sparse(
                adj_matrix
            ), "adj_matrix is a sparse matrix so please specify `is_sparse=True`."
            # TODO: look into static jax ndarray representation
            (self.adj_matrix,) = promote_shapes(
                adj_matrix, shape=batch_shape + adj_matrix.shape[-2:]
            )

        event_shape = jnp.shape(self.adj_matrix)[-1:]
        (self.loc,) = promote_shapes(loc, shape=batch_shape + event_shape)
        self.correlation, self.conditional_precision = promote_shapes(
            correlation, conditional_precision, shape=batch_shape
        )

        super(CAR, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

        if self._validate_args and (isinstance(adj_matrix, np.ndarray) or is_sparse):
            assert (
                self.adj_matrix.sum(axis=-1) > 0
            ).all() > 0, "all sites in adjacency matrix must have neighbours"

            if self.is_sparse:
                assert (
                    self.adj_matrix != self.adj_matrix.T
                ).nnz == 0, "adjacency matrix must be symmetric"
            else:
                assert np.array_equal(
                    self.adj_matrix, np.swapaxes(self.adj_matrix, -2, -1)
                ), "adjacency matrix must be symmetric"

    def sample(self, key, sample_shape=()):
        # TODO: look into a sparse sampling method
        mvn = MultivariateNormal(self.mean, precision_matrix=self.precision_matrix)
        return mvn.sample(key, sample_shape=sample_shape)

    @validate_sample
    def log_prob(self, value):
        phi = value - self.loc
        adj_matrix = self.adj_matrix

        if self.is_sparse:
            D = np.asarray(adj_matrix.sum(axis=-1)).squeeze(axis=-1)
            D_rsqrt = D ** (-0.5)

            adj_matrix_scaled = (
                adj_matrix.multiply(D_rsqrt).multiply(D_rsqrt[:, np.newaxis]).toarray()
            )

            adj_matrix = BCOO.from_scipy_sparse(adj_matrix)

        else:
            D = adj_matrix.sum(axis=-1)
            D_rsqrt = D ** (-0.5)

            adj_matrix_scaled = adj_matrix * (
                D_rsqrt[..., None, :] * D_rsqrt[..., None]
            )

        # TODO: look into sparse eignvalue methods
        if isinstance(adj_matrix_scaled, np.ndarray):
            lam = np.linalg.eigvalsh(adj_matrix_scaled)
        else:
            lam = jnp.linalg.eigvalsh(adj_matrix_scaled)

        n = D.shape[-1]

        logprec = n * jnp.log(self.conditional_precision)
        logdet = jnp.log1p(-jnp.expand_dims(self.correlation, -1) * lam).sum(-1)
        logdet = logdet + jnp.log(D).sum(-1)

        logquad = self.conditional_precision * jnp.sum(
            phi
            * (
                D * phi
                - jnp.expand_dims(self.correlation, -1)
                * (adj_matrix @ phi[..., jnp.newaxis]).squeeze(axis=-1)
            ),
            -1,
        )

        return 0.5 * (-n * jnp.log(2 * jnp.pi) + logprec + logdet - logquad)

    @property
    def mean(self):
        return jnp.broadcast_to(self.loc, self.shape())

    @lazy_property
    def precision_matrix(self):
        if self.is_sparse:
            adj_matrix = self.adj_matrix.toarray()
        else:
            adj_matrix = self.adj_matrix

        D = adj_matrix.sum(axis=-1, keepdims=True) * jnp.eye(adj_matrix.shape[-1])
        conditional_precision = jnp.expand_dims(self.conditional_precision, (-2, -1))
        correlation = jnp.expand_dims(self.correlation, (-2, -1))
        return conditional_precision * (D - correlation * adj_matrix)

    @staticmethod
    def infer_shapes(loc, correlation, conditional_precision, adj_matrix):
        event_shape = adj_matrix[-1:]
        batch_shape = lax.broadcast_shapes(
            loc[:-1], correlation, conditional_precision, adj_matrix[:-2]
        )
        return batch_shape, event_shape

    def tree_flatten(self):
        data, aux = super().tree_flatten()
        adj_matrix_data_idx = type(self).gather_pytree_data_fields().index("adj_matrix")
        adj_matrix_aux_idx = type(self).gather_pytree_aux_fields().index("adj_matrix")

        if not self.is_sparse:
            aux = list(aux)
            aux[adj_matrix_aux_idx] = None
            aux = tuple(aux)
        else:
            data = list(data)
            data[adj_matrix_data_idx] = None
            data = tuple(data)
        return data, aux

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        d = super().tree_unflatten(aux_data, params)
        if not d.is_sparse:
            adj_matrix_data_idx = cls.gather_pytree_data_fields().index("adj_matrix")
            setattr(d, "adj_matrix", params[adj_matrix_data_idx])
        else:
            adj_matrix_aux_idx = cls.gather_pytree_aux_fields().index("adj_matrix")
            setattr(d, "adj_matrix", aux_data[adj_matrix_aux_idx])
        return d


class MultivariateStudentT(Distribution):
    arg_constraints = {
        "df": constraints.positive,
        "loc": constraints.real_vector,
        "scale_tril": constraints.lower_cholesky,
    }
    support = constraints.real_vector
    reparametrized_params = ["df", "loc", "scale_tril"]
    pytree_data_fields = ("df", "loc", "scale_tril", "_chi2")

    def __init__(
        self,
        df,
        loc=0.0,
        scale_tril=None,
        validate_args=None,
    ):
        if jnp.ndim(loc) == 0:
            (loc,) = promote_shapes(loc, shape=(1,))
        batch_shape = lax.broadcast_shapes(
            jnp.shape(df), jnp.shape(loc)[:-1], jnp.shape(scale_tril)[:-2]
        )
        (self.df,) = promote_shapes(df, shape=batch_shape)
        (self.loc,) = promote_shapes(loc, shape=batch_shape + loc.shape[-1:])
        (self.scale_tril,) = promote_shapes(
            scale_tril, shape=batch_shape + scale_tril.shape[-2:]
        )
        event_shape = jnp.shape(self.scale_tril)[-1:]
        self._chi2 = Chi2(self.df)
        super(MultivariateStudentT, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        key_normal, key_chi2 = random.split(key)
        std_normal = random.normal(
            key_normal,
            shape=sample_shape + self.batch_shape + self.event_shape,
        )
        z = self._chi2.sample(key_chi2, sample_shape)
        y = std_normal * jnp.expand_dims(jnp.sqrt(self.df / z), -1)
        return self.loc + jnp.squeeze(
            jnp.matmul(self.scale_tril, y[..., jnp.newaxis]), axis=-1
        )

    @validate_sample
    def log_prob(self, value):
        n = self.scale_tril.shape[-1]
        Z = (
            tri_logabsdet(self.scale_tril)
            + 0.5 * n * jnp.log(self.df)
            + 0.5 * n * jnp.log(jnp.pi)
            + gammaln(0.5 * self.df)
            - gammaln(0.5 * (self.df + n))
        )
        M = _batch_mahalanobis(self.scale_tril, value - self.loc)
        return -0.5 * (self.df + n) * jnp.log1p(M / self.df) - Z

    @lazy_property
    def covariance_matrix(self):
        # NB: this is not covariance of this distribution;
        # the actual covariance is df / (df - 2) * covariance_matrix
        return jnp.matmul(self.scale_tril, jnp.swapaxes(self.scale_tril, -1, -2))

    @lazy_property
    def precision_matrix(self):
        identity = jnp.broadcast_to(
            jnp.eye(self.scale_tril.shape[-1]), self.scale_tril.shape
        )
        return cho_solve((self.scale_tril, True), identity)

    @property
    def mean(self):
        # for df <= 1. should be jnp.nan (keeping jnp.inf for consistency with scipy)
        return jnp.broadcast_to(
            jnp.where(jnp.expand_dims(self.df, -1) <= 1, jnp.inf, self.loc),
            self.shape(),
        )

    @property
    def variance(self):
        df = jnp.expand_dims(self.df, -1)
        var = jnp.power(self.scale_tril, 2).sum(-1) * (df / (df - 2))
        var = jnp.where(df > 2, var, jnp.inf)
        var = jnp.where(df <= 1, jnp.nan, var)
        return jnp.broadcast_to(var, self.batch_shape + self.event_shape)

    @staticmethod
    def infer_shapes(df, loc, scale_tril):
        event_shape = (scale_tril[-1],)
        batch_shape = lax.broadcast_shapes(df, loc[:-1], scale_tril[:-2])
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
    return jnp.linalg.cholesky(add_diag(K, 1))


def _batch_lowrank_logdet(W, D, capacitance_tril):
    r"""
    Uses "matrix determinant lemma"::
        log|W @ W.T + D| = log|C| + log|D|,
    where :math:`C` is the capacitance matrix :math:`I + W.T @ inv(D) @ W`, to compute
    the log determinant.
    """
    return 2 * tri_logabsdet(capacitance_tril) + jnp.log(D).sum(-1)


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
    pytree_data_fields = ("loc", "cov_factor", "cov_diag", "_capacitance_tril")

    def __init__(self, loc, cov_factor, cov_diag, *, validate_args=None):
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
        K = add_diag(K, 1)
        scale_tril = cov_diag_sqrt_unsqueeze * jnp.linalg.cholesky(K)
        return scale_tril

    @lazy_property
    def covariance_matrix(self):
        covariance_matrix = add_diag(
            jnp.matmul(self.cov_factor, jnp.swapaxes(self.cov_factor, -1, -2)),
            self.cov_diag,
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
        inverse_cov_diag = jnp.reciprocal(self.cov_diag)
        return add_diag(-jnp.matmul(jnp.swapaxes(A, -1, -2), A), inverse_cov_diag)

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

    def __init__(self, loc=0.0, scale=1.0, *, validate_args=None):
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
        return -0.5 * value_scaled**2 - normalize_term

    def cdf(self, value):
        scaled = (value - self.loc) / self.scale
        return ndtr(scaled)

    def log_cdf(self, value):
        return jax_norm.logcdf(value, loc=self.loc, scale=self.scale)

    def icdf(self, q):
        return self.loc + self.scale * ndtri(q)

    @property
    def mean(self):
        return jnp.broadcast_to(self.loc, self.batch_shape)

    @property
    def variance(self):
        return jnp.broadcast_to(self.scale**2, self.batch_shape)

    def entropy(self):
        return jnp.broadcast_to(
            (jnp.log(2 * np.pi * self.scale**2) + 1) / 2, self.batch_shape
        )


class Pareto(TransformedDistribution):
    arg_constraints = {"scale": constraints.positive, "alpha": constraints.positive}
    reparametrized_params = ["scale", "alpha"]

    def __init__(self, scale, alpha, *, validate_args=None):
        self.scale, self.alpha = promote_shapes(scale, alpha)
        batch_shape = lax.broadcast_shapes(jnp.shape(scale), jnp.shape(alpha))
        scale, alpha = (
            jnp.broadcast_to(scale, batch_shape),
            jnp.broadcast_to(alpha, batch_shape),
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
            (self.scale**2) * self.alpha, (self.alpha - 1) ** 2 * (self.alpha - 2)
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

    def entropy(self):
        return jnp.log(self.scale / self.alpha) + 1 + 1 / self.alpha


class RelaxedBernoulliLogits(TransformedDistribution):
    arg_constraints = {"temperature": constraints.positive, "logits": constraints.real}
    support = constraints.unit_interval

    def __init__(self, temperature, logits, *, validate_args=None):
        self.temperature, self.logits = promote_shapes(temperature, logits)
        base_dist = Logistic(logits / temperature, 1 / temperature)
        transforms = [SigmoidTransform()]
        super().__init__(base_dist, transforms, validate_args=validate_args)


def RelaxedBernoulli(temperature, probs=None, logits=None, *, validate_args=None):
    if probs is None and logits is None:
        raise ValueError("One of `probs` or `logits` must be specified.")
    if probs is not None:
        logits = _to_logits_bernoulli(probs)
    return RelaxedBernoulliLogits(temperature, logits, validate_args=validate_args)


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
        dtype = jnp.result_type(float)
        finfo = jnp.finfo(dtype)
        minval = finfo.tiny
        u = random.uniform(key, shape=sample_shape + self.batch_shape, minval=minval)
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
    pytree_data_fields = ("df", "loc", "scale", "_chi2")

    def __init__(self, df, loc=0.0, scale=1.0, *, validate_args=None):
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
        return -0.5 * (self.df + 1.0) * jnp.log1p(y**2.0 / self.df) - z

    @property
    def mean(self):
        # for df <= 1. should be jnp.nan (keeping jnp.inf for consistency with scipy)
        return jnp.broadcast_to(
            jnp.where(self.df <= 1, jnp.inf, self.loc), self.batch_shape
        )

    @property
    def variance(self):
        var = jnp.where(
            self.df > 2, jnp.divide(self.scale**2 * self.df, self.df - 2.0), jnp.inf
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
        return 0.5 * (
            1
            + jnp.sign(scaled)
            - jnp.sign(scaled) * betainc(0.5 * self.df, 0.5, beta_value)
        )

    def icdf(self, q):
        beta_value = betaincinv(0.5 * self.df, 0.5, 1 - jnp.abs(1 - 2 * q))
        scaled_squared = self.df * (1 / beta_value - 1)
        scaled = jnp.sign(q - 0.5) * jnp.sqrt(scaled_squared)
        return scaled * self.scale + self.loc

    def entropy(self):
        return jnp.broadcast_to(
            (self.df + 1) / 2 * (digamma((self.df + 1) / 2) - digamma(self.df / 2))
            + jnp.log(self.df) / 2
            + betaln(self.df / 2, 0.5)
            + jnp.log(self.scale),
            self.batch_shape,
        )


class Uniform(Distribution):
    arg_constraints = {"low": constraints.dependent, "high": constraints.dependent}
    reparametrized_params = ["low", "high"]
    pytree_data_fields = ("low", "high", "_support")

    def __init__(self, low=0.0, high=1.0, *, validate_args=None):
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
        return jnp.clip(cdf, 0.0, 1.0)

    def icdf(self, value):
        return self.low + value * (self.high - self.low)

    @property
    def mean(self):
        return self.low + (self.high - self.low) / 2.0

    @property
    def variance(self):
        return (self.high - self.low) ** 2 / 12.0

    @staticmethod
    def infer_shapes(low=(), high=()):
        batch_shape = lax.broadcast_shapes(low, high)
        event_shape = ()
        return batch_shape, event_shape

    def entropy(self):
        return jnp.log(self.high - self.low)


class Weibull(Distribution):
    arg_constraints = {
        "scale": constraints.positive,
        "concentration": constraints.positive,
    }
    support = constraints.positive
    reparametrized_params = ["scale", "concentration"]

    def __init__(self, scale, concentration, *, validate_args=None):
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
        return self.scale**2 * (
            jnp.exp(gammaln(1.0 + 2.0 / self.concentration))
            - jnp.exp(gammaln(1.0 + 1.0 / self.concentration)) ** 2
        )

    def entropy(self):
        return (
            jnp.euler_gamma * (1 - 1 / self.concentration)
            + jnp.log(self.scale / self.concentration)
            + 1
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
        "mean": constraints.open_interval(0.0, 1.0),
        "concentration": constraints.positive,
    }
    reparametrized_params = ["mean", "concentration"]
    support = constraints.unit_interval
    pytree_data_fields = ("concentration",)

    def __init__(self, mean, concentration, *, validate_args=None):
        self.concentration = jnp.broadcast_to(
            concentration, lax.broadcast_shapes(jnp.shape(concentration))
        )
        super().__init__(
            mean * concentration,
            (1.0 - mean) * concentration,
            validate_args=validate_args,
        )


class AsymmetricLaplaceQuantile(Distribution):
    """An alternative parameterization of AsymmetricLaplace commonly applied in
    Bayesian quantile regression.

    Instead of the `asymmetry` parameter employed by AsymmetricLaplace, to
    define the balance between left- versus right-hand sides of the
    distribution, this class utilizes a `quantile` parameter, which describes
    the proportion of probability density that falls to the left-hand side of
    the distribution.

    The `scale` parameter is also interpreted slightly differently than in
    AsymmetricLaplace. When `loc=0` and `scale=1`, AsymmetricLaplace(0,1,1)
    is equivalent to Laplace(0,1), while AsymmetricLaplaceQuantile(0,1,0.5) is
    equivalent to Laplace(0,2).
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "quantile": constraints.open_interval(0.0, 1.0),
    }
    reparametrized_params = ["loc", "scale", "quantile"]
    support = constraints.real
    pytree_data_fields = ("loc", "scale", "quantile", "_ald")

    def __init__(self, loc=0.0, scale=1.0, quantile=0.5, *, validate_args=None):
        batch_shape = lax.broadcast_shapes(
            jnp.shape(loc), jnp.shape(scale), jnp.shape(quantile)
        )
        self.loc, self.scale, self.quantile = promote_shapes(
            loc, scale, quantile, shape=batch_shape
        )
        super(AsymmetricLaplaceQuantile, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )
        asymmetry = (1 / ((1 / quantile) - 1)) ** 0.5
        scale_classic = scale * asymmetry / quantile
        self._ald = AsymmetricLaplace(loc=loc, scale=scale_classic, asymmetry=asymmetry)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self._ald.log_prob(value)

    def sample(self, key, sample_shape=()):
        return self._ald.sample(key, sample_shape=sample_shape)

    @property
    def mean(self):
        return self._ald.mean

    @property
    def variance(self):
        return self._ald.variance

    def cdf(self, value):
        return self._ald.cdf(value)

    def icdf(self, value):
        return self._ald.icdf(value)


class ZeroSumNormal(TransformedDistribution):
    r"""
    Zero Sum Normal distribution adapted from PyMC [1] as described in [2,3]. This is a Normal distribution where one or
    more axes are constrained to sum to zero (the last axis by default).

    .. math::
        \begin{align*}
        ZSN(\sigma) = N(0, \sigma^2 (I - \tfrac{1}{n}J)) \\
        \text{where} \ ~ J_{ij} = 1 \ ~ \text{and} \\
        n = \text{number of zero-sum axes}
        \end{align*}

    :param array_like scale: Standard deviation of the underlying normal distribution before the zerosum constraint is
        enforced.
    :param tuple event_shape: The event shape of the distribution, the axes of which get constrained to sum to zero.

    **Example:**

    .. doctest::

        >>> from numpy.testing import assert_allclose
        >>> from jax import random
        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.infer import MCMC, NUTS

        >>> N = 1000
        >>> n_categories = 20
        >>> rng_key = random.PRNGKey(0)
        >>> key1, key2, key3 = random.split(rng_key, 3)
        >>> category_ind = random.choice(key1, jnp.arange(n_categories), shape=(N,))
        >>> beta = random.normal(key2, shape=(n_categories,))
        >>> beta -= beta.mean(-1)
        >>> y = 5 + beta[category_ind] + random.normal(key3, shape=(N,))

        >>> def model(category_ind, y): # category_ind is an indexed categorical variable with 20 categories
        ...     N = len(category_ind)
        ...     alpha = numpyro.sample("alpha", dist.Normal(0, 2.5))
        ...     beta = numpyro.sample("beta", dist.ZeroSumNormal(1, event_shape=(n_categories,)))
        ...     sigma =  numpyro.sample("sigma", dist.Exponential(1))
        ...     with numpyro.plate("observations", N):
        ...         mu = alpha + beta[category_ind]
        ...         obs = numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)
        ...     return obs

        >>> nuts_kernel = NUTS(model=model, target_accept_prob=0.9)
        >>> mcmc = MCMC(
        ...     sampler=nuts_kernel,
        ...     num_samples=1_000, num_warmup=1_000, num_chains=4
        ... )
        >>> mcmc.run(random.PRNGKey(0), category_ind=category_ind, y=y)
        >>> posterior_samples = mcmc.get_samples()
        >>> # Confirm everything along last axis sums to zero
        >>> assert_allclose(posterior_samples['beta'].sum(-1), 0, atol=1e-3)

    **References**
    [1] https://github.com/pymc-devs/pymc/blob/6252d2e58dc211c913ee2e652a4058d271d48bbd/pymc/distributions/multivariate.py#L2637
    [2] https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.ZeroSumNormal.html
    [3] https://learnbayesstats.com/episode/74-optimizing-nuts-developing-zerosumnormal-distribution-adrian-seyboldt/
    """

    arg_constraints = {"scale": constraints.positive}
    reparametrized_params = ["scale"]

    def __init__(self, scale, event_shape, *, validate_args=None):
        event_ndim = len(event_shape)
        transformed_shape = tuple(size - 1 for size in event_shape)
        self.scale = scale
        super().__init__(
            Normal(0, scale).expand(transformed_shape).to_event(event_ndim),
            ZeroSumTransform(event_ndim),
            validate_args=validate_args,
        )

    @constraints.dependent_property(is_discrete=False)
    def support(self):
        return constraints.zero_sum(len(self.event_shape))

    @property
    def mean(self):
        return jnp.zeros(self.batch_shape + self.event_shape)

    @property
    def variance(self):
        event_ndim = len(self.event_shape)
        zero_sum_axes = tuple(range(-event_ndim, 0))
        theoretical_var = jnp.square(self.scale)
        for axis in zero_sum_axes:
            theoretical_var *= 1 - 1 / self.event_shape[axis]

        return jnp.broadcast_to(theoretical_var, self.batch_shape + self.event_shape)


class Wishart(TransformedDistribution):
    """
    Wishart distribution for covariance matrices.

    :param concentration: Positive concentration parameter analogous to the
        concentration of a :class:`Gamma` distribution. The concentration must be larger
        than the dimensionality of the scale matrix.
    :param scale_matrix: Scale matrix analogous to the inverse rate of a :class:`Gamma`
        distribution.
    :param rate_matrix: Rate matrix anaologous to the rate of a :class:`Gamma`
        distribution.
    :param scale_tril: Cholesky decomposition of the :code:`scale_matrix`.
    """

    arg_constraints = {
        "concentration": constraints.dependent(is_discrete=False),
        "scale_matrix": constraints.positive_definite,
        "rate_matrix": constraints.positive_definite,
        "scale_tril": constraints.lower_cholesky,
    }
    support = constraints.positive_definite
    reparametrized_params = [
        "scale_matrix",
        "rate_matrix",
        "scale_tril",
    ]

    def __init__(
        self,
        concentration,
        scale_matrix=None,
        rate_matrix=None,
        scale_tril=None,
        *,
        validate_args=None,
    ):
        base_dist = WishartCholesky(
            concentration,
            scale_matrix,
            rate_matrix,
            scale_tril,
            validate_args=validate_args,
        )
        super().__init__(
            base_dist, CholeskyTransform().inv, validate_args=validate_args
        )

    @lazy_property
    def concentration(self):
        return self.base_dist.concentration

    @lazy_property
    def scale_matrix(self):
        return self.base_dist.scale_matrix

    @lazy_property
    def rate_matrix(self):
        return self.base_dist.rate_matrix

    @lazy_property
    def scale_tril(self):
        return self.base_dist.scale_tril

    @lazy_property
    def mean(self):
        return self.concentration[..., None, None] * self.scale_matrix

    @lazy_property
    def variance(self):
        diag = jnp.diagonal(self.scale_matrix, axis1=-1, axis2=-2)
        return self.concentration[..., None, None] * (
            self.scale_matrix**2 + diag[..., :, None] * diag[..., None, :]
        )

    @staticmethod
    def infer_shapes(
        concentration=(), scale_matrix=None, rate_matrix=None, scale_tril=None
    ):
        return WishartCholesky.infer_shapes(
            concentration, scale_matrix, rate_matrix, scale_tril
        )

    def entropy(self):
        p = self.event_shape[-1]
        return (
            (p + 1) * tri_logabsdet(self.scale_tril)
            + p * (p + 1) / 2 * jnp.log(2)
            + multigammaln(self.concentration / 2, p)
            - (self.concentration - p - 1) / 2 * multidigamma(self.concentration / 2, p)
            + self.concentration * p / 2
        )


class WishartCholesky(Distribution):
    """
    Cholesky factor of a Wishart distribution for covariance matrices.

    :param concentration: Positive concentration parameter analogous to the
        concentration of a :class:`Gamma` distribution. The concentration must be larger
        than the dimensionality of the scale matrix.
    :param scale_matrix: Scale matrix analogous to the inverse rate of a :class:`Gamma`
        distribution.
    :param rate_matrix: Rate matrix anaologous to the rate of a :class:`Gamma`
        distribution.
    :param scale_tril: Cholesky decomposition of the :code:`scale_matrix`.
    """

    arg_constraints = {
        "concentration": constraints.dependent(is_discrete=False),
        "scale_matrix": constraints.positive_definite,
        "rate_matrix": constraints.positive_definite,
        "scale_tril": constraints.lower_cholesky,
    }
    support = constraints.lower_cholesky
    reparametrized_params = [
        "scale_matrix",
        "rate_matrix",
        "scale_tril",
    ]

    def __init__(
        self,
        concentration,
        scale_matrix=None,
        rate_matrix=None,
        scale_tril=None,
        *,
        validate_args=None,
    ):
        assert_one_of(
            scale_matrix=scale_matrix,
            rate_matrix=rate_matrix,
            scale_tril=scale_tril,
        )
        concentration = jnp.asarray(concentration)[..., None, None]
        if scale_matrix is not None:
            concentration, self.scale_matrix = promote_shapes(
                concentration, scale_matrix
            )
            self.scale_tril = jnp.linalg.cholesky(self.scale_matrix)
        elif rate_matrix is not None:
            concentration, self.rate_matrix = promote_shapes(concentration, rate_matrix)
            self.scale_tril = cholesky_of_inverse(self.rate_matrix)
        elif scale_tril is not None:
            concentration, self.scale_tril = promote_shapes(
                concentration, jnp.asarray(scale_tril)
            )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(concentration)[:-2], jnp.shape(self.scale_tril)[:-2]
        )
        event_shape = jnp.shape(self.scale_tril)[-2:]
        self.concentration = concentration[..., 0, 0]
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    @validate_sample
    def log_prob(self, value):
        # The log density of the Wishart distribution includes a term
        # t = trace(rate_matrix @ cov). Here, value = cholesky(cov) such that
        # t = trace(value.T @ rate_matrix @ value) by the cyclical property of the
        # trace. The rate matrix is the inverse scale matrix with Cholesky decomposition
        # scale_tril. Thus,
        # t = trace(value.T @ inv(scale_tril).T @ inv(scale_tril) @ value), and we can
        # rewrite as t = trace(x.T @ x) for x = inv(scale_tril) @ value which we can
        # obtain easily by solving a triangular system. x is again triangular such that
        # trace(x @ x.T) is equal to the sum of squares of elements.
        x = solve_triangular(*jnp.broadcast_arrays(self.scale_tril, value), lower=True)
        trace = jnp.square(x).sum(axis=(-1, -2))
        p = value.shape[-1]
        return (
            (self.concentration - p - 1) * tri_logabsdet(value)
            - trace / 2
            + p * (1 - self.concentration / 2) * jnp.log(2)
            - multigammaln(self.concentration / 2, p)
            - self.concentration * tri_logabsdet(self.scale_tril)
            # Part of the Jacobian of the Cholesky transformation.
            + jnp.sum(
                jnp.arange(p, 0, -1) * jnp.log(jnp.diagonal(value, axis1=-2, axis2=-1)),
                axis=-1,
            )
        )

    @lazy_property
    def scale_matrix(self):
        return jnp.matmul(self.scale_tril, self.scale_tril.mT)

    @lazy_property
    def rate_matrix(self):
        identity = jnp.broadcast_to(
            jnp.eye(self.scale_tril.shape[-1]), self.scale_tril.shape
        )
        return cho_solve((self.scale_tril, True), identity)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        # Sample using the Bartlett decomposition
        # (https://en.wikipedia.org/wiki/Wishart_distribution#Bartlett_decomposition).
        rng_diag, rng_offdiag = random.split(key)
        latent = jnp.zeros(sample_shape + self.batch_shape + self.event_shape)
        p = self.event_shape[-1]
        i = jnp.arange(p)
        latent = latent.at[..., i, i].set(
            jnp.sqrt(
                random.chisquare(
                    rng_diag, self.concentration[..., None] - i, latent.shape[:-1]
                )
            )
        )
        i, j = jnp.tril_indices(p, -1)
        assert i.size == p * (p - 1) // 2
        latent = latent.at[..., i, j].set(
            random.normal(rng_offdiag, latent.shape[:-2] + (i.size,))
        )
        return jnp.matmul(*jnp.broadcast_arrays(self.scale_tril, latent))

    @lazy_property
    def mean(self):
        # The mean follows from the Bartlett decomposition sampling. All off-diagonal
        # elements of the latent variable have zero expectation. The diagonal are the
        # expected square roots of chi^2 variables which can be expressed in terms of
        # gamma functions (see
        # https://en.wikipedia.org/wiki/Chi-squared_distribution#Noncentral_moments).
        k = self.concentration[..., None] - jnp.arange(self.scale_tril.shape[-1])
        sqrtchi2 = jnp.sqrt(2) * jnp.exp(gammaln((k + 1) / 2) - gammaln(k / 2))
        return self.scale_tril * sqrtchi2[..., None, :]

    @lazy_property
    def variance(self):
        # We have the same as for the mean except now the lower off-diagonals are one
        # due to the standard normal noise, and the diagonals are equal to the dof of
        # the chi^2 variables.
        i = jnp.arange(self.scale_tril.shape[-1])
        k = self.concentration[..., None] - i
        latent = jnp.tril(
            jnp.ones_like(k, shape=k.shape + (k.shape[-1],)).at[..., i, i].set(k)
        )
        return jnp.square(self.scale_tril) @ latent - jnp.square(self.mean)

    @staticmethod
    def infer_shapes(
        concentration=(), scale_matrix=None, rate_matrix=None, scale_tril=None
    ):
        assert_one_of(
            scale_matrix=scale_matrix,
            rate_matrix=rate_matrix,
            scale_tril=scale_tril,
        )
        for matrix in [scale_matrix, rate_matrix, scale_tril]:
            if matrix is not None:
                batch_shape = lax.broadcast_shapes(concentration, matrix[:-2])
                event_shape = matrix[-2:]
                return batch_shape, event_shape
