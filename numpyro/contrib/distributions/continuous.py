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


import jax.numpy as np
import jax.random as random
from jax import lax
from jax.scipy.special import gammaln

from numpyro.contrib.distributions.distribution import Distribution, TransformedDistribution
from numpyro.distributions import constraints
from numpyro.distributions.transforms import AbsTransform, ExpTransform, AffineTransform
from numpyro.distributions.util import get_dtypes, promote_shapes, standard_gamma


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

    def sample(self, key, size=()):
        return self._dirichlet.sample(key, size=size)[..., 0]

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
        return (self.concentration1 * self.concentration0 /
                (total ** 2 * (total + 1)))


class Cauchy(Distribution):
    reparametrized_params = ['loc', 'scale']
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = promote_shapes(loc, scale)
        batch_shape = lax.broadcast_shapes(np.shape(loc), np.shape(scale))
        super(Cauchy, self).__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, size=()):
        u = random.uniform(key, shape=size + self.batch_shape)
        eps = np.tan(np.pi * (u - 0.5))
        return self.loc + eps * self.scale

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return - np.log(np.pi) - np.log(self.scale) - np.log(1.0 + ((value - self.loc) / self.scale) ** 2)

    @property
    def mean(self):
        return np.broadcast_to(np.nan, self.batch_shape)

    @property
    def variance(self):
        return np.broadcast_to(np.nan, self.batch_shape)


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

    def sample(self, key, size=()):
        shape = size + self.batch_shape + self.event_shape
        gamma_samples = standard_gamma(key, self.concentration, shape=shape)
        return gamma_samples / np.sum(gamma_samples, axis=-1, keepdims=True)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        normalize_term = (gammaln(np.sum(self.concentration, axis=-1)) -
                          np.sum(gammaln(self.concentration), axis=-1))
        return (np.sum(np.log(value) * (self.concentration - 1.), axis=-1) +
                normalize_term)

    @property
    def mean(self):
        return np.broadcast_to(self.concentration / np.sum(self.concentration, axis=-1, keepdims=True),
                               self.batch_shape + self.event_shape)

    @property
    def variance(self):
        con0 = np.sum(self.concentration, axis=-1, keepdims=True)
        return np.broadcast_to(self.concentration * (con0 - self.concentration) /
                               (con0 ** 2 * (con0 + 1)),
                               self.batch_shape + self.event_shape)


class Gamma(Distribution):
    arg_constraints = {'concentration': constraints.positive,
                       'rate': constraints.positive}
    support = constraints.simplex

    def __init__(self, concentration, rate, validate_args=None):
        self.concentration, self.rate = promote_shapes(concentration, rate)
        batch_shape = lax.broadcast_shapes(np.shape(concentration), np.shape(rate))
        super(Gamma, self).__init__(batch_shape=batch_shape,
                                    validate_args=validate_args)

    def sample(self, key, size=()):
        shape = size + self.batch_shape + self.event_shape
        return standard_gamma(key, self.concentration, shape=shape) / self.rate

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        normalize_term = (self.concentration * np.log(self.rate) -
                          self.rate * value - gammaln(self.concentration))
        return (self.concentration - 1) * np.log(value) + normalize_term

    @property
    def mean(self):
        return np.broadcast_to(self.concentration / self.rate, self.batch_shape)

    @property
    def variance(self):
        return np.broadcast_to(self.concentration / np.power(self.rate, 2), self.batch_shape)


class Chi2(Gamma):
    arg_constraints = {'df': constraints.positive}

    def __init__(self, df, validate_args=None):
        super(Chi2, self).__init__(0.5 * df, 0.5, validate_args=validate_args)


class Exponential(Distribution):
    reparametrized_params = ['rate']
    arg_constraints = {'rate': constraints.positive}
    support = constraints.positive

    def __init__(self, rate, validate_args=None):
        self.rate = rate
        super(Exponential, self).__init__(batch_shape=np.shape(rate), validate_args=validate_args)

    def sample(self, key, size=()):
        u = random.uniform(key, shape=size + self.batch_shape)
        return -np.log1p(-u) / self.rate

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


class HalfCauchy(TransformedDistribution):
    reparametrized_params = ['scale']
    arg_constraints = {'scale': constraints.positive}
    support = constraints.positive

    def __init__(self, scale, validate_args=None):
        base_dist = Cauchy(0, scale)
        super(HalfCauchy, self).__init__(base_dist, AbsTransform(),
                                         validate_args=validate_args)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        log_prob = self.base_dist.log_prob(value) + np.log(2)
        value, log_prob = promote_shapes(value, log_prob)
        log_prob = np.where(value < 0, -np.inf, log_prob)
        return log_prob

    @property
    def mean(self):
        return np.broadcast_to(np.inf, self.batch_shape)

    @property
    def variance(self):
        return np.broadcast_to(np.inf, self.batch_shape)


class Normal(Distribution):
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    reparametrized_params = ['loc', 'scale']

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = promote_shapes(loc, scale)
        batch_shape = lax.broadcast_shapes(np.shape(loc), np.shape(scale))
        super(Normal, self).__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, size=()):
        eps = random.normal(key, shape=size + self.batch_shape)
        return self.loc + eps * self.scale

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return -((value - self.loc) ** 2) / (2.0 * self.scale ** 2) \
               - np.log(self.scale) - np.log(np.sqrt(2 * np.pi))

    @property
    def mean(self):
        return np.broadcast_to(self.loc, self.batch_shape)

    @property
    def variance(self):
        return np.broadcast_to(self.scale ** 2, self.batch_shape)


class LogNormal(TransformedDistribution):
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.positive
    reparametrized_params = ['loc', 'scale']

    def __init__(self, loc, scale, validate_args=None):
        base_dist = Normal(loc, scale)
        self.loc, self.scale = base_dist.loc, base_dist.scale
        super(LogNormal, self).__init__(base_dist, ExpTransform(), validate_args=validate_args)

    @property
    def mean(self):
        return np.exp(self.loc + self.scale ** 2 / 2)

    @property
    def variance(self):
        return (np.exp(self.scale ** 2) - 1) * np.exp(2 * self.loc + self.scale ** 2)


class Pareto(TransformedDistribution):
    arg_constraints = {'alpha': constraints.positive, 'scale': constraints.positive}
    support = constraints.real

    def __init__(self, scale, alpha, validate_args=None):
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

    @property
    def support(self):
        return constraints.greater_than(self.scale)


class Uniform(Distribution):
    arg_constraints = {'low': constraints.dependent, 'high': constraints.dependent}
    reparametrized_params = ['low', 'high']

    def __init__(self, low, high, validate_args=None):
        self.low, self.high = promote_shapes(low, high)
        batch_shape = lax.broadcast_shapes(np.shape(low), np.shape(high))
        super(Uniform, self).__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, size=()):
        size = size + self.batch_shape
        return self.low + random.uniform(key, shape=size) * (self.high - self.low)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        within_bounds = ((value >= self.low) & (value < self.high))
        return np.log(lax.convert_element_type(within_bounds, get_dtypes(self.low)[0])) - \
            np.log(self.high - self.low)

    @property
    def mean(self):
        return np.broadcast_to(self.low + (self.high - self.low) / 2., self.batch_shape)

    @property
    def variance(self):
        return np.broadcast_to((self.high - self.low) ** 2 / 12., self.batch_shape)

    @property
    def support(self):
        return constraints.interval(self.low, self.high)


class StudentT(Distribution):
    arg_constraints = {'df': constraints.positive, 'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    reparametrized_params = ['loc', 'scale']

    def __init__(self, df, loc=0., scale=1., validate_args=None):
        self.df, self.loc, self.scale = promote_shapes(df, loc, scale)
        self._chi2 = Chi2(self.df)
        batch_shape = lax.broadcast_shapes(np.shape(self.df), np.shape(self.loc), np.shape(self.scale))
        super(StudentT, self).__init__(batch_shape, validate_args=validate_args)

    def sample(self, key, size=()):
        std_normal = random.normal(key, shape=size + self.batch_shape)
        z = self._chi2.sample(key, size)
        y = std_normal * np.sqrt(self.df / z)
        return self.loc + self.scale * y

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        y = (value - self.loc) / self.scale
        z = (np.log(self.scale) + 0.5 * np.log(self.df) + 0.5 * np.log(np.pi) +
             gammaln(0.5 * self.df) - gammaln(0.5 * (self.df + 1.)))
        return -0.5 * (self.df + 1.) * np.log1p(y**2. / self.df) - z

    @property
    def mean(self):
        # for df <= 1. should be np.nan (keeping np.inf for consistency with scipy)
        return np.broadcast_to(np.where(self.df <= 1, np.inf, self.loc), self.batch_shape)

    @property
    def variance(self):
        var = np.where(self.df > 2, self.scale ** 2 * self.df / (self.df - 2.0), np.inf)
        var = np.where(self.df <= 1, np.nan, var)
        return np.broadcast_to(var, self.batch_shape)
