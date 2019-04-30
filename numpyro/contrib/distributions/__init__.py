# flake8: noqa

from jax import numpy as np

from numpyro.contrib.distributions.continuous import (
    Beta,
    Cauchy,
    Chi2,
    Dirichlet,
    Exponential,
    Gamma,
    HalfCauchy,
    LogNormal,
    Normal,
    Pareto,
    StudentT,
    Uniform
)
from numpyro.contrib.distributions.discrete import (
    Bernoulli,
    BernoulliWithLogits,
    Binomial,
    BinomialWithLogits,
    Categorical,
    CategoricalWithLogits,
    Multinomial,
    MultinomialWithLogits,
    Poisson
)

__all__ = [
    'Bernoulli',
    'BernoulliWithLogits',
    'Beta',
    'Binomial',
    'BinomialWithLogits',
    'Categorical',
    'CategoricalWithLogits',
    'Cauchy',
    'Chi2',
    'Dirichlet',
    'Exponential',
    'Gamma',
    'HalfCauchy',
    'LogNormal',
    'Multinomial',
    'MultinomialWithLogits',
    'Normal',
    'Pareto',
    'Poisson',
    'StudentT',
    'Uniform',
]

# TODO: remove before release


def get_bern(p, is_logits=False):
    if is_logits:
        return BernoulliWithLogits(p)
    else:
        return Bernoulli(p)


def get_binom(n, p, is_logits=False):
    if is_logits:
        return BinomialWithLogits(p, n)
    else:
        return Binomial(p, n)


def get_categorical(p, is_logits=False):
    if is_logits:
        return CategoricalWithLogits(p)
    else:
        return Categorical(p)


def get_multinomial(n, p, is_logits=False):
    if is_logits:
        return MultinomialWithLogits(p, n)
    else:
        return MultinomialWithLogits(p, n)


beta = Beta
bernoulli = get_bern
binom = get_binom
cauchy = lambda loc=0., scale=1.: Cauchy(loc, scale)
categorical = get_categorical
dirichlet = lambda alpha: Dirichlet(concentration=alpha)
expon = Exponential
gamma = lambda conc, scale=1.: Gamma(conc, rate=1./scale)
halfcauchy = lambda scale: HalfCauchy(scale)
lognorm = lambda s, scale=1.: LogNormal(loc=np.log(scale), scale=s)
multinomial = get_multinomial
norm = lambda loc=0., scale=1.: Normal(loc, scale)
pareto = lambda alpha, scale=1.: Pareto(scale, alpha)
t = lambda df, loc=0., scale=1.: StudentT(df, loc, scale)
uniform = Uniform
