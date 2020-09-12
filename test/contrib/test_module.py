# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import flax
import haiku as hk

import numpyro
from numpyro import handlers
from numpyro.contrib.module import flax_module, haiku_module

X = np.arange(100)
Y = 2 * X + 2


def haiku_model(x, y):
    linear_module = hk.transform(lambda x: hk.Linear(100)(x))
    nn = haiku_module("nn", linear_module, (100,))
    mean = nn(x)
    numpyro.sample("y", numpyro.distributions.Normal(mean, 0.1), obs=y)


def flax_model(x, y):
    linear_module = flax.nn.Dense.partial(features=100)
    nn = flax_module("nn", linear_module, (100,))
    mean = nn(x)
    numpyro.sample("y", numpyro.distributions.Normal(mean, 0.1), obs=y)


def test_flax_module():
    with handlers.trace() as flax_tr, handlers.seed(rng_seed=1):
        flax_model(X, Y)
    assert flax_tr["nn$params"]['value']['kernel'].shape == (100, 100)
    assert flax_tr["nn$params"]['value']['bias'].shape == (100,)


def test_haiku_module():
    with handlers.trace() as haiku_tr, handlers.seed(rng_seed=1):
        haiku_model(X, Y)
    assert haiku_tr["nn$params"]['value']['linear']['w'].shape == (100, 100)
    assert haiku_tr["nn$params"]['value']['linear']['b'].shape == (100,)
