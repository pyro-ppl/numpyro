# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import haiku as hk
import flax
from numpyro.contrib.module import haiku_module, flax_module
import numpy as np
from numpyro import handlers
import numpyro


X = np.array(list(range(100)))
Y = np.array([x*2 + 2.0 for x in list(range(100))])
param_keys = ['type', 'name', 'fn', 'args', 'kwargs', 'value', 'scale', 'cond_indep_stack']


def haiku_model(x, y):
    linear_module = hk.transform(lambda x: hk.Linear(1)(x))
    nn = haiku_module("nn", linear_module, 1)
    assert nn.args[0]
    mean = nn(x)
    numpyro.sample("y", numpyro.distributions.Normal(mean, 0.1), obs=y)


def flax_model(x, y):
    linear_module = flax.nn.Dense.partial(features=1)
    nn = flax_module("nn", linear_module, 1)
    mean = nn(x)
    numpyro.sample("y", numpyro.distributions.Normal(mean, 0.1), obs=y)

def test_module():
    with handlers.trace() as flax_tr, handlers.seed(rng_seed=1):
        flax_model(X, Y)
    flax_params = flax_tr["nn$params"]
    assert flax_params['args'][0]['kernel'].shape == (100, 100)
    assert flax_params['args'][0]['bias'].shape == (100,)

    with handlers.trace() as haiku_tr, handlers.seed(rng_seed=1):
        haiku_model(X, Y)
    haiku_params = haiku_tr["nn$params"]
    assert haiku_params['args'][0]['linear']['w'].shape == (100, 100)
    assert haiku_params['args'][0]['linear']['b'].shape == (100,)
