# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import haiku as hk
import flax
from numpyro.contrib import haiku_module, flax_module
import numpy as np
from numpyro import handlers
import numpyro


X = np.array(list(range(100)))
Y = np.array([x*2 + 2.0 for x in list(range(100))])


def haiku_model(x, y):
    linear_module = hk.transform(lambda x: hk.Linear(1)(x))
    nn = haiku_module("nn", linear_module, 1)
    mean = nn(x)
    numpyro.sample("y", numpyro.distributions.Normal(mean, 0.1), obs=y)

def flax_model(x, y):
    linear_module = flax.nn.Dense.partial(features=1)
    nn = flax_module("nn", linear_module, 1)
    mean = nn(x)
    numpyro.sample("y", numpyro.distributions.Normal(mean, 0.1), obs=y)


with handlers.trace() as tr, handlers.seed(rng_seed=1):
    # TODO: haiku_model currently doesn't allow params access
    flax_model(X, Y)

flax_params = tr["nn$params"]
assert flax_params['args'][0] == flax_params['value']
assert flax_params['value']['bias'].shape == (100,)
assert flax_params['value']['kernel'].shape == (100, 100)

# some assertation for the names and the shapes of parameters in `params`
# based on your definition of `nn`
