# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from jax import jit, lax, random
from jax.nn import softplus

import numpyro
from numpyro import handlers, optim
from numpyro.contrib.module import flax_module, haiku_module, random_module
import numpyro.distributions as dist
from numpyro.infer.autoguide import AutoDiagonalNormal
from numpyro.infer import ELBO, SVI


def haiku_model(x, y):
    import haiku as hk

    linear_module = hk.transform(lambda x: hk.Linear(100)(x))
    nn = haiku_module("nn", linear_module, (100,))
    mean = nn(x)
    numpyro.sample("y", numpyro.distributions.Normal(mean, 0.1), obs=y)


def flax_model(x, y):
    import flax

    linear_module = flax.nn.Dense.partial(features=100)
    nn = flax_module("nn", linear_module, (100,))
    mean = nn(x)
    numpyro.sample("y", numpyro.distributions.Normal(mean, 0.1), obs=y)


def test_flax_module():
    X = np.arange(100)
    Y = 2 * X + 2

    with handlers.trace() as flax_tr, handlers.seed(rng_seed=1):
        flax_model(X, Y)
    assert flax_tr["nn$params"]['value']['kernel'].shape == (100, 100)
    assert flax_tr["nn$params"]['value']['bias'].shape == (100,)


def test_haiku_module():
    X = np.arange(100)
    Y = 2 * X + 2

    with handlers.trace() as haiku_tr, handlers.seed(rng_seed=1):
        haiku_model(X, Y)
    assert haiku_tr["nn$params"]['value']['linear']['w'].shape == (100, 100)
    assert haiku_tr["nn$params"]['value']['linear']['b'].shape == (100,)


def test_random_module():
    # ported from https://github.com/ctallec/pyvarinf/blob/master/main_regression.ipynb
    from flax import nn

    class Model(nn.Module):
        def apply(self, x, n_units):
            x = nn.Dense(x, features=n_units)
            x = nn.relu(x)
            x = nn.Dense(x, features=n_units)
            x = nn.relu(x)
            mean = nn.Dense(x, features=1)
            rho = nn.Dense(x, features=1)
            return mean, rho

    def generate_data(n_samples):
        x = np.random.normal(size=(n_samples, 1))
        y = np.cos(x * 3) + np.random.normal(size=(n_samples, 1)) * np.abs(x) / 2
        return x, y

    n_train_data = 5000
    n_test_data = 100
    train_x, train_y = generate_data(n_train_data)
    test_x, test_y = generate_data(n_test_data)
    train_y = train_y / np.std(train_y)
    test_y = test_y / np.std(test_y)

    def model(x, y, batch_size=None):
        module = Model.partial(n_units=32)
        nn = random_module("nn", module, dist.Normal(0, 1e-1), input_shape=(1,))
        with numpyro.plate("batch", x.shape[0], subsample_size=batch_size):
            batch_x = numpyro.subsample(x, event_dim=1)
            batch_y = numpyro.subsample(y, event_dim=1)
            mean, rho = nn(batch_x)
            sigma = softplus(rho)
            numpyro.sample("obs", dist.Normal(mean, sigma), obs=batch_y)

    adam = optim.Adam(5e-3)
    guide = AutoDiagonalNormal(model)
    svi = SVI(model, guide, adam, ELBO())
    svi_state = svi.init(random.PRNGKey(0), train_x, train_y)
    update_fn = jit(svi.update)

    n_iterations = 30000
    batch_size = 256
    for e in range(n_iterations):
        svi_state, loss = update_fn(svi_state, train_x, train_y, batch_size=batch_size)

    params = svi.get_params(svi_state)
    # TODO: predict something from params
    assert set(params.keys()) == set(["auto_loc", "auto_scale", "nn$params"])
