# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import numpy as np
from numpy.testing import assert_allclose
import pytest

from jax import random, test_util

import numpyro
from numpyro import handlers
from numpyro.contrib.module import (
    ParamShape,
    _update_params,
    flax_module,
    haiku_module,
    random_flax_module,
    random_haiku_module,
)
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


def haiku_model_by_shape(x, y):
    import haiku as hk

    linear_module = hk.transform(lambda x: hk.Linear(100)(x))
    nn = haiku_module("nn", linear_module, input_shape=(100,))
    mean = nn(x)
    numpyro.sample("y", numpyro.distributions.Normal(mean, 0.1), obs=y)


def haiku_model_by_kwargs_1(x, y):
    import haiku as hk

    linear_module = hk.transform(lambda x: hk.Linear(100)(x))
    nn = haiku_module("nn", linear_module, x=x)
    mean = nn(x)
    numpyro.sample("y", numpyro.distributions.Normal(mean, 0.1), obs=y)


def haiku_model_by_kwargs_2(w, x, y):
    import haiku as hk

    class TestHaikuModule(hk.Module):
        def __init__(self, dim: int = 100):
            super().__init__()
            self._dim = dim
            return

        def __call__(self, w, x):
            l1 = hk.Linear(self._dim, name="w_linear")(w)
            l2 = hk.Linear(self._dim, name="x_linear")(x)
            return l1 + l2

    linear_module = hk.transform(lambda w, x: TestHaikuModule(100)(w, x))
    nn = haiku_module("nn", linear_module, w=w, x=x)
    mean = nn(w, x)
    numpyro.sample("y", numpyro.distributions.Normal(mean, 0.1), obs=y)


def flax_model_by_shape(x, y):
    import flax

    linear_module = flax.nn.Dense.partial(features=100)
    nn = flax_module("nn", linear_module, input_shape=(100,))
    mean = nn(x)
    numpyro.sample("y", numpyro.distributions.Normal(mean, 0.1), obs=y)


def flax_model_by_kwargs(x, y):
    import flax

    linear_module = flax.nn.Dense.partial(features=100)
    nn = flax_module("nn", linear_module, inputs=x)
    mean = nn(x)
    numpyro.sample("y", numpyro.distributions.Normal(mean, 0.1), obs=y)


def test_flax_module():
    X = np.arange(100).astype(np.float32)
    Y = 2 * X + 2

    with handlers.trace() as flax_tr, handlers.seed(rng_seed=1):
        flax_model_by_shape(X, Y)
    assert flax_tr["nn$params"]["value"]["kernel"].shape == (100, 100)
    assert flax_tr["nn$params"]["value"]["bias"].shape == (100,)

    with handlers.trace() as flax_tr, handlers.seed(rng_seed=1):
        flax_model_by_kwargs(X, Y)
    assert flax_tr["nn$params"]["value"]["kernel"].shape == (100, 100)
    assert flax_tr["nn$params"]["value"]["bias"].shape == (100,)


def test_haiku_module():
    W = np.arange(100).astype(np.float32)
    X = np.arange(100).astype(np.float32)
    Y = 2 * X + 2

    with handlers.trace() as haiku_tr, handlers.seed(rng_seed=1):
        haiku_model_by_shape(X, Y)
    assert haiku_tr["nn$params"]["value"]["linear"]["w"].shape == (100, 100)
    assert haiku_tr["nn$params"]["value"]["linear"]["b"].shape == (100,)

    with handlers.trace() as haiku_tr, handlers.seed(rng_seed=1):
        haiku_model_by_kwargs_1(X, Y)
    assert haiku_tr["nn$params"]["value"]["linear"]["w"].shape == (100, 100)
    assert haiku_tr["nn$params"]["value"]["linear"]["b"].shape == (100,)

    with handlers.trace() as haiku_tr, handlers.seed(rng_seed=1):
        haiku_model_by_kwargs_2(W, X, Y)
    assert haiku_tr["nn$params"]["value"]["test_haiku_module/w_linear"]["w"].shape == (
        100,
        100,
    )
    assert haiku_tr["nn$params"]["value"]["test_haiku_module/w_linear"]["b"].shape == (
        100,
    )
    assert haiku_tr["nn$params"]["value"]["test_haiku_module/x_linear"]["w"].shape == (
        100,
        100,
    )
    assert haiku_tr["nn$params"]["value"]["test_haiku_module/x_linear"]["b"].shape == (
        100,
    )


def test_update_params():
    params = {"a": {"b": {"c": {"d": 1}, "e": np.array(2)}, "f": np.ones(4)}}
    prior = {"a.b.c.d": dist.Delta(4), "a.f": dist.Delta(5)}
    new_params = deepcopy(params)
    with handlers.seed(rng_seed=0):
        _update_params(params, new_params, prior)
    assert params == {
        "a": {"b": {"c": {"d": ParamShape(())}, "e": 2}, "f": ParamShape((4,))}
    }
    test_util.check_eq(
        new_params,
        {
            "a": {
                "b": {"c": {"d": np.array(4.0)}, "e": np.array(2)},
                "f": np.full((4,), 5.0),
            }
        },
    )


@pytest.mark.parametrize("backend", ["flax", "haiku"])
@pytest.mark.parametrize("init", ["shape", "kwargs"])
def test_random_module__mcmc(backend, init):

    if backend == "flax":
        import flax

        linear_module = flax.nn.Dense.partial(features=1)
        bias_name = "bias"
        weight_name = "kernel"
        random_module = random_flax_module
        kwargs_name = "inputs"
    elif backend == "haiku":
        import haiku as hk

        linear_module = hk.transform(lambda x: hk.Linear(1)(x))
        bias_name = "linear.b"
        weight_name = "linear.w"
        random_module = random_haiku_module
        kwargs_name = "x"

    N, dim = 3000, 3
    num_warmup, num_samples = (1000, 1000)
    data = random.normal(random.PRNGKey(0), (N, dim))
    true_coefs = np.arange(1.0, dim + 1.0)
    logits = np.sum(true_coefs * data, axis=-1)
    labels = dist.Bernoulli(logits=logits).sample(random.PRNGKey(1))

    if init == "shape":
        kwargs = {"input_shape": (3,)}
    elif init == "kwargs":
        kwargs = {kwargs_name: data}

    def model(data, labels):
        nn = random_module(
            "nn",
            linear_module,
            {bias_name: dist.Cauchy(), weight_name: dist.Normal()},
            **kwargs
        )
        logits = nn(data).squeeze(-1)
        numpyro.sample("y", dist.Bernoulli(logits=logits), obs=labels)

    kernel = NUTS(model=model)
    mcmc = MCMC(
        kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=False
    )
    mcmc.run(random.PRNGKey(2), data, labels)
    mcmc.print_summary()
    samples = mcmc.get_samples()
    assert set(samples.keys()) == {
        "nn/{}".format(bias_name),
        "nn/{}".format(weight_name),
    }
    assert_allclose(
        np.mean(samples["nn/{}".format(weight_name)].squeeze(-1), 0),
        true_coefs,
        atol=0.22,
    )
