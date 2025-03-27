# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
import sys
from typing import Optional

import numpy as np
from numpy.testing import assert_allclose
import pytest

import jax
from jax import random
import jax.numpy as jnp

import numpyro
from numpyro import handlers
from numpyro.contrib.module import (
    ParamShape,
    _update_params,
    eqx_module,
    flax_module,
    haiku_module,
    nnx_module,
    random_eqx_module,
    random_flax_module,
    random_haiku_module,
    random_nnx_module,
)
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.primitives import mutable as numpyro_mutable

pytestmark = pytest.mark.filterwarnings(
    "ignore:jax.tree_.+ is deprecated:FutureWarning"
)


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

    linear_module = flax.linen.Dense(features=100)
    nn = flax_module("nn", linear_module, input_shape=(100,))
    mean = nn(x)
    numpyro.sample("y", numpyro.distributions.Normal(mean, 0.1), obs=y)


def flax_model_by_kwargs(x, y):
    import flax

    linear_module = flax.linen.Dense(features=100)
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

    jax.tree.all(
        jax.tree.map(
            assert_allclose,
            new_params,
            {
                "a": {
                    "b": {"c": {"d": np.array(4.0)}, "e": np.array(2)},
                    "f": np.full((4,), 5.0),
                }
            },
        )
    )


@pytest.mark.parametrize("backend", ["flax", "haiku"])
@pytest.mark.parametrize("init", ["args", "shape", "kwargs"])
@pytest.mark.parametrize("callable_prior", [True, False])
def test_random_module_mcmc(backend, init, callable_prior):
    if backend == "flax":
        import flax

        linear_module = flax.linen.Dense(features=1)
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
        args = ()
        kwargs = {"input_shape": (3,)}
    elif init == "kwargs":
        args = ()
        kwargs = {kwargs_name: data}
    elif init == "args":
        args = (np.ones(3, dtype=np.float32),)
        kwargs = {}

    if callable_prior:

        def prior(name, shape):
            return dist.Cauchy() if name == bias_name else dist.Normal()
    else:
        prior = {bias_name: dist.Cauchy(), weight_name: dist.Normal()}

    def model(data, labels):
        nn = random_module("nn", linear_module, prior, *args, **kwargs)
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


@pytest.mark.parametrize("dropout", [True, False])
@pytest.mark.parametrize("batchnorm", [True, False])
def test_haiku_state_dropout_smoke(dropout, batchnorm):
    import haiku as hk

    def fn(x):
        if dropout:
            x = hk.dropout(hk.next_rng_key(), 0.5, x)
        if batchnorm:
            x = hk.BatchNorm(create_offset=True, create_scale=True, decay_rate=0.001)(
                x, is_training=True
            )
        return x

    def model():
        transform = hk.transform_with_state if batchnorm else hk.transform
        nn = haiku_module("nn", transform(fn), apply_rng=dropout, input_shape=(4, 3))
        x = numpyro.sample("x", dist.Normal(0, 1).expand([4, 3]).to_event(2))
        if dropout:
            y = nn(numpyro.prng_key(), x)
        else:
            y = nn(x)
        numpyro.deterministic("y", y)

    with handlers.trace(model) as tr, handlers.seed(rng_seed=0):
        model()

    if batchnorm:
        assert set(tr.keys()) == {"nn$params", "nn$state", "x", "y"}
        assert tr["nn$state"]["type"] == "mutable"
    else:
        assert set(tr.keys()) == {"nn$params", "x", "y"}

    # test svi
    guide = AutoDelta(model)
    svi = SVI(model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
    svi.run(random.PRNGKey(100), 10)


@pytest.mark.parametrize("dropout", [True, False])
@pytest.mark.parametrize("batchnorm", [True, False])
def test_flax_state_dropout_smoke(dropout, batchnorm):
    import flax.linen as nn

    class Net(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(10)(x)
            if dropout:
                x = nn.Dropout(0.5, deterministic=False)(x)
            if batchnorm:
                x = nn.BatchNorm(
                    use_bias=True,
                    use_scale=True,
                    momentum=0.999,
                    use_running_average=False,
                )(x)
            return x

    def model():
        net = flax_module(
            "nn",
            Net(),
            apply_rng=["dropout"] if dropout else None,
            mutable=["batch_stats"] if batchnorm else None,
            input_shape=(4, 3),
        )
        x = numpyro.sample("x", dist.Normal(0, 1).expand([4, 3]).to_event(2))
        if dropout:
            y = net(x, rngs={"dropout": numpyro.prng_key()})
        else:
            y = net(x)
        numpyro.deterministic("y", y)

    with handlers.trace(model) as tr, handlers.seed(rng_seed=0):
        model()

    if batchnorm:
        assert set(tr.keys()) == {"nn$params", "nn$state", "x", "y"}
        assert tr["nn$state"]["type"] == "mutable"
    else:
        assert set(tr.keys()) == {"nn$params", "x", "y"}

    # test svi
    guide = AutoDelta(model)
    svi = SVI(model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
    svi.run(random.PRNGKey(100), 10)


@pytest.mark.skipif(sys.version_info[:2] == (3, 9), reason="Skipping on Python 3.9")
def test_nnx_module():
    from flax import nnx

    X = np.arange(100).astype(np.float32)
    Y = 2 * X + 2

    class Linear(nnx.Module):
        def __init__(self, din, dout, *, rngs):
            self.w = nnx.Param(jax.random.uniform(rngs.params(), (din, dout)))
            self.bias = nnx.Param(jnp.zeros((dout,)))

        def __call__(self, x):
            w_val = self.w.value
            bias_val = self.bias.value
            return x @ w_val + bias_val

    # Eager initialization of the Linear module outside the model
    rng_key = random.PRNGKey(1)
    linear_module = Linear(din=100, dout=100, rngs=nnx.Rngs(params=rng_key))

    # Extract parameters and state for inspection
    _, params_state = nnx.split(linear_module, nnx.Param)
    params_dict = nnx.to_pure_dict(params_state)

    # Verify parameters were created correctly
    assert "w" in params_dict
    assert "bias" in params_dict
    assert params_dict["w"].shape == (100, 100)
    assert params_dict["bias"].shape == (100,)

    # Define a model using eager initialization
    def nnx_model_eager(x, y):
        # Use the pre-initialized Linear module
        nn = nnx_module("nn", linear_module)
        mean = nn(x)
        numpyro.sample("y", numpyro.distributions.Normal(mean, 0.1), obs=y)

    with handlers.trace() as nnx_tr, handlers.seed(rng_seed=1):
        nnx_model_eager(X, Y)

    assert "w" in nnx_tr["nn$params"]["value"]
    assert "bias" in nnx_tr["nn$params"]["value"]
    assert nnx_tr["nn$params"]["value"]["w"].shape == (100, 100)
    assert nnx_tr["nn$params"]["value"]["bias"].shape == (100,)


@pytest.mark.skipif(sys.version_info[:2] == (3, 9), reason="Skipping on Python 3.9")
@pytest.mark.parametrize(
    argnames="dropout", argvalues=[True, False], ids=["dropout", "no_dropout"]
)
@pytest.mark.parametrize(
    argnames="batchnorm", argvalues=[True, False], ids=["batchnorm", "no_batchnorm"]
)
def test_nnx_state_dropout_smoke(dropout, batchnorm):
    from flax import nnx

    class Net(nnx.Module):
        def __init__(self, *, rngs):
            if batchnorm:
                # Use feature dimension 3 to match the input shape (4, 3)
                self.bn = nnx.BatchNorm(3, rngs=rngs)
            if dropout:
                # Create dropout with deterministic=True to disable dropout
                self.dropout = nnx.Dropout(rate=0.5, deterministic=True, rngs=rngs)

        def __call__(self, x, *, rngs=None):
            if dropout:
                # Use deterministic=True to disable dropout
                x = self.dropout(x, deterministic=True)

            if batchnorm:
                x = self.bn(x)

            return x

    # Eager initialization of the Net module outside the model
    rng_key = random.PRNGKey(0)
    net_module = Net(rngs=nnx.Rngs(params=rng_key))

    # Extract parameters and state for inspection
    _, state = nnx.split(net_module)

    def model():
        # Use the pre-initialized module
        nn = nnx_module("nn", net_module)

        x = numpyro.sample("x", dist.Normal(0, 1).expand([4, 3]).to_event(2))
        y = nn(x)
        numpyro.deterministic("y", y)

    with handlers.trace(model) as tr, handlers.seed(rng_seed=0):
        model()

    assert set(tr.keys()) == {"nn$params", "nn$state", "x", "y"}
    assert tr["nn$state"]["type"] == "mutable"

    # test svi
    guide = AutoDelta(model)
    svi = SVI(model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
    svi.run(random.PRNGKey(100), 10)


@pytest.mark.skipif(sys.version_info[:2] == (3, 9), reason="Skipping on Python 3.9")
@pytest.mark.parametrize("callable_prior", [True, False])
def test_random_nnx_module_mcmc(callable_prior):
    from flax import nnx

    class Linear(nnx.Module):
        def __init__(self, din, dout, *, rngs):
            self.w = nnx.Param(jax.random.uniform(rngs.params(), (din, dout)))
            self.b = nnx.Param(jnp.zeros((dout,)))

        def __call__(self, x):
            w_val = self.w
            b_val = self.b
            return x @ w_val + b_val

    N, dim = 3000, 3
    data = random.normal(random.PRNGKey(0), (N, dim))
    true_coefs = np.arange(1.0, dim + 1.0)
    logits = np.sum(true_coefs * data, axis=-1)
    labels = dist.Bernoulli(logits=logits).sample(random.PRNGKey(1))

    if callable_prior:

        def prior(name, shape):
            return dist.Cauchy() if name == "b" else dist.Normal()
    else:
        prior = {"w": dist.Normal(), "b": dist.Cauchy()}

    # Create a pre-initialized module for eager initialization
    rng_key = random.PRNGKey(0)
    linear_module = Linear(din=dim, dout=1, rngs=nnx.Rngs(params=rng_key))

    # Extract parameters and state for inspection
    _, params_state = nnx.split(linear_module, nnx.Param)
    params_dict = nnx.to_pure_dict(params_state)

    # Verify parameters were created correctly
    assert "w" in params_dict
    assert "b" in params_dict
    assert params_dict["w"].shape == (dim, 1)
    assert params_dict["b"].shape == (1,)

    def model(data, labels=None):
        # Use the pre-initialized module with eager initialization
        nn = random_nnx_module("nn", linear_module, prior)
        logits = nn(data).squeeze(-1)
        return numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=labels)

    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=2, num_samples=2, progress_bar=False)
    mcmc.run(random.PRNGKey(0), data, labels)
    samples = mcmc.get_samples()
    assert "nn/b" in samples
    assert "nn/w" in samples


@pytest.mark.skipif(sys.version_info[:2] == (3, 9), reason="Skipping on Python 3.9")
def test_eqx_module():
    import equinox as eqx

    X = np.arange(100).astype(np.float32)[None]
    Y = 2 * X + 2

    linear_module = eqx.nn.Linear(
        in_features=100, out_features=100, key=random.PRNGKey(0)
    )

    # Verify parameters were created correctly
    assert hasattr(linear_module, "weight")
    assert hasattr(linear_module, "bias")
    assert linear_module.weight.shape == (100, 100)
    assert linear_module.bias.shape == (100,)

    # Define a model using eager initialization
    def eqx_model_eager(x, y):
        # Use the pre-initialized Linear module
        nn = eqx_module("nn", linear_module)
        mean = jax.vmap(nn)(x)
        numpyro.sample("y", numpyro.distributions.Normal(mean, 0.1), obs=y)

    with handlers.trace() as eqx_tr, handlers.seed(rng_seed=1):
        eqx_model_eager(X, Y)

    assert hasattr(eqx_tr["nn$params"]["value"], "weight")
    assert hasattr(eqx_tr["nn$params"]["value"], "bias")
    assert eqx_tr["nn$params"]["value"].weight.shape == (100, 100)
    assert eqx_tr["nn$params"]["value"].bias.shape == (100,)


@pytest.mark.skipif(sys.version_info[:2] == (3, 9), reason="Skipping on Python 3.9")
@pytest.mark.parametrize(
    argnames="dropout", argvalues=[True, False], ids=["dropout", "no_dropout"]
)
@pytest.mark.parametrize(
    argnames="batchnorm", argvalues=[True, False], ids=["batchnorm", "no_batchnorm"]
)
def test_eqx_state_dropout_smoke(dropout, batchnorm):
    import equinox as eqx

    class Net(eqx.Module):
        bn: Optional[eqx.nn.BatchNorm]
        dropout: Optional[eqx.nn.Dropout]

        def __init__(self, key):
            # Use feature dimension 3 to match the input shape (4, 3)
            self.bn = eqx.nn.BatchNorm(3, axis_name="batch") if batchnorm else None
            # Create dropout with inference=True to disable dropout
            self.dropout = eqx.nn.Dropout(p=0.5, inference=True) if dropout else None

        def __call__(self, x, state):
            if dropout:
                # Use deterministic=True to disable dropout
                x = self.dropout(x, inference=True)

            if batchnorm:
                x, state = self.bn(x, state)

            return x, state

    # Eager initialization of the Net module outside the model
    net_module, eager_state = eqx.nn.make_with_state(Net)(key=random.PRNGKey(0))  # noqa: E1111

    def model():
        # Use the pre-initialized module
        nn = eqx_module("nn", net_module)
        mutable_holder = numpyro_mutable("nn$state", {"state": eager_state})

        x = numpyro.sample("x", dist.Normal(0, 1).expand([4, 3]).to_event(2))

        batched_nn = jax.vmap(
            nn, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
        )
        y, state = batched_nn(x, mutable_holder["state"])
        mutable_holder["state"] = state

        numpyro.deterministic("y", y)

    with handlers.trace(model) as tr, handlers.seed(rng_seed=0):
        model()

    assert set(tr.keys()) == {"nn$params", "nn$state", "x", "y"}
    assert tr["nn$state"]["type"] == "mutable"

    # test svi - trace error with AutoDelta
    guide = AutoDelta(model)
    svi = SVI(model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
    svi.run(random.PRNGKey(100), 10)


@pytest.mark.skipif(sys.version_info[:2] == (3, 9), reason="Skipping on Python 3.9")
@pytest.mark.parametrize("callable_prior", [True, False])
def test_random_eqx_module_mcmc(callable_prior):
    import equinox as eqx

    N, dim = 3000, 3
    data = random.normal(random.PRNGKey(0), (N, dim))
    true_coefs = np.arange(1.0, dim + 1.0)
    logits = np.sum(true_coefs * data, axis=-1)
    labels = dist.Bernoulli(logits=logits).sample(random.PRNGKey(1))

    if callable_prior:

        def prior(name, shape):
            return dist.Cauchy() if name == "bias" else dist.Normal()
    else:
        prior = {"weight": dist.Normal(), "bias": dist.Cauchy()}

    # Create a pre-initialized module for eager initialization
    rng_key = random.PRNGKey(0)
    linear_module = eqx.nn.Linear(in_features=dim, out_features=1, key=rng_key)

    def model(data, labels=None):
        # Use the pre-initialized module with eager initialization
        nn = random_eqx_module("nn", linear_module, prior=prior)
        logits = jax.vmap(nn)(data).squeeze(-1)
        return numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=labels)

    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=2, num_samples=2, progress_bar=False)
    mcmc.run(random.PRNGKey(0), data, labels)
    samples = mcmc.get_samples()
    assert "nn/bias" in samples
    assert "nn/weight" in samples
