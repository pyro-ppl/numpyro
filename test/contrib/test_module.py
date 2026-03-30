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
    nnx_module,
    random_eqx_module,
    random_flax_module,
    random_nnx_module,
)
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.primitives import mutable as numpyro_mutable

pytestmark = pytest.mark.filterwarnings(
    "ignore:jax.tree_.+ is deprecated:FutureWarning"
)


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


def make_batchnorm():
    import equinox as eqx
    from equinox import __version__

    if __version__ >= "0.13":
        return eqx.nn.BatchNorm(3, axis_name="batch", mode="batch")

    return eqx.nn.BatchNorm(3, axis_name="batch")


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


@pytest.mark.parametrize("init", ["args", "shape", "kwargs"])
@pytest.mark.parametrize("callable_prior", [True, False])
def test_random_module_mcmc(init, callable_prior):
    import flax

    linear_module = flax.linen.Dense(features=1)
    bias_name = "bias"
    weight_name = "kernel"
    random_module = random_flax_module
    kwargs_name = "inputs"

    N, dim = 3000, 3
    num_warmup, num_samples = (1000, 1000)
    data = random.normal(random.key(0), (N, dim))
    true_coefs = np.arange(1.0, dim + 1.0)
    logits = np.sum(true_coefs * data, axis=-1)
    labels = dist.Bernoulli(logits=logits).sample(random.key(1))

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
    mcmc.run(random.key(2), data, labels)
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
    svi.run(random.key(100), 10)


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
            w_val = self.w[...]
            bias_val = self.bias[...]
            return x @ w_val + bias_val

    # Eager initialization of the Linear module outside the model
    rng_key = random.key(1)
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
    net_module = Net(rngs=nnx.Rngs(params=0, dropout=1))

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

    key_set = {"nn$params", "x", "y"}
    if batchnorm or dropout:
        key_set.add("nn$state")
    assert set(tr.keys()) == key_set
    if batchnorm or dropout:
        assert tr["nn$state"]["type"] == "mutable"

    # test svi
    guide = AutoDelta(model)
    svi = SVI(model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
    svi.run(random.key(100), 10)


@pytest.mark.skipif(sys.version_info[:2] == (3, 9), reason="Skipping on Python 3.9")
@pytest.mark.parametrize(
    argnames="callable_prior",
    argvalues=[True, False],
    ids=["callable_prior", "dict_prior"],
)
@pytest.mark.parametrize(
    argnames="scope_divider",
    argvalues=["/", "|"],
    ids=["scope_divider=/", "scope_divider=|"],
)
def test_random_nnx_module_mcmc(callable_prior, scope_divider: str):
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
    data = random.normal(random.key(0), (N, dim))
    true_coefs = np.arange(1.0, dim + 1.0)
    logits = np.sum(true_coefs * data, axis=-1)
    labels = dist.Bernoulli(logits=logits).sample(random.key(1))

    if callable_prior:

        def prior(name, shape):
            return dist.Cauchy() if name == "b" else dist.Normal()
    else:
        prior = {"w": dist.Normal(), "b": dist.Cauchy()}

    # Create a pre-initialized module for eager initialization
    rng_key = random.key(0)
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
        nn = random_nnx_module("nn", linear_module, prior, scope_divider=scope_divider)
        logits = nn(data).squeeze(-1)
        return numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=labels)

    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=2, num_samples=2, progress_bar=False)
    mcmc.run(random.key(0), data, labels)
    samples = mcmc.get_samples()
    assert f"nn{scope_divider}b" in samples
    assert f"nn{scope_divider}w" in samples


@pytest.mark.skipif(sys.version_info[:2] == (3, 9), reason="Skipping on Python 3.9")
@pytest.mark.parametrize(
    argnames="scope_divider",
    argvalues=["/", "|"],
    ids=["scope_divider=/", "scope_divider=|"],
)
def test_random_nnx_module_mcmc_sequence_params(scope_divider: str):
    from flax import nnx

    class MLP(nnx.Module):
        def __init__(self, din, dout, hidden_layers, *, rngs, activation=jax.nn.relu):
            self.activation = activation
            self.layers = nnx.List([])
            layer_dims = [din] + hidden_layers + [dout]
            for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
                self.layers.append(nnx.Linear(in_dim, out_dim, rngs=rngs))

        def __call__(self, x):
            for layer in self.layers[:-1]:
                x = self.activation(layer(x))
            return self.layers[-1](x)

    N, dim = 3000, 3
    data = random.normal(random.key(0), (N, dim))
    true_coefs = np.arange(1.0, dim + 1.0)
    logits = np.sum(true_coefs * data, axis=-1)
    labels = dist.Bernoulli(logits=logits).sample(random.key(1))

    rng_key = random.key(0)
    nn_module = MLP(
        din=dim, dout=1, hidden_layers=[8, 8], rngs=nnx.Rngs(params=rng_key)
    )

    def prior(name, shape):
        return dist.Cauchy() if name == "bias" else dist.Normal()

    def model(data, labels=None):
        # Use the pre-initialized module with eager initialization
        nn = random_nnx_module(
            "nn", nn_module, prior=prior, scope_divider=scope_divider
        )
        logits = nn(data).squeeze(-1)
        return numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=labels)

    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=1, num_samples=1, progress_bar=False)
    mcmc.run(random.key(0), data, labels)
    samples = mcmc.get_samples()

    # check both layers have parameters in the samples
    assert f"nn{scope_divider}layers.0.bias" in samples
    assert f"nn{scope_divider}layers.1.bias" in samples


@pytest.mark.parametrize("use_deterministic", [True, False])
def test_random_nnx_module_mcmc_with_mutable_state(use_deterministic):
    from flax import nnx

    class NNXModel(nnx.Module):
        def __init__(self):
            self.linear = nnx.Linear(10, 1, rngs=nnx.Rngs(0))
            self.mutable = nnx.Variable(0)

        def __call__(self, x):
            return self.linear(x)

    nn_module = NNXModel()

    def model(x, y=None):
        random_model = random_nnx_module("model", nn_module, dist.Normal(0, 1))
        pred = random_model(x)
        with numpyro.plate("plate", size=x.shape[0]):
            if use_deterministic:
                pred = numpyro.deterministic("pred", pred)
            numpyro.sample("obs", dist.Normal(pred, 1.0).to_event(1), obs=y)

    x = jax.random.uniform(jax.random.key(0), shape=(10, 10))
    y = jax.random.uniform(jax.random.key(0), shape=(10, 1))

    mcmc = MCMC(NUTS(model), num_warmup=5, num_samples=5, progress_bar=False)
    with jax.check_tracer_leaks(True):
        mcmc.run(jax.random.key(0), x, y)
    samples = mcmc.get_samples()
    assert "model/linear.kernel" in samples
    assert "model/linear.bias" in samples


@pytest.mark.skipif(sys.version_info[:2] == (3, 9), reason="Skipping on Python 3.9")
def test_eqx_module():
    import equinox as eqx

    X = np.arange(100).astype(np.float32)[None]
    Y = 2 * X + 2

    linear_module = eqx.nn.Linear(in_features=100, out_features=100, key=random.key(0))

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
            self.bn = make_batchnorm() if batchnorm else None
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
    net_module, eager_state = eqx.nn.make_with_state(Net)(key=random.key(0))
    x = dist.Normal(0, 1).expand([4, 3]).to_event(2).sample(random.key(0))

    def model():
        # Use the pre-initialized module
        nn = eqx_module("nn", net_module)
        mutable_holder = numpyro_mutable("nn$state", {"state": eager_state})

        batched_nn = jax.vmap(
            nn, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
        )
        y, new_state = batched_nn(x, mutable_holder["state"])
        mutable_holder["state"] = new_state

        numpyro.deterministic("y", y)

    with handlers.trace(model) as tr, handlers.seed(rng_seed=0):
        model()

    assert set(tr.keys()) == {"nn$params", "nn$state", "y"}
    assert tr["nn$state"]["type"] == "mutable"

    # test svi - trace error with AutoDelta
    guide = AutoDelta(model)
    svi = SVI(model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
    svi.run(random.key(100), 10)


@pytest.mark.skipif(sys.version_info[:2] == (3, 9), reason="Skipping on Python 3.9")
@pytest.mark.parametrize(
    argnames="scope_divider",
    argvalues=["/", "|"],
    ids=["scope_divider=/", "scope_divider=|"],
)
@pytest.mark.parametrize(
    argnames="callable_prior",
    argvalues=[True, False],
    ids=["callable_prior", "dict_prior"],
)
def test_random_eqx_module_mcmc(callable_prior, scope_divider: str):
    import equinox as eqx

    N, dim = 3000, 3
    data = random.normal(random.key(0), (N, dim))
    true_coefs = np.arange(1.0, dim + 1.0)
    logits = np.sum(true_coefs * data, axis=-1)
    labels = dist.Bernoulli(logits=logits).sample(random.key(1))

    if callable_prior:

        def prior(name, shape):
            return dist.Cauchy() if name == "bias" else dist.Normal()
    else:
        prior = {"weight": dist.Normal(), "bias": dist.Cauchy()}

    # Create a pre-initialized module for eager initialization
    rng_key = random.key(0)
    linear_module = eqx.nn.Linear(in_features=dim, out_features=1, key=rng_key)

    def model(data, labels=None):
        # Use the pre-initialized module with eager initialization
        nn = random_eqx_module(
            "nn", linear_module, prior=prior, scope_divider=scope_divider
        )
        logits = jax.vmap(nn)(data).squeeze(-1)
        return numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=labels)

    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=2, num_samples=2, progress_bar=False)
    mcmc.run(random.key(0), data, labels)
    samples = mcmc.get_samples()
    assert f"nn{scope_divider}bias" in samples
    assert f"nn{scope_divider}weight" in samples


@pytest.mark.skipif(sys.version_info[:2] == (3, 9), reason="Skipping on Python 3.9")
@pytest.mark.parametrize(
    argnames="scope_divider",
    argvalues=["/", "|"],
    ids=["scope_divider=/", "scope_divider=|"],
)
def test_random_eqx_module_mcmc_sequence_params(scope_divider: str):
    import equinox as eqx

    class MLP(eqx.Module):
        layers: list

        def __init__(
            self,
            in_size: int,
            out_size: int,
            hidden_layers: list[int],
            key: jax.random.key,
        ):
            keys = jax.random.split(key, len(hidden_layers))
            self.layers = []

            # Create all linear layers
            self.layers = []
            layer_dims = [in_size] + list(hidden_layers) + [out_size]
            for i, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
                self.layers.append(eqx.nn.Linear(in_dim, out_dim, key=keys[i]))

        def __call__(self, x):
            for layer in self.layers[:-1]:
                x = jax.nn.relu(layer(x))
            return self.layers[-1](x)  # Final layer, no activation

    N, dim = 3000, 3
    data = random.normal(random.key(0), (N, dim))
    true_coefs = np.arange(1.0, dim + 1.0)
    logits = np.sum(true_coefs * data, axis=-1)
    labels = dist.Bernoulli(logits=logits).sample(random.key(1))

    rng_key = random.key(0)
    nn_module = MLP(in_size=dim, out_size=1, hidden_layers=[8, 8], key=rng_key)

    def prior(name, shape):
        return dist.Cauchy() if name == "bias" else dist.Normal()

    def model(data, labels=None):
        # Use the pre-initialized module with eager initialization
        nn = random_eqx_module(
            "nn", nn_module, prior=prior, scope_divider=scope_divider
        )
        logits = jax.vmap(nn)(data).squeeze(-1)
        return numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=labels)

    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=1, num_samples=1, progress_bar=False)
    mcmc.run(random.key(0), data, labels)
    samples = mcmc.get_samples()

    # check both layers have parameters in the samples
    assert f"nn{scope_divider}layers[0].bias" in samples
    assert f"nn{scope_divider}layers[1].bias" in samples
