# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

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
    SafeRngs,
    _update_params,
    _update_state_with_params,
    flax_module,
    haiku_module,
    nnx_module,
    random_flax_module,
    random_haiku_module,
    random_nnx_module,
)
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta

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


def nnx_model_by_shape(x, y):
    from flax import nnx

    class Linear(nnx.Module):
        def __init__(self, din, dout, *, rngs):
            self.w = nnx.Param(jax.random.uniform(rngs.params(), (din, dout)))
            self.bias = nnx.Param(jnp.zeros((dout,)))

        def __call__(self, x):
            return x @ self.w + self.bias

    # Pass input_shape separately - it will be handled properly by nnx_module
    nn = nnx_module("nn", Linear, din=100, dout=100, input_shape=(100,))
    mean = nn(x)
    numpyro.sample("y", numpyro.distributions.Normal(mean, 0.1), obs=y)


def nnx_model_by_kwargs(x, y):
    from flax import nnx

    class Linear(nnx.Module):
        def __init__(self, din, dout, *, rngs):
            self.w = nnx.Param(jax.random.uniform(rngs.params(), (din, dout)))
            self.bias = nnx.Param(jnp.zeros((dout,)))

        def __call__(self, x):
            return x @ self.w + self.bias

    # Directly initialize with dimensions
    input_dim = x.shape[0]
    # Don't pass x directly to nnx_module's constructor
    nn = nnx_module("nn", Linear, din=input_dim, dout=100)
    mean = nn(x)
    numpyro.sample("y", numpyro.distributions.Normal(mean, 0.1), obs=y)


@pytest.mark.parametrize("init", ["shape", "kwargs"])
def test_nnx_module(init):
    X = np.arange(100).astype(np.float32)
    Y = 2 * X + 2

    if init == "shape":
        with handlers.trace() as nnx_tr, handlers.seed(rng_seed=1):
            nnx_model_by_shape(X, Y)
        assert "w" in nnx_tr["nn$params"]["value"]
        assert "bias" in nnx_tr["nn$params"]["value"]
        assert nnx_tr["nn$params"]["value"]["w"].shape == (100, 100)
        assert nnx_tr["nn$params"]["value"]["bias"].shape == (100,)

    elif init == "kwargs":
        with handlers.trace() as nnx_tr, handlers.seed(rng_seed=1):
            nnx_model_by_kwargs(X, Y)
        assert "w" in nnx_tr["nn$params"]["value"]
        assert "bias" in nnx_tr["nn$params"]["value"]
        assert nnx_tr["nn$params"]["value"]["w"].shape == (100, 100)
        assert nnx_tr["nn$params"]["value"]["bias"].shape == (100,)


@pytest.mark.parametrize("dropout", [True, False])
@pytest.mark.parametrize("batchnorm", [True, False])
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

    def model():
        apply_rng = ["dropout"] if dropout else None
        mutable = ["batch_stats"] if batchnorm else None

        # Pass input_shape separately to nnx_module
        nn = nnx_module(
            "nn", Net, apply_rng=apply_rng, mutable=mutable, input_shape=(4, 3)
        )

        x = numpyro.sample("x", dist.Normal(0, 1).expand([4, 3]).to_event(2))

        if dropout:
            y = nn(x, rngs={"dropout": numpyro.prng_key()})
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


@pytest.mark.parametrize("callable_prior", [True, False])
def test_random_nnx_module_mcmc(callable_prior):
    from flax import nnx

    class Linear(nnx.Module):
        def __init__(self, din, dout, *, rngs):
            self.w = nnx.Param(jax.random.uniform(rngs.params(), (din, dout)))
            self.bias = nnx.Param(jnp.zeros((dout,)))

        def __call__(self, x):
            return x @ self.w + self.bias

    N, dim = 3000, 3
    num_warmup, num_samples = (1000, 1000)
    data = random.normal(random.PRNGKey(0), (N, dim))
    true_coefs = np.arange(1.0, dim + 1.0)
    logits = np.sum(true_coefs * data, axis=-1)
    labels = dist.Bernoulli(logits=logits).sample(random.PRNGKey(1))

    if callable_prior:

        def prior(name, shape):
            return dist.Cauchy() if name == "bias" else dist.Normal()
    else:
        prior = {"bias": dist.Cauchy(), "w": dist.Normal()}

    def model(data, labels):
        # Pass input_shape separately
        nn = random_nnx_module("nn", Linear, prior, din=dim, dout=1, input_shape=(dim,))
        logits = nn(data).squeeze(-1)
        numpyro.sample("y", dist.Bernoulli(logits=logits), obs=labels)

    kernel = NUTS(model=model)
    mcmc = MCMC(
        kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=False
    )
    mcmc.run(random.PRNGKey(2), data, labels)
    mcmc.print_summary()
    samples = mcmc.get_samples()

    # Check that we have the expected parameter keys
    assert "nn/bias" in samples
    assert "nn/w" in samples
    # The nn$params key is also expected in the samples
    assert "nn$params" in samples
    assert_allclose(
        np.mean(samples["nn/w"].squeeze(-1), 0),
        true_coefs,
        atol=0.22,
    )


def test_nnx_cnn_module():
    """Test a convolutional neural network with NNX module in a NumPyro model."""
    from flax import nnx
    import jax.nn as nn

    # Define a CNN module
    class CNN(nnx.Module):
        def __init__(self, *, rngs):
            # Define convolutional layers
            self.conv1 = nnx.Conv(
                in_features=1,  # Input channels
                out_features=16,  # Output channels
                kernel_size=(3, 3),
                padding="SAME",
                rngs=rngs,
            )
            self.conv2 = nnx.Conv(
                in_features=16,  # Input channels from previous layer
                out_features=32,  # Output channels
                kernel_size=(3, 3),
                padding="SAME",
                rngs=rngs,
            )
            # Define linear layers
            self.linear1 = nnx.Linear(
                in_features=32 * 7 * 7, out_features=64, rngs=rngs
            )
            self.linear2 = nnx.Linear(in_features=64, out_features=10, rngs=rngs)
            # Batch normalization
            self.bn1 = nnx.BatchNorm(16, rngs=rngs)
            self.bn2 = nnx.BatchNorm(32, rngs=rngs)

        def __call__(self, x, *, rngs=None, training=True):
            # First conv block
            x = self.conv1(x)
            x = self.bn1(x, use_running_average=not training)
            x = nn.relu(x)
            x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))

            # Second conv block
            x = self.conv2(x)
            x = self.bn2(x, use_running_average=not training)
            x = nn.relu(x)
            x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))

            # Flatten and linear layers
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
            x = self.linear1(x)
            x = nn.relu(x)
            x = self.linear2(x)
            return x

    # Create a simple image classification model
    def model(images, labels=None):
        batch_size, height, width, channels = images.shape

        # Create the CNN module
        cnn = nnx_module(
            "cnn",
            CNN,
            mutable=["batch_stats"],
            input_shape=(batch_size, height, width, channels),
        )

        # Get logits from the CNN
        logits = cnn(images)

        # Use the mean of sequence outputs for classification
        mean_logits = jnp.mean(logits, axis=1)  # (batch_size,)

        # Sample from categorical distribution
        with numpyro.plate("data", batch_size):
            return numpyro.sample(
                "obs", dist.Categorical(logits=mean_logits), obs=labels
            )

    # Create dummy data
    batch_size, height, width, channels = 4, 28, 28, 1
    images = jnp.ones((batch_size, height, width, channels))
    labels = jnp.zeros(batch_size, dtype=jnp.int32)

    # Test the model
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        model(images, labels)

    # Check that parameters were created
    assert "cnn$params" in tr
    assert "cnn$state" in tr

    # Check parameter shapes
    params = tr["cnn$params"]["value"]
    # Check that key parameters exist
    assert "conv1.kernel" in params
    assert "conv2.kernel" in params
    assert "linear1.kernel" in params
    assert "linear2.kernel" in params
    assert "bn1.scale" in params
    assert "bn2.scale" in params

    # Test with SVI
    guide = AutoDelta(model)
    svi = SVI(model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
    _ = svi.run(random.PRNGKey(0), 2, images, labels)

    # Test with random module for MCMC
    def random_model(images, labels=None):
        batch_size, height, width, channels = images.shape

        # Define prior distributions
        prior = {
            "conv1.kernel": dist.Normal(0, 0.1),
            "conv1.bias": dist.Normal(0, 0.1),
            "conv2.kernel": dist.Normal(0, 0.1),
            "conv2.bias": dist.Normal(0, 0.1),
            "linear1.kernel": dist.Normal(0, 0.1),
            "linear1.bias": dist.Normal(0, 0.1),
            "linear2.kernel": dist.Normal(0, 0.1),
            "linear2.bias": dist.Normal(0, 0.1),
        }

        # Create random CNN module
        cnn = random_nnx_module(
            "cnn", CNN, prior, input_shape=(batch_size, height, width, channels)
        )

        # Get logits from the CNN
        logits = cnn(images)

        # Use the mean of sequence outputs for classification (no squeeze needed)
        mean_logits = jnp.mean(logits, axis=1)  # (batch_size,)

        # Sample from categorical distribution to match the model function
        with numpyro.plate("data", batch_size):
            return numpyro.sample(
                "obs", dist.Categorical(logits=mean_logits), obs=labels
            )

    # Test that the random model runs without errors
    with handlers.seed(rng_seed=0):
        # Just check that it runs without errors
        random_model(images, labels)

    # Test MCMC inference
    kernel = NUTS(model=random_model)
    mcmc = MCMC(kernel, num_warmup=2, num_samples=2, progress_bar=False)
    mcmc.run(random.PRNGKey(2), images, labels)

    # Check that we can access posterior samples
    samples = mcmc.get_samples()
    assert "cnn$params" in samples
    params = samples["cnn$params"]
    assert "conv1.kernel" in params
    assert "conv2.kernel" in params
    assert "linear1.kernel" in params
    assert "linear2.kernel" in params
    assert params["conv1.kernel"].shape[0] == 2  # num_samples
    assert params["conv2.kernel"].shape[0] == 2
    assert params["linear1.kernel"].shape[0] == 2
    assert params["linear2.kernel"].shape[0] == 2


def test_nnx_rnn_module():
    """Test a recurrent neural network with NNX module in a NumPyro model."""
    from flax import nnx

    # Define a simple RNN module
    class SimpleRNN(nnx.Module):
        def __init__(self, input_size, hidden_size, *, rngs):
            self.input_size = input_size
            self.hidden_size = hidden_size

            # Input-to-hidden and hidden-to-hidden weights
            self.W_ih = nnx.Param(
                jax.random.normal(rngs.params(), (input_size, hidden_size)) * 0.1
            )
            self.W_hh = nnx.Param(
                jax.random.normal(rngs.params(), (hidden_size, hidden_size)) * 0.1
            )
            self.b_ih = nnx.Param(jnp.zeros((hidden_size,)))
            self.b_hh = nnx.Param(jnp.zeros((hidden_size,)))

            # Output projection
            self.output_proj = nnx.Linear(
                in_features=hidden_size, out_features=1, rngs=rngs
            )

        def __call__(self, x):
            # x shape: (batch_size, seq_len, input_size)
            batch_size, seq_len, _ = x.shape

            # Initialize hidden state
            h = jnp.zeros((batch_size, self.hidden_size))

            # Process sequence
            for t in range(seq_len):
                # Get input at current timestep
                x_t = x[:, t, :]

                # Update hidden state
                h = jnp.tanh(x_t @ self.W_ih + self.b_ih + h @ self.W_hh + self.b_hh)

            # Project final hidden state to output
            return self.output_proj(h)

    # Define a time series prediction model
    def model(sequences, targets=None):
        batch_size, seq_len, input_size = sequences.shape

        # Create the RNN module
        rnn = nnx_module(
            "rnn",
            SimpleRNN,
            input_size=input_size,
            hidden_size=32,
            input_shape=(batch_size, seq_len, input_size),
        )

        # Get predictions from the RNN (using only the final output)
        predictions = rnn(sequences)  # (batch_size, 1)

        # Sample observations
        with numpyro.plate("batch", batch_size):
            return numpyro.sample(
                "obs", dist.Normal(predictions.squeeze(-1), 0.1), obs=targets
            )

    # Create dummy data
    batch_size, seq_len, input_size = 4, 10, 5
    sequences = jnp.ones((batch_size, seq_len, input_size))
    targets = jnp.zeros(batch_size)  # Only one target per sequence

    # Test the model
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        model(sequences, targets)

    # Check that parameters were created
    assert "rnn$params" in tr
    params = tr["rnn$params"]["value"]

    # Check parameter shapes
    assert "W_ih" in params
    assert "W_hh" in params
    assert params["W_ih"].shape == (input_size, 32)
    assert params["W_hh"].shape == (32, 32)

    # Test with SVI
    guide = AutoDelta(model)
    svi = SVI(model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
    _ = svi.run(random.PRNGKey(0), 2, sequences, targets)

    # Test with random module for MCMC
    def random_model(sequences, targets=None):
        batch_size, seq_len, input_size = sequences.shape

        # Define prior distributions
        prior = {
            "W_ih": dist.Normal(0, 0.1),
            "W_hh": dist.Normal(0, 0.1),
            "b_ih": dist.Normal(0, 0.01),
            "b_hh": dist.Normal(0, 0.01),
            "output_proj.kernel": dist.Normal(0, 0.1),
            "output_proj.bias": dist.Normal(0, 0.01),
        }

        # Create random RNN module
        rnn = random_nnx_module(
            "rnn",
            SimpleRNN,
            prior,
            input_size=input_size,
            hidden_size=32,
            input_shape=(batch_size, seq_len, input_size),
        )

        # Get predictions from the RNN
        predictions = rnn(sequences)

        # Sample observations
        with numpyro.plate("batch", batch_size):
            return numpyro.sample(
                "obs", dist.Normal(predictions.squeeze(-1), 0.1), obs=targets
            )

    # Test that the random model runs without errors
    with handlers.seed(rng_seed=0):
        # Just check that it runs without errors
        random_model(sequences, targets)

    # Test MCMC inference
    kernel = NUTS(model=random_model)
    mcmc = MCMC(kernel, num_warmup=2, num_samples=2, progress_bar=False)
    mcmc.run(random.PRNGKey(2), sequences, targets)

    # Check that we can access posterior samples
    samples = mcmc.get_samples()
    assert "rnn$params" in samples
    params = samples["rnn$params"]
    assert "W_ih" in params
    assert "W_hh" in params
    assert params["W_ih"].shape == (2, input_size, 32)  # (num_samples, *param_shape)
    assert params["W_hh"].shape == (2, 32, 32)


def test_nnx_transformer_module():
    """Test a transformer-like architecture with NNX module in a NumPyro model."""
    from flax import nnx
    import jax.nn as nn

    # Define a simple transformer module with a flatter structure
    class SimpleTransformer(nnx.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, *, rngs):
            # Linear projections for attention
            self.query = nnx.Linear(
                in_features=input_dim, out_features=hidden_dim, rngs=rngs
            )
            self.key = nnx.Linear(
                in_features=input_dim, out_features=hidden_dim, rngs=rngs
            )
            self.value = nnx.Linear(
                in_features=input_dim, out_features=hidden_dim, rngs=rngs
            )

            # Output projections
            self.attention_output = nnx.Linear(
                in_features=hidden_dim, out_features=input_dim, rngs=rngs
            )
            self.ffn1 = nnx.Linear(
                in_features=input_dim, out_features=hidden_dim, rngs=rngs
            )
            self.ffn2 = nnx.Linear(
                in_features=hidden_dim, out_features=input_dim, rngs=rngs
            )

            # Layer normalization
            self.norm1 = nnx.LayerNorm(input_dim, rngs=rngs)
            self.norm2 = nnx.LayerNorm(input_dim, rngs=rngs)

            # Final output projection
            self.output = nnx.Linear(
                in_features=input_dim, out_features=output_dim, rngs=rngs
            )

            # Store dimensions
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim

        def __call__(self, x):
            batch_size, seq_len, _ = x.shape

            # Self-attention block
            residual = x
            x_norm = self.norm1(x)

            # Compute query, key, value
            q = self.query(x_norm)
            k = self.key(x_norm)
            v = self.value(x_norm)

            # Simple attention mechanism
            attention_scores = jnp.matmul(q, jnp.transpose(k, (0, 2, 1))) / jnp.sqrt(
                self.hidden_dim
            )
            attention_weights = nn.softmax(attention_scores, axis=-1)
            attention_output = jnp.matmul(attention_weights, v)

            # Apply output projection and residual connection
            x = residual + self.attention_output(attention_output)

            # Feed-forward block
            residual = x
            x_norm = self.norm2(x)
            x = residual + self.ffn2(nn.gelu(self.ffn1(x_norm)))

            # Final output projection
            return self.output(x)

    # Define a sequence classification model
    def model(sequences, labels=None):
        batch_size, seq_len, input_dim = sequences.shape

        # Create the transformer module
        transformer = nnx_module(
            "transformer",
            SimpleTransformer,
            input_dim=input_dim,
            hidden_dim=input_dim * 2,
            output_dim=1,
            input_shape=(batch_size, seq_len, input_dim),
        )

        # Get logits from the transformer
        logits = transformer(sequences)

        # Use the mean of sequence outputs for classification (no squeeze needed)
        mean_logits = jnp.mean(logits, axis=1)  # (batch_size,)

        # Sample from categorical distribution to match the model function
        with numpyro.plate("data", batch_size):
            return numpyro.sample(
                "obs", dist.Categorical(logits=mean_logits), obs=labels
            )

    # Create dummy data
    batch_size, seq_len, input_dim = 4, 16, 32
    sequences = jnp.ones((batch_size, seq_len, input_dim))
    labels = jnp.zeros(batch_size, dtype=jnp.int32)

    # Test the model
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        model(sequences, labels)

    # Check that parameters were created
    assert "transformer$params" in tr
    params = tr["transformer$params"]["value"]

    # Check parameter structure - should have flattened parameters
    assert "query.kernel" in params
    assert "query.bias" in params
    assert "key.kernel" in params
    assert "key.bias" in params
    assert "value.kernel" in params
    assert "value.bias" in params
    assert "attention_output.kernel" in params
    assert "attention_output.bias" in params
    assert "ffn1.kernel" in params
    assert "ffn1.bias" in params
    assert "ffn2.kernel" in params
    assert "ffn2.bias" in params
    assert "norm1.scale" in params
    assert "norm1.bias" in params
    assert "norm2.scale" in params
    assert "norm2.bias" in params
    assert "output.kernel" in params
    assert "output.bias" in params

    # Test with SVI
    guide = AutoDelta(model)
    svi = SVI(model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
    _ = svi.run(random.PRNGKey(0), 2, sequences, labels)

    # Test with random module for MCMC
    def random_model(sequences, labels=None):
        batch_size, seq_len, input_dim = sequences.shape

        # Define prior distributions
        prior = {
            "query.kernel": dist.Normal(0, 0.1),
            "query.bias": dist.Normal(0, 0.1),
            "key.kernel": dist.Normal(0, 0.1),
            "key.bias": dist.Normal(0, 0.1),
            "value.kernel": dist.Normal(0, 0.1),
            "value.bias": dist.Normal(0, 0.1),
            "attention_output.kernel": dist.Normal(0, 0.1),
            "attention_output.bias": dist.Normal(0, 0.1),
            "ffn1.kernel": dist.Normal(0, 0.1),
            "ffn1.bias": dist.Normal(0, 0.1),
            "ffn2.kernel": dist.Normal(0, 0.1),
            "ffn2.bias": dist.Normal(0, 0.1),
            "norm1.scale": dist.Normal(0, 0.1),
            "norm1.bias": dist.Normal(0, 0.1),
            "norm2.scale": dist.Normal(0, 0.1),
            "norm2.bias": dist.Normal(0, 0.1),
            "output.kernel": dist.Normal(0, 0.1),
            "output.bias": dist.Normal(0, 0.1),
        }

        # Create random transformer module
        transformer = random_nnx_module(
            "transformer",
            SimpleTransformer,
            prior,
            input_dim=input_dim,
            hidden_dim=input_dim * 2,
            output_dim=1,
            input_shape=(batch_size, seq_len, input_dim),
        )

        # Get logits from the transformer
        logits = transformer(sequences)

        # Use the mean of sequence outputs for classification (no squeeze needed)
        mean_logits = jnp.mean(logits, axis=1)  # (batch_size,)

        # Sample from categorical distribution to match the model function
        with numpyro.plate("data", batch_size):
            return numpyro.sample(
                "obs", dist.Categorical(logits=mean_logits), obs=labels
            )

    # Test that the random model runs without errors
    with handlers.seed(rng_seed=0):
        # Just check that it runs without errors
        random_model(sequences, labels)

    # Test MCMC inference
    kernel = NUTS(model=random_model)
    mcmc = MCMC(kernel, num_warmup=2, num_samples=2, progress_bar=False)
    mcmc.run(random.PRNGKey(2), sequences, labels)

    # Check that we can access posterior samples
    samples = mcmc.get_samples()
    assert "transformer$params" in samples
    params = samples["transformer$params"]
    assert "query.kernel" in params
    assert "key.kernel" in params
    assert "value.kernel" in params
    assert params["query.kernel"].shape == (
        2,
        input_dim,
        input_dim * 2,
    )  # (num_samples, *param_shape)
    assert params["key.kernel"].shape == (2, input_dim, input_dim * 2)
    assert params["value.kernel"].shape == (2, input_dim, input_dim * 2)


def test_update_state_with_params():
    """Test the _update_state_with_params helper function for updating state with parameters."""
    from flax import nnx
    import jax.numpy as jnp

    # Create a simple module with parameters
    class SimpleModule(nnx.Module):
        def __init__(self, *, rngs):
            self.weight = nnx.Param(jnp.zeros((2, 2)))
            self.bias = nnx.Param(jnp.zeros(2))
            self.nested = NestedModule(rngs=rngs)

        def __call__(self, x):
            return x @ self.weight + self.bias

    class NestedModule(nnx.Module):
        def __init__(self, *, rngs):
            self.weight = nnx.Param(jnp.zeros((2, 2)))
            self.bias = nnx.Param(jnp.zeros(2))
            self.deep_nested = DeepNestedModule(rngs=rngs)

        def __call__(self, x):
            return x @ self.weight + self.bias

    class DeepNestedModule(nnx.Module):
        def __init__(self, *, rngs):
            self.weight = nnx.Param(jnp.zeros((2, 2)))
            self.bias = nnx.Param(jnp.zeros(2))

        def __call__(self, x):
            return x @ self.weight + self.bias

    # Create a module and get its state
    module = SimpleModule(rngs=nnx.Rngs(0))
    _, state = nnx.split(module)

    # Test Case 1: Basic parameter update
    params = {
        "weight": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        "bias": jnp.array([0.1, 0.2]),
        "nested.weight": jnp.array([[5.0, 6.0], [7.0, 8.0]]),
        "nested.bias": jnp.array([0.3, 0.4]),
    }

    # Update the state with the parameters
    updated_state = _update_state_with_params(state, params)

    # Check that the state was updated correctly
    assert jnp.array_equal(updated_state["weight"].value, params["weight"])
    assert jnp.array_equal(updated_state["bias"].value, params["bias"])
    assert jnp.array_equal(
        updated_state["nested"]["weight"].value, params["nested.weight"]
    )
    assert jnp.array_equal(updated_state["nested"]["bias"].value, params["nested.bias"])

    # Test Case 2: Deep nested parameters
    deep_params = {
        "nested.deep_nested.weight": jnp.array([[9.0, 10.0], [11.0, 12.0]]),
        "nested.deep_nested.bias": jnp.array([0.5, 0.6]),
    }

    updated_state = _update_state_with_params(state, deep_params)

    # Check deep nested parameters
    assert jnp.array_equal(
        updated_state["nested"]["deep_nested"]["weight"].value,
        deep_params["nested.deep_nested.weight"],
    )
    assert jnp.array_equal(
        updated_state["nested"]["deep_nested"]["bias"].value,
        deep_params["nested.deep_nested.bias"],
    )

    # Test Case 3: Mixed parameter update (flat and nested)
    mixed_params = {
        "weight": jnp.array([[13.0, 14.0], [15.0, 16.0]]),
        "nested.deep_nested.bias": jnp.array([0.7, 0.8]),
    }

    updated_state = _update_state_with_params(state, mixed_params)

    # Check mixed parameters
    assert jnp.array_equal(updated_state["weight"].value, mixed_params["weight"])
    assert jnp.array_equal(
        updated_state["nested"]["deep_nested"]["bias"].value,
        mixed_params["nested.deep_nested.bias"],
    )

    # Test Case 4: Non-existent parameters (should not raise errors)
    nonexistent_params = {
        "nonexistent": jnp.array([1.0, 2.0]),
        "nested.nonexistent": jnp.array([3.0, 4.0]),
        "nonexistent.nested": jnp.array([5.0, 6.0]),
    }

    # This should not raise errors
    updated_state = _update_state_with_params(state, nonexistent_params)

    # Original parameters should remain unchanged
    assert jnp.array_equal(updated_state["weight"].value, mixed_params["weight"])
    assert jnp.array_equal(
        updated_state["nested"]["deep_nested"]["bias"].value,
        mixed_params["nested.deep_nested.bias"],
    )

    # Test Case 5: Empty parameters dictionary
    empty_params = {}

    updated_state = _update_state_with_params(state, empty_params)

    # State should remain unchanged
    assert jnp.array_equal(updated_state["weight"].value, mixed_params["weight"])
    assert jnp.array_equal(
        updated_state["nested"]["deep_nested"]["bias"].value,
        mixed_params["nested.deep_nested.bias"],
    )


def test_nnx_module_model_surgery():
    """Test the model surgery approach used in the nnx_module function."""
    from flax import nnx
    import jax.numpy as jnp

    import numpyro
    from numpyro import handlers
    from numpyro.contrib.module import nnx_module

    # Create a simple module with parameters
    class SimpleModule(nnx.Module):
        def __init__(self, din, dout, *, rngs):
            self.weight = nnx.Param(jnp.zeros((din, dout)))
            self.bias = nnx.Param(jnp.zeros(dout))

        def __call__(self, x):
            return x @ self.weight + self.bias

    # Define a model that uses the module
    def model(x, y=None):
        # Create the module
        nn = nnx_module("nn", SimpleModule, din=3, dout=1)

        # Apply the module to get predictions
        mean = nn(x)

        # Sample from a normal distribution
        with numpyro.plate("data", x.shape[0]):
            return numpyro.sample("y", numpyro.distributions.Normal(mean, 0.1), obs=y)

    # Create some test data
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = jnp.array([1.0, 2.0])

    # Run the model once to initialize parameters
    with handlers.seed(rng_seed=0):
        handlers.trace(model).get_trace(x, y)

    # Run the model again to check that parameters are reused
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        model(x, y)

    # Check that parameters exist and have the right shape
    assert "nn$params" in tr
    params = tr["nn$params"]["value"]
    assert "weight" in params
    assert "bias" in params
    assert params["weight"].shape == (3, 1)
    assert params["bias"].shape == (1,)

    # Now modify the parameters and check that they're used correctly
    modified_params = {
        "weight": jnp.array([[1.0], [2.0], [3.0]]),
        "bias": jnp.array([0.5]),
    }

    # Create a model with modified parameters
    def modified_model(x):
        with handlers.substitute(data={"nn$params": modified_params}):
            nn = nnx_module("nn", SimpleModule, din=3, dout=1)
            return nn(x)

    # Run the modified model
    with handlers.seed(rng_seed=0):
        result = modified_model(x)

    # Check that the result matches what we expect with the modified parameters
    expected = x @ modified_params["weight"] + modified_params["bias"]
    assert jnp.allclose(result, expected)


def test_init_and_access():
    """Test initialization and access of SafeRngs."""
    # Create a test key
    key = jax.random.PRNGKey(0)

    # Initialize with a single key
    rngs = SafeRngs(params=key)
    assert rngs["params"] is key
    assert rngs.params() is key

    # Initialize with multiple keys
    key2 = jax.random.PRNGKey(1)
    rngs = SafeRngs(params=key, dropout=key2)
    assert rngs["params"] is key
    assert rngs["dropout"] is key2
    assert rngs.params() is key

    # Test non-existent key
    assert rngs["non_existent"] is None
