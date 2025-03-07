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
            # Handle different Python versions by accessing the value attribute if needed
            w_val = self.w.value if hasattr(self.w, "value") else self.w
            bias_val = self.bias.value if hasattr(self.bias, "value") else self.bias
            return x @ w_val + bias_val

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
            # Handle different Python versions by accessing the value attribute if needed
            w_val = self.w.value if hasattr(self.w, "value") else self.w
            bias_val = self.bias.value if hasattr(self.bias, "value") else self.bias
            return x @ w_val + bias_val

    # Directly initialize with dimensions
    input_dim = x.shape[0]
    # Don't pass x directly to nnx_module's constructor
    nn = nnx_module("nn", Linear, din=input_dim, dout=100)
    mean = nn(x)
    numpyro.sample("y", numpyro.distributions.Normal(mean, 0.1), obs=y)


def test_nnx_module():
    from flax import nnx

    X = np.arange(100).astype(np.float32)
    Y = 2 * X + 2

    class Linear(nnx.Module):
        def __init__(self, din, dout, *, rngs):
            self.w = nnx.Param(jax.random.uniform(rngs.params(), (din, dout)))
            self.bias = nnx.Param(jnp.zeros((dout,)))

        def __call__(self, x):
            # Handle different Python versions by accessing the value attribute if needed
            w_val = self.w.value if hasattr(self.w, "value") else self.w
            bias_val = self.bias.value if hasattr(self.bias, "value") else self.bias
            return x @ w_val + bias_val

    # Eager initialization of the Linear module outside the model
    rng_key = random.PRNGKey(1)
    linear_module = Linear(din=100, dout=100, rngs=nnx.Rngs(params=rng_key))

    # Extract parameters and state for inspection
    _, params_state = nnx.split(linear_module, nnx.Param)
    params_dict = {
        ".".join(str(p) for p in path): param.value
        for path, param in dict(params_state.flat_state()).items()
    }

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

    # Eager initialization of the Net module outside the model
    rng_key = random.PRNGKey(0)
    net_module = Net(rngs=nnx.Rngs(params=rng_key))

    # Extract parameters and state for inspection
    _, state = nnx.split(net_module)

    def model():
        # Use the pre-initialized module
        nn = nnx_module("nn", net_module)

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
            self.b = nnx.Param(jnp.zeros((dout,)))

        def __call__(self, x):
            # Handle different Python versions by accessing the value attribute if needed
            w_val = self.w.value if hasattr(self.w, "value") else self.w
            b_val = self.b.value if hasattr(self.b, "value") else self.b
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
    params_dict = {
        ".".join(str(p) for p in path): param.value
        for path, param in dict(params_state.flat_state()).items()
    }

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
    assert "nn$params" in samples
    assert "w" in samples["nn$params"]
    assert "b" in samples["nn$params"]


def test_nnx_transformer_module():
    """Test a transformer-like architecture with NNX module in a NumPyro model."""
    from flax import nnx
    import jax.nn as nn

    # Create dummy data
    batch_size, seq_len, input_dim = 4, 16, 32
    sequences = jnp.ones((batch_size, seq_len, input_dim))
    labels = jnp.zeros(batch_size, dtype=jnp.int32)

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

    # Eager initialization of the transformer module outside the model
    rng_key = random.PRNGKey(0)
    transformer_module = SimpleTransformer(
        input_dim=input_dim,
        hidden_dim=input_dim * 2,
        output_dim=1,
        rngs=nnx.Rngs(params=rng_key),
    )

    # Define a sequence classification model
    def model(sequences, labels=None):
        batch_size = sequences.shape[0]

        # Use the pre-initialized transformer module
        transformer = nnx_module("transformer", transformer_module)

        # Get logits from the transformer
        logits = transformer(sequences)

        # Use the mean of sequence outputs for classification
        mean_logits = jnp.mean(logits, axis=1)  # (batch_size,)

        # Sample from categorical distribution
        with numpyro.plate("data", batch_size):
            return numpyro.sample(
                "obs", dist.Categorical(logits=mean_logits), obs=labels
            )

    # Test the model
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        model(sequences, labels)

    # Check that parameters were created
    assert "transformer$params" in tr
    params = tr["transformer$params"]["value"]
    assert "query.kernel" in params
    assert "key.kernel" in params
    assert "value.kernel" in params

    # Test with SVI
    guide = AutoDelta(model)
    svi = SVI(model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
    _ = svi.run(random.PRNGKey(0), 2, sequences, labels)

    # Define prior distributions for random module
    prior = {
        "query.kernel": dist.Normal(0, 0.1),
        "query.bias": dist.Normal(0, 0.01),
        "key.kernel": dist.Normal(0, 0.1),
        "key.bias": dist.Normal(0, 0.01),
        "value.kernel": dist.Normal(0, 0.1),
        "value.bias": dist.Normal(0, 0.01),
    }

    # Test with random module for MCMC
    def random_model(sequences, labels=None):
        batch_size = sequences.shape[0]

        # Create random transformer module using the pre-initialized module
        transformer = random_nnx_module("transformer", transformer_module, prior)

        # Get logits from the transformer
        logits = transformer(sequences)

        # Use the mean of sequence outputs for classification
        mean_logits = jnp.mean(logits, axis=1)

        # Sample from categorical distribution
        with numpyro.plate("data", batch_size):
            return numpyro.sample(
                "obs", dist.Categorical(logits=mean_logits), obs=labels
            )

    # Test that the random model runs without errors
    with handlers.seed(rng_seed=0):
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
    assert params["query.kernel"].shape[0] == 2  # num_samples
    assert params["key.kernel"].shape[0] == 2
    assert params["value.kernel"].shape[0] == 2


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


def test_nnx_cnn_module():
    """Test a convolutional neural network with NNX module in a NumPyro model."""
    from flax import nnx
    import jax.nn as nn

    # Create dummy data
    batch_size, height, width, channels = 4, 28, 28, 1
    images = jnp.ones((batch_size, height, width, channels))
    labels = jnp.zeros(batch_size, dtype=jnp.int32)

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

    # Eager initialization of the CNN module outside the model
    rng_key = random.PRNGKey(0)
    cnn_module = CNN(rngs=nnx.Rngs(params=rng_key))

    # Create a simple image classification model
    def model(images, labels=None):
        batch_size = images.shape[0]

        # Use the pre-initialized CNN module
        cnn = nnx_module("cnn", cnn_module)

        # Get logits from the CNN
        logits = cnn(images)

        # Use the mean of sequence outputs for classification
        mean_logits = jnp.mean(logits, axis=1)  # (batch_size,)

        # Sample from categorical distribution
        with numpyro.plate("data", batch_size):
            return numpyro.sample(
                "obs", dist.Categorical(logits=mean_logits), obs=labels
            )

    # Test the model
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        model(images, labels)

    # Check that parameters were created
    assert "cnn$params" in tr
    assert "cnn$state" in tr
    assert tr["cnn$state"]["type"] == "mutable"

    # Test with SVI
    guide = AutoDelta(model)
    svi = SVI(model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
    _ = svi.run(random.PRNGKey(0), 2, images, labels)

    # Define prior distributions for random module
    prior = {
        "conv1.kernel": dist.Normal(0, 0.1),
        "conv1.bias": dist.Normal(0, 0.01),
        "conv2.kernel": dist.Normal(0, 0.1),
        "conv2.bias": dist.Normal(0, 0.01),
        "linear1.kernel": dist.Normal(0, 0.1),
        "linear1.bias": dist.Normal(0, 0.01),
        "linear2.kernel": dist.Normal(0, 0.1),
        "linear2.bias": dist.Normal(0, 0.01),
        "bn1.scale": dist.Normal(1, 0.1),
        "bn1.bias": dist.Normal(0, 0.1),
        "bn2.scale": dist.Normal(1, 0.1),
        "bn2.bias": dist.Normal(0, 0.1),
    }

    # Test with random module for MCMC
    def random_model(images, labels=None):
        batch_size = images.shape[0]

        # Create random CNN module using the pre-initialized module
        cnn = random_nnx_module("cnn", cnn_module, prior, mutable=["batch_stats"])

        # Get logits from the CNN
        logits = cnn(images)

        # Use the mean of sequence outputs for classification
        mean_logits = jnp.mean(logits, axis=1)

        # Sample from categorical distribution
        with numpyro.plate("data", batch_size):
            return numpyro.sample(
                "obs", dist.Categorical(logits=mean_logits), obs=labels
            )

    # Test that the random model runs without errors
    with handlers.seed(rng_seed=0):
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
    assert params["conv1.kernel"].shape[0] == 2  # num_samples
    assert params["conv2.kernel"].shape[0] == 2
