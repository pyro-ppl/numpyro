# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from flax import linen as nn
from jax import numpy as jnp

import numpyro
from numpyro.contrib.module import flax_module
import numpyro.distributions as dist


def cross_entropy_loss(y_pred, y):
    log_p = jnp.log(y_pred)
    log_not_p = jnp.log1p(-y_pred)
    return -y * log_p - (1.0 - y) * log_not_p


class BaselineNet(nn.Module):
    hidden_size: int = 512

    @nn.compact
    def __call__(self, x):
        batch_size, _, _ = x.shape
        y_hat = nn.relu(nn.Dense(self.hidden_size)(x.reshape(-1, 196)))
        y_hat = nn.relu(nn.Dense(self.hidden_size)(y_hat))
        y_hat = nn.Dense(784)(y_hat)
        y_hat = nn.sigmoid(y_hat).reshape((-1, 28, 28))
        return y_hat


class Encoder(nn.Module):
    hidden_size: int = 512
    latent_dim: int = 256

    @nn.compact
    def __call__(self, x, y):
        z = jnp.concatenate([x.reshape(-1, 196), y.reshape(-1, 784)], axis=-1)
        hidden = nn.relu(nn.Dense(self.hidden_size)(z))
        hidden = nn.relu(nn.Dense(self.hidden_size)(hidden))
        z_loc = nn.Dense(self.latent_dim)(hidden)
        z_scale = jnp.exp(nn.Dense(self.latent_dim)(hidden))
        return z_loc, z_scale


class Decoder(nn.Module):
    hidden_size: int = 512
    latent_dim: int = 256

    @nn.compact
    def __call__(self, z):
        y_hat = nn.relu(nn.Dense(self.hidden_size)(z))
        y_hat = nn.relu(nn.Dense(self.hidden_size)(y_hat))
        y_hat = nn.Dense(784)(y_hat)
        y_hat = nn.sigmoid(y_hat).reshape((-1, 28, 28))
        return y_hat


def cvae_model(x, y=None):
    baseline_net = flax_module(
        "baseline", BaselineNet(), x=jnp.ones((1, 14, 14), dtype=jnp.float32)
    )
    prior_net = flax_module(
        "prior_net",
        Encoder(),
        x=jnp.ones((1, 14, 14), dtype=jnp.float32),
        y=jnp.ones((1, 28, 28), dtype=jnp.float32),
    )
    generation_net = flax_module(
        "generation_net", Decoder(), z=jnp.ones((1, 256), dtype=jnp.float32)
    )

    y_hat = baseline_net(x)
    z_loc, z_scale = prior_net(x, y_hat)
    z = numpyro.sample("z", dist.Normal(z_loc, z_scale))
    loc = generation_net(z)

    if y is None:
        numpyro.deterministic("y", loc)
    else:
        numpyro.sample("y", dist.Bernoulli(loc), obs=y)
    return loc


def cvae_guide(x, y=None):
    recognition_net = flax_module(
        "recognition_net",
        Encoder(),
        x=jnp.ones((1, 14, 14), dtype=jnp.float32),
        y=jnp.ones((1, 28, 28), dtype=jnp.float32),
    )
    loc, scale = recognition_net(x, y)
    numpyro.sample("z", dist.Normal(loc, scale))
