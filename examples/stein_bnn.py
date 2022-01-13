# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Bayesian Neural Network with SteinVI
=============================================
We demonstrate how to use SteinVI to predict housing prices using a BNN for the Boston Housing prices dataset
from the UCI regression benchmarks.
"""

import argparse
from collections import namedtuple
import datetime
from functools import partial
from time import time

from sklearn.model_selection import train_test_split

from jax import random
import jax.numpy as jnp

import numpyro
from numpyro.contrib.einstein import RBFKernel, SteinVI
from numpyro.distributions import Gamma, Normal
from numpyro.examples.datasets import BOSTON_HOUSING, load_dataset
from numpyro.infer import Predictive, Trace_ELBO, init_to_uniform
from numpyro.infer.autoguide import AutoDelta
from numpyro.optim import Adagrad

DataState = namedtuple("data", ["xtr", "xte", "ytr", "yte"])


def load_data() -> DataState:
    _, fetch = load_dataset(BOSTON_HOUSING, shuffle=False)
    x, y = fetch()
    xtr, xte, ytr, yte = train_test_split(x, y, train_size=0.90)

    return DataState(*map(partial(jnp.array, dtype=float), (xtr, xte, ytr, yte)))


def normalize(val, mean=None, std=None):
    """Normalize data to zero mean, unit variance"""
    if mean is None and std is None:
        # Only use training data to estimate mean and std.
        std = jnp.std(val, 0, keepdims=True)
        std = jnp.where(std == 0, 1.0, std)
        mean = jnp.mean(val, 0, keepdims=True)
    return (val - mean) / std, mean, std


def model(x, y=None, hidden_dim=50, subsample_size=100):
    """BNN described in section 5 of [1].

    **References:**
        1. *Stein variational gradient descent: A general purpose bayesian inference algorithm*
        Qiang Liu and Dilin Wang (2016).
    """

    prec_nn = numpyro.sample(
        "prec_nn", Gamma(1.0, 0.1)
    )  # hyper prior for precision of nn weights and biases

    n, m = x.shape

    b1 = numpyro.sample(
        "nn_b1",
        Normal(
            jnp.zeros(
                hidden_dim,
            ),
            1.0 / prec_nn,
        ),
    )  # prior l1 bias term
    b2 = numpyro.sample("nn_b2", Normal(0.0, 1.0 / prec_nn))  # prior output bias term

    w1 = numpyro.sample(
        "nn_w1", Normal(jnp.zeros((m, hidden_dim)), 1.0 / prec_nn)
    )  # prior l1 weights
    w2 = numpyro.sample(
        "nn_w2", Normal(jnp.zeros(hidden_dim), 1.0 / prec_nn)
    )  # prior output weights

    prec_obs = numpyro.sample(
        "prec_obs", Gamma(1.0, 0.1)
    )  # precision prior on observations
    with numpyro.plate(
        "data",
        x.shape[0],
        subsample_size=subsample_size,
        subsample_scale=subsample_size,  # scale up the subsample factor
        dim=-1,
    ):
        batch_x = numpyro.subsample(x, event_dim=1)
        if y is not None:
            batch_y = numpyro.subsample(y, event_dim=0)
        else:
            batch_y = y

        numpyro.sample(
            "y",
            Normal(
                jnp.maximum(batch_x @ w1 + b1, 0) @ w2 + b2, 1.0 / prec_obs
            ),  # 1 hidden layer with ReLU activation
            obs=batch_y,
        )


def main(args):
    data = load_data()

    inf_key, pred_key, data_key = random.split(random.PRNGKey(args.rng_key), 3)
    x, xtr_mean, xtr_std = normalize(data.xtr)
    y, ytr_mean, ytr_std = normalize(data.ytr)

    rng_key, inf_key = random.split(inf_key)

    stein = SteinVI(
        model,
        AutoDelta(model),
        Adagrad(0.05),
        Trace_ELBO(num_particles=20),
        RBFKernel(),
        init_strategy=partial(init_to_uniform, radius=0.1),
        repulsion_temperature=args.repulsion,
        num_particles=args.num_particles,
    )
    start = time()

    # use keyword params for static (shape etc.)!
    result = stein.run(
        rng_key,
        args.max_iter,
        x,
        y,
        hidden_dim=args.hidden_dim,
        subsample_size=args.subsample_size,
        progress_bar=args.progress_bar,
    )
    time_taken = time() - start

    pred = Predictive(
        model,
        guide=stein.guide,
        params=stein.get_params(result.state),
        num_samples=1,
        num_particles=args.num_particles,
    )
    xte, _, _ = normalize(data.xte, xtr_mean, xtr_std)
    preds = pred(pred_key, xte, subsample_size=xte.shape[0])["y"].reshape(
        -1, xte.shape[0]
    )

    y_pred = jnp.mean(preds, 0) * ytr_std + ytr_mean

    rmse = jnp.sqrt(jnp.mean((y_pred - data.yte) ** 2))

    print(fr"Time taken: {datetime.timedelta(seconds=int(time_taken))}")
    print(fr"RMSE: {rmse}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample-size", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--repulsion", type=float, default=1.0)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--num-particles", type=int, default=100)
    parser.add_argument("--progress-bar", type=bool, default=True)
    parser.add_argument("--rng-key", type=int, default=142)
    parser.add_argument("--device", default="cpu", choices=["gpu", "cpu"])
    parser.add_argument("--hidden-dim", default=50, type=int)

    args = parser.parse_args()

    numpyro.set_platform(args.device)

    main(args)
