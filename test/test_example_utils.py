# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp

from numpyro.examples.datasets import (
    BASEBALL,
    COVTYPE,
    JSB_CHORALES,
    MNIST,
    SP500,
    load_dataset,
)
from numpyro.util import fori_loop


def test_baseball_data_load():
    init, fetch = load_dataset(BASEBALL, split="train", shuffle=False)
    num_batches, idx = init()
    dataset = fetch(0, idx)
    assert jnp.shape(dataset[0]) == (18, 2)
    assert jnp.shape(dataset[1]) == (18,)


def test_covtype_data_load():
    _, fetch = load_dataset(COVTYPE, shuffle=False)
    x, y = fetch()
    assert jnp.shape(x) == (581012, 54)
    assert jnp.shape(y) == (581012,)


def test_mnist_data_load():
    def mean_pixels(i, mean_pix):
        batch, _ = fetch(i, idx)
        return mean_pix + jnp.sum(batch) / batch.size

    init, fetch = load_dataset(MNIST, batch_size=128, split="train")
    num_batches, idx = init()
    assert fori_loop(0, num_batches, mean_pixels, jnp.float32(0.0)) / num_batches < 0.15


def test_sp500_data_load():
    _, fetch = load_dataset(SP500, split="train", shuffle=False)
    date, value = fetch()
    assert jnp.shape(date) == jnp.shape(date) == (2427,)


def test_jsb_chorales():
    _, fetch = load_dataset(JSB_CHORALES, split="train", shuffle=False)
    lengths, sequences = fetch()
    assert jnp.shape(lengths) == (229,)
    assert jnp.shape(sequences) == (229, 129, 88)
