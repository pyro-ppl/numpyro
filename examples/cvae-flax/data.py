# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from jax import lax

from numpyro.examples.datasets import _load


def load_dataset(
    dset, batch_size=None, split="train", shuffle=True, num_datapoints=None, seed=23
):
    data = _load(dset, num_datapoints)
    if isinstance(data, dict):
        arrays = data[split]
    num_records = len(arrays[0])
    idxs = np.arange(num_records)
    if not batch_size:
        batch_size = num_records

    Y = arrays[0]
    X = Y.copy()
    _, m, n = X.shape

    X = X[:, (m // 2) :, : (n // 2)]
    arrays = (X, Y)

    def init():
        np.random.seed(seed)
        return (
            num_records // batch_size,
            np.random.permutation(idxs) if shuffle else idxs,
        )

    def get_batch(i=0, idxs=idxs):
        ret_idx = lax.dynamic_slice_in_dim(idxs, i * batch_size, batch_size)
        d = tuple(
            np.take(a, ret_idx, axis=0)
            if isinstance(a, list)
            else lax.index_take(a, (ret_idx,), axes=(0,))
            for a in arrays
        )
        return d

    return init, get_batch
