# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import io
from urllib.error import HTTPError

import pytest

import jax.numpy as jnp

import numpyro.examples.datasets as datasets
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
    assert jnp.shape(date) == jnp.shape(date) == (2517,)


def test_jsb_chorales():
    _, fetch = load_dataset(JSB_CHORALES, split="train", shuffle=False)
    lengths, sequences = fetch()
    assert jnp.shape(lengths) == (229,)
    assert jnp.shape(sequences) == (229, 129, 88)


def test_download_retries_on_http_429(tmp_path, monkeypatch):
    test_dset = datasets.dset("toy", ["https://example.com/file.txt"])
    monkeypatch.setattr(datasets, "DATA_DIR", str(tmp_path))

    attempts = {"count": 0}
    sleep_calls = []

    def fake_sleep(delay):
        sleep_calls.append(delay)

    def fake_urlopen(request):
        url = request.full_url
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise HTTPError(url, 429, "Too Many Requests", {"Retry-After": "0"}, None)
        return io.BytesIO(b"ok")

    monkeypatch.setattr(datasets.time, "sleep", fake_sleep)
    monkeypatch.setattr(datasets, "urlopen", fake_urlopen)

    datasets._download(test_dset)
    assert (tmp_path / "file.txt").exists()
    assert attempts["count"] == 3
    assert sleep_calls == [0.0, 0.0]


def test_download_raises_immediately_on_non_transient_http_error(tmp_path, monkeypatch):
    test_dset = datasets.dset("toy", ["https://example.com/missing.txt"])
    monkeypatch.setattr(datasets, "DATA_DIR", str(tmp_path))

    attempts = {"count": 0}

    def fake_urlopen(request):
        url = request.full_url
        attempts["count"] += 1
        raise HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(datasets, "urlopen", fake_urlopen)

    with pytest.raises(HTTPError):
        datasets._download(test_dset)
    assert attempts["count"] == 1


def test_download_attempts_github_raw_fallback(tmp_path, monkeypatch):
    blob_url = "https://github.com/pyro-ppl/datasets/blob/master/data.txt?raw=true"
    fallback_url = "https://raw.githubusercontent.com/pyro-ppl/datasets/master/data.txt"
    test_dset = datasets.dset("toy", [blob_url])
    monkeypatch.setattr(datasets, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(datasets, "_DOWNLOAD_MAX_RETRIES", 1)

    attempted_urls = []

    def fake_urlopen(request):
        url = request.full_url
        attempted_urls.append(url)
        if url == blob_url:
            raise HTTPError(url, 429, "Too Many Requests", {"Retry-After": "0"}, None)
        return io.BytesIO(b"ok")

    monkeypatch.setattr(datasets, "urlopen", fake_urlopen)

    datasets._download(test_dset)
    assert attempted_urls == [blob_url, fallback_url]
