# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.fftpack import next_fast_len

from numpyro.diagnostics import (
    _fft_next_fast_len,
    autocorrelation,
    autocovariance,
    effective_sample_size,
    gelman_rubin,
    hpdi,
    split_gelman_rubin,
)


@pytest.mark.parametrize(
    "statistics, input_shape, output_shape",
    [
        (autocorrelation, (10,), (10,)),
        (autocorrelation, (10, 3), (10, 3)),
        (autocovariance, (10,), (10,)),
        (autocovariance, (10, 3), (10, 3)),
        (hpdi, (10,), (2,)),
        (hpdi, (10, 3), (2, 3)),
        (gelman_rubin, (4, 10), ()),
        (gelman_rubin, (4, 10, 3), (3,)),
        (split_gelman_rubin, (4, 10), ()),
        (split_gelman_rubin, (4, 10, 3), (3,)),
        (effective_sample_size, (4, 10), ()),
        (effective_sample_size, (4, 10, 3), (3,)),
    ],
)
def test_shape(statistics, input_shape, output_shape):
    x = np.random.normal(size=input_shape)
    y = statistics(x)
    assert y.shape == output_shape

    # test correct batch calculation
    if x.shape[-1] == 3:
        for i in range(3):
            assert_allclose(statistics(x[..., i]), y[..., i])


@pytest.mark.parametrize("target", [433, 124, 25, 300, 1, 3, 7])
def test_fft_next_fast_len(target):
    assert _fft_next_fast_len(target) == next_fast_len(target)


def test_hpdi():
    x = np.random.normal(size=20000)
    assert_allclose(hpdi(x, prob=0.8), np.quantile(x, [0.1, 0.9]), atol=0.01)

    x = np.random.exponential(size=20000)
    assert_allclose(hpdi(x, prob=0.2), np.array([0.0, 0.22]), atol=0.01)


def test_autocorrelation():
    x = np.arange(10.0)
    actual = autocorrelation(x)
    expected = np.array([1, 0.78, 0.52, 0.21, -0.13, -0.52, -0.94, -1.4, -1.91, -2.45])
    assert_allclose(actual, expected, atol=0.01)


def test_autocovariance():
    x = np.arange(10.0)
    actual = autocovariance(x)
    expected = np.array(
        [8.25, 6.42, 4.25, 1.75, -1.08, -4.25, -7.75, -11.58, -15.75, -20.25]
    )
    assert_allclose(actual, expected, atol=0.01)


def test_gelman_rubin():
    # only need to test precision for small data
    x = np.empty((2, 10))
    x[0, :] = np.arange(10.0)
    x[1, :] = np.arange(10.0) + 1

    r_hat = gelman_rubin(x)
    assert_allclose(r_hat, 0.98, atol=0.01)


def test_split_gelman_rubin_agree_with_gelman_rubin():
    x = np.random.normal(size=(2, 10))
    r_hat1 = gelman_rubin(x.reshape(2, 2, 5).reshape(4, 5))
    r_hat2 = split_gelman_rubin(x)
    assert_allclose(r_hat1, r_hat2)


def test_effective_sample_size():
    x = np.arange(1000.0).reshape(100, 10)
    assert_allclose(effective_sample_size(x), 52.64, atol=0.01)
