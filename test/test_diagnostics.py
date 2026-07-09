# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numpy.testing import assert_, assert_allclose
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
    actual = autocorrelation(x, bias=False)
    expected = np.array([1, 0.78, 0.52, 0.21, -0.13, -0.52, -0.94, -1.4, -1.91, -2.45])
    assert_allclose(actual, expected, atol=0.01)

    actual = autocorrelation(x, bias=True)
    expected = expected * np.arange(len(x), 0.0, -1) / len(x)
    assert_allclose(actual, expected, atol=0.01)

    # the unbiased estimator has variance O(1) at large lags
    x = np.random.normal(size=20000)
    ac = autocorrelation(x, bias=False)
    assert_(np.any(np.abs(ac[-100:]) > 0.1))

    ac = autocorrelation(x, bias=True)
    assert_allclose(np.abs(ac[-100:]), 0.0, atol=0.01)


def test_autocovariance():
    x = np.arange(10.0)
    actual = autocovariance(x, bias=False)
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
    assert_allclose(effective_sample_size(x, bias=False), 52.64, atol=0.01)


def test_weighted_summary():
    from numpyro.diagnostics import summary

    np.random.seed(42)
    x = np.random.normal(loc=5.0, scale=2.0, size=(4, 500))

    # unweighted should match standard summary
    unweighted = summary(x, group_by_chain=True)
    assert "r_hat" in unweighted["Param:0"]
    assert_allclose(unweighted["Param:0"]["mean"], x.mean(), atol=0.2)
    assert_allclose(unweighted["Param:0"]["std"], x.std(ddof=1), atol=0.15)

    # uniform log_weights should give same result as unweighted
    uniform_lw = np.zeros(x.shape[0] * x.shape[1])
    weighted = summary(x, group_by_chain=True, log_weights=uniform_lw)
    assert "r_hat" not in weighted["Param:0"]
    assert_allclose(weighted["Param:0"]["mean"], unweighted["Param:0"]["mean"], atol=0.01)
    assert_allclose(weighted["Param:0"]["std"], unweighted["Param:0"]["std"], atol=0.01)
    assert_allclose(weighted["Param:0"]["median"], unweighted["Param:0"]["median"], atol=0.01)
    assert_allclose(weighted["Param:0"]["n_eff"], 2000.0, atol=1.0)

    # non-uniform weights should return different values
    nonuniform_lw = np.cos(np.linspace(0, 2 * np.pi, 2000))
    weighted2 = summary(x, group_by_chain=True, log_weights=nonuniform_lw)
    assert "r_hat" not in weighted2["Param:0"]
    assert weighted2["Param:0"]["n_eff"] < 2000.0

    # group_by_chain=False should work too
    x_2d = np.random.normal(size=(1000,))
    s = summary(x_2d, group_by_chain=False, log_weights=np.zeros(1000))
    assert "r_hat" not in s["Param:0"]


def test_weighted_summary_multivariate():
    from numpyro.diagnostics import summary

    np.random.seed(7)
    x = np.random.normal(size=(4, 200, 3))
    lw = np.zeros((4, 200))

    s = summary(x, group_by_chain=True, log_weights=lw.ravel())
    assert "r_hat" not in s["Param:0"]
    assert s["Param:0"]["mean"].shape == (3,)
    assert_allclose(s["Param:0"]["n_eff"], 800.0, atol=1.0)


def test_print_summary_log_weights():
    import io
    import sys

    from numpyro.diagnostics import print_summary

    np.random.seed(1)
    x = np.random.normal(size=(2, 100))
    lw = np.zeros(200)

    captured = io.StringIO()
    sys.stdout = captured
    try:
        print_summary(x, log_weights=lw)
    finally:
        sys.stdout = sys.__stdout__

    output = captured.getvalue()
    assert "mean" in output
    assert "n_eff" in output
    assert "r_hat" not in output
