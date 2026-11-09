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
    print_summary,
    split_gelman_rubin,
    summary,
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


def test_hpdi_full_mass():
    x = np.random.normal(size=(100, 3))
    assert_allclose(hpdi(x, prob=1.0), np.stack([x.min(0), x.max(0)]))
    stats = summary(x, prob=1.0, group_by_chain=False)["Param:0"]
    assert_allclose(stats["0.0%"], x.min(0))
    assert_allclose(stats["100.0%"], x.max(0))


def test_hpdi_invalid_prob():
    x = np.random.normal(size=100)
    with pytest.raises(ValueError, match="prob"):
        hpdi(x, prob=1.5)
    with pytest.raises(ValueError, match="prob"):
        hpdi(x, prob=-0.1)


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


@pytest.mark.parametrize("num_draws", [500, 501])
def test_summary_median_matches_numpy(num_draws):
    x = np.random.RandomState(0).randn(2, num_draws, 3)
    stats = summary({"x": x})["x"]
    assert_allclose(stats["median"], np.median(x.reshape(-1, 3), axis=0))


def test_summary_median_propagates_nan():
    x = np.random.RandomState(0).randn(2, 100, 3)
    x[0, 5, 1] = np.nan
    stats = summary({"x": x})["x"]
    assert np.isnan(stats["median"][1])
    assert not np.isnan(stats["median"][[0, 2]]).any()


@pytest.mark.parametrize("prob_true", [0.3, 0.8])
@pytest.mark.parametrize("weighted", [False, True])
def test_summary_boolean_samples(prob_true, weighted):
    x = np.random.RandomState(0).rand(2, 100, 3) < prob_true
    log_weights = np.random.RandomState(1).randn(*x.shape[:2]) if weighted else None
    stats = summary({"x": x}, log_weights=log_weights)["x"]
    expected = summary({"x": x.astype(np.int32)}, log_weights=log_weights)["x"]
    for key, value in expected.items():
        assert_allclose(stats[key], value, err_msg=key)
    if not weighted:
        assert_allclose(stats["median"], np.median(x.reshape(-1, 3), axis=0))


def test_hpdi_boolean_samples():
    x = np.random.RandomState(0).rand(1000, 3) < np.array([0.02, 0.5, 0.98])
    assert_allclose(hpdi(x, prob=0.9), [[0, 0, 1], [0, 1, 1]])


@pytest.mark.parametrize("group_by_chain, num_draws", [(True, 500), (False, 501)])
def test_summary_equal_log_weights_match_unweighted(group_by_chain, num_draws):
    rng = np.random.RandomState(0)
    shape = (2, num_draws) if group_by_chain else (num_draws,)
    x = rng.randn(*shape, 3)
    log_weights = np.full(shape, 123.4)
    kwargs = {"group_by_chain": group_by_chain}
    expected = summary({"x": x}, **kwargs)["x"]
    actual = summary({"x": x}, log_weights=log_weights, **kwargs)["x"]
    assert "r_hat" not in actual
    for key in ["mean", "std", "median", "5.0%", "95.0%"]:
        assert_allclose(actual[key], expected[key], rtol=1e-6)
    assert_allclose(actual["n_eff"], np.full(3, x.size // 3))


def test_summary_integer_weights_match_repeated_draws():
    # weighting draws by integer counts is the same as repeating them
    rng = np.random.RandomState(1)
    x = rng.randn(2, 200, 2)
    counts = rng.randint(1, 6, size=(2, 200))
    repeated = np.repeat(x.reshape(1, -1, 2), counts.reshape(-1), axis=1)
    actual = summary({"x": x}, prob=0.8, log_weights=np.log(counts) + 1000.0)["x"]
    expected = summary({"x": repeated}, prob=0.8)["x"]
    for key in ["mean", "median", "10.0%", "90.0%"]:
        assert_allclose(actual[key], expected[key], rtol=1e-6)
    expected_std = np.sqrt(
        np.diag(np.cov(x.reshape(-1, 2).T, aweights=counts.reshape(-1), ddof=1))
    )
    assert_allclose(actual["std"], expected_std, rtol=1e-6)
    assert_allclose(actual["n_eff"], counts.sum() ** 2 / (counts**2).sum())


def test_summary_log_weights_ignore_zero_weight_draws():
    rng = np.random.RandomState(3)
    x = rng.randn(1, 300)
    log_weights = np.zeros((1, 300))
    log_weights[0, ::3] = -np.inf
    x_kept = x[:, np.isfinite(log_weights[0])]
    actual = summary({"x": x}, log_weights=log_weights)["x"]
    expected = summary({"x": x_kept})["x"]
    for key in ["mean", "std", "median", "5.0%", "95.0%"]:
        assert_allclose(actual[key], expected[key], rtol=1e-6)
    assert_allclose(actual["n_eff"], 200)


def test_summary_log_weights_bad_shape():
    with pytest.raises(ValueError, match="log_weights"):
        summary({"x": np.zeros((2, 10))}, log_weights=np.zeros(10))


def test_print_summary_log_weights(capsys):
    x = np.random.RandomState(4).randn(100, 2)
    print_summary({"x": x}, group_by_chain=False, log_weights=np.zeros(100))
    out = capsys.readouterr().out
    assert "n_eff" in out and "r_hat" not in out
