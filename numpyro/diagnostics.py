# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
This provides a small set of utilities in NumPyro that are used to diagnose posterior samples.
"""

from collections import OrderedDict
from itertools import product
from typing import Optional, Union, cast

import numpy as np
from numpy.typing import NDArray

import jax
from jax import device_get

try:
    import scipy.fft

    def _rfft(x: NDArray, n: int, axis: int) -> NDArray:
        return scipy.fft.rfft(x, n=n, axis=axis, workers=-1)

    def _irfft(x: NDArray, n: int, axis: int) -> NDArray:
        return scipy.fft.irfft(x, n=n, axis=axis, workers=-1)
except ImportError:  # pragma: no cover
    # scipy is not a direct dependency of numpyro (only a transitive one via jax)
    def _rfft(x: NDArray, n: int, axis: int) -> NDArray:
        return np.fft.rfft(x, n=n, axis=axis)

    def _irfft(x: NDArray, n: int, axis: int) -> NDArray:
        return np.fft.irfft(x, n=n, axis=axis)


__all__ = [
    "autocorrelation",
    "autocovariance",
    "effective_sample_size",
    "gelman_rubin",
    "hpdi",
    "split_gelman_rubin",
    "print_summary",
]


def _compute_chain_variance_stats(x: NDArray) -> tuple[NDArray, NDArray]:
    # compute within-chain variance and variance estimator
    # input has shape C x N x sample_shape
    C, N = x.shape[:2]
    chain_var = x.var(axis=1, ddof=1)
    var_within = chain_var.mean(axis=0)
    var_estimator = var_within * (N - 1) / N
    if x.shape[0] > 1:
        chain_mean = x.mean(axis=1)
        var_between = chain_mean.var(axis=0, ddof=1)
        var_estimator = var_estimator + var_between
    else:
        var_within = var_estimator
    return var_within, var_estimator


def gelman_rubin(x: NDArray) -> NDArray:
    """
    Computes R-hat over chains of samples ``x``, where the first dimension of
    ``x`` is chain dimension and the second dimension of ``x`` is draw dimension.
    It is required that ``x.shape[0] >= 2`` and ``x.shape[1] >= 2``.

    :param numpy.ndarray x: the input array.
    :return: R-hat of ``x``.
    :rtype: numpy.ndarray
    """
    assert x.ndim >= 2
    assert x.shape[0] >= 2
    assert x.shape[1] >= 2
    var_within, var_estimator = _compute_chain_variance_stats(x)
    with np.errstate(invalid="ignore", divide="ignore"):
        rhat = np.sqrt(var_estimator / var_within)
    return rhat


def split_gelman_rubin(x: NDArray) -> NDArray:
    """
    Computes split R-hat over chains of samples ``x``, where the first dimension
    of ``x`` is chain dimension and the second dimension of ``x`` is draw dimension.
    It is required that ``x.shape[1] >= 4``.

    :param numpy.ndarray x: the input array.
    :return: split R-hat of ``x``.
    :rtype: numpy.ndarray
    """
    assert x.ndim >= 2
    assert x.shape[1] >= 4

    N_half = x.shape[1] // 2
    new_input = np.concatenate([x[:, :N_half], x[:, -N_half:]], axis=0)
    split_rhat = gelman_rubin(new_input)
    return split_rhat


def _fft_next_fast_len(target: int) -> int:
    # find the smallest number >= N such that the only divisors are 2, 3, 5
    # works just like scipy.fftpack.next_fast_len
    if target <= 2:
        return target
    while True:
        m = target
        while m % 2 == 0:
            m //= 2
        while m % 3 == 0:
            m //= 3
        while m % 5 == 0:
            m //= 5
        if m == 1:
            return target
        target += 1


def autocorrelation(x: NDArray, axis: int = 0, bias: bool = True) -> NDArray:
    """
    Computes the autocorrelation of samples at dimension ``axis``.

    :param numpy.ndarray x: the input array.
    :param int axis: the dimension to calculate autocorrelation.
    :param bias: whether to use a biased estimator.
    :return: autocorrelation of ``x``.
    :rtype: numpy.ndarray
    """
    autocov = autocovariance(x, axis=axis, bias=bias)
    # normalize by the lag-0 autocovariance (which is the same for the biased
    # and unbiased estimators)
    with np.errstate(invalid="ignore", divide="ignore"):
        return autocov / np.take(autocov, [0], axis=axis)


def autocovariance(x: NDArray, axis: int = 0, bias: bool = True) -> NDArray:
    """
    Computes the autocovariance of samples at dimension ``axis``.

    :param numpy.ndarray x: the input array.
    :param int axis: the dimension to calculate autocovariance.
    :param bias: whether to use a biased estimator.
    :return: autocovariance of ``x``.
    :rtype: numpy.ndarray
    """
    # Ref: https://en.wikipedia.org/wiki/Autocorrelation#Efficient_computation
    # Adapted from Stan implementation
    # https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/autocorrelation.hpp
    N = x.shape[axis]
    M = _fft_next_fast_len(N)
    M2 = 2 * M

    # transpose axis with -1 for Fourier transform
    x = np.swapaxes(x, axis, -1)

    # centering x
    centered_signal = x - x.mean(axis=-1, keepdims=True)
    # np.fft promotes to double precision; keep doing so explicitly because
    # scipy.fft computes in the input precision
    centered_signal = centered_signal.astype(np.float64, copy=False)

    # Fourier transform
    freqvec = _rfft(centered_signal, n=M2, axis=-1)
    # take square of magnitude of freqvec (or freqvec x freqvec*)
    freqvec_gram = freqvec * np.conjugate(freqvec)
    # inverse Fourier transform
    autocov = _irfft(freqvec_gram, n=M2, axis=-1)

    # truncate the result and normalize by the number of terms in each lag sum,
    # then transpose back to original shape

    # the unbiased estimator is known to have "wild" tails, due to few samples at longer lags.
    # see Geyer (1992) and Priestley (1981) for a discussion. also note that it is only strictly
    # unbiased when the mean is known, whereas we it estimate from samples here.
    if bias:
        autocov = autocov[..., :N] / N
    else:
        autocov = autocov[..., :N] / np.arange(N, 0.0, -1)
    return np.swapaxes(autocov, axis, -1)


def effective_sample_size(x: NDArray, bias: bool = True) -> NDArray:
    """
    Computes effective sample size of input ``x``, where the first dimension of
    ``x`` is chain dimension and the second dimension of ``x`` is draw dimension.

    **References:**

    1. *Introduction to Markov Chain Monte Carlo*,
       Charles J. Geyer
    2. *Stan Reference Manual version 2.18*,
       Stan Development Team

    :param numpy.ndarray x: the input array.
    :param bias: whether to use a biased estimator of the autocovariance.
    :return: effective sample size of ``x``.
    :rtype: numpy.ndarray
    """
    x = device_get(x)
    assert x.ndim >= 2
    assert x.shape[1] >= 2

    # find autocovariance for each chain at lag k
    gamma_k_c = autocovariance(x, axis=1, bias=bias)

    # find autocorrelation at lag k (from Stan reference)
    var_within, var_estimator = _compute_chain_variance_stats(x)
    rho_k = 1.0 - (var_within - gamma_k_c.mean(axis=0)) / var_estimator
    # correlation at lag 0 is always 1
    rho_k[0] = 1.0

    # initial positive sequence (formula 1.18 in [1]) applied for autocorrelation
    Rho_k = rho_k[:-1:2, ...] + rho_k[1::2, ...]

    # initial monotone (decreasing) sequence
    Rho_init = Rho_k[:1]
    Rho_k = np.concatenate(
        [
            Rho_init,
            np.minimum.accumulate(np.clip(Rho_k[1:, ...], 0, None), axis=0),
        ],
        axis=0,
    )

    tau = -1.0 + 2.0 * Rho_k.sum(axis=0)
    n_eff = np.prod(x.shape[:2]) / tau
    return n_eff


def hpdi(x: NDArray, prob: float = 0.90, axis: int = 0) -> NDArray:
    """
    Computes "highest posterior density interval" (HPDI) which is the narrowest
    interval with probability mass ``prob``.

    :param numpy.ndarray x: the input array.
    :param float prob: the probability mass of samples within the interval.
    :param int axis: the dimension to calculate hpdi.
    :return: Array containing the lower and upper bounds of the HPDI along
        the specified axis. The output has the same shape as ``x`` except
        that the size along ``axis`` is 2 (lower bound first, upper bound second).
    :rtype: numpy.ndarray
    """
    x = np.swapaxes(x, axis, 0)
    sorted_x = np.sort(x, axis=0)
    hpd_left, hpd_right = _hpdi_of_sorted(sorted_x, prob)
    hpd_left = np.swapaxes(hpd_left, axis, 0)
    hpd_right = np.swapaxes(hpd_right, axis, 0)
    return np.concatenate([hpd_left, hpd_right], axis=axis)


def _hpdi_of_sorted(sorted_x: NDArray, prob: float) -> tuple[NDArray, NDArray]:
    # compute the hpdi bounds of samples already sorted along axis 0; both
    # bounds are returned with a leading axis of size 1
    mass = sorted_x.shape[0]
    index_length = int(prob * mass)
    intervals_left = sorted_x[: (mass - index_length)]
    intervals_right = sorted_x[index_length:]
    intervals_length = intervals_right - intervals_left
    index_start = intervals_length.argmin(axis=0)
    index_end = index_start + index_length
    hpd_left = np.take_along_axis(sorted_x, index_start[None, ...], axis=0)
    hpd_right = np.take_along_axis(sorted_x, index_end[None, ...], axis=0)
    return hpd_left, hpd_right


def _normalized_weights(log_weights: NDArray) -> NDArray:
    # turn unnormalized log weights (num_chains x num_samples) into normalized
    # weights of shape (num_chains * num_samples,); the max is subtracted before
    # exponentiating so that large log weights do not overflow
    log_weights = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    max_log_weight = log_weights.max()
    if not np.isfinite(max_log_weight):
        raise ValueError(
            "`log_weights` must contain at least one finite value and no nan or +inf."
        )
    weights = np.exp(log_weights - max_log_weight)
    return weights / weights.sum()


def _weighted_stats(
    x: NDArray, weights: NDArray, prob: float
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    # weighted mean, std, median and hpdi of `x` (shape num_draws x sample_shape)
    # under normalized `weights` (shape num_draws)
    # the weights are float64, so cast back to the dtype of the unweighted path
    dtype = np.result_type(x.dtype, np.float32)
    w = weights.reshape((-1,) + (1,) * (x.ndim - 1))
    mean = (w * x).sum(axis=0)
    # unbiased estimator for reliability weights, the same as
    # np.cov(x, aweights=weights, ddof=1); it reduces to ddof=1 for equal weights
    with np.errstate(divide="ignore", invalid="ignore"):
        var = (w * (x - mean) ** 2).sum(axis=0) / (1.0 - np.square(weights).sum())
    std = np.sqrt(var).astype(dtype)
    mean = mean.astype(dtype)

    n = x.shape[0]
    sort_idx = np.argsort(x, axis=0)
    sorted_x = np.take_along_axis(x, sort_idx, axis=0).reshape(n, -1)
    sorted_w = weights[sort_idx].reshape(n, -1)
    # mass up to and including each sorted draw, and strictly before it
    cum_w = np.cumsum(sorted_w, axis=0)
    cum_w_before = cum_w - sorted_w
    # slack for round-off in the cumulative sums, so that equal weights give
    # exactly the same median and hpdi as the unweighted statistics
    tol = 16 * n * np.finfo(np.float64).eps

    # lower and upper weighted medians; these are the two middle draws (or the
    # middle draw twice) when the weights are equal
    lo = np.minimum((cum_w >= 0.5 - tol).argmax(axis=0), n - 1)
    hi = np.minimum((cum_w > 0.5 + tol).argmax(axis=0), n - 1)
    cols = np.arange(sorted_x.shape[1])
    median = ((sorted_x[lo, cols] + sorted_x[hi, cols]) / 2).astype(dtype)

    hpd = np.empty((2, sorted_x.shape[1]), dtype=sorted_x.dtype)
    for k in cols:
        xs, cw, cw_before = sorted_x[:, k], cum_w[:, k], cum_w_before[:, k]
        # narrowest interval [xs[i], xs[j]] whose mass exceeds `prob`; for each
        # left end i, j is the first draw at which the mass from i exceeds `prob`
        right = np.searchsorted(cw, cw_before + prob + tol, side="right")
        valid = right < n
        lengths = np.where(valid, xs[np.minimum(right, n - 1)] - xs, np.inf)
        start = int(np.argmin(lengths)) if valid.any() else 0
        hpd[:, k] = xs[start], xs[min(right[start], n - 1)]

    sample_shape = x.shape[1:]
    median = median.reshape(sample_shape)
    median = np.where(np.isnan(sorted_x[-1].reshape(sample_shape)), np.nan, median)
    return mean, std, median, hpd.reshape((2,) + sample_shape)


def summary(
    samples: Union[dict, np.ndarray],
    prob: float = 0.90,
    group_by_chain: bool = True,
    log_weights: Optional[NDArray] = None,
) -> dict:
    """
    Returns a summary table displaying diagnostics of ``samples`` from the
    posterior. The diagnostics displayed are mean, standard deviation, median,
    the 90% Credibility Interval :func:`~numpyro.diagnostics.hpdi`,
    :func:`~numpyro.diagnostics.effective_sample_size`, and
    :func:`~numpyro.diagnostics.split_gelman_rubin`.

    When ``log_weights`` is given (e.g. for importance sampling or nested
    sampling), the mean, standard deviation, median and HPDI are the weighted
    versions of these statistics, ``n_eff`` is the effective sample size of the
    weights, :math:`(\\sum_i w_i)^2 / \\sum_i w_i^2`, and ``r_hat`` is not
    reported.

    :param samples: a collection of input samples with left most dimension is chain
        dimension and second to left most dimension is draw dimension.
    :type samples: dict or numpy.ndarray
    :param float prob: the probability mass of samples within the HPDI interval.
    :param bool group_by_chain: If True, each variable in `samples` will be treated
        as having shape `num_chains x num_samples x sample_shape`. Otherwise, the
        corresponding shape will be `num_samples x sample_shape` (i.e. without
        chain dimension).
    :param numpy.ndarray log_weights: optional unnormalized log weights of the
        draws, with shape `num_chains x num_samples` if `group_by_chain` is True
        and `num_samples` otherwise.
    """
    if not group_by_chain:
        samples = jax.tree.map(lambda x: x[None, ...], samples)
        if log_weights is not None:
            log_weights = np.asarray(device_get(log_weights))[None, ...]
    if not isinstance(samples, dict):
        samples = {
            "Param:{}".format(i): v for i, v in enumerate(jax.tree.flatten(samples)[0])
        }
    samples = cast(dict[str, np.ndarray], samples)
    if log_weights is not None:
        log_weights = np.asarray(device_get(log_weights))
        weights = _normalized_weights(log_weights)
        # effective sample size of the weights, shared by all sites
        weights_n_eff = 1.0 / np.square(weights).sum()

    summary_dict = {}
    for name, value in samples.items():
        if len(value) == 0:
            continue
        value = device_get(value)
        value_flat = np.reshape(value, (-1,) + value.shape[2:])
        hpd_lower = "{:.1f}%".format(50 * (1 - prob))
        hpd_upper = "{:.1f}%".format(50 * (1 + prob))
        if log_weights is not None:
            if log_weights.shape != value.shape[:2]:
                raise ValueError(
                    "Expected `log_weights` to have shape {} (the chain and draw"
                    " dimensions of `samples`), but got {}.".format(
                        value.shape[:2], log_weights.shape
                    )
                )
            mean, std, median, hpd = _weighted_stats(value_flat, weights, prob)
            n_eff = np.full(mean.shape, weights_n_eff)
            summary_dict[name] = OrderedDict(
                [
                    ("mean", mean),
                    ("std", std),
                    ("median", median),
                    (hpd_lower, hpd[0]),
                    (hpd_upper, hpd[1]),
                    ("n_eff", n_eff),
                ]
            )
            continue
        mean = value_flat.mean(axis=0)
        std = value_flat.std(axis=0, ddof=1)
        # sort once and reuse the result for both the median and the hpdi
        sorted_flat = np.sort(value_flat, axis=0)
        n = sorted_flat.shape[0]
        median = (sorted_flat[(n - 1) // 2] + sorted_flat[n // 2]) / 2
        # like np.median, return nan wherever any draw is nan (sorting puts
        # nans last, so it suffices to check the last draw)
        median = np.where(np.isnan(sorted_flat[-1]), median.dtype.type(np.nan), median)
        hpd = np.concatenate(_hpdi_of_sorted(sorted_flat, prob), axis=0)
        n_eff = effective_sample_size(value)
        r_hat = split_gelman_rubin(value)
        summary_dict[name] = OrderedDict(
            [
                ("mean", mean),
                ("std", std),
                ("median", median),
                (hpd_lower, hpd[0]),
                (hpd_upper, hpd[1]),
                ("n_eff", n_eff),
                ("r_hat", r_hat),
            ]
        )
    return summary_dict


def print_summary(
    samples: Union[dict, NDArray],
    prob: float = 0.90,
    group_by_chain: bool = True,
    log_weights: Optional[NDArray] = None,
) -> None:
    """
    Prints a summary table displaying diagnostics of ``samples`` from the
    posterior. The diagnostics displayed are mean, standard deviation, median,
    the 90% Credibility Interval :func:`~numpyro.diagnostics.hpdi`,
    :func:`~numpyro.diagnostics.effective_sample_size`, and
    :func:`~numpyro.diagnostics.split_gelman_rubin`.

    :param samples: a collection of input samples with left most dimension is chain
        dimension and second to left most dimension is draw dimension.
    :type samples: dict or numpy.ndarray
    :param float prob: the probability mass of samples within the HPDI interval.
    :param bool group_by_chain: If True, each variable in `samples` will be treated
        as having shape `num_chains x num_samples x sample_shape`. Otherwise, the
        corresponding shape will be `num_samples x sample_shape` (i.e. without
        chain dimension).
    :param numpy.ndarray log_weights: optional unnormalized log weights of the
        draws; see :func:`~numpyro.diagnostics.summary`.
    """
    if not group_by_chain:
        samples = jax.tree.map(lambda x: x[None, ...], samples)
        if log_weights is not None:
            log_weights = np.asarray(device_get(log_weights))[None, ...]
    if not isinstance(samples, dict):
        samples = {
            "Param:{}".format(i): v for i, v in enumerate(jax.tree.flatten(samples)[0])
        }
    samples = cast(dict[str, np.ndarray], samples)
    summary_dict = summary(samples, prob, group_by_chain=True, log_weights=log_weights)
    if not summary_dict:
        return

    row_names = {
        k: k + "[" + ",".join(map(lambda x: str(x - 1), v.shape[2:])) + "]"
        for k, v in samples.items()
    }
    max_len = max(max(map(lambda x: len(x), row_names.values())), 10)
    name_format = "{:>" + str(max_len) + "}"
    columns = [""] + list(list(summary_dict.values())[0].keys())
    header_format = name_format + " {:>9}" * (len(columns) - 1)

    print()
    print(header_format.format(*columns))

    row_format = name_format + " {:>9.2f}" * (len(columns) - 1)
    for name, stats_dict in summary_dict.items():
        shape = stats_dict["mean"].shape
        if len(shape) == 0:
            print(row_format.format(name, *stats_dict.values()))
        else:
            for idx in product(*map(range, shape)):
                idx_str = "[{}]".format(",".join(map(str, idx)))
                print(
                    row_format.format(
                        name + idx_str, *[v[idx] for v in stats_dict.values()]
                    )
                )
    print()
