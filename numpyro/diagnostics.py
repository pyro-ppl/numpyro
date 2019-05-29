from itertools import product

import numpy as onp

from jax import device_get, tree_flatten


def _compute_chain_variance_stats(x):
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


def gelman_rubin(x):
    """
    Computes R-hat over chains of samples ``x``, where the first dimension of
    ``x`` is chain dimension and the second dimension of ``x`` is draw dimension.
    It is required that ``input.shape[0] >= 2`` and ``input.shape[1] >= 2``.

    :param numpy.ndarray x: the input array.
    :return: R-hat of ``x``.
    :rtype: numpy.ndarray
    """
    assert x.ndim >= 2
    assert x.shape[0] >= 2
    assert x.shape[1] >= 2
    var_within, var_estimator = _compute_chain_variance_stats(x)
    rhat = onp.sqrt(var_estimator / var_within)
    return rhat


def split_gelman_rubin(x):
    """
    Computes split R-hat over chains of samples ``x``, where the first dimension
    of ``x`` is chain dimension and the second dimension of ``x`` is draw dimension.
    It is required that ``input.shape[1] >= 4``.

    :param numpy.ndarray x: the input array.
    :return: split R-hat of ``x``.
    :rtype: numpy.ndarray
    """
    assert x.ndim >= 2
    assert x.shape[1] >= 4

    N_half = x.shape[1] // 2
    new_input = onp.concatenate([x[:, :N_half], x[:, -N_half:]], axis=0)
    split_rhat = gelman_rubin(new_input)
    return split_rhat


def _fft_next_fast_len(target):
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


def autocorrelation(x, axis=0):
    """
    Computes the autocorrelation of samples at dimension ``axis``.

    :param numpy.ndarray x: the input array.
    :param int axis: the dimension to calculate autocorrelation.
    :return: autocorrelation of ``x``.
    :rtype: numpy.ndarray
    """
    # Ref: https://en.wikipedia.org/wiki/Autocorrelation#Efficient_computation
    # Adapted from Stan implementation
    # https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/autocorrelation.hpp
    N = x.shape[axis]
    M = _fft_next_fast_len(N)
    M2 = 2 * M

    # transpose axis with -1 for Fourier transform
    x = onp.swapaxes(x, axis, -1)

    # centering x
    centered_signal = x - x.mean(axis=-1, keepdims=True)

    # Fourier transform
    freqvec = onp.fft.rfft(centered_signal, n=M2, axis=-1)
    # take square of magnitude of freqvec (or freqvec x freqvec*)
    freqvec_gram = freqvec * onp.conjugate(freqvec)
    # inverse Fourier transform
    autocorr = onp.fft.irfft(freqvec_gram, n=M2, axis=-1)

    # truncate and normalize the result, then transpose back to original shape
    autocorr = autocorr[..., :N]
    autocorr = autocorr / onp.arange(N, 0., -1)
    autocorr = autocorr / autocorr[..., :1]
    return onp.swapaxes(autocorr, axis, -1)


def autocovariance(x, axis=0):
    """
    Computes the autocovariance of samples at dimension ``axis``.

    :param numpy.ndarray x: the input array.
    :param int axis: the dimension to calculate autocovariance.
    :return: autocovariance of ``x``.
    :rtype: numpy.ndarray
    """
    return autocorrelation(x, axis) * x.var(axis=axis, keepdims=True)


def effective_sample_size(x):
    """
    Computes effective sample size of input ``x``, where the first dimension of
    ``x`` is chain dimension and the second dimension of ``x`` is draw dimension.

    **References:**

    1. *Introduction to Markov Chain Monte Carlo*,
       Charles J. Geyer
    2. *Stan Reference Manual version 2.18*,
       Stan Development Team

    :param numpy.ndarray x: the input array.
    :return: effective sample size of ``x``.
    :rtype: numpy.ndarray
    """
    assert x.ndim >= 2
    assert x.shape[1] >= 2

    # find autocovariance for each chain at lag k
    gamma_k_c = autocovariance(x, axis=1)

    # find autocorrelation at lag k (from Stan reference)
    var_within, var_estimator = _compute_chain_variance_stats(x)
    rho_k = 1. - (var_within - gamma_k_c.mean(axis=0)) / var_estimator
    # correlation at lag 0 is always 1
    rho_k[0] = 1.

    # initial positive sequence (formula 1.18 in [1]) applied for autocorrelation
    Rho_k = rho_k[:-1:2, ...] + rho_k[1::2, ...]

    # initial monotone (decreasing) sequence
    Rho_init = Rho_k[:1]
    Rho_k = onp.concatenate(
        [Rho_init, onp.minimum.accumulate(onp.clip(Rho_k[1:, ...], a_min=0, a_max=None), axis=0)],
        axis=0
    )

    tau = -1. + 2. * Rho_k.sum(axis=0)
    n_eff = onp.prod(x.shape[:2]) / tau
    return n_eff


def hpdi(x, prob=0.89, axis=0):
    """
    Computes "highest posterior density interval" (HPDI) which is the narrowest
    interval with probability mass ``prob``.

    :param numpy.ndarray x: the input array.
    :param float prob: the probability mass of samples within the interval.
    :param int axis: the dimension to calculate hpdi.
    :return: quantiles of ``input`` at ``(1 - probs) / 2`` and
        ``(1 + probs) / 2``.
    :rtype: numpy.ndarray
    """
    x = onp.swapaxes(x, axis, 0)
    sorted_x = onp.sort(x, axis=0)
    mass = x.shape[0]
    index_length = int(prob * mass)
    intervals_left = sorted_x[:(mass - index_length)]
    intervals_right = sorted_x[index_length:]
    intervals_length = intervals_right - intervals_left
    index_start = intervals_length.argmin(axis=0)
    index_end = index_start + index_length
    hpd_left = onp.take_along_axis(sorted_x, index_start[None, ...], axis=0)
    hpd_left = onp.swapaxes(hpd_left, axis, 0)
    hpd_right = onp.take_along_axis(sorted_x, index_end[None, ...], axis=0)
    hpd_right = onp.swapaxes(hpd_right, axis, 0)
    return onp.concatenate([hpd_left, hpd_right], axis=axis)


def summary(samples, prob=0.89):
    """
    Prints a summary table for diagnostics of ``samples``.

    :param samples: a collection of input samples.
    :param float prob: the probability mass of samples within the HPDI interval.
    """
    # FIXME: handle variable with str len > 20
    header_format = '{:>20} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}'
    columns = ['', 'mean', 'sd', '{:.1f}%'.format(50 * (1 - prob)),
               '{:.1f}%'.format(50 * (1 + prob)), 'n_eff', 'Rhat']
    print('\n')
    print(header_format.format(*columns))

    # FIXME: maybe allow a `digits` arg to set how many floatting points are needed?
    row_format = '{:>20} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}'
    if not isinstance(samples, dict):
        samples = {'Param:{}'.format(i): v for i, v in enumerate(tree_flatten(samples)[0])}
    # TODO: support summary for chains of samples
    for name, value in samples.items():
        value = device_get(value)
        mean = value.mean(axis=0)
        sd = value.std(axis=0, ddof=1)
        hpd = hpdi(value, prob=prob)
        n_eff = effective_sample_size(value[None, ...])
        r_hat = split_gelman_rubin(value[None, ...])
        shape = value.shape[1:]
        if len(shape) == 0:
            print(row_format.format(name, mean, sd, hpd[0], hpd[1], n_eff, r_hat))
        else:
            for idx in product(*map(range, shape)):
                idx_str = '[{}]'.format(','.join(map(str, idx)))
                print(row_format.format(name + idx_str, mean[idx], sd[idx],
                                        hpd[0][idx], hpd[1][idx], n_eff[idx], r_hat[idx]))
