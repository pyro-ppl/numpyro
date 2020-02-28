# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def eval_mae(pred, truth):
    """
    Evaluate mean absolute error, using sample median as point estimate.

    :param np.ndarray pred: Forecasted samples.
    :param np.ndarray truth: Ground truth.
    :rtype: float
    """
    pred = np.median(pred, 0)
    return np.mean(np.abs(pred - truth)).item()


def eval_rmse(pred, truth):
    """
    Evaluate root mean squared error, using sample mean as point estimate.

    :param np.ndarray pred: Forecasted samples.
    :param np.ndarray truth: Ground truth.
    :rtype: float
    """
    pred = np.mean(pred, 0)
    return np.sqrt(np.mean(np.square(pred - truth))).item()


def eval_crps(pred, truth):
    """
    Evaluate continuous ranked probability score, averaged over all data
    elements.

    :param np.ndarray pred: Forecasted samples.
    :param np.ndarray truth: Ground truth.
    :rtype: float
    """
    # ref: https://github.com/pyro-ppl/pyro/pull/2045
    num_samples = pred.shape[0]
    pred = np.sort(pred, axis=0)
    diff = pred[1:] - pred[:-1]
    weight = np.arange(1, num_samples) * np.arange(num_samples - 1, 0, -1)
    weight = weight.reshape(weight.shape + (1,) * truth.ndim)
    crps_empirical = np.mean(np.abs(pred - truth), 0) \
        - (diff * weight).sum(axis=0) / num_samples ** 2
    return np.mean(crps_empirical).item()
