# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod

from jax import lax
import jax.numpy as np

import numpyro
import numpyro.distributions as dist


# src: https://github.com/cbergmeir/Rlgt/blob/master/Rlgt/R/rlgtcontrol.R
DEFAULT_HYPERPARAMETERS = {
    "cauchy_sd_div": 150,
    "min_nu": 2,
    "max_nu": 20,
    "min_pow_trend": -0.5,
    "max_pow_trend": 1,
    "pow_trend_alpha": 1,
    "pow_trend_beta": 1,
    "pow_season_alpha": 1,
    "pow_season_beta": 1,
    "min_sigma": 1e-10,
    "min_val": 1e-30,
    "max_val": 1e38,
}


def scan_exp_val(y, init_s, level_sm, s_sm, coef_trend, pow_trend, pow_season):
    seasonality = init_s.shape[0]

    def scan_fn(carry, t):
        level, s, moving_sum = carry
        season = s[0] * level ** pow_season
        exp_val = level + coef_trend * level ** pow_trend + season
        exp_val = np.clip(exp_val, a_min=0)

        moving_sum = moving_sum + y[t] - np.where(t >= seasonality, y[t - seasonality], 0.)
        level_p = np.where(t >= seasonality, moving_sum / seasonality, y[t] - season)
        level = level_sm * level_p + (1 - level_sm) * level
        level = np.clip(level, a_min=0)
        new_s = (s_sm * (y[t] - level) / season + (1 - s_sm)) * s[0]
        s = np.concatenate([s[1:], new_s[None]], axis=0)
        return (level, s, moving_sum), exp_val

    level_init = y[0]
    s_init = np.concatenate([init_s[1:], init_s[:1]], axis=0)
    moving_sum = level_init
    (last_level, last_s, moving_sum), exp_vals = lax.scan(
        scan_fn, (level_init, s_init, moving_sum), np.arange(1, y.shape[0]))
    return exp_vals, last_level, last_s


def sgt(y, seasonality, seasonality2=1):
    # heuristically, standard derivation of Cauchy prior depends on the max value of data
    cauchy_sd = np.max(y) / 150

    nu = numpyro.sample("nu", dist.Uniform(2, 20))
    powx = numpyro.sample("powx", dist.Uniform(0, 1))
    sigma = numpyro.sample("sigma", dist.HalfCauchy(cauchy_sd))
    offset_sigma = numpyro.sample("offset_sigma", dist.TruncatedCauchy(low=1e-10, loc=1e-10,
                                                                       scale=cauchy_sd))

    coef_trend = numpyro.sample("coef_trend", dist.Cauchy(0, cauchy_sd))
    pow_trend_beta = numpyro.sample("pow_trend_beta", dist.Beta(1, 1))
    # pow_trend takes values from -0.5 to 1
    pow_trend = 1.5 * pow_trend_beta - 0.5
    pow_season = numpyro.sample("pow_season", dist.Beta(1, 1))

    level_sm = numpyro.sample("level_sm", dist.Beta(1, 2))
    s_sm = numpyro.sample("s_sm", dist.Uniform(0, 1))
    init_s = numpyro.sample("init_s", dist.Cauchy(0, y[:seasonality] * 0.3))

    exp_val, last_level, last_s = scan_exp_val(
        y, init_s, level_sm, s_sm, coef_trend, pow_trend, pow_season)
    omega = sigma * exp_val ** powx + offset_sigma
    numpyro.sample("y", dist.StudentT(nu, exp_val, omega), obs=y[1:])
    # we return last `level` and last `s` for forecasting
    return last_level, last_s


class ForecastingModel:
    def __init__(self,
                 seasonality=1, seasonality2=2,
                 seasonality_type="multiplicative",
                 error_size_method="std",
                 level_method="HW"):
        self.seasonality = seasonality
        self.seasonality2 = seasonality2

    @property
    def model(self):
        return self._model

    def __call__(self, data, covariates):
        self.model(data, covariates)


class SGT(ForecastingModel):
    def __init__(self):
        pass

        model = 
