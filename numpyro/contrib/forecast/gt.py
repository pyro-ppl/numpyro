# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from jax import lax
import jax.numpy as np

import numpyro
import numpyro.distributions as dist

from .forecaster import ForecastingModel


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


class GT:
    def __init__(self,
                 seasonality=1, seasonality2=1,
                 seasonality_type="multiplicative",  # generalized
                 error_size_method="std",  # innov
                 level_method="HW",  # seasAvg, HW_sAvg
                 hyperparameters={}):
        assert seasonality >= seasonality2
        self.seasonality = seasonality
        self.seasonality2 = seasonality2
        assert seasonality_type in ["multiplicative", "generalized"]
        self.seasonality_type = seasonality_type
        assert error_size_method in ["std", "innov"]
        self.error_size_method = error_size_method
        assert level_method in ["HW", "seasAvg", "HW_sAvg"]
        self.level_method = level_method
        self.hyperparameters = {**DEFAULT_HYPERPARAMETERS, **hyperparameters}

    def __call__(self, data, covariates):
        if self.seasonality == 1 and self.seasonality2 == 1:
            return lgt(data, covariates)
        elif self.seasonality2 == 1:
            return sgt(data, covariates)
        else:
            return s2gt(data, covariates)


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


# src: https://github.com/cbergmeir/Rlgt/blob/master/Rlgt/src/stan_files/SGT.stan
# TODO: fractional seasonality
def sgt(y, xreg, seasonality, seasonality_type, error_size_method, level_method, hyperparameters):
    N = y.shape[0]
    xreg_train, xreg_test = xreg[:N], xreg[N:]
    nu = numpyro.param("nu",
                       (hyperparameters["min_nu"] + hyperparameters["max_nu"]) / 2,
                       dist.constraints.interval(hyperparameters["min_nu"],
                                                 hyperparameters["max_nu"]))
    powx = numpyro.param("powx", 0.5, dist.constraints.unit_interval)
    s_sm = numpyro.param("sSm", 0.5, dist.constraints.unit_interval)

    cauchy_sd = np.max(y) / hyperparameters["cauchy_sd_div"]
    sigma = numpyro.sample("sigma", dist.HalfCauchy(cauchy_sd))
    offset_sigma = numpyro.sample("offsetSigma",
                                  dist.TruncatedCauchy(hyperparameters["min_sigma"],
                                                       hyperparameters["min_sigma"],
                                                       cauchy_sd))
    coef_trend = numpyro.sample("coefTrend", dist.Cauchy(0, cauchy_sd))
    pow_trend_beta = numpyro.sample("powTrendBeta",
                                    dist.Beta(hyperparameters["pow_trend_alpha"],
                                              hyperparameters["pow_trend_beta"]))
    pow_trend = (hyperparameters["max_pow_trend"] - hyperparameters["min_pow_trend "]) \
        * pow_trend_beta + hyperparameters["min_pow_trend"]
    lev_sm = numpyro.sample("levSm", dist.Beta(1, 2))

    if xreg.size > 0:
        reg_cauchy_sd = np.mean(y) / np.mean(xreg, 0) / hyperparameters["cauchy_sd_div"]
        reg_coef = numpyro.sample("regCoef", dist.Cauchy(0, reg_cauchy_sd))
        reg0_cauchy_sd = np.mean(reg_cauchy_sd) * 10
        reg_offset = numpyro.sample("regOffset", dist.Cauchy(0, reg0_cauchy_sd))
        r = np.dot(xreg_train, reg_coef) + reg_offset
    else:
        r = np.zeros(N)

    if seasonality_type == "multiplicative":
        init_s = numpyro.sample("init_s", dist.Cauchy(0, 4))
    else:
        pow_season = numpyro.sample("pow_season",
                                    dist.Beta(hyperparameters["pow_season_alpha"],
                                              hyperparameters["pow_season_beta"]))
        init_s = numpyro.sample("init_s", dist.Cauchy(0, y[:seasonality] * 0.3))

    if level_method == "HW_sAvg":
        llevSm = numpyro.param("sSm", 0.5, dist.constraints.unit_interval)
    else:
        llevSm = None

    exp_val, last_level, last_s = scan_exp_val(
        y, init_s, level_sm, s_sm, coef_trend, pow_trend, pow_season)

    if error_size_method == "innov":
        innov_size_init = numpyro.sample("innovSizeInit",
                                         dist.TruncatedCauchy(0, y[0] / 100, cauchy_sd))
        omega = sigma * smoothed_innov_size + offset_sigma
    else:
        omega = sigma * exp_val ** powx + offset_sigma

    numpyro.sample("y", dist.StudentT(nu, exp_val, omega), obs=y[1:])

    # TODO: forecast on xreg_test
