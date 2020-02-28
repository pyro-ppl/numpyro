# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from jax import lax, nn
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


class GlobalTrendModel:
    def __init__(self,
                 seasonality=1, seasonality2=1,
                 use_smoothed_error=False,
                 generalized_seasonality=False,
                 level_method="HW",  # seasAvg, HW_sAvg
                 hyperparameters={}):
        assert seasonality2 == 1 or seasonality2 > seasonality
        self.seasonality = seasonality
        self.seasonality2 = seasonality2
        self.use_smoothed_error = use_smoothed_error
        self.generalized_seasonality = generalized_seasonality
        assert level_method in ["HW", "seasAvg", "HW_sAvg"]
        self.level_method = level_method
        self.hyperparameters = {**DEFAULT_HYPERPARAMETERS, **hyperparameters}

    def __call__(self, data, covariates):
        hypers = self.hyperparameters
        N = data.shape[0]
        duration = covariates.shape[0]
        assert N >= self.seasonality + self.seasonality2
        assert duration >= N
        cauchy_sd = np.max(data) / hypers["cauchy_sd_div"]

        if covariates.shape[-1] > 0:
            reg_cauchy_sd = np.mean(data) / np.mean(covariates, 0) / hypers["cauchy_sd_div"]
            reg_coef = numpyro.sample("reg_coef", dist.Cauchy(0, reg_cauchy_sd))
            reg0_cauchy_sd = np.mean(reg_cauchy_sd) * 10
            reg_offset = numpyro.sample("reg_offset", dist.Cauchy(0, reg0_cauchy_sd))
            r = np.dot(covariates, reg_coef) + reg_offset
        else:
            r = np.zeros(covariates.shape[0])
        y = data - r[:N]

        coef_trend = numpyro.sample("coef_trend", dist.Cauchy(0, cauchy_sd))
        pow_trend_beta = numpyro.sample("pow_trend_beta",
                                        dist.Beta(hypers["pow_trend_alpha"],
                                                  hypers["pow_trend_beta"]))
        pow_trend = (hypers["max_pow_trend"] - hypers["min_pow_trend"]) * pow_trend_beta \
            + hypers["min_pow_trend"]
        lev_sm = numpyro.sample("lev_sm", dist.Beta(1, 2))

        innov_sm = innov_size_init = None
        if self.use_smoothed_error:
            innov_sm = numpyro.sample("innov_sm", dist.Uniform())
            innov_size_init = numpyro.sample("innov_size_init",
                                             dist.TruncatedCauchy(0, data[0] / 100, cauchy_sd))

        # TODO: fractional seasonality
        if self.seasonality > 1:
            s_sm = numpyro.sample("sSm", dist.Uniform())
            if self.generalized_seasonality:
                init_s = numpyro.sample("init_s", dist.Cauchy(0, data[:self.seasonality] * 0.3))
                pow_season = numpyro.sample("pow_season",
                                            dist.Beta(hypers["pow_season_alpha"],
                                                      hypers["pow_season_beta"]))
            else:
                init_s = numpyro.sample("init_s", dist.Cauchy(0, 4))
                pow_season = None

            llev_sm = self.level_method
            if llev_sm == "HW_sAvg":
                llev_sm = numpyro.sample("llev_sm", dist.Uniform())

            if self.seasonality2 > self.seasonality:
                s2_sm = numpyro.sample("s2_sm", dist.Uniform())
                if self.generalized_seasonality:
                    init_s2 = numpyro.sample("init_s2",
                                             dist.Cauchy(0, data[:self.seasonality2] * 0.3))
                    pow_season2 = numpyro.sample("pow_season2",
                                                 dist.Beta(hypers["pow_season_alpha"],
                                                           hypers["pow_season_beta"]))
                else:
                    init_s2 = numpyro.sample("init_s2", dist.Cauchy(0, 4))
                    pow_season2 = None
                exp_val, smoothed_innov_size = s2gt_scan(y, duration, coef_trend, pow_trend,
                                                         lev_sm, innov_sm, innov_size_init,
                                                         s_sm, init_s, pow_season,
                                                         s2_sm, init_s2, pow_season2, llev_sm)
            else:
                exp_val, smoothed_innov_size = sgt_scan(y, duration, coef_trend, pow_trend,
                                                        lev_sm, innov_sm, innov_size_init,
                                                        s_sm, init_s, pow_season, llev_sm)
        else:
            loc_trend_fract = numpyro.sample("loc_trend_fract", dist.Uniform())
            b_sm = numpyro.sample("b_sm", dist.Uniform())
            b_init = numpyro.sample("b_init", dist.Cauchy(0, cauchy_sd))
            exp_val, smoothed_innov_size = lgt_scan(y, duration, coef_trend, pow_trend,
                                                    lev_sm, innov_sm, innov_size_init,
                                                    loc_trend_fract, b_sm, b_init)

        exp_val = exp_val + r[1:]
        if duration > N:  # forecasting
            exp_val = np.clip(exp_val[N - 1:], hypers["min_val"], hypers["max_val"])
            if smoothed_innov_size is not None:
                smoothed_innov_size = smoothed_innov_size[N - 1:]

        nu = numpyro.sample("nu", dist.Uniform(hypers["min_nu"], hypers["max_nu"]))
        sigma = numpyro.sample("sigma", dist.HalfCauchy(cauchy_sd))
        offset_sigma = numpyro.sample("offset_sigma", dist.HalfCauchy(cauchy_sd)) \
            + hypers["min_sigma"]
        if self.use_smoothed_error:
            omega = sigma * smoothed_innov_size + offset_sigma
        else:
            powx = numpyro.sample("powx", dist.Uniform())
            omega = sigma * exp_val ** powx + offset_sigma

        if covariates.shape[0] == N:  # training
            numpyro.sample("y", dist.StudentT(nu, exp_val, omega), obs=data[1:])
        else:  # forecasting
            numpyro.sample("y", dist.StudentT(nu, exp_val, omega))


# src: https://github.com/cbergmeir/Rlgt/blob/master/Rlgt/src/stan_files/LGT.stan
def lgt_scan(y, duration, coef_trend, pow_trend, lev_sm, innov_sm, innov_size_init,
             loc_trend_fract, b_sm, b_init):

    def scan_fn(carry, t):
        level, b, smoothed_innov_size = carry
        if duration > N:
            level = np.clip(level, 0.)  # prevent NaN when forecasting

        exp_val = level + coef_trend * level ** pow_trend + loc_trend_fract * b
        y_t = np.where(t >= N, exp_val, y[t])
        new_level = lev_sm * y_t + (1 - lev_sm) * level
        b = b_sm * (new_level - level) + (1 - b_sm) * b

        innov_size = smoothed_innov_size
        if innov_sm is not None:
            innov_size = innov_sm * np.abs(y_t - exp_val) + (1 - innov_sm) * smoothed_innov_size
            innov_size = np.where(t >= N, smoothed_innov_size, innov_size)
        return (new_level, b, innov_size), (exp_val, smoothed_innov_size)

    N = y.shape[0]
    _, (exp_val, smoothed_innov_size) = lax.scan(
        scan_fn, (y[0], b_init, innov_size_init), np.arange(1, duration))
    return exp_val, smoothed_innov_size


# src: https://github.com/cbergmeir/Rlgt/blob/master/Rlgt/src/stan_files/SGT.stan
def sgt_scan(y, duration, coef_trend, pow_trend, lev_sm, innov_sm, innov_size_init,
             s_sm, init_s, pow_season, llev_sm):

    def scan_fn(carry, t):
        level, s, l0, moving_sum, y_season, smoothed_innov_size = carry
        level = np.clip(level, 0.) if duration > N else level  # prevent NaN when forecasting

        if pow_season is None:
            exp_val = (level + coef_trend * level ** pow_trend) * s[0]
            y_t = np.where(t >= N, exp_val, y[t])
            new_level_p = y_t / s[0]
        else:
            season = s[0] * level ** pow_season
            exp_val = level + coef_trend * level ** pow_trend + season
            y_t = np.where(t >= N, exp_val, y[t])
            new_level_p = y_t - season

        l0 = lev_sm * new_level_p + (1 - lev_sm) * l0
        if isinstance(llev_sm, str) and llev_sm == "HW":
            level = l0
        else:
            y_t_prev_season = y_season[0] if y_season is not None else y[t - seasonality]
            moving_sum = moving_sum + y_t - np.where(t >= seasonality, y_t_prev_season, 0.)
            if isinstance(llev_sm, str):  # seasAvg
                level = lev_sm * moving_sum / seasonality + (1 - lev_sm) * level
            else:
                level = llev_sm * l0 + (1 - llev_sm) * moving_sum / seasonality
            level = np.where(t >= seasonality, level, l0)

        if pow_season is None:
            seasonality_p = s_sm * y_t / level + (1 - s_sm) * s[0]
        else:
            seasonality_p = (s_sm * (y_t - level) / season + (1 - s_sm)) * s[0]
        s = np.concatenate([s[1:], np.where(t >= N, s[0], seasonality_p)[None]])
        y_season = None if duration == N else np.concatenate([y_season[1:], y_t[None]])

        innov_size = smoothed_innov_size
        if innov_sm is not None:
            innov_size = innov_sm * np.abs(y_t - exp_val) + (1 - innov_sm) * smoothed_innov_size
            innov_size = np.where(t >= N, smoothed_innov_size, innov_size)
        return (level, s, l0, moving_sum, y_season, innov_size), (exp_val, smoothed_innov_size)

    N = y.shape[0]
    seasonality = init_s.shape[0]
    s = np.concatenate([init_s[1:], init_s[:1]])
    l0 = y[0]
    if pow_season is None:
        s = nn.softmax(s) * seasonality
        l0 = l0 / s[-1]  # y[0] / softmax(init_s)[0] / seasonality
    y_season = None if duration == N else np.concatenate([np.zeros(seasonality - 1), y[:1]])
    _, (exp_val, smoothed_innov_size) = lax.scan(
        scan_fn, (l0, s, l0, y[0], y_season, innov_size_init), np.arange(1, duration))
    return exp_val, smoothed_innov_size


# src: https://github.com/cbergmeir/Rlgt/blob/master/Rlgt/src/stan_files/S2GT.stan
def s2gt_scan(y, duration, coef_trend, pow_trend, lev_sm, innov_sm, innov_size_init,
              s_sm, init_s, pow_season, s2_sm, init_s2, pow_season2, llev_sm):

    def scan_fn(carry, t):
        level, s, s2, l0, moving_sum, y_season, smoothed_innov_size = carry
        level = np.clip(level, 0.) if duration > N else level  # prevent NaN when forecasting

        if pow_season is None:
            s_s2 = s[0] * s2[0]
            exp_val = (level + coef_trend * level ** pow_trend) * s_s2
            y_t = np.where(t >= N, exp_val, y[t])
            new_level_p = y_t / s_s2
        else:
            season = s[0] * level ** pow_season
            season2 = s2[0] * level ** pow_season2
            s_s2 = season + season2
            exp_val = level + coef_trend * level ** pow_trend + s_s2
            y_t = np.where(t >= N, exp_val, y[t])
            new_level_p = y_t - s_s2

        l0 = lev_sm * new_level_p + (1 - lev_sm) * l0
        if isinstance(llev_sm, str) and llev_sm == "HW":
            level = l0
        else:
            y_t_prev_season = y_season[0] if y_season is not None else y[t - seasonality2]
            moving_sum = moving_sum + y_t - np.where(t >= seasonality2, y_t_prev_season, 0.)
            if isinstance(llev_sm, str):  # seasAvg
                level = lev_sm * moving_sum / seasonality2 + (1 - lev_sm) * level
            else:
                level = llev_sm * l0 + (1 - llev_sm) * moving_sum / seasonality2
            level = np.where(t >= seasonality2, level, l0)

        if pow_season is None:
            seasonality_p = s_sm * y_t / (level * s2[0]) + (1 - s_sm) * s[0]
            seasonality_p2 = s2_sm * y_t / (level * s[0]) + (1 - s2_sm) * s2[0]
        else:
            seasonality_p = (s_sm * (y_t - level - season2) / season + (1 - s_sm)) * s[0]
            seasonality_p2 = (s2_sm * (y_t - level - season) / season2 + (1 - s2_sm)) * s2[0]
        s = np.concatenate([s[1:], np.where(t >= N, s[0], seasonality_p)[None]])
        s2 = np.concatenate([s2[1:], np.where(t >= N, s2[0], seasonality_p2)[None]])
        y_season = None if duration == N else np.concatenate([y_season[1:], y_t[None]])

        innov_size = smoothed_innov_size
        if innov_sm is not None:
            innov_size = innov_sm * np.abs(y_t - exp_val) + (1 - innov_sm) * smoothed_innov_size
            innov_size = np.where(t >= N, smoothed_innov_size, innov_size)
        return (level, s, s2, l0, moving_sum, y_season, innov_size), (exp_val, smoothed_innov_size)

    N = y.shape[0]
    seasonality = init_s.shape[0]
    seasonality2 = init_s2.shape[0]
    s = np.concatenate([init_s[1:], init_s[:1]])
    s2 = np.concatenate([init_s2[1:], init_s2[:1]])
    l0 = y[0]
    if pow_season is None:
        s = nn.softmax(s) * seasonality
        s2 = nn.softmax(s2) * seasonality2
        l0 = l0 / (s[-1] * s2[-1])
    y_season = None if duration == N else np.concatenate(np.zeros(seasonality2 - 1), y[:1])
    _, (exp_val, smoothed_innov_size) = lax.scan(
        scan_fn, (l0, s, s2, l0, y[0], y_season, innov_size_init), np.arange(1, duration))
    return exp_val, smoothed_innov_size
