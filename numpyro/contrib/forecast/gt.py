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


class GT:
    def __init__(self,
                 seasonality=1, seasonality2=1,
                 generalized_seasonality=False,
                 use_smoothed_error=False,
                 level_method="HW",  # seasAvg, HW_sAvg
                 hyperparameters={}):
        assert seasonality >= seasonality2
        self.seasonality = seasonality
        self.seasonality2 = seasonality2
        self.generalized_seasonality = generalized_seasonality
        self.use_smoothed_error = use_smoothed_error
        assert level_method in ["HW", "seasAvg", "HW_sAvg"]
        self.level_method = level_method
        self.hyperparameters = {**DEFAULT_HYPERPARAMETERS, **hyperparameters}

    def __call__(self, data, covariates):
        hypers = self.hyperparameters
        N = data.shape[0]
        duration = covariates.shape[0]
        assert covariates.shape[0] >= N
        assert N >= self.seasonality
        cauchy_sd = np.max(data) / hypers["cauchy_sd_div"]

        offset_sigma = numpyro.sample("offsetSigma", dist.HalfCauchy(cauchy_sd)) \
            + hypers["min_sigma"]
        coef_trend = numpyro.sample("coefTrend", dist.Cauchy(0, cauchy_sd))
        pow_trend_beta = numpyro.sample("powTrendBeta",
                                        dist.Beta(hypers["pow_trend_alpha"],
                                                  hypers["pow_trend_beta"]))
        pow_trend = (hypers["max_pow_trend"] - hypers["min_pow_trend "]) * pow_trend_beta \
            + hypers["min_pow_trend"]
        lev_sm = numpyro.sample("levSm", dist.Beta(1, 2))

        if covariates.shape[-1] > 0:
            reg_cauchy_sd = np.mean(data) / np.mean(covariates, 0) / hypers["cauchy_sd_div"]
            reg_coef = numpyro.sample("regCoef", dist.Cauchy(0, reg_cauchy_sd))
            reg0_cauchy_sd = np.mean(reg_cauchy_sd) * 10
            reg_offset = numpyro.sample("regOffset", dist.Cauchy(0, reg0_cauchy_sd))
            r = np.dot(covariates, reg_coef) + reg_offset
        else:
            r = np.zeros(covariates.shape[0])
        y = data - r[:N]

        innov_sm = innov_size_init = None
        if self.use_smoothed_error:
            innov_sm = numpyro.sample("innovSm", dist.Uniform())
            innov_size_init = numpyro.sample("innovSizeInit",
                                             dist.TruncatedCauchy(0, data[0] / 100, cauchy_sd))

        # TODO: fractional seasonality
        if self.seasonality > 1:
            s_sm = numpyro.sample("sSm", dist.Uniform())
            if self.generalized_seasonality:
                init_s = numpyro.sample("initS", dist.Cauchy(0, data[:self.seasonality] * 0.3))
                pow_season = numpyro.sample("powSeason",
                                            dist.Beta(hypers["pow_season_alpha"],
                                                      hypers["pow_season_beta"]))
            else:
                init_s = numpyro.sample("initS", dist.Cauchy(0, 4))
                pow_season = None

            llev_sm = self.level_method
            if llev_sm == "HW_sAvg":
                llev_sm = numpyro.sample("llevSm", dist.Uniform())

            if self.seasonality2 > 1:
                pass  # TODO: add S2GT parameters
            else:
                exp_val, smoothed_innov_size = sgt_scan(y, duration, coef_trend, pow_trend,
                                                        lev_sm, innov_sm, innov_size_init,
                                                        s_sm, init_s, pow_season, llev_sm)
        else:
            loc_trend_fract = numpyro.sample("locTrendFract", dist.Uniform())
            b_sm = numpyro.sample("bSm", dist.Uniform())
            b_init = numpyro.sample("bInit", dist.Cauchy(0, cauchy_sd))
            exp_val, smoothed_innov_size = lgt_scan(y, duration, coef_trend, pow_trend,
                                                    lev_sm, innov_sm, innov_size_init,
                                                    loc_trend_fract, b_sm, b_init)

        exp_val = exp_val + r
        if duration > N:  # forecasting
            exp_val = np.clip(exp_val[N:], hypers["min_val"], hypers["max_val"])
            if smoothed_innov_size is not None:
                smoothed_innov_size = smoothed_innov_size[N:]

        nu = numpyro.sample("nu", dist.Uniform(hypers["min_nu"], hypers["max_nu"]))
        sigma = numpyro.sample("sigma", dist.HalfCauchy(cauchy_sd))
        if self.use_smoothed_error:
            omega = sigma * smoothed_innov_size + offset_sigma
        else:
            powx = numpyro.param("powx", 0.5, dist.constraints.unit_interval)
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
        level, s, l0, moving_sum, smoothed_innov_size, y_season = carry
        if duration > N:
            level = np.clip(level, 0.)  # prevent NaN when forecasting

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

        if innov_sm is not None:
            innov_size = innov_sm * np.abs(y_t - exp_val) + (1 - innov_sm) * smoothed_innov_size
            innov_size = np.where(t >= N, smoothed_innov_size, innov_size)

        if y_season is not None:
            y_season = np.concatenate(y_season[1:], y_t[None])
        return (level, s, l0, moving_sum, innov_size, y_season), (exp_val, smoothed_innov_size)

    N = y.shape[0]
    seasonality = init_s.shape[0]
    s = np.concatenate([init_s[1:], init_s[:1]])
    l0 = y[0]
    if pow_season is None:
        s = nn.softmax(s) * seasonality
        l0 = l0 / s[-1]  # y[0] / softmax(init_s)[0] / seasonality
    y_season = None if duration == N else np.concatenate(np.zeros(seasonality - 1), y[:1])
    _, (exp_val, smoothed_innov_size) = lax.scan(
        scan_fn, (l0, s, l0, y[0], innov_size_init, y_season), np.arange(1, duration))
    return exp_val, smoothed_innov_size
