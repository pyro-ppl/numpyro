# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from jax import random

import numpyro
from numpyro.infer import MCMC, NUTS, Predictive


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


class ForecastingModel:
    def __init__(self,
                 seasonality=1, seasonality2=2,
                 seasonality_type="multiplicative",
                 error_size_method="std",
                 level_method="HW",
                 hyperparameters={}):
        self.hyperparameters = {**DEFAULT_HYPERPARAMETERS, **hyperparameters}
        # TODO: update seasonality method

    def __call__(self, data, covariates=None):
        raise NotImplementedError


class Forecaster:
    def __init__(self, model, data, covariates,
                 num_warmup=5000, num_samples=5000, num_chains=1):
        if num_chains > 1:
            numpyro.set_host_device_count(num_chains)
        mcmc = MCMC(NUTS(model), num_warmup, num_samples, num_chains)
        mcmc.run(random.PRNGKey(0), data, covariates)
        self._samples = mcmc.get_samples()

    def __call__(self, data, covariates, num_samples=None):
        # TODO: resample self._samples if num_samples != None
        predictive = Predictive(self.model)
        return predictive(random.PRNGKey(0), data, covariates)
