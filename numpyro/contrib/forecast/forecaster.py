# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from jax import random

import numpyro
from numpyro.infer import MCMC, NUTS, Predictive


class Forecaster:
    """
    Forecaster for univariate time series models. These models should take
    two arguments: `data` and `covariates`.

    On initialization, this will run MCMC to get posterior samples of the model.

    After construction, this can be called to generate sample forecasts.

    :param callable model: A forecasting model.
    :param data: A 1D time series data.
    :param covariates: An array of covariates with time dimension -2. For models not
        using covariates, pass a shaped empty array ``np.empty((duration, 0))``.
    :param int num_warmup: number of MCMC warmup steps.
    :param int num_samples: number of MCMC samples.
    :param int num_chains: number of parallel MCMC chains.
    :param int seed: initial random seed.
    """
    def __init__(self, model, data, covariates=None, *,
                 num_warmup=2500, num_samples=2500, num_chains=1, seed=0):
        self.model = model
        covariates = np.empty((data.shape[0], 0)) if covariates is None else covariates
        key_mcmc, self.rng_key = random.split(random.PRNGKey(seed))
        if num_chains > 1:
            numpyro.set_host_device_count(num_chains)
        mcmc = MCMC(NUTS(model), num_warmup, num_samples, num_chains)
        mcmc.run(key_mcmc, data, covariates)
        mcmc.print_summary()
        self._samples = mcmc.get_samples()

    def __call__(self, data, covariates):
        key_forecast, self.rng_key = random.split(self.rng_key)
        samples = self._samples  # XXX: should we resample?
        forecast = Predictive(self.model, samples)(key_forecast, data, covariates)["y"]
        return forecast.copy()
