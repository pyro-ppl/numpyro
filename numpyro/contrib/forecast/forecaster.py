# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from jax import random

import numpyro
from numpyro.infer import MCMC, NUTS, Predictive


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
        samples = self._samples
        return Predictive(self.model, samples)(random.PRNGKey(0), data, covariates)
