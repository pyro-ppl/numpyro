# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

import jax
import jax.numpy as jnp
from jax import random

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.dcc.dcc import DCC

PRIOR_MEAN, PRIOR_STD = 0.0, 1.0
LIKELIHOOD1_STD = 2.0
LIKELIHOOD2_STD = 0.62177


def model(y):
    z = numpyro.sample("z", dist.Normal(PRIOR_MEAN, PRIOR_STD))
    model1 = numpyro.sample("model1", dist.Bernoulli(0.5), infer={"branching": True})
    sigma = LIKELIHOOD1_STD if model1 == 0 else LIKELIHOOD2_STD
    with numpyro.plate("data", y.shape[0]):
        numpyro.sample("obs", dist.Normal(z, sigma), obs=y)


def log_marginal_likelihood(data, likelihood_std, prior_mean, prior_std):
    """
    Calculate the marginal likelihood of a model with Normal likelihood, unknown mean,
    and Normal prior.

    Taken from Section 2.5 at https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf.
    """
    num_data = data.shape[0]
    likelihood_var = jnp.square(likelihood_std)
    prior_var = jnp.square(prior_std)

    first_term = (
        jnp.log(likelihood_std)
        - num_data * jnp.log(jnp.sqrt(2 * math.pi) * likelihood_std)
        + 0.5 * jnp.log(num_data * prior_var + likelihood_var)
    )
    second_term = -(jnp.sum(jnp.square(data)) / (2 * likelihood_var)) - (
        jnp.square(prior_mean) / (2 * prior_var)
    )
    third_term = (
        (prior_var * jnp.square(num_data) * jnp.square(jnp.mean(data)) / likelihood_var)
        + (likelihood_var * jnp.square(prior_mean) / prior_var)
        + 2 * num_data * jnp.mean(data) * prior_mean
    ) / (2 * (num_data * prior_var + likelihood_var))
    return first_term + second_term + third_term


def main():
    rng_key = random.PRNGKey(0)

    rng_key, subkey = random.split(rng_key)
    y_train = dist.Normal(0, 1).sample(subkey, (200,))

    mcmc_kwargs = dict(
        num_warmup=500,
        num_samples=1000,
        num_chains=2,
    )

    dcc = DCC(model, mcmc_kwargs=mcmc_kwargs)
    rng_key, subkey = random.split(rng_key)
    dcc_result = dcc.run(subkey, y_train)
    slp_weights = jnp.array([dcc_result.slp_weights["0"], dcc_result.slp_weights["1"]])
    assert jnp.allclose(1.0, jnp.sum(slp_weights))

    slp1_lml = log_marginal_likelihood(y_train, LIKELIHOOD1_STD, PRIOR_MEAN, PRIOR_STD)
    slp2_lml = log_marginal_likelihood(y_train, LIKELIHOOD2_STD, PRIOR_MEAN, PRIOR_STD)
    lmls = jnp.array([slp1_lml, slp2_lml])
    analytic_weights = jnp.exp(lmls - jax.scipy.special.logsumexp(lmls))
    assert jnp.allclose(analytic_weights, slp_weights)


if __name__ == "__main__":
    main()
