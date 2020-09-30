"""
Proportion Test
===============
You are managing a business and want to test if calling your customers will
increase their chance of making a purchase. You get 100,000 customer records and call
roughly half of them and record if they make a purchase in the next three months.
You do the same for the half that did not get called. After three months, the data is in -
did calling help?

This example answers this question by estimating a logistic regression model where the
covariates are whether the customer got called and their gender. We place a multivariate
normal prior on the regression coefficients. We report the 95% highest posterior
density interval for the effect of making a call.
"""


import argparse
import os
from typing import Tuple

from jax import random
import jax.numpy as jnp
from jax.scipy.special import expit

import numpyro
from numpyro.diagnostics import hpdi
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


def make_dataset(rng_key) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Make simulated dataset where potential customers who get a
    sales calls have ~2% higher chance of making another purchase.
    """
    key1, key2, key3 = random.split(rng_key, 3)

    num_calls = 51342
    num_no_calls = 48658

    made_purchase_got_called = dist.Bernoulli(0.084).sample(key1, sample_shape=(num_calls,))
    made_purchase_no_calls = dist.Bernoulli(0.061).sample(key2, sample_shape=(num_no_calls,))

    made_purchase = jnp.concatenate([made_purchase_got_called, made_purchase_no_calls])

    is_female = dist.Bernoulli(0.5).sample(key3, sample_shape=(num_calls + num_no_calls,))
    got_called = jnp.concatenate([jnp.ones(num_calls), jnp.zeros(num_no_calls)])
    design_matrix = jnp.hstack([jnp.ones((num_no_calls + num_calls, 1)),
                               got_called.reshape(-1, 1),
                               is_female.reshape(-1, 1)])

    return design_matrix, made_purchase


def model(design_matrix: jnp.ndarray, outcome: jnp.ndarray = None) -> None:
    """
    Model definition: Log odds of making a purchase is a linear combination
    of covariates. Specify a Normal prior over regression coefficients.
    :param design_matrix: Covariates. All categorical variables have been one-hot
        encoded.
    :param outcome: Binary response variable. In this case, whether or not the
        customer made a purchase.
    """

    beta = numpyro.sample('coefficients', dist.MultivariateNormal(loc=0.,
                                                                  covariance_matrix=jnp.eye(design_matrix.shape[1])))
    logits = design_matrix.dot(beta)

    with numpyro.plate('data', design_matrix.shape[0]):
        numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=outcome)


def print_results(coef: jnp.ndarray, interval_size: float = 0.95) -> None:
    """
    Print the confidence interval for the effect size with interval_size
    probability mass.
    """

    baseline_response = expit(coef[:, 0])
    response_with_calls = expit(coef[:, 0] + coef[:, 1])

    impact_on_probability = hpdi(response_with_calls - baseline_response, prob=interval_size)

    effect_of_gender = hpdi(coef[:, 2], prob=interval_size)

    print(f"There is a {interval_size * 100}% probability that calling customers "
          "increases the chance they'll make a purchase by "
          f"{(100 * impact_on_probability[0]):.2} to {(100 * impact_on_probability[1]):.2} percentage points."
          )

    print(f"There is a {interval_size * 100}% probability the effect of gender on the log odds of conversion "
          f"lies in the interval ({effect_of_gender[0]:.2}, {effect_of_gender[1]:.2f})."
          " Since this interval contains 0, we can conclude gender does not impact the conversion rate.")


def run_inference(design_matrix: jnp.ndarray, outcome: jnp.ndarray,
                  rng_key: jnp.ndarray,
                  num_warmup: int,
                  num_samples: int, num_chains: int,
                  interval_size: float = 0.95) -> None:
    """
    Estimate the effect size.
    """

    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup, num_samples, num_chains,
                progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True)
    mcmc.run(rng_key, design_matrix, outcome)

    # 0th column is intercept (not getting called)
    # 1st column is effect of getting called
    # 2nd column is effect of gender (should be none since assigned at random)
    coef = mcmc.get_samples()['coefficients']
    print_results(coef, interval_size)


def main(args):
    rng_key, _ = random.split(random.PRNGKey(3))
    design_matrix, response = make_dataset(rng_key)
    run_inference(design_matrix, response, rng_key,
                  args.num_warmup,
                  args.num_samples,
                  args.num_chains,
                  args.interval_size)


if __name__ == '__main__':
    assert numpyro.__version__.startswith('0.4.0')
    parser = argparse.ArgumentParser(description='Testing whether  ')
    parser.add_argument('-n', '--num-samples', nargs='?', default=500, type=int)
    parser.add_argument('--num-warmup', nargs='?', default=1500, type=int)
    parser.add_argument('--num-chains', nargs='?', default=1, type=int)
    parser.add_argument('--interval-size', nargs='?', default=0.95, type=float)
    parser.add_argument('--device', default='cpu', type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
