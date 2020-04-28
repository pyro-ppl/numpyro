"""
Proportion Test
===============================
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
from typing import Any, Callable, Tuple

import jax.numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import random
from jax.scipy.special import expit
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS
from scipy import stats


def make_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """
    Make simulated dataset where potential customers who get a 
    sales calls have ~2% higher chance of making another purchase.
    """
    made_purchase_got_called = stats.bernoulli.rvs(0.084, size=51342)
    made_purchase_no_calls = stats.bernoulli.rvs(0.061, size=48658)
    
    made_purchase = np.concatenate([made_purchase_got_called, made_purchase_no_calls])
    
    is_female = stats.bernoulli.rvs(0.5, size=100000)
    got_called = np.concatenate([np.ones(51342), np.zeros(48658)])
    design_matrix = np.hstack([np.ones((100000, 1)),
                               got_called.reshape(-1, 1),
                               is_female.reshape(-1, 1)])
    
    return design_matrix, made_purchase


def model(design_matrix: np.ndarray, outcome: np.ndarray = None) -> None:
    """
    Model definition: Log odds of making a purchase is a linear combination
    of covariates. Specify aNormal prior over regression coefficients.
    :param design_matrix: Covariates. All categorical variables have been one-hot
        encoded.
    :param outcome: Binary response variable. In this case, whether or not the
        customer made a purchase.
    """
    
    # Use multivariate normal prior in case we want to add more covariates later.
    beta = numpyro.sample('coefficients', dist.MultivariateNormal(loc=0.,
                                                                  covariance_matrix=np.eye(design_matrix.shape[1])))
    logits = design_matrix.dot(beta)
    
    with numpyro.plate('data', design_matrix.shape[0]):
            numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=outcome)
    

def print_results(coef: np.ndarray, interval_size: float = 0.95) -> None:
    """
    Print the confidence interval for the effect size with interval_size
    probability mass.
    """

    baseline = hpdi(coef[:, 0], prob=0.95)
    effect_size = hpdi(coef[:, 1], prob=0.95)
    
    baseline_response = expit(baseline)
    response_with_calls = expit(baseline + effect_size)
    
    impact_on_probability = response_with_calls - baseline_response
    
    print(f"There is a {interval_size * 100}% probability that calling customers "
          "increases the chance they'll make a purchase by "
          f"{(100 * impact_on_probability[0]):.2} to {(100 * impact_on_probability[1]):.2} percentage points."
         )
    

def run_inference(design_matrix: np.ndarray, outcome: np.ndarray,
                  rng_key: np.ndarray,
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
    design_matrix, response = make_dataset()
    rng_key, rng_key_predict = random.split(random.PRNGKey(1))
    run_inference(design_matrix, response, rng_key,
                  args.num_samples,
                  args.num_warmup,
                  args.num_chains,
                  args.interval_size)


if __name__ == '__main__':
    assert numpyro.__version__.startswith('0.2.4')
    parser = argparse.ArgumentParser(description='Testing whether  ')
    parser.add_argument('-n', '--num-samples', nargs='?', default=2000, type=int)
    parser.add_argument('--num-warmup', nargs='?', default=500, type=int)
    parser.add_argument('--num-chains', nargs='?', default=1, type=int)
    parser.add_argument('--interval-size', nargs='?', default=0.95, type=float)
    parser.add_argument('--device', default='cpu', type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
