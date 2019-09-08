import argparse
import os

import numpy as onp

from jax import random, vmap
from jax.config import config as jax_config
import jax.numpy as np

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.examples.datasets import UCBADMIT, load_dataset
from numpyro.mcmc import MCMC, NUTS


"""
The UCBadmit data is sourced from the study [1] of gender biased in graduate admissions at
UC Berkeley in Fall 1973:

    dept | male | applications | admit
   ------|------|--------------|-------
     0      1         825         512
     0      0         108          89
     1      1         560         353
     1      0          25          17
     2      1         325         120
     2      0         593         202
     3      1         417         138
     3      0         375         131
     4      1         191          53
     4      0         393          94
     5      1         373          22
     5      0         341          24

This example replicates the multilevel model `m_glmm5` at [3], which is used to evaluate whether
the data contain evidence of gender biased in admissions accross departments. This is a form of
Generalized Linear Mixed Models for binomial regression problem, which models
 - varying intercepts accross departments,
 - varying slopes (or the effects of being male) accross departments,
 - correlation between intercepts and slopes,
and uses non-centered parameterization (or whitening).

A more comprehensive explanation for binomial regression and non-centered parameterization can be
found in Chapter 10 (Counting and Classification) and Chapter 13 (Adventures in Covariance) of [2].

[1] Bickel, P. J., Hammel, E. A., and O'Connell, J. W. (1975), "Sex Bias in Graduate Admissions:
    Data from Berkeley", Science, 187(4175), 398-404.
[2] McElreath, R. (2018), "Statistical Rethinking: A Bayesian Course with Examples in R and Stan",
    Chapman and Hall/CRC.
[3] "https://github.com/rmcelreath/rethinking/tree/Experimental#multilevel-model-formulas"
"""


def glmm(dept, male, applications, admit):
    v_mu = numpyro.sample('v_mu', dist.Normal(0, np.array([4., 1.])))

    sigma = numpyro.sample('sigma', dist.HalfNormal(np.ones(2)))
    L_Rho = numpyro.sample('L_Rho', dist.LKJCholesky(2))
    scale_tril = sigma[..., np.newaxis] * L_Rho
    # non-centered parameterization
    num_dept = len(onp.unique(dept))
    z = numpyro.sample('z', dist.Normal(np.zeros((num_dept, 2)), 1))
    v = np.dot(scale_tril, z.T).T

    logits = v_mu[0] + v[dept, 0] + (v_mu[1] + v[dept, 1]) * male
    numpyro.sample('admit', dist.Binomial(applications, logits=logits), obs=admit)


def run_inference(dept, male, applications, admit, rng, args):
    kernel = NUTS(glmm)
    mcmc = MCMC(kernel, args.num_warmup, args.num_samples, args.num_chains)
    mcmc.run(rng, dept, male, applications, admit)
    return mcmc.get_samples()


def predict(dept, male, applications, z, rng):
    model = handlers.substitute(handlers.seed(glmm, rng), z)
    model_trace = handlers.trace(model).get_trace(dept, male, applications, admit=None)
    return model_trace['admit']['fn'].probs


def print_results(header, preds, dept, male, probs):
    columns = ['Dept', 'Male', 'ActualProb', 'Pred(p25)', 'Pred(p50)', 'Pred(p75)']
    header_format = '{:>10} {:>10} {:>10} {:>10} {:>10} {:>10}'
    row_format = '{:>10.0f} {:>10.0f} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}'
    quantiles = onp.quantile(preds, [0.25, 0.5, 0.75], axis=0)
    print('\n', header, '\n')
    print(header_format.format(*columns))
    for i in range(len(dept)):
        print(row_format.format(dept[i], male[i], probs[i], *quantiles[:, i]), '\n')


def main(args):
    jax_config.update('jax_platform_name', args.device)
    _, fetch_train = load_dataset(UCBADMIT, split='train', shuffle=False)
    dept, male, applications, admit = fetch_train()
    rng, rng_predict = random.split(random.PRNGKey(1))
    zs = run_inference(dept, male, applications, admit, rng, args)
    rngs = random.split(rng_predict, args.num_samples * args.num_chains)
    pred_probs = vmap(lambda z, rng: predict(dept, male, applications, z, rng))(zs, rngs)
    header = '=' * 30 + 'glmm - TRAIN' + '=' * 30
    print_results(header, pred_probs, dept, male, admit / applications)


if __name__ == '__main__':
    assert numpyro.__version__.startswith('0.2.0')
    parser = argparse.ArgumentParser(description='UCBadmit gender discrimination using HMC')
    parser.add_argument('-n', '--num-samples', nargs='?', default=2000, type=int)
    parser.add_argument('--num-warmup', nargs='?', default=500, type=int)
    parser.add_argument('--num-chains', nargs='?', default=1, type=int)
    parser.add_argument('--device', default='cpu', type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    if args.device == 'cpu' and args.num_chains <= os.cpu_count():
        os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count={}'.format(
            args.num_chains)

    main(args)
