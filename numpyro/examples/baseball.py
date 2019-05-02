import argparse
import numpyro.distributions as dist
import jax.numpy as np
import jax.random as random

from numpyro.examples.datasets import load_dataset, BASEBALL
from numpyro.handlers import sample, substitute, seed
from numpyro.hmc_util import initialize_model
from numpyro.mcmc import hmc
from numpyro.util import fori_collect, control_flow_prims_disabled

"""
Original example from 

Example has been adapted from [1]. It demonstrates how to do Bayesian inference using
NUTS (or, HMC) in Pyro, and use of some common inference utilities.

As in the Stan tutorial, this uses the small baseball dataset of Efron and Morris [2]
to estimate players' batting average which is the fraction of times a player got a
base hit out of the number of times they went up at bat.

The dataset separates the initial 45 at-bats statistics from the remaining season.
We use the hits data from the initial 45 at-bats to estimate the batting average
for each player. We then use the remaining season's data to validate the predictions
from our models.

Three models are evaluated:
 - Complete pooling model: The success probability of scoring a hit is shared
     amongst all players.
 - No pooling model: Each individual player's success probability is distinct and
     there is no data sharing amongst players.
 - Partial pooling model: A hierarchical model with partial data sharing.


We recommend Radford Neal's tutorial on HMC ([3]) to users who would like to get a
more comprehensive understanding of HMC and its variants, and to [4] for details on
the No U-Turn Sampler, which provides an efficient and automated way (i.e. limited
hyper-parameters) of running HMC on different problems.

[1] Carpenter B. (2016), ["Hierarchical Partial Pooling for Repeated Binary Trials"]
    (http://mc-stan.org/users/documentation/case-studies/pool-binary-trials.html).
[2] Efron B., Morris C. (1975), "Data analysis using Stein's estimator and its
    generalizations", J. Amer. Statist. Assoc., 70, 311-319.
[3] Neal, R. (2012), "MCMC using Hamiltonian Dynamics",
    (https://arxiv.org/pdf/1206.1901.pdf)
[4] Hoffman, M. D. and Gelman, A. (2014), "The No-U-turn sampler: Adaptively setting
    path lengths in Hamiltonian Monte Carlo", (https://arxiv.org/abs/1111.4246)
"""


DATA_URL = "https://d2fefpcigoriu7.cloudfront.net/datasets/EfronMorrisBB.txt"


def fully_pooled(at_bats, hits=None):
    r"""
    Number of hits in $K$ at bats for each player has a Binomial
    distribution with a common probability of success, $\phi$.

    :param (np.DeviceArray) at_bats: Number of at bats for each player.
    :param (np.DeviceArray) hits: Number of hits for the given at bats.
    :return: Number of hits predicted by the model.
    """
    phi_prior = dist.Uniform(np.array([0.]), np.array([1.]))
    phi = sample("phi", phi_prior)
    return sample("obs", dist.Binomial(phi, at_bats), obs=hits)


def not_pooled(at_bats, hits=None):
    r"""
    Number of hits in $K$ at bats for each player has a Binomial
    distribution with independent probability of success, $\phi_i$.

    :param (np.DeviceArray) at_bats: Number of at bats for each player.
    :param (np.DeviceArray) hits: Number of hits for the given at bats.
    :return: Number of hits predicted by the model.
    """
    num_players = at_bats.shape[0]
    phi_prior = dist.Uniform(np.zeros((num_players,)),
                             np.ones((num_players,)))
    phi = sample("phi", phi_prior)
    return sample("obs", dist.Binomial(phi, at_bats), obs=hits)


def partially_pooled(at_bats, hits=None):
    r"""
    Number of hits has a Binomial distribution with independent
    probability of success, $\phi_i$. Each $\phi_i$ follows a Beta
    distribution with concentration parameters $c_1$ and $c_2$, where
    $c_1 = m * kappa$, $c_2 = (1 - m) * kappa$, $m ~ Uniform(0, 1)$,
    and $kappa ~ Pareto(1, 1.5)$.

    :param (np.DeviceArray) at_bats: Number of at bats for each player.
    :param (np.DeviceArray) hits: Number of hits for the given at bats.
    :return: Number of hits predicted by the model.
    """
    num_players = at_bats.shape[0]
    m = sample("m", dist.Uniform(np.array([0.]), np.array([1.])))
    kappa = sample("kappa", dist.Pareto(np.array([1.]), np.array([1.5])))
    shape = np.shape(kappa)[:np.ndim(kappa) - 1] + (num_players,)
    phi_prior = dist.Beta(np.broadcast_to(m * kappa, shape),
                          np.broadcast_to((1 - m) * kappa, shape))
    phi = sample("phi", phi_prior)
    return sample("obs", dist.Binomial(phi, at_bats), obs=hits)


def partially_pooled_with_logit(at_bats, hits=None):
    r"""
    Number of hits has a Binomial distribution with a logit link function.
    The logits $\alpha$ for each player is normally distributed with the
    mean and scale parameters sharing a common prior.

    :param (np.DeviceArray) at_bats: Number of at bats for each player.
    :param (np.DeviceArray) hits: Number of hits for the given at bats.
    :return: Number of hits predicted by the model.
    """
    num_players = at_bats.shape[0]
    loc = sample("loc", dist.Normal(np.array([-1.]), np.array([1.])))
    scale = sample("scale", dist.HalfCauchy(np.array([1.])))
    shape = np.shape(loc)[:np.ndim(loc) - 1] + (num_players,)
    alpha = sample("alpha", dist.Normal(np.broadcast_to(loc, shape),
                                        np.broadcast_to(scale, shape)))
    return sample("obs", dist.BinomialWithLogits(alpha, at_bats), obs=hits)


def run_inference(model, at_bats, hits, rng, args):
    init_params, potential_fn, transform_fn = initialize_model(rng, model,
                                                               (at_bats, hits), {})
    with control_flow_prims_disabled():
        init_kernel, sample_kernel = hmc(potential_fn, algo='NUTS')
        hmc_state = init_kernel(init_params, args.num_warmup_steps)
        hmc_states = fori_collect(args.num_samples, sample_kernel, hmc_state,
                                  transform=lambda hmc_state: transform_fn(hmc_state.z),
                                  progbar=True)
    return hmc_states


def predict(model, at_bats, z, rng, player_names, hits):
    model_name = model.__name__
    model = substitute(seed(model, rng), z)
    predictions = model(at_bats)
    mean, std = np.mean(predictions, 0), np.std(predictions, 0)
    print_out('=' * 15 + model_name + '=' * 15,
              mean,
              std,
              player_names,
              hits)


def print_out(header, mean, std, player_names, hits):
    columns = ['', 'Mean', 'Std', 'ActualHits']
    header_format = '{:>20} {:>7} {:>7} {:>14}'
    row_format = '{:>20} {:>7.2f} {:>7.2f} {:>14}'
    print('\n\n')
    print(header)
    print(header_format.format(*columns))
    for i, p in enumerate(player_names):
        print(row_format.format(p, mean[i], std[i], hits[i]))
    print('\n\n')


def main(args):
    _, fetch = load_dataset(BASEBALL, split='train', shuffle=False)
    train, player_names = fetch(0)
    at_bats, hits = train[:, 0], train[:, 1]
    for i, model in enumerate((#fully_pooled,
                               not_pooled,
                               #partially_pooled,
                               # partially_pooled_with_logit,
                               )):
        rng, rng_predict = random.split(random.PRNGKey(i))
        zs = run_inference(model, at_bats, hits, random.PRNGKey(i), args)
        predict(model, at_bats, zs, rng_predict, player_names, hits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseball batting average using HMC")
    parser.add_argument("-n", "--num-samples", nargs="?", default=20, type=int)
    parser.add_argument("--num-warmup-steps", nargs='?', default=10, type=int)
    parser.add_argument('--device', default='cpu', type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()
    main(args)
