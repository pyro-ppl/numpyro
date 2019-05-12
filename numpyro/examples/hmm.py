import argparse

import numpy as onp

import jax.numpy as np
from jax import lax, random
from jax.config import config as jax_config
from jax.scipy.special import logsumexp

import numpyro.distributions as dist
from numpyro.handlers import sample
from numpyro.hmc_util import initialize_model
from numpyro.mcmc import hmc
from numpyro.util import fori_collect


"""
In this example, we will follow [1] to construct a semi-supervised Hidden Markov
Model for a generative model with observations are words and latent variables
are categories. Instead of automatically marginalizing all discrete latent
variables (as in [2]), we will use the "forward algorithm" (which exploits the
conditional independent of a Markov model - see [3]) to iteratively do this
marginalization.

The semi-supervised problem is chosen instead of an unsupervised one because it
is hard to make the inference works for an unsupervised model (see the
discussion [4]). On the other hand, this example also illustrates the usage of
JAX's `lax.scan` primitive. The primitive will greatly improve compiling and
sampling speed for the model.

[1] https://mc-stan.org/docs/2_19/stan-users-guide/hmms-section.html
[2] http://pyro.ai/examples/hmm.html
[3] https://en.wikipedia.org/wiki/Forward_algorithm
[4] https://discourse.pymc.io/t/how-to-marginalized-markov-chain-with-categorical/2230
"""


def simulate_data(rng, num_categories, num_words, num_supervised_data, num_unsupervised_data):
    rng, rng_transition, rng_emission = random.split(rng, 3)

    transition_prior = np.ones(num_categories)
    emission_prior = np.repeat(0.1, num_words)

    transition_prob = dist.Dirichlet(transition_prior).sample(key=rng_transition,
                                                              size=(num_categories,))
    emission_prob = dist.Dirichlet(emission_prior).sample(key=rng_emission, size=(num_categories,))

    start_prob = np.repeat(1. / num_categories, num_categories)
    categories, words = [], []
    for t in range(num_supervised_data + num_unsupervised_data):
        rng, rng_transition, rng_emission = random.split(rng, 3)
        if t == 0 or t == num_supervised_data:
            category = dist.Categorical(start_prob).sample(key=rng_transition)
        else:
            category = dist.Categorical(transition_prob[category]).sample(key=rng_transition)
        word = dist.Categorical(emission_prob[category]).sample(key=rng_emission)
        categories.append(category)
        words.append(word)

    # split into supervised data and unsupervised data
    categories, words = np.stack(categories), np.stack(words)
    supervised_categories = categories[:num_supervised_data]
    supervised_words = words[:num_supervised_data]
    unsupervised_words = words[num_supervised_data:]
    return (transition_prior, emission_prior, transition_prob, emission_prob,
            supervised_categories, supervised_words, unsupervised_words)


def forward_one_step(prev_log_prob, curr_word, transition_log_prob, emission_log_prob):
    log_prob_tmp = np.expand_dims(prev_log_prob, axis=1) + transition_log_prob
    log_prob = log_prob_tmp + emission_log_prob[:, curr_word]
    return logsumexp(log_prob, axis=0)


def forward_log_prob(init_log_prob, words, transition_lob_prob, emission_log_prob):
    # Note: The following naive implementation will make it very slow to compile
    # and do inference. So we use lax.scan instead.
    #
    # >>> log_prob = init_log_prob
    # >>> for word in words:
    # ...     log_prob = forward_one_step(log_prob, word, transition_log_prob, emission_log_prob)
    def scan_fn(log_prob, word):
        return forward_one_step(log_prob, word, transition_lob_prob, emission_log_prob), np.ones(())

    log_prob, _ = lax.scan(scan_fn, init_log_prob, words)
    return log_prob


def semi_supervised_hmm(transition_prior, emission_prior,
                        supervised_categories, supervised_words,
                        unsupervised_words):
    num_categories, num_words = transition_prior.shape[0], emission_prior.shape[0]
    transition_prob = sample('transition_prob', dist.Dirichlet(
        np.broadcast_to(transition_prior, (num_categories, num_categories))))
    emission_prob = sample('emission_prob', dist.Dirichlet(
        np.broadcast_to(emission_prior, (num_categories, num_words))))

    # models supervised data
    sample('supervised_categories', dist.Categorical(transition_prob[supervised_categories[:-1]]),
           obs=supervised_categories[1:])
    sample('supervised_words', dist.Categorical(emission_prob[supervised_categories]),
           obs=supervised_words)

    # computes log prob of unsupervised data
    transition_log_prob = np.log(transition_prob)
    emission_log_prob = np.log(emission_prob)
    init_log_prob = emission_log_prob[:, unsupervised_words[0]]
    log_prob = forward_log_prob(init_log_prob, unsupervised_words[1:],
                                transition_log_prob, emission_log_prob)
    # inject log_prob to potential function
    return sample('forward_log_prob', dist.Multinomial(logits=log_prob), obs=0)


def run_inference(transition_prior, emission_prior, supervised_categories, supervised_words,
                  unsupervised_words, rng, args):
    init_params, potential_fn, transform_fn = initialize_model(
        rng,
        semi_supervised_hmm,
        (transition_prior, emission_prior, supervised_categories,
         supervised_words, unsupervised_words),
        {}
    )
    init_kernel, sample_kernel = hmc(potential_fn, algo='NUTS')
    hmc_state = init_kernel(init_params, args.num_warmup_steps)
    hmc_states = fori_collect(args.num_samples, sample_kernel, hmc_state,
                              transform=lambda state: transform_fn(state.z))
    return hmc_states


def print_results(posterior, transition_prob):
    header = semi_supervised_hmm.__name__ + ' - TRAIN'
    columns = ['', 'ActualProb', 'Pred(p25)', 'Pred(p50)', 'Pred(p75)']
    header_format = '{:>20} {:>10} {:>10} {:>10} {:>10}'
    row_format = '{:>20} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}'
    quantiles = onp.quantile(posterior, [0.25, 0.5, 0.75], axis=0)
    print('\n', '=' * 20 + header + '=' * 20, '\n')
    print(header_format.format(*columns))
    for i in range(transition_prob.shape[0]):
        for j in range(transition_prob.shape[1]):
            idx = 'transition[{},{}]'.format(i, j)
            print(row_format.format(idx, transition_prob[i, j], *quantiles[:, i, j]), '\n')


def main(args):
    jax_config.update('jax_platform_name', args.device)
    print('Simulating data...')
    (transition_prior, emission_prior, transition_prob, emission_prob,
     supervised_categories, supervised_words, unsupervised_words) = simulate_data(
        random.PRNGKey(1),
        num_categories=3,
        num_words=10,
        num_supervised_data=100,
        num_unsupervised_data=500,
    )
    print('Starting inference...')
    zs = run_inference(transition_prior, emission_prior,
                       supervised_categories, supervised_words, unsupervised_words,
                       random.PRNGKey(2), args)
    print_results(zs['transition_prob'], transition_prob)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semi-supervised Hidden Markov Model')
    parser.add_argument('-n', '--num-samples', nargs='?', default=1000, type=int)
    parser.add_argument('--num-warmup-steps', nargs='?', default=500, type=int)
    parser.add_argument('--device', default='cpu', type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()
    main(args)
