import argparse
import time

import numpy as onp

# numpyro
import jax
from jax import numpy as np
import numpyro
import numpyro.distributions as ndist

# pyro
import torch
import pyro
import pyro.distributions as tdist

# stan
import pystan


########################################
# simulate data
########################################

def simulate_data(num_categories, num_words, num_supervised_data, num_unsupervised_data):
    onp.random.seed(2019)
    transition_prior = onp.ones(num_categories)
    emission_prior = onp.repeat(0.1, num_words)
    transition_prob = onp.random.dirichlet(transition_prior, num_categories)
    emission_prob = onp.random.dirichlet(emission_prior, num_categories)
    start_prob = onp.repeat(1. / num_categories, num_categories)
    categories, words = [], []
    for t in range(num_supervised_data + num_unsupervised_data):
        if t == 0 or t == num_supervised_data:
            category = onp.random.choice(num_categories, p=start_prob)
        else:
            category = onp.random.choice(num_categories, p=transition_prob[category])
        word = onp.random.choice(num_words, p=emission_prob[category])
        categories.append(category)
        words.append(word)

    # split into supervised data and unsupervised data
    categories, words = onp.stack(categories), onp.stack(words)
    supervised_categories = categories[:num_supervised_data]
    supervised_words = words[:num_supervised_data]
    unsupervised_words = words[num_supervised_data:]
    return transition_prior, emission_prior, supervised_categories, supervised_words, unsupervised_words


########################################
# NumPyro model and inference
########################################

def forward_one_step(prev_log_prob, curr_word, transition_log_prob, emission_log_prob):
    log_prob_tmp = np.expand_dims(prev_log_prob, axis=1) + transition_log_prob
    log_prob = log_prob_tmp + emission_log_prob[:, curr_word]
    return jax.scipy.special.logsumexp(log_prob, axis=0)


def forward_log_prob(words, transition_log_prob, emission_log_prob):
    def scan_fn(log_prob, word):
        return forward_one_step(log_prob, word, transition_log_prob, emission_log_prob), np.zeros((0,))

    init_log_prob = emission_log_prob[:, words[0]]
    log_prob, _ = jax.lax.scan(scan_fn, init_log_prob, words[1:])
    return log_prob


def semi_supervised_hmm(transition_prior, emission_prior, supervised_categories, supervised_words, unsupervised_words):
    num_categories = transition_prior.shape[0]
    transition_prob = numpyro.sample('transition_prob', ndist.Dirichlet(transition_prior),
                                     sample_shape=(num_categories,))
    emission_prob = numpyro.sample('emission_prob', ndist.Dirichlet(emission_prior),
                                   sample_shape=(num_categories,))

    numpyro.sample('supervised_categories', ndist.Categorical(transition_prob[supervised_categories[:-1]]),
                   obs=supervised_categories[1:])
    numpyro.sample('supervised_words', ndist.Categorical(emission_prob[supervised_categories]),
                   obs=supervised_words)

    log_prob = forward_log_prob(unsupervised_words, np.log(transition_prob), np.log(emission_prob))
    numpyro.factor('forward_log_prob', jax.scipy.special.logsumexp(log_prob))


def numpyro_inference(data, args):
    rng_key = jax.random.PRNGKey(2019)
    kernel = numpyro.infer.NUTS(semi_supervised_hmm)
    mcmc = numpyro.infer.MCMC(kernel, args.num_warmup, num_chains=args.num_chains)
    mcmc.run(rng_key, *data, extra_fields=('num_steps',))
    mcmc.num_samples = args.num_samples
    mcmc._warmup_state.i.copy()  # make sure no jax async affects tic
    tic = time.time()
    mcmc.run(rng_key, *data, extra_fields=('num_steps',), reuse_warmup_state=True)
    mcmc.print_summary()
    toc = time.time()
    print('\nMCMC (numpyro) elapsed time:', toc - tic)
    num_leapfrogs = np.sum(mcmc.get_extra_fields()['num_steps'])
    print('num leapfrogs', num_leapfrogs)
    print('time per leapfrog', (toc - tic) / num_leapfrogs)
    n_effs = [numpyro.diagnostics.effective_sample_size(jax.device_get(v))
              for k, v in mcmc.get_samples(group_by_chain=True).items()]
    n_effs = onp.concatenate([onp.reshape(x, -1) for x in n_effs])
    n_eff_mean = sum(n_effs) / len(n_effs)
    print('time per effective sample', (toc - tic) / n_eff_mean)


########################################
# Pyro model and inference
########################################

def pyro_inference(data, args):
    pass


########################################
# Stan model and inference
########################################

def stan_inference(data, args):
    pass


########################################
# Main
########################################

def main(args):
    data = simulate_data(args.num_categories, args.num_words, args.num_supervised, args.num_unsupervised)
    if args.backend == 'numpyro':
        numpyro_inference(data, args)
    elif args.backend == 'pyro':
        pyro_inference(data, args)
    elif args.backend == 'stan':
        stan_inference(data, args)


if __name__ == "__main__":
    assert numpyro.__version__.startswith('0.2.1')
    parser = argparse.ArgumentParser(description="Gaussian Process example")
    parser.add_argument("-n", "--num-samples", nargs="?", default=1000, type=int)
    parser.add_argument("--num-warmup", nargs='?', default=1000, type=int)
    parser.add_argument("--num-chains", nargs='?', default=1, type=int)
    parser.add_argument("--num-categories", nargs='?', default=3, type=int)
    parser.add_argument("--num-words", nargs='?', default=10, type=int)
    parser.add_argument("--num-supervised", nargs='?', default=100, type=int)
    parser.add_argument("--num-unsupervised", nargs='?', default=500, type=int)
    parser.add_argument("--device", default='cpu', type=str, help='use "cpu" or "gpu".')
    parser.add_argument("--backend", default='numpyro', type=str, help='either "numpyro", "pyro", or "stan"')
    parser.add_argument("--x64", action="store_true")
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", args.x64)
    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)
    tt = torch.cuda if args.device == "gpu" else torch
    torch.set_default_tensor_type(tt.DoubleTensor if args.x64 else tt.FloatTensor)

    main(args)
