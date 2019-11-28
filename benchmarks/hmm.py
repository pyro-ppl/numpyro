import argparse
import os
import subprocess
import sys
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
import pyro.distributions as pdist

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
    categories, words = [], []
    for t in range(num_supervised_data + num_unsupervised_data):
        if t == 0 or t == num_supervised_data:
            category = onp.random.choice(num_categories)
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

def _forward_one_step(prev_log_prob, curr_word, transition_log_prob, emission_log_prob):
    log_prob = np.expand_dims(prev_log_prob, axis=1) + transition_log_prob + emission_log_prob[:, curr_word]
    return jax.scipy.special.logsumexp(log_prob, axis=0)


def _forward_log_prob(words, transition_log_prob, emission_log_prob):
    def scan_fn(log_prob, word):
        return _forward_one_step(log_prob, word, transition_log_prob, emission_log_prob), np.zeros((0,))

    init_log_prob = emission_log_prob[:, words[0]]
    log_prob, _ = jax.lax.scan(scan_fn, init_log_prob, words[1:])
    return jax.scipy.special.logsumexp(log_prob)


def numpyro_model(transition_prior, emission_prior, supervised_categories, supervised_words, unsupervised_words):
    num_categories = transition_prior.shape[0]
    with numpyro.plate('K', num_categories):
        transition_prob = numpyro.sample('transition_prob', ndist.Dirichlet(transition_prior))
        emission_prob = numpyro.sample('emission_prob', ndist.Dirichlet(emission_prior))

    numpyro.sample('supervised_categories', ndist.Categorical(transition_prob[supervised_categories[:-1]]),
                   obs=supervised_categories[1:])
    numpyro.sample('supervised_words', ndist.Categorical(emission_prob[supervised_categories]),
                   obs=supervised_words)

    log_prob = _forward_log_prob(unsupervised_words, np.log(transition_prob), np.log(emission_prob))
    numpyro.factor('forward_log_prob', log_prob)


def numpyro_inference(data, args):
    rng_key = jax.random.PRNGKey(args.seed)
    kernel = numpyro.infer.NUTS(numpyro_model)
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
    print('mean n_eff', n_eff_mean)
    print('time per effective sample', (toc - tic) / n_eff_mean)


########################################
# Pyro model and inference
########################################

def _pyro_forward_one_step(prev_log_prob, curr_word, transition_log_prob, emission_log_prob):
    log_prob = prev_log_prob.unsqueeze(dim=1) + transition_log_prob + emission_log_prob[:, curr_word]
    return log_prob.logsumexp(dim=0)


@torch.jit.script
def _pyro_forward_log_prob(words, transition_log_prob, emission_log_prob):
    log_prob = emission_log_prob[:, words[0]]
    for t in range(1, len(words)):
        log_prob = _pyro_forward_one_step(log_prob, words[t], transition_log_prob, emission_log_prob)
    return log_prob.logsumexp()


def pyro_model(transition_prior, emission_prior, supervised_categories, supervised_words, unsupervised_words):
    num_categories = transition_prior.shape[0]
    with pyro.plate("K", num_categories):
        transition_prob = pyro.sample('transition_prob', pdist.Dirichlet(transition_prior))
        emission_prob = pyro.sample('emission_prob', pdist.Dirichlet(emission_prior))

    pyro.sample('supervised_categories', pdist.Categorical(transition_prob[supervised_categories[:-1]]),
                obs=supervised_categories[1:])
    pyro.sample('supervised_words', pdist.Categorical(emission_prob[supervised_categories]),
                obs=supervised_words)

    log_prob = _pyro_forward_log_prob(unsupervised_words, transition_prob.log(), emission_prob.log())
    numpyro.factor('forward_log_prob', log_prob)


def pyro_inference(data, args):
    pyro.set_rng_seed(args.seed)
    kernel = pyro.infer.NUTS(pyro_model, jit_compile=True, ignore_jit_warnings=True)
    mcmc = pyro.infer.MCMC(kernel, num_samples=args.num_samples, warmup_steps=args.num_warmup,
                           num_chains=args.num_chains)
    tic = time.time()
    mcmc.run(*data)
    toc = time.time()
    mcmc.print_summary()
    print('\nMCMC (pyro) elapsed time:', toc - tic)
    # TODO: patch Pyro to record num_steps...


########################################
# Stan model and inference
########################################

def stan_model():
    model_code = """
        data {
          int<lower=1> K;  // num categories
          int<lower=1> V;  // num words
          int<lower=0> T;  // num supervised items
          int<lower=1> T_unsup;  // num unsupervised items
          int<lower=1,upper=V> w[T]; // words
          int<lower=1,upper=K> z[T]; // categories
          int<lower=1,upper=V> u[T_unsup]; // unsup words
          vector<lower=0>[K] alpha;  // transit prior
          vector<lower=0>[V] beta;   // emit prior
        }
        parameters {
          simplex[K] theta[K];  // transit probs
          simplex[V] phi[K];    // emit probs
        }
        model {
          for (k in 1:K)
            theta[k] ~ dirichlet(alpha);
          for (k in 1:K)
            phi[k] ~ dirichlet(beta);
          for (t in 1:T)
            w[t] ~ categorical(phi[z[t]]);
          for (t in 2:T)
            z[t] ~ categorical(theta[z[t-1]]);

          {
            // forward algorithm computes log p(u|...)
            real acc[K];
            real gamma[T_unsup,K];
            for (k in 1:K)
              gamma[1,k] = log(phi[k,u[1]]);
            for (t in 2:T_unsup) {
              for (k in 1:K) {
                for (j in 1:K)
                  acc[j] = gamma[t-1,j] + log(theta[j,k]) + log(phi[k,u[t]]);
                gamma[t,k] = log_sum_exp(acc);
              }
            }
            target += log_sum_exp(gamma[T_unsup]);
          }
        }
    """
    return pystan.StanModel(model_code=model_code)


def _set_logging(filename):
    tee = subprocess.Popen(['tee', '/tmp/' + filename + '.txt'], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())


def _get_pystan_sampling_time(filename):
    secs = None
    with open('/tmp/' + filename + '.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if 'seconds (Sampling)' in line:
                secs = float(line.split()[0])
                break
    return secs


def stan_inference(data, args):
    transition_prior, emission_prior, supervised_categories, supervised_words, unsupervised_words = data
    stan_data = {"K": transition_prior.shape[0], "V": emission_prior.shape[0],
                 "T": supervised_words.shape[0], "T_unsup": unsupervised_words.shape[0],
                 "alpha": transition_prior, "beta": emission_prior,
                 "w": supervised_words + 1, "z": supervised_categories + 1, "u": unsupervised_words + 1}
    sm = stan_model()

    log_filename = 'hmm.txt'
    _set_logging(log_filename)
    tic = time.time()
    fit = sm.sampling(data=stan_data, iter=args.num_samples + args.num_warmup, warmup=args.num_warmup,
                      chains=args.num_chains, seed=args.seed)
    toc = time.time()
    print(fit)
    print('\nMCMC (stan) elapsed time:', toc - tic)
    sampling_time = _get_pystan_sampling_time(log_filename)
    num_leapfrogs = fit.get_sampler_params(inc_warmup=False)[0]["n_leapfrog__"].sum()
    print('num leapfrogs', num_leapfrogs)
    print('time per leapfrog', sampling_time / num_leapfrogs)
    summary = fit.summary(pars=('theta', 'phi'))['summary']
    n_effs = [row[8] for row in summary]
    n_eff_mean = sum(n_effs) / len(n_effs)
    print('mean n_eff', n_eff_mean)
    print('time per effective sample', sampling_time / n_eff_mean)


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
    parser = argparse.ArgumentParser(description="HMM example")
    parser.add_argument("-n", "--num-samples", nargs="?", default=1000, type=int)
    parser.add_argument("--num-warmup", nargs='?', default=1000, type=int)
    parser.add_argument("--num-chains", nargs='?', default=1, type=int)
    parser.add_argument("--num-categories", nargs='?', default=3, type=int)
    parser.add_argument("--num-words", nargs='?', default=10, type=int)
    parser.add_argument("--num-supervised", nargs='?', default=100, type=int)
    parser.add_argument("--num-unsupervised", nargs='?', default=500, type=int)
    parser.add_argument("--seed", nargs='?', default=2019, type=int)
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
