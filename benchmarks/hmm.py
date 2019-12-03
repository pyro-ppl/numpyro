import argparse
import os
import subprocess
import sys
import time
import warnings

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


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


########################################
# simulate data
########################################

def simulate_data(num_categories, num_words, num_supervised_data, num_unsupervised_data):
    onp.random.seed(0)
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
    mcmc = numpyro.infer.MCMC(kernel, args.num_warmup, args.num_samples,
                              num_chains=args.num_chains, progress_bar=not args.disable_progbar)
    tic = time.time()
    mcmc._compile(rng_key, *data, extra_fields=('num_steps',))
    print('MCMC (numpyro) compiling time:', time.time() - tic, '\n')
    tic = time.time()
    mcmc.warmup(rng_key, *data, extra_fields=('num_steps',))
    mcmc.num_samples = args.num_samples
    rng_key = mcmc._warmup_state.rng_key.copy()
    tic_run = time.time()
    mcmc.run(rng_key, *data, extra_fields=('num_steps',))
    mcmc._last_state.rng_key.copy()
    toc = time.time()
    mcmc.print_summary()
    print('\nMCMC (numpyro) elapsed time:', toc - tic)
    sampling_time = toc - tic_run
    num_leapfrogs = np.sum(mcmc.get_extra_fields()['num_steps'])
    print('num leapfrogs', num_leapfrogs)
    time_per_leapfrog = sampling_time / num_leapfrogs
    print('time per leapfrog', time_per_leapfrog)
    n_effs = [numpyro.diagnostics.effective_sample_size(jax.device_get(v))
              for k, v in mcmc.get_samples(group_by_chain=True).items()]
    n_effs = onp.concatenate([onp.reshape(x, -1) for x in n_effs])
    n_eff_mean = sum(n_effs) / len(n_effs)
    print('mean n_eff', n_eff_mean)
    time_per_eff_sample = sampling_time / n_eff_mean
    print('time per effective sample', time_per_eff_sample)
    return num_leapfrogs, n_eff_mean, toc - tic, time_per_leapfrog, time_per_eff_sample


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
    return log_prob.logsumexp(dim=0)


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


_NUM_LEAPFROGS = 0
_SAMPLING_PHASE_TIC = None


@numpyro.patch.patch_dependency('pyro.ops.integrator._kinetic_grad', root_module=pyro)
def _kinetic_grad(inverse_mass_matrix, r):
    global _NUM_LEAPFROGS
    if _SAMPLING_PHASE_TIC is not None:
        _NUM_LEAPFROGS += 1

    r_flat = torch.cat([r[site_name].reshape(-1) for site_name in sorted(r)])
    if inverse_mass_matrix.dim() == 1:
        grads_flat = inverse_mass_matrix * r_flat
    else:
        grads_flat = inverse_mass_matrix.matmul(r_flat)

    grads = {}
    pos = 0
    for site_name in sorted(r):
        next_pos = pos + r[site_name].numel()
        grads[site_name] = grads_flat[pos:next_pos].reshape(r[site_name].shape)
        pos = next_pos
    assert pos == grads_flat.size(0)
    return grads


@numpyro.patch.patch_dependency('pyro.infer.mcmc.api._gen_samples', root_module=pyro)
def _mcmc_gen_samples(kernel, warmup_steps, num_samples, hook, chain_id, *args, **kwargs):
    global _SAMPLING_PHASE_TIC

    kernel.setup(warmup_steps, *args, **kwargs)
    params = kernel.initial_params
    # yield structure (key, value.shape) of params
    yield {k: v.shape for k, v in params.items()}
    for i in range(warmup_steps):
        params = kernel.sample(params)
        hook(kernel, params, 'Warmup [{}]'.format(chain_id) if chain_id is not None else 'Warmup', i)

    _SAMPLING_PHASE_TIC = time.time()

    for i in range(num_samples):
        params = kernel.sample(params)
        hook(kernel, params, 'Sample [{}]'.format(chain_id) if chain_id is not None else 'Sample', i)
        yield torch.cat([params[site].reshape(-1) for site in sorted(params)]) if params else torch.tensor([])
    yield kernel.diagnostics()
    kernel.cleanup()


def pyro_inference(data, args):
    pyro.set_rng_seed(args.seed)
    pyro_data = [torch.Tensor(x) if i <= 1 else torch.Tensor(x).long() for i, x in enumerate(data)]
    warnings.warn("Pyro is slow, so we fix step_size=0.1, num_samples=100,"
                  " and no warmup adaptation.")
    kernel = pyro.infer.NUTS(pyro_model, step_size=0.1, adapt_step_size=False,
                             jit_compile=True, ignore_jit_warnings=True)
    mcmc = pyro.infer.MCMC(kernel, num_samples=100, warmup_steps=0,
                           num_chains=args.num_chains)
    tic = time.time()
    mcmc.run(*pyro_data)
    toc = time.time()
    mcmc.summary()
    print('\nMCMC (pyro) elapsed time:', toc - tic)
    print('num leapfrogs', _NUM_LEAPFROGS)
    time_per_leapfrog = (toc - _SAMPLING_PHASE_TIC) / _NUM_LEAPFROGS
    print('time per leapfrog', time_per_leapfrog)
    n_effs = [pyro.ops.stats.effective_sample_size(v).detach().cpu().numpy()
              for k, v in mcmc.get_samples(group_by_chain=True).items()]
    n_effs = onp.concatenate([onp.reshape(x, -1) for x in n_effs])
    n_eff_mean = sum(n_effs) / len(n_effs)
    print('mean n_eff', n_eff_mean)
    time_per_eff_sample = (toc - _SAMPLING_PHASE_TIC) / n_eff_mean
    print('time per effective sample', time_per_eff_sample)
    return _NUM_LEAPFROGS, n_eff_mean, toc - tic, time_per_leapfrog, time_per_eff_sample


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
    time_per_leapfrog = sampling_time / num_leapfrogs
    print('time per leapfrog', time_per_leapfrog)
    summary = fit.summary(pars=('theta', 'phi'))['summary']
    n_effs = [row[8] for row in summary]
    n_eff_mean = sum(n_effs) / len(n_effs)
    print('mean n_eff', n_eff_mean)
    time_per_eff_sample = sampling_time / n_eff_mean
    print('time per effective sample', time_per_eff_sample)
    return num_leapfrogs, n_eff_mean, toc - tic, time_per_leapfrog, time_per_eff_sample


########################################
# Main
########################################

def main(args):
    data = simulate_data(args.num_categories, args.num_words, args.num_supervised, args.num_unsupervised)
    if args.backend == 'numpyro':
        result = numpyro_inference(data, args)
    elif args.backend == 'pyro':
        result = pyro_inference(data, args)
    elif args.backend == 'stan':
        result = stan_inference(data, args)

    out_filename = 'hmm_{}_{}_seed={}.txt'.format(args.backend,
                                                  args.device,
                                                  args.seed)
    with open(os.path.join(DATA_DIR, out_filename), 'w') as f:
        f.write('\t'.join(['num_leapfrog', 'n_eff', 'total_time', 'time_per_leapfrog', 'time_per_eff_sample']))
        f.write('\n')
        f.write('\t'.join([str(x) for x in result]))
        f.write('\n')


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
    parser.add_argument("--disable-progbar", action="store_true")
    args = parser.parse_args()

    numpyro.enable_x64(args.x64)
    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)
    tt = torch.cuda if args.device == "gpu" else torch
    torch.set_default_tensor_type(tt.DoubleTensor if args.x64 else tt.FloatTensor)

    main(args)
