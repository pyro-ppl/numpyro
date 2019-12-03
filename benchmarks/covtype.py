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
from numpyro.examples.datasets import COVTYPE, load_dataset

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

def get_data():
    _, fetch = load_dataset(COVTYPE, shuffle=False)
    features, labels = fetch()

    # normalize features and add intercept
    features = (features - features.mean(0)) / features.std(0)
    features = onp.hstack([features, onp.ones((features.shape[0], 1))])

    # make binary feature
    _, counts = onp.unique(labels, return_counts=True)
    specific_category = onp.argmax(counts)
    labels = (labels == specific_category)

    N, dim = features.shape
    print("Data shape:", features.shape)
    print("Label distribution: {} has label 1, {} has label 0"
          .format(labels.sum(), N - labels.sum()))
    return features, labels.astype(int)


########################################
# NumPyro model and inference
########################################

def numpyro_model(features, labels):
    coefs = numpyro.sample('coefs', ndist.Normal(0, 1), sample_shape=features.shape[1:])
    return numpyro.sample('obs', ndist.Bernoulli(logits=np.dot(features, coefs)), obs=labels)


def numpyro_inference(data, args):
    rng_key = jax.random.PRNGKey(args.seed)
    kernel = numpyro.infer.NUTS(numpyro_model, step_size=0.0015, adapt_step_size=False)
    mcmc = numpyro.infer.MCMC(kernel, args.num_warmup, args.num_samples,
                              num_chains=args.num_chains, progress_bar=not args.disable_progbar)
    tic = time.time()
    mcmc._compile(rng_key, *data, extra_fields=('num_steps',))
    print('MCMC (numpyro) compiling time:', time.time() - tic, '\n')
    tic = time.time()
    mcmc.run(rng_key, *data, extra_fields=('num_steps',))
    mcmc._last_state.rng_key.copy()
    toc = time.time()
    mcmc.print_summary()
    print('\nMCMC (numpyro) elapsed time:', toc - tic)
    sampling_time = toc - tic
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

def pyro_model(features, labels):
    with pyro.plate("dim", features.shape[1]):
        coefs = pyro.sample('coefs', pdist.Normal(0, 1))
    logits = torch.matmul(features, coefs.unsqueeze(-1)).squeeze(-1)
    return pyro.sample('obs', pdist.Bernoulli(logits=logits), obs=labels)


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
    print(data[0].shape, data[1].shape)
    pyro_data = [torch.Tensor(x) for i, x in enumerate(data)]
    kernel = pyro.infer.NUTS(pyro_model, step_size=0.0015, adapt_step_size=False,
                             jit_compile=True, ignore_jit_warnings=True)
    mcmc = pyro.infer.MCMC(kernel, num_samples=args.num_samples, warmup_steps=args.num_warmup,
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
            int<lower=1> D;
            int<lower=0> N;
            matrix[N, D] x;
            int<lower=0,upper=1> y[N];
        }
        parameters {
            vector[D] beta;
        }
        model {
            y ~ bernoulli_logit(x * beta);
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
    features, labels = data
    stan_data = {"D": features.shape[1], "N": features.shape[0], "x": features, "y": labels}

    tic = time.time()
    sm = stan_model()
    print('MCMC (stan) compiling time:', time.time() - tic, '\n')

    log_filename = 'covtype.txt'
    _set_logging(log_filename)
    tic = time.time()
    fit = sm.sampling(data=stan_data, iter=args.num_samples + args.num_warmup, warmup=args.num_warmup,
                      chains=args.num_chains, seed=args.seed, control={'adapt_engaged': False, 'stepsize': 0.0015})
    toc = time.time()
    print(fit)
    print('\nMCMC (stan) elapsed time:', toc - tic)
    sampling_time = _get_pystan_sampling_time(log_filename)
    num_leapfrogs = fit.get_sampler_params(inc_warmup=False)[0]["n_leapfrog__"].sum()
    print('num leapfrogs', num_leapfrogs)
    time_per_leapfrog = sampling_time / num_leapfrogs
    print('time per leapfrog', time_per_leapfrog)
    summary = fit.summary(pars=('beta',))['summary']
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
    data = get_data()
    if args.backend == 'numpyro':
        result = numpyro_inference(data, args)
    elif args.backend == 'pyro':
        result = pyro_inference(data, args)
    elif args.backend == 'stan':
        result = stan_inference(data, args)
    elif args.backend == 'edward':
        pass

    out_filename = 'covtype_{}_{}_seed={}.txt'.format(args.backend,
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
    parser.add_argument("-n", "--num-samples", nargs="?", default=30, type=int)
    parser.add_argument("--num-warmup", nargs='?', default=0, type=int)
    parser.add_argument("--num-chains", nargs='?', default=1, type=int)
    parser.add_argument("--seed", nargs='?', default=2019, type=int)
    parser.add_argument("--device", default='cpu', type=str, help='use "cpu" or "gpu".')
    parser.add_argument("--backend", default='numpyro', type=str, help='either "numpyro", "pyro", "stan", or "edward"')
    parser.add_argument("--x64", action="store_true")
    parser.add_argument("--disable-progbar", action="store_true")
    args = parser.parse_args()

    numpyro.enable_x64(args.x64)
    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)
    tt = torch.cuda if args.device == "gpu" else torch
    torch.set_default_tensor_type(tt.DoubleTensor if args.x64 else tt.FloatTensor)

    main(args)
