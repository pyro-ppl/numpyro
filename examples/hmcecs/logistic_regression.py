import os
import pathlib
import pickle
from datetime import datetime
from time import time

import jax.numpy as jnp
import numpy as np
from jax import random, device_get
from pandas_plink import read_plink1_bin
from sklearn.datasets import load_breast_cancer

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.examples.datasets import _load_higgs
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, init_to_median, init_to_value, HMC
from numpyro.infer.hmc_gibbs import HMCECS, perturbed_method, variational_proxy, taylor_proxy
from numpyro.infer.util import _predictive

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"


def summary(dataset, name, mcmc, sample_time, svi_time=0., plates={}):
    n_eff_mean = np.mean([numpyro.diagnostics.effective_sample_size(device_get(v))
                          for k, v in mcmc.get_samples(True).items() if k not in plates])
    pickle.dump(mcmc.get_samples(True), open(f'{dataset}/{name}_posterior_samples.pkl', 'wb'))
    step_field = 'num_steps' if name in ['hmc', 'nuts'] else 'hmc_state.num_steps'
    num_step = np.sum(mcmc.get_extra_fields()[step_field])
    accpt_prob = np.mean(mcmc.get_extra_fields()['accept_prob']) if 'ecs' in name else 1.

    with open(f'{dataset}/{name}_chain_stats.txt', 'w') as f:
        print('sample_time', 'svi_time', 'n_eff_mean', 'gibbs_accpt_prob', 'tot_num_steps', 'time_per_step',
              'time_per_eff', sep=',', file=f)
        print(sample_time, svi_time, n_eff_mean, accpt_prob, num_step, sample_time / num_step, sample_time / n_eff_mean,
              sep=',', file=f)


def higgs_data():
    obs, data = _load_higgs()
    return data, obs


def breast_cancer_data():
    dataset = load_breast_cancer()
    feats = dataset.data
    feats = (feats - feats.mean(0)) / feats.std(0)
    feats = jnp.hstack((feats, jnp.ones((feats.shape[0], 1))))
    return feats, dataset.target


def copsac_data():
    data_folder = pathlib.Path('data')
    bim_file = str(data_folder / 'Sim_data_3.bim')
    fam_file = str(data_folder / 'Sim_data_3.fam')
    bed_file = str(data_folder / 'Sim_data_3.bed')
    data = read_plink1_bin(bed_file, bim_file, fam_file)

    return jnp.array(data.values), jnp.array(data['trait'].astype(int))


def model(features, obs, subsample_size):
    n, m = features.shape
    theta = numpyro.sample('theta', dist.continuous.Normal(jnp.zeros(m), .5 * jnp.ones(m)))
    with numpyro.plate('N', n, subsample_size=subsample_size):
        batch_feats = numpyro.subsample(features, event_dim=1)
        batch_obs = numpyro.subsample(obs, event_dim=0)
        numpyro.sample('obs', dist.Bernoulli(logits=theta @ batch_feats.T), obs=batch_obs)


def guide(feature, obs, subsample_size):
    _, m = feature.shape
    mean = numpyro.param('mean', jnp.zeros(m), constraints=constraints.real)
    # var = numpyro.param('var', jnp.ones(m), constraints=constraints.positive)
    numpyro.sample('theta', dist.continuous.Normal(mean, .5))


def hmcecs_model(dataset, data, obs, subsample_size, proxy_name='vari'):
    model_args, model_kwargs = (data, obs, subsample_size), {}

    svi_key, proxy_key, estimator_key, mcmc_key = random.split(random.PRNGKey(0), 4)
    optimizer = numpyro.optim.Adam(step_size=5e-5)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    start = time()
    svi_result = svi.run(svi_key, 1000, *model_args)
    svi_time = time() - start

    pickle.dump(svi_result.params, open(f'{dataset}/svi_params.pkl', 'wb'))
    params = svi_result.params

    proxy_key, ref_key = random.split(proxy_key)
    ref_params = _predictive(ref_key, guide, {}, (1,), return_sites='', parallel=True,
                             model_args=model_args, model_kwargs=model_kwargs)
    ref_params.pop('mean')
    if proxy_name == 'taylor':
        proxy_fn = taylor_proxy(ref_params)

    else:
        proxy_fn = variational_proxy(guide, params)

    # Compute HMCECS
    kernel = HMCECS(NUTS(model), proxy=proxy_fn)
    mcmc = MCMC(kernel, 1000, 1000)
    start = time()
    mcmc.run(random.PRNGKey(3), data, obs, subsample_size, extra_fields=("accept_prob",
                                                                         "hmc_state.num_steps"))
    summary(dataset, f'ecs_{proxy_name}', mcmc, time() - start, svi_time=svi_time, plates={'N': ''})
    return ref_params


def plain_log_reg_model(features, obs):
    n, m = features.shape
    theta = numpyro.sample('theta', dist.continuous.Normal(jnp.zeros(m), .5 * jnp.ones(m)))
    numpyro.sample('obs', dist.Bernoulli(logits=theta @ features.T), obs=obs)


def nuts(dataset, data, obs, ref_param):
    kernel = NUTS(plain_log_reg_model, trajectory_length=1.2, init_strategy=init_to_value(values=ref_param))
    mcmc = MCMC(kernel, 1000, 1000)
    mcmc._compile(random.PRNGKey(0), data, obs, extra_fields=("num_steps",))
    start = time()
    mcmc.run(random.PRNGKey(0), data, obs, extra_fields=('num_steps',))
    summary(dataset, 'nuts', mcmc, time() - start)


def hmc(dataset, data, obs, ref_param):
    kernel = HMC(plain_log_reg_model, trajectory_length=1.2, init_strategy=init_to_value(values=ref_param))
    mcmc = MCMC(kernel, 1000, 1000)
    mcmc._compile(random.PRNGKey(0), data, obs, extra_fields=("num_steps",))
    start = time()
    mcmc.run(random.PRNGKey(0), data, obs, extra_fields=('num_steps',))
    summary(dataset, 'hmc', mcmc, time() - start)


if __name__ == '__main__':

    load_data = {'higgs': higgs_data}  # 'breast': breast_cancer_data  , 'copsac': copsac_data}
    subsample_sizes = {'higgs': 1300, 'breast': 75, }  # 'copsac': 1000,
    data, obs = breast_cancer_data()

    for platform in ['gpu', 'cpu']:
        numpyro.set_platform(platform)
        for dataset in load_data.keys():
            dir = f'{platform}_{dataset}_{datetime.now().strftime("%Y_%m_%d_%H%M%S")}'
            if not os.path.exists(dir):
                os.mkdir(dir)
            data, obs = load_data[dataset]()
            ref_param = hmcecs_model(dir, data, obs, subsample_sizes[dataset], proxy_name='taylor')
            ref_param = hmcecs_model(dir, data, obs, subsample_sizes[dataset], proxy_name='variational')
            hmc(dir, data, obs, ref_param)
            nuts(dir, data, obs, ref_param)
