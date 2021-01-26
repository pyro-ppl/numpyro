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
from numpyro.contrib.ecs import ECS
from numpyro.distributions import constraints
from numpyro.examples.datasets import _load_higgs
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, init_to_sample

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

numpyro.set_platform("gpu")


def summary(dataset, name, mcmc, sample_time, svi_time=0.):
    n_eff_mean = np.mean([numpyro.diagnostics.effective_sample_size(device_get(v))
                          for v in mcmc.get_samples(True).values()])
    pickle.dump(mcmc.get_samples(True), open(f'{dataset}/{name}_posterior_samples.pkl', 'wb'))
    step_field = 'num_steps' if name == 'hmc' else 'hmc_state.num_steps'
    num_step = np.sum(mcmc.get_extra_fields()[step_field])
    accpt_prob = 1.
    if name == 'ecs':
        accpt_prob = np.mean(mcmc.get_extra_fields()['accept_prob'])

    with open(f'{dataset}/{name}_chain_stats.txt', 'w') as f:
        print('sample_time', 'svi_time', 'n_eff_mean', 'gibbs_accpt_prob', 'tot_num_steps', 'time_per_step',
              'time_per_eff',
              sep=',', file=f)
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


def log_reg_model(features, obs, subsample_size):
    n, m = features.shape
    theta = numpyro.sample('theta', dist.continuous.Normal(jnp.zeros(m), .5 * jnp.ones(m)))
    with numpyro.plate('N', n, subsample_size=subsample_size):
        batch_feats = numpyro.subsample(features, event_dim=1)
        batch_obs = numpyro.subsample(obs, event_dim=0)
        numpyro.sample('obs', dist.Bernoulli(logits=theta @ batch_feats.T), obs=batch_obs)


def log_reg_guide(feature, obs, subsample_size):
    _, m = feature.shape
    mean = numpyro.param('mean', jnp.zeros(m), constraints=constraints.real)
    # var = numpyro.param('var', jnp.ones(m), constraints=constraints.positive)
    numpyro.sample('theta', dist.continuous.Normal(mean, .5))


def hmcecs_model(dataset, data, obs, subsample_size):
    optimizer = numpyro.optim.Adam(step_size=5e-5)
    svi = SVI(log_reg_model, log_reg_guide, optimizer, loss=Trace_ELBO())
    start = time()
    svi_result = svi.run(random.PRNGKey(2), 1000, data, obs, subsample_size, )
    svi_time = time() - start

    pickle.dump(svi_result.params, open(f'{dataset}/svi_params.pkl', 'wb'))
    params = svi_result.params

    # Compute HMCECS
    kernel = ECS(NUTS(log_reg_model),
                 proxy='variational',
                 model_struct={'obs': ['theta']},
                 ref=params,
                 guide=log_reg_guide)
    mcmc = MCMC(kernel, 10000, 10000)
    start = time()
    mcmc.run(random.PRNGKey(3), data, obs, subsample_size, extra_fields=("accept_prob",
                                                                         "hmc_state.accept_prob",
                                                                         "hmc_state.num_steps"))
    print(mcmc.get_extra_fields(True)['hmc_state.accept_prob'])
    summary(dataset, 'ecs', mcmc, time() - start, svi_time=svi_time)


def plain_log_reg_model(features, obs):
    n, m = features.shape
    theta = numpyro.sample('theta', dist.continuous.Normal(jnp.zeros(m), .5 * jnp.ones(m)))
    numpyro.sample('obs', dist.Bernoulli(logits=theta @ features.T), obs=obs)


def hmc(dataset, data, obs):
    kernel = NUTS(plain_log_reg_model, init_strategy=init_to_sample)
    mcmc = MCMC(kernel, 100, 200)
    mcmc._compile(random.PRNGKey(0), data, obs, extra_fields=("num_steps",))
    start = time()
    mcmc.run(random.PRNGKey(0), data, obs, extra_fields=('num_steps',))
    summary(dataset, 'hmc', mcmc, time() - start)


if __name__ == '__main__':

    datasets = ('copsac',)
    load_data = {'breast': breast_cancer_data, 'higgs': higgs_data, 'copsac': copsac_data}
    subsample_sizes = {'breast': 75, 'higgs': 1300, 'copsac': 1000}
    data, obs = breast_cancer_data()

    for dataset in datasets:
        dir = f'{dataset}_{datetime.now().strftime("%Y_%m_%d_%H%M%S")}'
        if not os.path.exists(dir):
            os.mkdir(dir)
        data, obs = load_data[dataset]()
        hmcecs_model(dir, data, obs, subsample_sizes[dataset])
        hmc(dir, data, obs)
