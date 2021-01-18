import jax.numpy as jnp
from jax import random
from sklearn.datasets import load_breast_cancer

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.ecs import ECS
from numpyro.distributions import constraints
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO


def breast_cancer_data():
    dataset = load_breast_cancer()
    feats = dataset.data
    feats = (feats - feats.mean(0)) / feats.std(0)
    feats = jnp.hstack((feats, jnp.ones((feats.shape[0], 1))))
    return feats, dataset.target


def log_reg_model(features, obs):
    n, m = features.shape
    theta = numpyro.sample('theta', dist.continuous.Normal(jnp.zeros(m), .5 * jnp.ones(m)))
    with numpyro.plate('N', n, subsample_size=75):
        batch_feats = numpyro.subsample(features, event_dim=1)
        batch_obs = numpyro.subsample(obs, event_dim=0)
        numpyro.sample('obs', dist.Bernoulli(logits=theta @ batch_feats.T), obs=batch_obs)


def log_reg_guide(feature, obs):
    _, m = feature.shape
    mean = numpyro.param('mean', jnp.zeros(m), constraints=constraints.real)
    var = numpyro.param('var', jnp.ones(m), constraints=constraints.positive)
    numpyro.sample('theta', dist.continuous.Normal(mean, var))


def hmcecs_model(data, obs):
    optimizer = numpyro.optim.Adam(step_size=0.005)
    svi = SVI(log_reg_model, log_reg_guide, optimizer, loss=Trace_ELBO())
    svi_result = svi.run(random.PRNGKey(1), 1000, data, obs)

    # Compute HMCECS
    kernel = ECS(NUTS(log_reg_model),
                 proxy='variational',
                 model_struct={'obs': ['theta']},
                 ref=svi_result.params,
                 guide=svi.guide)
    mcmc = MCMC(kernel, 1500, 8500)
    mcmc.run(random.PRNGKey(0), data, obs, extra_fields=("accept_prob",))
    mcmc.print_summary(exclude_deterministic=False)

def plain_log_reg_model(features, obs):
    n, m = features.shape
    theta = numpyro.sample('theta', dist.continuous.Normal(jnp.zeros(m), .5 * jnp.ones(m)))
    numpyro.sample('obs', dist.Bernoulli(logits=theta @ features.T), obs=obs)

def hmc(data, obs):
    kernel = NUTS(log_reg_model)
    mcmc = MCMC(kernel, 1500, 8500)
    mcmc.run(random.PRNGKey(0), data, obs, extra_fields=("accept_prob",))
    mcmc.print_summary(exclude_deterministic=False)


if __name__ == '__main__':
    data, obs = breast_cancer_data()
    # hmcecs_model(data, obs)
    hmc(data, obs)