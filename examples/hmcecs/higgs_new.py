import jax.numpy as jnp
from jax import random
from sklearn.datasets import load_breast_cancer

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.ecs import ECS
from numpyro.contrib.ecs_utils import difference_estimator_fn, taylor_proxy
from numpyro.infer import MCMC, NUTS


def breast_cancer_data():
    dataset = load_breast_cancer()
    feats = dataset.data
    feats = (feats - feats.mean(0)) / feats.std(0)
    feats = jnp.hstack((feats, jnp.ones((feats.shape[0], 1))))
    return feats, dataset.target


def log_reg_model(features, obs):
    n, m = features.shape
    theta = numpyro.sample('theta', dist.continuous.Normal(jnp.zeros(m), .5 * jnp.ones(m)))
    with numpyro.plate('N', n, subsample_size=75) as idx:
        batch_feats = numpyro.subsample(features, event_dim=1)
        batch_obs = numpyro.subsample(obs, event_dim=1)
        numpyro.sample('obs', dist.Bernoulli(logits=jnp.matmul(batch_feats, theta)), obs=batch_obs)


def plain_log_reg_model(features, obs):
    n, m = features.shape
    theta = numpyro.sample('theta', dist.continuous.Normal(jnp.zeros(m), 2 * jnp.ones(m)))
    numpyro.sample('obs', dist.Bernoulli(logits=jnp.matmul(features, theta)), obs=obs)


if __name__ == '__main__':
    data, obs = breast_cancer_data()

    # Get reference parameters
    kernel = NUTS(plain_log_reg_model)
    mcmc = MCMC(kernel, 500, 500)
    mcmc.run(random.PRNGKey(1), data, obs)
    z_ref = {k: v.mean(0) for k, v in mcmc.get_samples().items()}

    # Compute HMCECS
    kernel = ECS(NUTS(log_reg_model), estimator_fn=difference_estimator_fn, proxy_gen_fn=taylor_proxy, z_ref=z_ref)
    mcmc = MCMC(kernel, 500, 500)
    mcmc.run(random.PRNGKey(0), data, obs, extra_fields=("accept_prob",))
    mcmc.print_summary(exclude_deterministic=False)
