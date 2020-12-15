""" Logistic regression model as implemetned in https://arxiv.org/pdf/1708.00955.pdf with Higgs Dataset """
# !/usr/bin/env python
from collections import namedtuple

import jax
import jax.numpy as jnp
import jax.numpy as np_jax
from jax.tree_util import tree_map
from sklearn.model_selection import train_test_split

import numpyro
import numpyro.distributions as dist
from examples.logistic_hmcecs_svi import svi_map
from numpyro import optim
from numpyro.contrib.autoguide_hmcecs import AutoDiagonalNormal
from numpyro.contrib.hmcecs import HMCECS
from numpyro.diagnostics import summary
from numpyro.examples.datasets import _load_higgs
from numpyro.infer import NUTS, MCMC
from numpyro.infer.elbo import Trace_ELBO
from numpyro.infer.svi import SVI

numpyro.set_platform("gpu")

DataLoaderState = namedtuple("DataLoaderState", ('iteration', 'rng_key', 'indexes', 'max_iter'))


def dataloader(*xs, batch_size=32, train_size=None, test_size=None, shuffle=True):
    assert len(xs) > 1
    splitxs = train_test_split(*xs, train_size=train_size, test_size=test_size)
    trainxs, testxs = splitxs[0::2], splitxs[1::2]
    max_train_iter, max_test_iter = len(trainxs[0]) // batch_size, len(testxs[0]) // batch_size

    def make_dataset(dxs, max_iter):
        def init(rng_key):
            return DataLoaderState(0, rng_key, jnp.arange(len(dxs[0])), max_iter)

        def next_step(state):

            iteration = state.iteration % state.max_iter
            batch = tuple(x[state.indexes[iteration * batch_size:(iteration + 1) * batch_size]]
                          for x in dxs)
            if iteration + 1 == state.max_iter:
                shuffle_rng_key, rng_key = jax.random.split(state.rng_key)
                if shuffle:
                    indexes = jax.random.shuffle(shuffle_rng_key, state.indexes)
                else:
                    indexes = state.indexes
                return batch, DataLoaderState(state.iteration + 1, rng_key, indexes, state.max_iter)
            else:
                return batch, DataLoaderState(state.iteration + 1, state.rng_key, state.indexes, state.max_iter)

        return init, next_step

    return make_dataset(trainxs, max_train_iter), make_dataset(testxs, max_test_iter), testxs


def svi_map(model, rng_key, feats, obs, num_epochs, batch_size):
    guide = AutoDiagonalNormal(model)
    svi = SVI(model, guide, optim.Adam(0.0003), loss=Trace_ELBO())
    svi_rng_key, data_rng_key = jax.random.split(rng_key)
    (init_train, next_train), _, _ = dataloader(feats, obs, train_size=0.9, batch_size=batch_size)
    batch_fn = jax.jit(svi.update)
    svi_state = None
    data_state = init_train(data_rng_key)
    num_batches = 0
    for _ in range(num_epochs):
        for j in range(data_state.max_iter):
            xs, data_state = next_train(data_state)
            if svi_state is None:
                svi_state = svi.init(svi_rng_key, *xs)
            svi_state, _ = batch_fn(svi_state, *xs)
            num_batches += 1
    return svi, svi_state


def infer_nuts(rng_key, features, obs, samples, warmup):
    kernel = NUTS(model=logistic_regression, target_accept_prob=0.8)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples)
    mcmc.run(rng_key, features, obs)
    samples = mcmc.get_samples()
    samples = tree_map(lambda x: x[None, ...], samples)
    r_hat_average = np_jax.sum(summary(samples)["theta"]["r_hat"]) / len(summary(samples)["theta"]["r_hat"])

    return mcmc.get_samples(), r_hat_average


def infer_hmcecs(rng_key, obs, features, m=None, g=None, n_samples=None, warmup=None, algo="NUTS",
                 subsample_method=None, map_method=None, proxy="taylor", estimator=None, num_epochs=None):
    hmcecs_key, map_key = jax.random.split(rng_key)
    n, _ = features.shape

    svi = None
    if map_method == "nuts":
        samples, r_hat_average = infer_nuts(map_key, features, obs, samples=10, warmup=5)
        z_ref = {key: value.mean(0) for key, value in samples.items()}
    elif map_method == "svi":
        map_key, post_key = jax.random.split(map_key)
        svi, svi_state = svi_map(logistic_regression,
                                 map_key,
                                 feats=features,
                                 obs=obs,
                                 num_epochs=num_epochs,
                                 batch_size=256)
        z_ref = svi.guide.sample_posterior(post_key, svi.get_params(svi_state), (100,))
        z_ref = {name: value.mean(0) for name, value in z_ref.items()}

    kernel = HMCECS(model=logistic_regression, z_ref=z_ref, m=m, g=g, algo=algo.upper(),
                    subsample_method=subsample_method, proxy=proxy, svi_fn=svi,
                    estimator=estimator, target_accept_prob=0.8)

    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=n_samples, num_chains=1)
    mcmc.run(rng_key, features, obs)

    return mcmc.get_samples()


def logistic_regression(features, obs):
    n, m = features.shape
    theta = numpyro.sample('theta', dist.continuous.Normal(jnp.zeros(m), 2 * jnp.ones(m)))
    numpyro.sample('obs', dist.Bernoulli(logits=jnp.matmul(features, theta)), obs=obs)


def higgs_data():
    return _load_higgs()


if __name__ == '__main__':
    rng_key = jax.random.PRNGKey(37)
    obs, feats = higgs_data()
    num_examples = 1000

    est_posterior_ECS = infer_hmcecs(rng_key, obs[:num_examples], feats[:num_examples],
                                     n_samples=10,
                                     warmup=5,
                                     m=30, g=5,
                                     algo='nuts',
                                     subsample_method="perturb",
                                     proxy='svi',
                                     estimator='',
                                     map_method='svi',
                                     num_epochs=100)
