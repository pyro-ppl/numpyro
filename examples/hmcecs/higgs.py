import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax
from sklearn.datasets import load_breast_cancer

import numpyro
import numpyro.distributions as dist
from examples.logistic_hmcecs_svi import svi_map
from numpyro import optim
from numpyro.contrib.autoguide_hmcecs import AutoDiagonalNormal
from numpyro.contrib.hmcecs import HMCECS
from numpyro.infer import NUTS, MCMC
from numpyro.infer.elbo import ELBO
from numpyro.infer.svi import SVI
from numpyro.util import fori_loop

numpyro.set_platform("cpu")


def load_dataset(observations, features, batch_size=None, shuffle=True):
    num_records = observations.shape[0]
    idxs = jnp.arange(num_records)
    if not batch_size:
        batch_size = num_records

    def init():
        return num_records // batch_size, np.random.permutation(idxs) if shuffle else idxs

    def get_batch(i=0, idxs=idxs):
        ret_idx = lax.dynamic_slice_in_dim(idxs, i * batch_size, batch_size)
        batch_obs = jnp.take(observations, ret_idx, axis=0)
        batch_feats = jnp.take(features, ret_idx, axis=0)
        return batch_obs, batch_feats

    return init, get_batch


def svi_map(model, rng_key, feats, obs, num_epochs, batch_size):
    @jit
    def epoch_train(svi_state):
        def body_fn(i, val):
            batch_obs, batch_feats = train_fetch(i, train_idx)
            loss_sum, svi_state = val
            svi_state, loss = svi.update(svi_state, batch_feats, batch_obs)
            loss_sum += loss
            return loss_sum, svi_state

        return fori_loop(0, num_train, body_fn, (0., svi_state))

    n, _ = feats.shape
    guide = AutoDiagonalNormal(model)
    svi = SVI(model, guide, optim.Adam(0.0003), loss=ELBO())
    svi_state = svi.init(rng_key, feats, obs)
    train_init, train_fetch = load_dataset(obs, feats, batch_size=batch_size)

    for i in range(num_epochs):
        num_train, train_idx = train_init()
        train_loss, svi_state = epoch_train(svi_state)
    return svi.get_params(svi_state), svi, svi_state


def breast_cancer_data():
    """ Logistic regression model as implemetned in https://arxiv.org/pdf/1708.00955.pdf with Higgs Dataset """
    dataset = load_breast_cancer()
    feats = dataset.data
    feats = (feats - feats.mean(0)) / feats.std(0)
    feats = jnp.hstack((feats, jnp.ones((feats.shape[0], 1))))

    return feats, dataset.target


def model(feats, obs):
    """  Logistic regression model """
    n, m = feats.shape
    theta = numpyro.sample('theta', dist.continuous.Normal(jnp.zeros(m), 2 * jnp.ones(m)))
    numpyro.sample('obs', dist.Bernoulli(logits=jnp.matmul(feats, theta)), obs=obs)


def infer_hmcecs(rng_key, feats, obs, m=None, g=None, n_samples=None, warmup=None, algo="NUTS", subsample_method=None,
                 map_method=None, proxy="taylor", estimator=None, num_epochs=None, postprocess_fn=None):
    hmcecs_key, map_key = jax.random.split(rng_key)
    n, _ = feats.shape

    if map_method == "SVI":
        factor_SVI = obs.shape[0]
        batch_size = 32
        map_key, post_key = jax.random.split(map_key)
        z_ref, svi, svi_state = svi_map(model, map_key, feats=feats[:factor_SVI], obs=obs[:factor_SVI],
                                        num_epochs=num_epochs, batch_size=batch_size)
        z_ref = svi.guide.sample_posterior(post_key, svi.get_params(svi_state), (100,))
        z_ref = {name: value.mean(0) for name, value in z_ref.items()}
    else:
        svi = None
        map_samples = 10
        map_warmup = 5
        if map_method == "NUTS":
            kernel = NUTS(model=model, target_accept_prob=0.8)
        if map_method == 'HMC':
            kernel = NUTS(model=model, target_accept_prob=0.8)
        mcmc = MCMC(kernel, num_warmup=map_warmup, num_samples=map_samples)
        mcmc.run(rng_key, feats, obs)
        samples = mcmc.get_samples()
        z_ref = {key: value.mean(0) for key, value in samples.items()}

    extra_fields = []
    if estimator == "poisson":
        postprocess_fn = None
        extra_fields = ("sign",)

    kernel = HMCECS(model=model, z_ref=z_ref, m=m, g=g, algo=algo, subsample_method=subsample_method, proxy=proxy,
                    svi_fn=svi, estimator=estimator, target_accept_prob=0.8)

    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=n_samples, num_chains=1, postprocess_fn=postprocess_fn)
    mcmc.run(rng_key, feats, obs, extra_fields=extra_fields)

    return mcmc.get_samples()


if __name__ == '__main__':
    num_samples = 10
    num_warmup = 5
    ecs_algo = 'NUTS'
    ecs_proxy = 'taylor'
    estimator = 'perturb'
    map_init = 'SVI'
    epochs = 1000
    rng_key = jax.random.PRNGKey(37)

    feats, obs = breast_cancer_data()

    n, = obs.shape
    m = int(jnp.sqrt(n))
    g = 5

    infer_hmcecs(rng_key, feats=feats, obs=obs, n_samples=num_samples,
                 warmup=num_warmup, m=m, g=g, algo=ecs_algo, subsample_method="perturb",
                 proxy=ecs_proxy, estimator=estimator, map_method=map_init, num_epochs=epochs)
