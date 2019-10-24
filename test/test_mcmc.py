import os

from numpy.testing import assert_allclose
import pytest

from jax import pmap, random, vmap
from jax.lib import xla_bridge
import jax.numpy as np

import numpyro
import numpyro.distributions as dist
from numpyro.infer.mcmc import hmc, mcmc
from numpyro.infer.util import initialize_model
from numpyro.util import fori_collect


@pytest.mark.parametrize('algo', ['HMC', 'NUTS'])
def test_logistic_regression(algo):
    N, dim = 3000, 3
    warmup_steps, num_samples = 1000, 8000
    data = random.normal(random.PRNGKey(0), (N, dim))
    true_coefs = np.arange(1., dim + 1.)
    logits = np.sum(true_coefs * data, axis=-1)
    labels = dist.Bernoulli(logits=logits).sample(random.PRNGKey(1))

    def model(labels):
        coefs = numpyro.sample('coefs', dist.Normal(np.zeros(dim), np.ones(dim)))
        logits = np.sum(coefs * data, axis=-1)
        return numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=labels)

    init_params, potential_fn, constrain_fn = initialize_model(random.PRNGKey(2), model, labels)
    samples = mcmc(warmup_steps, num_samples, init_params, sampler='hmc', algo=algo,
                   potential_fn=potential_fn, trajectory_length=10, constrain_fn=constrain_fn)
    assert_allclose(np.mean(samples['coefs'], 0), true_coefs, atol=0.22)

    if 'JAX_ENABLE_x64' in os.environ:
        assert samples['coefs'].dtype == np.float64


@pytest.mark.parametrize('algo', ['HMC', 'NUTS'])
def test_beta_bernoulli(algo):
    warmup_steps, num_samples = 500, 20000

    def model(data):
        alpha = np.array([1.1, 1.1])
        beta = np.array([1.1, 1.1])
        p_latent = numpyro.sample('p_latent', dist.Beta(alpha, beta))
        numpyro.sample('obs', dist.Bernoulli(p_latent), obs=data)
        return p_latent

    true_probs = np.array([0.9, 0.1])
    data = dist.Bernoulli(true_probs).sample(random.PRNGKey(1), (1000, 2))
    init_params, potential_fn, constrain_fn = initialize_model(random.PRNGKey(2), model, data)
    init_kernel, sample_kernel = hmc(potential_fn, algo=algo)
    hmc_state = init_kernel(init_params,
                            trajectory_length=1.,
                            num_warmup=warmup_steps,
                            progbar=False)
    samples = fori_collect(0, num_samples, sample_kernel, hmc_state,
                           transform=lambda x: constrain_fn(x.z),
                           progbar=False)
    assert_allclose(np.mean(samples['p_latent'], 0), true_probs, atol=0.05)

    if 'JAX_ENABLE_x64' in os.environ:
        assert samples['p_latent'].dtype == np.float64


@pytest.mark.parametrize('algo', ['HMC', 'NUTS'])
@pytest.mark.parametrize('map_fn', [vmap, pmap])
@pytest.mark.skipif('JAX_ENABLE_x64' in os.environ, reason='skip x64 test')
def test_map(algo, map_fn):
    if map_fn is pmap and xla_bridge.device_count() == 1:
        pytest.skip('pmap test requires device_count greater than 1.')

    true_mean, true_std = 1., 2.
    warmup_steps, num_samples = 1000, 8000

    def potential_fn(z):
        return 0.5 * np.sum(((z - true_mean) / true_std) ** 2)

    init_kernel, sample_kernel = hmc(potential_fn, algo=algo)
    init_params = np.array([0., -1.])
    rng_keys = random.split(random.PRNGKey(0), 2)

    init_kernel_map = map_fn(lambda init_param, rng_key: init_kernel(
        init_param, trajectory_length=9, num_warmup=warmup_steps, progbar=False, rng_key=rng_key))
    init_states = init_kernel_map(init_params, rng_keys)

    fori_collect_map = map_fn(lambda hmc_state: fori_collect(0, num_samples, sample_kernel, hmc_state,
                                                             transform=lambda x: x.z, progbar=False))
    chain_samples = fori_collect_map(init_states)

    assert_allclose(np.mean(chain_samples, axis=1), np.repeat(true_mean, 2), rtol=0.05)
    assert_allclose(np.std(chain_samples, axis=1), np.repeat(true_std, 2), rtol=0.05)
