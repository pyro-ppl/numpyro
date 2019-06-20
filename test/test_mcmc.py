import os

import numpy as onp
from numpy.testing import assert_allclose
import pytest

from jax import pmap, random
from jax.lib import xla_bridge
import jax.numpy as np
from jax.scipy.special import logit

import numpyro.distributions as dist
from numpyro.handlers import sample
from numpyro.hmc_util import initialize_model
from numpyro.mcmc import hmc, mcmc
from numpyro.util import fori_collect


@pytest.mark.parametrize('algo', ['HMC', 'NUTS'])
@pytest.mark.parametrize('dense_mass', [False, True])
def test_unnormalized_normal(algo, dense_mass):
    true_mean, true_std = 1., 2.
    warmup_steps, num_samples = 1000, 8000

    def potential_fn(z):
        return 0.5 * np.sum(((z - true_mean) / true_std) ** 2)

    init_kernel, sample_kernel = hmc(potential_fn, algo=algo)
    init_params = np.array(0.)
    hmc_state = init_kernel(init_params,
                            trajectory_length=9,
                            num_warmup=warmup_steps,
                            dense_mass=dense_mass)
    hmc_states = fori_collect(0, num_samples, sample_kernel, hmc_state,
                              transform=lambda x: x.z)
    assert_allclose(np.mean(hmc_states), true_mean, rtol=0.05)
    assert_allclose(np.std(hmc_states), true_std, rtol=0.05)

    if 'JAX_ENABLE_x64' in os.environ:
        assert hmc_states.dtype == np.float64


def test_correlated_mvn():
    # This requires dense mass matrix estimation.
    D = 5

    warmup_steps, num_samples = 5000, 8000

    true_mean = 0.
    a = np.tril(0.5 * np.fliplr(np.eye(D)) + 0.1 * np.exp(random.normal(random.PRNGKey(0), shape=(D, D))))
    true_cov = np.dot(a, a.T)
    true_prec = np.linalg.inv(true_cov)

    def potential_fn(z):
        return 0.5 * np.dot(z.T, np.dot(true_prec, z))

    init_params = np.zeros(D)
    samples = mcmc(warmup_steps, num_samples, init_params, potential_fn=potential_fn, dense_mass=True)
    assert_allclose(np.mean(samples), true_mean, atol=0.02)
    assert onp.sum(onp.abs(onp.cov(samples.T) - true_cov)) / D**2 < 0.02


@pytest.mark.parametrize('algo', ['HMC', 'NUTS'])
def test_logistic_regression(algo):
    N, dim = 3000, 3
    warmup_steps, num_samples = 1000, 8000
    data = random.normal(random.PRNGKey(0), (N, dim))
    true_coefs = np.arange(1., dim + 1.)
    logits = np.sum(true_coefs * data, axis=-1)
    labels = dist.Bernoulli(logits=logits).sample(random.PRNGKey(1))

    def model(labels):
        coefs = sample('coefs', dist.Normal(np.zeros(dim), np.ones(dim)))
        logits = np.sum(coefs * data, axis=-1)
        return sample('obs', dist.Bernoulli(logits=logits), obs=labels)

    init_params, potential_fn, constrain_fn = initialize_model(random.PRNGKey(2), model, labels)
    samples = mcmc(warmup_steps, num_samples, init_params, sampler='hmc',
                   potential_fn=potential_fn, trajectory_length=10, constrain_fn=constrain_fn)
    assert_allclose(np.mean(samples['coefs'], 0), true_coefs, atol=0.21)

    if 'JAX_ENABLE_x64' in os.environ:
        assert samples['coefs'].dtype == np.float64


@pytest.mark.parametrize('algo', ['HMC', 'NUTS'])
def test_beta_bernoulli(algo):
    warmup_steps, num_samples = 500, 20000

    def model(data):
        alpha = np.array([1.1, 1.1])
        beta = np.array([1.1, 1.1])
        p_latent = sample('p_latent', dist.Beta(alpha, beta))
        sample('obs', dist.Bernoulli(p_latent), obs=data)
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
@pytest.mark.parametrize('dense_mass', [False, True])
def test_dirichlet_categorical(algo, dense_mass):
    warmup_steps, num_samples = 100, 20000

    def model(data):
        concentration = np.array([1.0, 1.0, 1.0])
        p_latent = sample('p_latent', dist.Dirichlet(concentration))
        sample('obs', dist.Categorical(p_latent), obs=data)
        return p_latent

    true_probs = np.array([0.1, 0.6, 0.3])
    data = dist.Categorical(true_probs).sample(random.PRNGKey(1), (2000,))
    init_params, potential_fn, constrain_fn = initialize_model(random.PRNGKey(2), model, data)
    samples = mcmc(warmup_steps, num_samples, init_params, constrain_fn=constrain_fn, progbar=False,
                   print_summary=False, potential_fn=potential_fn, algo=algo, trajectory_length=1.,
                   dense_mass=dense_mass)
    assert_allclose(np.mean(samples['p_latent'], 0), true_probs, atol=0.02)

    if 'JAX_ENABLE_x64' in os.environ:
        assert samples['p_latent'].dtype == np.float64


@pytest.mark.xfail(reason='TODO: Fix this flaky test.')
def test_change_point():
    # Ref: https://forum.pyro.ai/t/i-dont-understand-why-nuts-code-is-not-working-bayesian-hackers-mail/696
    warmup_steps, num_samples = 500, 3000

    def model(data):
        alpha = 1 / np.mean(data)
        lambda1 = sample('lambda1', dist.Exponential(alpha))
        lambda2 = sample('lambda2', dist.Exponential(alpha))
        tau = sample('tau', dist.Uniform(0, 1))
        lambda12 = np.where(np.arange(len(data)) < tau * len(data), lambda1, lambda2)
        sample('obs', dist.Poisson(lambda12), obs=data)

    count_data = np.array([
        13,  24,   8,  24,   7,  35,  14,  11,  15,  11,  22,  22,  11,  57,
        11,  19,  29,   6,  19,  12,  22,  12,  18,  72,  32,   9,   7,  13,
        19,  23,  27,  20,   6,  17,  13,  10,  14,   6,  16,  15,   7,   2,
        15,  15,  19,  70,  49,   7,  53,  22,  21,  31,  19,  11,  18,  20,
        12,  35,  17,  23,  17,   4,   2,  31,  30,  13,  27,   0,  39,  37,
        5,  14,  13,  22,
    ])
    init_params, potential_fn, constrain_fn = initialize_model(random.PRNGKey(4), model, count_data)
    init_kernel, sample_kernel = hmc(potential_fn)
    hmc_state = init_kernel(init_params, num_warmup=warmup_steps)
    samples = fori_collect(0, num_samples, sample_kernel, hmc_state,
                           transform=lambda x: constrain_fn(x.z))
    tau_posterior = (samples['tau'] * len(count_data)).astype("int")
    tau_values, counts = onp.unique(tau_posterior, return_counts=True)
    mode_ind = np.argmax(counts)
    mode = tau_values[mode_ind]
    assert mode == 44

    if 'JAX_ENABLE_x64' in os.environ:
        assert samples['lambda1'].dtype == np.float64
        assert samples['lambda2'].dtype == np.float64
        assert samples['tau'].dtype == np.float64


@pytest.mark.parametrize('with_logits', ['True', 'False'])
def test_binomial_stable(with_logits):
    # Ref: https://github.com/pyro-ppl/pyro/issues/1706
    warmup_steps, num_samples = 200, 200

    def model(data):
        p = sample('p', dist.Beta(1., 1.))
        if with_logits:
            logits = logit(p)
            sample('obs', dist.Binomial(data['n'], logits=logits), obs=data['x'])
        else:
            sample('obs', dist.Binomial(data['n'], probs=p), obs=data['x'])

    data = {'n': 5000000, 'x': 3849}
    init_params, potential_fn, constrain_fn = initialize_model(random.PRNGKey(2), model, data)
    init_kernel, sample_kernel = hmc(potential_fn)
    hmc_state = init_kernel(init_params, num_warmup=warmup_steps)
    samples = fori_collect(0, num_samples, sample_kernel, hmc_state,
                           transform=lambda x: constrain_fn(x.z))

    assert_allclose(np.mean(samples['p'], 0), data['x'] / data['n'], rtol=0.05)

    if 'JAX_ENABLE_x64' in os.environ:
        assert samples['p'].dtype == np.float64


@pytest.mark.parametrize('algo', ['HMC', 'NUTS'])
def test_pmap(algo):
    if xla_bridge.device_count() == 1:
        pytest.skip('pmap test requires device_count greater than 1.')

    true_mean, true_std = 1., 2.
    warmup_steps, num_samples = 1000, 8000

    def potential_fn(z):
        return 0.5 * np.sum(((z - true_mean) / true_std) ** 2)

    init_kernel, sample_kernel = hmc(potential_fn, algo=algo)
    init_params = np.array([0., -1.])
    rngs = random.split(random.PRNGKey(0), 2)

    init_kernel_pmap = pmap(lambda init_param, rng: init_kernel(
        init_param, trajectory_length=9, num_warmup=warmup_steps, progbar=False, rng=rng))
    init_states = init_kernel_pmap(init_params, rngs)

    fori_collect_pmap = pmap(lambda hmc_state: fori_collect(0, num_samples, sample_kernel, hmc_state,
                                                            transform=lambda x: x.z, progbar=False))
    chain_samples = fori_collect_pmap(init_states)

    assert_allclose(np.mean(chain_samples, axis=1), np.repeat(true_mean, 2), rtol=0.05)
    assert_allclose(np.std(chain_samples, axis=1), np.repeat(true_std, 2), rtol=0.05)


@pytest.mark.filterwarnings("ignore:There are not enough devices:UserWarning")
def test_chain():
    N, dim = 3000, 3
    num_warmup, num_samples = 5000, 5000
    data = random.normal(random.PRNGKey(0), (N, dim))
    true_coefs = np.arange(1., dim + 1.)
    logits = np.sum(true_coefs * data, axis=-1)
    labels = dist.Bernoulli(logits=logits).sample(random.PRNGKey(1))

    def model(labels):
        coefs = sample('coefs', dist.Normal(np.zeros(dim), np.ones(dim)))
        logits = np.sum(coefs * data, axis=-1)
        return sample('obs', dist.Bernoulli(logits=logits), obs=labels)

    rngs = random.split(random.PRNGKey(2), 2)
    init_params, potential_fn, constrain_fn = initialize_model(rngs, model, labels)
    samples = mcmc(num_warmup, num_samples, init_params, num_chains=2,
                   potential_fn=potential_fn, constrain_fn=constrain_fn)

    assert samples['coefs'].shape[0] == 2 * num_samples
    assert_allclose(np.mean(samples['coefs'], 0), true_coefs, atol=0.21)
