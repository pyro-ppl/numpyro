# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as onp
from jax.test_util import check_close
from numpy.testing import assert_allclose
import pytest

from jax import jit, pmap, random, vmap
from jax.lib import xla_bridge
import jax.numpy as np
from jax.scipy.special import logit

import numpyro
from numpyro.contrib.reparam import TransformReparam, reparam
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import HMC, MCMC, NUTS, SA
from numpyro.infer.mcmc import hmc, _get_proposal_loc_and_scale, _numpy_delete
from numpyro.infer.util import initialize_model
from numpyro.util import fori_collect


@pytest.mark.parametrize('kernel_cls', [HMC, NUTS, SA])
@pytest.mark.parametrize('dense_mass', [False, True])
def test_unnormalized_normal_x64(kernel_cls, dense_mass):
    true_mean, true_std = 1., 0.5
    warmup_steps, num_samples = (100000, 100000) if kernel_cls is SA else (1000, 8000)

    def potential_fn(z):
        return 0.5 * np.sum(((z - true_mean) / true_std) ** 2)

    init_params = np.array(0.)
    if kernel_cls is SA:
        kernel = SA(potential_fn=potential_fn, dense_mass=dense_mass)
    else:
        kernel = kernel_cls(potential_fn=potential_fn, trajectory_length=8, dense_mass=dense_mass)
    mcmc = MCMC(kernel, warmup_steps, num_samples, progress_bar=False)
    mcmc.run(random.PRNGKey(0), init_params=init_params)
    mcmc.print_summary()
    hmc_states = mcmc.get_samples()
    assert_allclose(np.mean(hmc_states), true_mean, rtol=0.05)
    assert_allclose(np.std(hmc_states), true_std, rtol=0.05)

    if 'JAX_ENABLE_X64' in os.environ:
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
    kernel = NUTS(potential_fn=potential_fn, dense_mass=True)
    mcmc = MCMC(kernel, warmup_steps, num_samples)
    mcmc.run(random.PRNGKey(0), init_params=init_params)
    samples = mcmc.get_samples()
    assert_allclose(np.mean(samples), true_mean, atol=0.02)
    assert onp.sum(onp.abs(onp.cov(samples.T) - true_cov)) / D**2 < 0.02


@pytest.mark.parametrize('kernel_cls', [HMC, NUTS, SA])
def test_logistic_regression_x64(kernel_cls):
    N, dim = 3000, 3
    warmup_steps, num_samples = (100000, 100000) if kernel_cls is SA else (1000, 8000)
    data = random.normal(random.PRNGKey(0), (N, dim))
    true_coefs = np.arange(1., dim + 1.)
    logits = np.sum(true_coefs * data, axis=-1)
    labels = dist.Bernoulli(logits=logits).sample(random.PRNGKey(1))

    def model(labels):
        coefs = numpyro.sample('coefs', dist.Normal(np.zeros(dim), np.ones(dim)))
        logits = numpyro.deterministic('logits', np.sum(coefs * data, axis=-1))
        return numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=labels)

    if kernel_cls is SA:
        kernel = SA(model=model, adapt_state_size=9)
    else:
        kernel = kernel_cls(model=model, trajectory_length=8, find_heuristic_step_size=True)
    mcmc = MCMC(kernel, warmup_steps, num_samples, progress_bar=False)
    mcmc.run(random.PRNGKey(2), labels)
    mcmc.print_summary()
    samples = mcmc.get_samples()
    assert samples['logits'].shape == (num_samples, N)
    assert_allclose(np.mean(samples['coefs'], 0), true_coefs, atol=0.22)

    if 'JAX_ENABLE_X64' in os.environ:
        assert samples['coefs'].dtype == np.float64


def test_uniform_normal():
    true_coef = 0.9
    num_warmup, num_samples = 1000, 1000

    def model(data):
        alpha = numpyro.sample('alpha', dist.Uniform(0, 1))
        with reparam(config={'loc': TransformReparam()}):
            loc = numpyro.sample('loc', dist.Uniform(0, alpha))
        numpyro.sample('obs', dist.Normal(loc, 0.1), obs=data)

    data = true_coef + random.normal(random.PRNGKey(0), (1000,))
    kernel = NUTS(model=model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.warmup(random.PRNGKey(2), data, collect_warmup=True)
    warmup_samples = mcmc.get_samples()
    mcmc.run(random.PRNGKey(3), data)
    samples = mcmc.get_samples()
    assert len(warmup_samples['loc']) == num_warmup
    assert len(samples['loc']) == num_samples
    assert_allclose(np.mean(samples['loc'], 0), true_coef, atol=0.05)


def test_improper_normal():
    true_coef = 0.9

    def model(data):
        alpha = numpyro.sample('alpha', dist.Uniform(0, 1))
        loc = numpyro.param('loc', 0., constraint=constraints.interval(0., alpha))
        numpyro.sample('obs', dist.Normal(loc, 0.1), obs=data)

    data = true_coef + random.normal(random.PRNGKey(0), (1000,))
    kernel = NUTS(model=model)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000)
    mcmc.run(random.PRNGKey(0), data)
    samples = mcmc.get_samples()
    assert_allclose(np.mean(samples['loc'], 0), true_coef, atol=0.05)


@pytest.mark.parametrize('kernel_cls', [HMC, NUTS, SA])
def test_beta_bernoulli_x64(kernel_cls):
    warmup_steps, num_samples = (100000, 100000) if kernel_cls is SA else (500, 20000)

    def model(data):
        alpha = np.array([1.1, 1.1])
        beta = np.array([1.1, 1.1])
        p_latent = numpyro.sample('p_latent', dist.Beta(alpha, beta))
        numpyro.sample('obs', dist.Bernoulli(p_latent), obs=data)
        return p_latent

    true_probs = np.array([0.9, 0.1])
    data = dist.Bernoulli(true_probs).sample(random.PRNGKey(1), (1000, 2))
    if kernel_cls is SA:
        kernel = SA(model=model)
    else:
        kernel = kernel_cls(model=model, trajectory_length=1.)
    mcmc = MCMC(kernel, num_warmup=warmup_steps, num_samples=num_samples, progress_bar=False)
    mcmc.run(random.PRNGKey(2), data)
    mcmc.print_summary()
    samples = mcmc.get_samples()
    assert_allclose(np.mean(samples['p_latent'], 0), true_probs, atol=0.05)

    if 'JAX_ENABLE_X64' in os.environ:
        assert samples['p_latent'].dtype == np.float64


@pytest.mark.parametrize('kernel_cls', [HMC, NUTS])
@pytest.mark.parametrize('dense_mass', [False, True])
def test_dirichlet_categorical_x64(kernel_cls, dense_mass):
    warmup_steps, num_samples = 100, 20000

    def model(data):
        concentration = np.array([1.0, 1.0, 1.0])
        p_latent = numpyro.sample('p_latent', dist.Dirichlet(concentration))
        numpyro.sample('obs', dist.Categorical(p_latent), obs=data)
        return p_latent

    true_probs = np.array([0.1, 0.6, 0.3])
    data = dist.Categorical(true_probs).sample(random.PRNGKey(1), (2000,))
    kernel = kernel_cls(model, trajectory_length=1., dense_mass=dense_mass)
    mcmc = MCMC(kernel, warmup_steps, num_samples, progress_bar=False)
    mcmc.run(random.PRNGKey(2), data)
    samples = mcmc.get_samples()
    assert_allclose(np.mean(samples['p_latent'], 0), true_probs, atol=0.02)

    if 'JAX_ENABLE_X64' in os.environ:
        assert samples['p_latent'].dtype == np.float64


def test_change_point_x64():
    # Ref: https://forum.pyro.ai/t/i-dont-understand-why-nuts-code-is-not-working-bayesian-hackers-mail/696
    warmup_steps, num_samples = 500, 3000

    def model(data):
        alpha = 1 / np.mean(data)
        lambda1 = numpyro.sample('lambda1', dist.Exponential(alpha))
        lambda2 = numpyro.sample('lambda2', dist.Exponential(alpha))
        tau = numpyro.sample('tau', dist.Uniform(0, 1))
        lambda12 = np.where(np.arange(len(data)) < tau * len(data), lambda1, lambda2)
        numpyro.sample('obs', dist.Poisson(lambda12), obs=data)

    count_data = np.array([
        13,  24,   8,  24,   7,  35,  14,  11,  15,  11,  22,  22,  11,  57,
        11,  19,  29,   6,  19,  12,  22,  12,  18,  72,  32,   9,   7,  13,
        19,  23,  27,  20,   6,  17,  13,  10,  14,   6,  16,  15,   7,   2,
        15,  15,  19,  70,  49,   7,  53,  22,  21,  31,  19,  11,  18,  20,
        12,  35,  17,  23,  17,   4,   2,  31,  30,  13,  27,   0,  39,  37,
        5,  14,  13,  22,
    ])
    kernel = NUTS(model=model)
    mcmc = MCMC(kernel, warmup_steps, num_samples)
    mcmc.run(random.PRNGKey(4), count_data)
    samples = mcmc.get_samples()
    tau_posterior = (samples['tau'] * len(count_data)).astype(np.int32)
    tau_values, counts = onp.unique(tau_posterior, return_counts=True)
    mode_ind = np.argmax(counts)
    mode = tau_values[mode_ind]
    assert mode == 44

    if 'JAX_ENABLE_X64' in os.environ:
        assert samples['lambda1'].dtype == np.float64
        assert samples['lambda2'].dtype == np.float64
        assert samples['tau'].dtype == np.float64


@pytest.mark.parametrize('with_logits', ['True', 'False'])
def test_binomial_stable_x64(with_logits):
    # Ref: https://github.com/pyro-ppl/pyro/issues/1706
    warmup_steps, num_samples = 200, 200

    def model(data):
        p = numpyro.sample('p', dist.Beta(1., 1.))
        if with_logits:
            logits = logit(p)
            numpyro.sample('obs', dist.Binomial(data['n'], logits=logits), obs=data['x'])
        else:
            numpyro.sample('obs', dist.Binomial(data['n'], probs=p), obs=data['x'])

    data = {'n': 5000000, 'x': 3849}
    kernel = NUTS(model=model)
    mcmc = MCMC(kernel, warmup_steps, num_samples)
    mcmc.run(random.PRNGKey(2), data)
    samples = mcmc.get_samples()
    assert_allclose(np.mean(samples['p'], 0), data['x'] / data['n'], rtol=0.05)

    if 'JAX_ENABLE_X64' in os.environ:
        assert samples['p'].dtype == np.float64


def test_improper_prior():
    true_mean, true_std = 1., 2.
    num_warmup, num_samples = 1000, 8000

    def model(data):
        mean = numpyro.param('mean', 0.)
        std = numpyro.param('std', 1., constraint=constraints.positive)
        return numpyro.sample('obs', dist.Normal(mean, std), obs=data)

    data = dist.Normal(true_mean, true_std).sample(random.PRNGKey(1), (2000,))
    kernel = NUTS(model=model)
    mcmc = MCMC(kernel, num_warmup, num_samples)
    mcmc.warmup(random.PRNGKey(2), data)
    mcmc.run(random.PRNGKey(2), data)
    samples = mcmc.get_samples()
    assert_allclose(np.mean(samples['mean']), true_mean, rtol=0.05)
    assert_allclose(np.mean(samples['std']), true_std, rtol=0.05)


def test_mcmc_progbar():
    true_mean, true_std = 1., 2.
    num_warmup, num_samples = 10, 10

    def model(data):
        mean = numpyro.param('mean', 0.)
        std = numpyro.param('std', 1., constraint=constraints.positive)
        return numpyro.sample('obs', dist.Normal(mean, std), obs=data)

    data = dist.Normal(true_mean, true_std).sample(random.PRNGKey(1), (2000,))
    kernel = NUTS(model=model)
    mcmc = MCMC(kernel, num_warmup, num_samples)
    mcmc.warmup(random.PRNGKey(2), data)
    mcmc.run(random.PRNGKey(3), data)
    mcmc1 = MCMC(kernel, num_warmup, num_samples, progress_bar=False)
    mcmc1.run(random.PRNGKey(2), data)

    with pytest.raises(AssertionError):
        check_close(mcmc1.get_samples(), mcmc.get_samples(), atol=1e-4, rtol=1e-4)
    mcmc1.warmup(random.PRNGKey(2), data)
    mcmc1.run(random.PRNGKey(3), data)
    check_close(mcmc1.get_samples(), mcmc.get_samples(), atol=1e-4, rtol=1e-4)
    check_close(mcmc1._warmup_state, mcmc._warmup_state, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize('kernel_cls', [HMC, NUTS])
@pytest.mark.parametrize('adapt_step_size', [True, False])
def test_diverging(kernel_cls, adapt_step_size):
    data = random.normal(random.PRNGKey(0), (1000,))

    def model(data):
        loc = numpyro.sample('loc', dist.Normal(0., 1.))
        numpyro.sample('obs', dist.Normal(loc, 1), obs=data)

    kernel = kernel_cls(model, step_size=10., adapt_step_size=adapt_step_size, adapt_mass_matrix=False)
    num_warmup = num_samples = 1000
    mcmc = MCMC(kernel, num_warmup, num_samples)
    mcmc.warmup(random.PRNGKey(1), data, extra_fields=['diverging'], collect_warmup=True)
    warmup_divergences = mcmc.get_extra_fields()['diverging'].sum()
    mcmc.run(random.PRNGKey(2), data, extra_fields=['diverging'])
    num_divergences = warmup_divergences + mcmc.get_extra_fields()['diverging'].sum()
    if adapt_step_size:
        assert num_divergences <= num_warmup
    else:
        assert_allclose(num_divergences, num_warmup + num_samples)


def test_prior_with_sample_shape():
    data = {
        "J": 8,
        "y": np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]),
        "sigma": np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]),
    }

    def schools_model():
        mu = numpyro.sample('mu', dist.Normal(0, 5))
        tau = numpyro.sample('tau', dist.HalfCauchy(5))
        theta = numpyro.sample('theta', dist.Normal(mu, tau), sample_shape=(data['J'],))
        numpyro.sample('obs', dist.Normal(theta, data['sigma']), obs=data['y'])

    num_samples = 500
    mcmc = MCMC(NUTS(schools_model), num_warmup=500, num_samples=num_samples)
    mcmc.run(random.PRNGKey(0))
    assert mcmc.get_samples()['theta'].shape == (num_samples, data['J'])


@pytest.mark.parametrize('num_chains', [1, 2])
@pytest.mark.parametrize('chain_method', ['parallel', 'sequential', 'vectorized'])
@pytest.mark.parametrize('progress_bar', [True, False])
@pytest.mark.filterwarnings("ignore:There are not enough devices:UserWarning")
def test_empty_model(num_chains, chain_method, progress_bar):
    def model():
        pass

    mcmc = MCMC(NUTS(model), num_warmup=10, num_samples=10, num_chains=num_chains,
                chain_method=chain_method, progress_bar=progress_bar)
    mcmc.run(random.PRNGKey(0))
    assert mcmc.get_samples() == {}


@pytest.mark.parametrize('use_init_params', [False, True])
@pytest.mark.parametrize('chain_method', ['parallel', 'sequential', 'vectorized'])
@pytest.mark.skipif('XLA_FLAGS' not in os.environ, reason='without this mark, we have duplicated tests in Travis')
def test_chain(use_init_params, chain_method):
    N, dim = 3000, 3
    num_chains = 2
    num_warmup, num_samples = 5000, 5000
    data = random.normal(random.PRNGKey(0), (N, dim))
    true_coefs = np.arange(1., dim + 1.)
    logits = np.sum(true_coefs * data, axis=-1)
    labels = dist.Bernoulli(logits=logits).sample(random.PRNGKey(1))

    def model(labels):
        coefs = numpyro.sample('coefs', dist.Normal(np.zeros(dim), np.ones(dim)))
        logits = np.sum(coefs * data, axis=-1)
        return numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=labels)

    kernel = NUTS(model=model)
    mcmc = MCMC(kernel, num_warmup, num_samples, num_chains=num_chains)
    mcmc.chain_method = chain_method
    init_params = None if not use_init_params else \
        {'coefs': np.tile(np.ones(dim), num_chains).reshape(num_chains, dim)}
    mcmc.run(random.PRNGKey(2), labels, init_params=init_params)
    samples_flat = mcmc.get_samples()
    assert samples_flat['coefs'].shape[0] == num_chains * num_samples
    samples = mcmc.get_samples(group_by_chain=True)
    assert samples['coefs'].shape[:2] == (num_chains, num_samples)
    assert_allclose(np.mean(samples_flat['coefs'], 0), true_coefs, atol=0.21)


@pytest.mark.parametrize('kernel_cls', [HMC, NUTS])
@pytest.mark.parametrize('chain_method', [
    pytest.param('parallel', marks=pytest.mark.xfail(
        reason='jit+pmap does not work in CPU yet')),
    'sequential',
    'vectorized',
])
@pytest.mark.skipif('CI' in os.environ, reason="Compiling time the whole sampling process is slow.")
def test_chain_inside_jit(kernel_cls, chain_method):
    # NB: this feature is useful for consensus MC.
    # Caution: compiling time will be slow (~ 90s)
    if chain_method == 'parallel' and xla_bridge.device_count() == 1:
        pytest.skip('parallel method requires device_count greater than 1.')
    warmup_steps, num_samples = 100, 2000
    # Here are settings which is currently supported.
    rng_key = random.PRNGKey(2)
    step_size = 1.
    target_accept_prob = 0.8
    trajectory_length = 1.
    # Not supported yet:
    #   + adapt_step_size
    #   + adapt_mass_matrix
    #   + max_tree_depth
    #   + num_warmup
    #   + num_samples

    def model(data):
        concentration = np.array([1.0, 1.0, 1.0])
        p_latent = numpyro.sample('p_latent', dist.Dirichlet(concentration))
        numpyro.sample('obs', dist.Categorical(p_latent), obs=data)
        return p_latent

    @jit
    def get_samples(rng_key, data, step_size, trajectory_length, target_accept_prob):
        kernel = kernel_cls(model, step_size=step_size, trajectory_length=trajectory_length,
                            target_accept_prob=target_accept_prob)
        mcmc = MCMC(kernel, warmup_steps, num_samples, num_chains=2, chain_method=chain_method,
                    progress_bar=False)
        mcmc.run(rng_key, data)
        return mcmc.get_samples()

    true_probs = np.array([0.1, 0.6, 0.3])
    data = dist.Categorical(true_probs).sample(random.PRNGKey(1), (2000,))
    samples = get_samples(rng_key, data, step_size, trajectory_length, target_accept_prob)
    assert_allclose(np.mean(samples['p_latent'], 0), true_probs, atol=0.02)


@pytest.mark.parametrize('chain_method', [
    'sequential',
    'parallel',
    'vectorized',
])
@pytest.mark.parametrize('compile_args', [
    False,
    True
])
@pytest.mark.skipif('CI' in os.environ, reason="Compiling time the whole sampling process is slow.")
def test_chain_smoke(chain_method, compile_args):
    def model(data):
        concentration = np.array([1.0, 1.0, 1.0])
        p_latent = numpyro.sample('p_latent', dist.Dirichlet(concentration))
        numpyro.sample('obs', dist.Categorical(p_latent), obs=data)
        return p_latent

    data = dist.Categorical(np.array([0.1, 0.6, 0.3])).sample(random.PRNGKey(1), (2000,))
    kernel = NUTS(model)
    mcmc = MCMC(kernel, 2, 5, num_chains=2, chain_method=chain_method, jit_model_args=compile_args)
    mcmc.warmup(random.PRNGKey(0), data)
    mcmc.run(random.PRNGKey(1), data)


def test_extra_fields():
    def model():
        numpyro.sample('x', dist.Normal(0, 1), sample_shape=(5,))

    mcmc = MCMC(NUTS(model), 1000, 1000)
    mcmc.run(random.PRNGKey(0), extra_fields=('num_steps', 'adapt_state.step_size'))
    samples = mcmc.get_samples(group_by_chain=True)
    assert samples['x'].shape == (1, 1000, 5)
    stats = mcmc.get_extra_fields(group_by_chain=True)
    assert 'num_steps' in stats
    assert stats['num_steps'].shape == (1, 1000)
    assert 'adapt_state.step_size' in stats
    assert stats['adapt_state.step_size'].shape == (1, 1000)


@pytest.mark.parametrize('algo', ['HMC', 'NUTS'])
def test_functional_beta_bernoulli_x64(algo):
    warmup_steps, num_samples = 500, 20000

    def model(data):
        alpha = np.array([1.1, 1.1])
        beta = np.array([1.1, 1.1])
        p_latent = numpyro.sample('p_latent', dist.Beta(alpha, beta))
        numpyro.sample('obs', dist.Bernoulli(p_latent), obs=data)
        return p_latent

    true_probs = np.array([0.9, 0.1])
    data = dist.Bernoulli(true_probs).sample(random.PRNGKey(1), (1000, 2))
    init_params, potential_fn, constrain_fn = initialize_model(random.PRNGKey(2), model, model_args=(data,))
    init_kernel, sample_kernel = hmc(potential_fn, algo=algo)
    hmc_state = init_kernel(init_params,
                            trajectory_length=1.,
                            num_warmup=warmup_steps)
    samples = fori_collect(0, num_samples, sample_kernel, hmc_state,
                           transform=lambda x: constrain_fn(x.z))
    assert_allclose(np.mean(samples['p_latent'], 0), true_probs, atol=0.05)

    if 'JAX_ENABLE_X64' in os.environ:
        assert samples['p_latent'].dtype == np.float64


@pytest.mark.parametrize('algo', ['HMC', 'NUTS'])
@pytest.mark.parametrize('map_fn', [vmap, pmap])
@pytest.mark.skipif('XLA_FLAGS' not in os.environ, reason='without this mark, we have duplicated tests in Travis')
def test_functional_map(algo, map_fn):
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
        init_param, trajectory_length=9, num_warmup=warmup_steps, rng_key=rng_key))
    init_states = init_kernel_map(init_params, rng_keys)

    fori_collect_map = map_fn(lambda hmc_state: fori_collect(0, num_samples, sample_kernel, hmc_state,
                                                             transform=lambda x: x.z, progbar=False))
    chain_samples = fori_collect_map(init_states)

    assert_allclose(np.mean(chain_samples, axis=1), np.repeat(true_mean, 2), rtol=0.06)
    assert_allclose(np.std(chain_samples, axis=1), np.repeat(true_std, 2), rtol=0.06)


@pytest.mark.parametrize('jit_args', [False, True])
@pytest.mark.parametrize('shape', [50, 100])
def test_reuse_mcmc_run(jit_args, shape):
    y1 = onp.random.normal(3, 0.1, (100,))
    y2 = onp.random.normal(-3, 0.1, (shape,))

    def model(y_obs):
        mu = numpyro.sample('mu', dist.Normal(0., 1.))
        sigma = numpyro.sample("sigma", dist.HalfCauchy(3.))
        numpyro.sample("y", dist.Normal(mu, sigma), obs=y_obs)

    # Run MCMC on zero observations.
    kernel = NUTS(model)
    mcmc = MCMC(kernel, 300, 500, jit_model_args=jit_args)
    mcmc.run(random.PRNGKey(32), y1)

    # Re-run on new data - should be much faster.
    mcmc.run(random.PRNGKey(32), y2)
    assert_allclose(mcmc.get_samples()['mu'].mean(), -3., atol=0.1)


@pytest.mark.parametrize('jit_args', [False, True])
def test_model_with_multiple_exec_paths(jit_args):
    def model(a=None, b=None, z=None):
        int_term = numpyro.sample('a', dist.Normal(0., 0.2))
        x_term, y_term = 0., 0.
        if a is not None:
            x = numpyro.sample('x', dist.HalfNormal(0.5))
            x_term = a * x
        if b is not None:
            y = numpyro.sample('y', dist.HalfNormal(0.5))
            y_term = b * y
        sigma = numpyro.sample('sigma', dist.Exponential(1.))
        mu = int_term + x_term + y_term
        numpyro.sample('obs', dist.Normal(mu, sigma), obs=z)

    a = np.exp(onp.random.randn(10))
    b = np.exp(onp.random.randn(10))
    z = onp.random.randn(10)

    # Run MCMC on zero observations.
    kernel = NUTS(model)
    mcmc = MCMC(kernel, 20, 10, jit_model_args=jit_args)
    mcmc.run(random.PRNGKey(1), a, b=None, z=z)
    mcmc.run(random.PRNGKey(2), a=None, b=b, z=z)
    mcmc.run(random.PRNGKey(3), a=a, b=b, z=z)


@pytest.mark.parametrize('num_chains', [1, 2])
@pytest.mark.parametrize('chain_method', ['parallel', 'sequential', 'vectorized'])
@pytest.mark.parametrize('progress_bar', [True, False])
def test_compile_warmup_run(num_chains, chain_method, progress_bar):
    def model():
        numpyro.sample("x", dist.Normal(0, 1))

    if num_chains == 1 and chain_method in ['sequential', 'vectorized']:
        pytest.skip('duplicated test')
    if num_chains > 1 and chain_method == 'parallel':
        pytest.skip('duplicated test')

    rng_key = random.PRNGKey(0)
    num_samples = 10
    mcmc = MCMC(NUTS(model), 10, num_samples, num_chains,
                chain_method=chain_method, progress_bar=progress_bar)

    mcmc.run(rng_key)
    expected_samples = mcmc.get_samples()["x"]

    mcmc._compile(rng_key)
    # no delay after compiling
    mcmc.warmup(rng_key)
    mcmc.run(mcmc._warmup_state.rng_key)
    actual_samples = mcmc.get_samples()["x"]

    assert_allclose(actual_samples, expected_samples)

    # test for reproducible
    if num_chains > 1:
        mcmc = MCMC(NUTS(model), 10, num_samples, 1, progress_bar=progress_bar)
        rng_key = random.split(rng_key)[0]
        mcmc.run(rng_key)
        first_chain_samples = mcmc.get_samples()["x"]
        assert_allclose(actual_samples[:num_samples], first_chain_samples, atol=1e-5)


@pytest.mark.parametrize('dense_mass', [True, False])
def test_get_proposal_loc_and_scale(dense_mass):
    N = 10
    dim = 3
    samples = random.normal(random.PRNGKey(0), (N, dim))
    loc = np.mean(samples[:-1], 0)
    if dense_mass:
        scale = np.linalg.cholesky(np.cov(samples[:-1], rowvar=False, bias=True))
    else:
        scale = np.std(samples[:-1], 0)
    actual_loc, actual_scale = _get_proposal_loc_and_scale(samples[:-1], loc, scale, samples[-1])
    expected_loc, expected_scale = [], []
    for i in range(N - 1):
        samples_i = onp.delete(samples, i, axis=0)
        expected_loc.append(np.mean(samples_i, 0))
        if dense_mass:
            expected_scale.append(np.linalg.cholesky(np.cov(samples_i, rowvar=False, bias=True)))
        else:
            expected_scale.append(np.std(samples_i, 0))
    expected_loc = np.stack(expected_loc)
    expected_scale = np.stack(expected_scale)
    assert_allclose(actual_loc, expected_loc, rtol=1e-4)
    assert_allclose(actual_scale, expected_scale, atol=1e-6, rtol=0.05)


@pytest.mark.parametrize('shape', [(4,), (3, 2)])
@pytest.mark.parametrize('idx', [0, 1, 2])
def test_numpy_delete(shape, idx):
    x = random.normal(random.PRNGKey(0), shape)
    expected = onp.delete(x, idx, axis=0)
    actual = _numpy_delete(x, idx)
    assert_allclose(actual, expected)
