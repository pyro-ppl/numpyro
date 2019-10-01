import os

import numpy as onp
from numpy.testing import assert_allclose
import pytest

from jax import jit, random
from jax.lib import xla_bridge
import jax.numpy as np
from jax.scipy.special import logit

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.mcmc import HMC, MCMC, NUTS


@pytest.mark.parametrize('kernel_cls', [HMC, NUTS])
@pytest.mark.parametrize('dense_mass', [False, True])
def test_unnormalized_normal(kernel_cls, dense_mass):
    true_mean, true_std = 1., 2.
    warmup_steps, num_samples = 1000, 8000

    def potential_fn(z):
        return 0.5 * np.sum(((z - true_mean) / true_std) ** 2)

    init_params = np.array(0.)
    kernel = kernel_cls(potential_fn=potential_fn, trajectory_length=9, dense_mass=dense_mass)
    mcmc = MCMC(kernel, warmup_steps, num_samples)
    mcmc.run(random.PRNGKey(0), init_params=init_params)
    hmc_states = mcmc.get_samples()
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
    kernel = NUTS(potential_fn=potential_fn, dense_mass=True)
    mcmc = MCMC(kernel, warmup_steps, num_samples)
    mcmc.run(random.PRNGKey(0), init_params=init_params)
    samples = mcmc.get_samples()
    assert_allclose(np.mean(samples), true_mean, atol=0.02)
    assert onp.sum(onp.abs(onp.cov(samples.T) - true_cov)) / D**2 < 0.02


@pytest.mark.parametrize('kernel_cls', [HMC, NUTS])
def test_logistic_regression(kernel_cls):
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

    kernel = kernel_cls(model=model, trajectory_length=10)
    mcmc = MCMC(kernel, warmup_steps, num_samples)
    mcmc.run(random.PRNGKey(2), labels)
    samples = mcmc.get_samples()
    assert_allclose(np.mean(samples['coefs'], 0), true_coefs, atol=0.22)

    if 'JAX_ENABLE_x64' in os.environ:
        assert samples['coefs'].dtype == np.float64


def test_uniform_normal():
    true_coef = 0.9
    num_warmup, num_samples = 1000, 1000

    def model(data):
        alpha = numpyro.sample('alpha', dist.Uniform(0, 1))
        loc = numpyro.sample('loc', dist.Uniform(0, alpha))
        numpyro.sample('obs', dist.Normal(loc, 0.1), obs=data)

    data = true_coef + random.normal(random.PRNGKey(0), (1000,))
    kernel = NUTS(model=model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(random.PRNGKey(2), data, collect_warmup=True,
             collect_fields=('z', 'num_steps', 'adapt_state.step_size'))
    samples = mcmc.get_samples()
    assert len(samples[0]['loc']) == num_warmup + num_samples
    assert_allclose(np.mean(samples[0]['loc'], 0), true_coef, atol=0.05)


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


@pytest.mark.parametrize('kernel_cls', [HMC, NUTS])
def test_beta_bernoulli(kernel_cls):
    warmup_steps, num_samples = 500, 20000

    def model(data):
        alpha = np.array([1.1, 1.1])
        beta = np.array([1.1, 1.1])
        p_latent = numpyro.sample('p_latent', dist.Beta(alpha, beta))
        numpyro.sample('obs', dist.Bernoulli(p_latent), obs=data)
        return p_latent

    true_probs = np.array([0.9, 0.1])
    data = dist.Bernoulli(true_probs).sample(random.PRNGKey(1), (1000, 2))
    kernel = kernel_cls(model=model, trajectory_length=1.)
    mcmc = MCMC(kernel, num_warmup=warmup_steps, num_samples=num_samples, progress_bar=False)
    mcmc.run(random.PRNGKey(2), data)
    samples = mcmc.get_samples()
    assert_allclose(np.mean(samples['p_latent'], 0), true_probs, atol=0.05)

    if 'JAX_ENABLE_x64' in os.environ:
        assert samples['p_latent'].dtype == np.float64


@pytest.mark.parametrize('kernel_cls', [HMC, NUTS])
@pytest.mark.parametrize('dense_mass', [False, True])
def test_dirichlet_categorical(kernel_cls, dense_mass):
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

    if 'JAX_ENABLE_x64' in os.environ:
        assert samples['p_latent'].dtype == np.float64


@pytest.mark.xfail(reason='TODO: Fix this flaky test.')
def test_change_point():
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

    if 'JAX_ENABLE_x64' in os.environ:
        assert samples['lambda1'].dtype == np.float64
        assert samples['lambda2'].dtype == np.float64
        assert samples['tau'].dtype == np.float64


@pytest.mark.parametrize('with_logits', ['True', 'False'])
def test_binomial_stable(with_logits):
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

    if 'JAX_ENABLE_x64' in os.environ:
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
    mcmc.run(random.PRNGKey(2), data)
    samples = mcmc.get_samples()
    assert_allclose(np.mean(samples['mean']), true_mean, rtol=0.05)
    assert_allclose(np.mean(samples['std']), true_std, rtol=0.05)


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
    mcmc.run(random.PRNGKey(1), data, collect_fields=('z', 'diverging'), collect_warmup=True)
    num_divergences = mcmc.get_samples()[1].sum()
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
    samples = mcmc.get_samples()
    assert samples['coefs'].shape[0] == num_chains * num_samples
    assert_allclose(np.mean(samples['coefs'], 0), true_coefs, atol=0.21)


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
    rng = random.PRNGKey(2)
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
    def get_samples(rng, data, step_size, trajectory_length, target_accept_prob):
        kernel = kernel_cls(model, step_size=step_size, trajectory_length=trajectory_length,
                            target_accept_prob=target_accept_prob)
        mcmc = MCMC(kernel, warmup_steps, num_samples, num_chains=2, chain_method=chain_method,
                    progress_bar=False)
        mcmc.run(rng, data)
        return mcmc.get_samples()

    true_probs = np.array([0.1, 0.6, 0.3])
    data = dist.Categorical(true_probs).sample(random.PRNGKey(1), (2000,))
    samples = get_samples(rng, data, step_size, trajectory_length, target_accept_prob)
    assert_allclose(np.mean(samples['p_latent'], 0), true_probs, atol=0.02)
