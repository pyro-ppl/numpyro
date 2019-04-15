import pytest
from numpy.testing import assert_allclose

import jax.numpy as np
from jax import jit, random

import numpyro.distributions as dist
from numpyro.handlers import sample
from numpyro.hmc_util import initialize_model
from numpyro.mcmc import hmc
from numpyro.util import tscan


# TODO: add test for diag_mass=False
@pytest.mark.parametrize('algo', ['HMC', 'NUTS'])
def test_unnormalized_normal(algo):
    true_mean, true_std = 1., 2.
    warmup_steps, num_samples = 1000, 8000

    def potential_fn(z):
        return 0.5 * np.sum(((z - true_mean) / true_std) ** 2)

    init_kernel, sample_kernel = hmc(potential_fn, algo=algo)
    init_samples = np.array(0.)
    hmc_state = init_kernel(init_samples,
                            num_steps=10,
                            num_warmup_steps=warmup_steps)
    hmc_states = tscan(lambda state, i: sample_kernel(state), hmc_state, np.arange(num_samples),
                       transform=lambda x: x.z)
    assert_allclose(np.mean(hmc_states), true_mean, rtol=0.05)
    assert_allclose(np.std(hmc_states), true_std, rtol=0.05)


@pytest.mark.parametrize('algo', ['HMC', 'NUTS'])
def test_logistic_regression(algo):
    N, dim = 3000, 3
    warmup_steps, num_samples = 1000, 8000
    data = random.normal(random.PRNGKey(0), (N, dim))
    true_coefs = np.arange(1., dim + 1.)
    logits = np.sum(true_coefs * data, axis=-1)
    labels = dist.bernoulli(logits, is_logits=True).rvs(random_state=random.PRNGKey(1))

    def model(labels):
        coefs = sample('coefs', dist.norm(np.zeros(dim), np.ones(dim)))
        logits = np.sum(coefs * data, axis=-1)
        return sample('obs', dist.bernoulli(logits, is_logits=True), obs=labels)

    init_params, potential_fn, transforms = initialize_model(random.PRNGKey(2), model, (labels,), {})
    init_kernel, sample_kernel = hmc(potential_fn, algo=algo)
    hmc_state = init_kernel(init_params,
                            step_size=0.1,
                            num_steps=15,
                            num_warmup_steps=warmup_steps)
    sample_kernel = jit(sample_kernel)
    hmc_states = tscan(lambda state, i: sample_kernel(state), hmc_state, np.arange(num_samples),
                       transform=lambda x: {k: transforms[k](v) for k, v in x.z.items()})
    assert_allclose(np.mean(hmc_states['coefs'], 0), true_coefs, atol=0.2)
