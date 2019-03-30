from numpy.testing import assert_allclose

import jax.numpy as np
from jax import lax, jit

from numpyro.mcmc import hmc_kernel


def test_normal():
    true_mean, true_std = 1., 2.
    warmup_steps, num_samples = 1000, 8000

    def potential_fn(z):
        return 0.5 * np.sum(((z[0] - true_mean) / true_std) ** 2)

    def kinetic_fn(r, m_inv):
        return 0.5 * np.sum(m_inv * r[0] ** 2)

    init_kernel, sample_kernel = hmc_kernel(potential_fn, kinetic_fn)
    init_samples = [np.array([0.])]
    hmc_state = init_kernel(init_samples,
                            num_warmup_steps=warmup_steps)
    sample_kernel = jit(sample_kernel)
    hmc_states = lax.scan(sample_kernel, hmc_state, np.arange(num_samples))
    zs = hmc_states.z[0]
    assert_allclose(np.mean(zs), true_mean, rtol=0.05)
    assert_allclose(np.std(zs), true_std, rtol=0.05)
