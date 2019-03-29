import jax
import pytest

import jax.numpy as np
from jax import jit, lax

from numpyro.mcmc import hmc_kernel


@pytest.mark.parametrize('adapt_mass_matrix', [False, True])
@jax.disable_jit()
def test_normal(adapt_mass_matrix):
    def potential_fn(z):
        return - 0.5 * np.sum((z[0] - 1.) ** 2 / 4.)

    def kinetic_fn(r, m_inv):
        return 0.5 * np.sum(m_inv * r[0] ** 2)

    init_kernel, sample_kernel = hmc_kernel(potential_fn, kinetic_fn)
    init_samples = [np.array([0.])]
    hmc_state = init_kernel(init_samples,
                            num_warmup_steps=50,
                            adapt_mass_matrix=adapt_mass_matrix)

    sample_kernel = sample_kernel
    hmc_states = lax.scan(sample_kernel, hmc_state, np.arange(50))
    print(hmc_states)
