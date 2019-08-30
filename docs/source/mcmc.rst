Markov Chain Monte Carlo (MCMC)
===============================

Hamiltonian Monte Carlo
-----------------------

.. autoclass:: numpyro.mcmc.MCMC
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. autoclass:: numpyro.mcmc.HMC
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. autoclass:: numpyro.mcmc.NUTS
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. autofunction:: numpyro.mcmc.hmc

.. autofunction:: numpyro.mcmc.hmc.init_kernel

.. autofunction:: numpyro.mcmc.hmc.sample_kernel

.. autodata:: numpyro.mcmc.HMCState


MCMC Utilities
--------------

.. autofunction:: numpyro.hmc_util.initialize_model

.. autofunction:: numpyro.util.fori_collect

.. autofunction:: numpyro.diagnostics.summary

.. autofunction:: numpyro.hmc_util.consensus

.. autofunction:: numpyro.hmc_util.parametric

.. autofunction:: numpyro.hmc_util.parametric_draws
