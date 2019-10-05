Markov Chain Monte Carlo (MCMC)
===============================

Hamiltonian Monte Carlo
-----------------------

.. autoclass:: numpyro.infer.mcmc.MCMC
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. autoclass:: numpyro.infer.mcmc.HMC
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. autoclass:: numpyro.infer.mcmc.NUTS
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. autofunction:: numpyro.infer.mcmc.hmc

.. autofunction:: numpyro.infer.mcmc.hmc.init_kernel

.. autofunction:: numpyro.infer.mcmc.hmc.sample_kernel

.. autodata:: numpyro.infer.mcmc.HMCState


MCMC Utilities
--------------

.. autofunction:: numpyro.hmc_util.initialize_model

.. autofunction:: numpyro.util.fori_collect

.. autofunction:: numpyro.hmc_util.consensus

.. autofunction:: numpyro.hmc_util.parametric

.. autofunction:: numpyro.hmc_util.parametric_draws
