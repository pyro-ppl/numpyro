Markov Chain Monte Carlo (MCMC)
===============================

Hamiltonian Monte Carlo
-----------------------

.. autoclass:: numpyro.infer.mcmc.MCMC
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. autoclass:: numpyro.infer.hmc.HMC
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. autoclass:: numpyro.infer.hmc.NUTS
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. autoclass:: numpyro.infer.sa.SA
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. autofunction:: numpyro.infer.hmc.hmc

.. autofunction:: numpyro.infer.hmc.hmc.init_kernel

.. autofunction:: numpyro.infer.hmc.hmc.sample_kernel

.. autodata:: numpyro.infer.hmc.HMCState

.. autodata:: numpyro.infer.sa.SAState


MCMC Utilities
--------------

.. autofunction:: numpyro.infer.util.initialize_model

.. autofunction:: numpyro.util.fori_collect

.. autofunction:: numpyro.infer.hmc_util.consensus

.. autofunction:: numpyro.infer.hmc_util.parametric

.. autofunction:: numpyro.infer.hmc_util.parametric_draws
