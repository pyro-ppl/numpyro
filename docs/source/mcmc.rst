Markov Chain Monte Carlo (MCMC)
===============================

.. autoclass:: numpyro.infer.mcmc.MCMC
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

MCMC Kernels
------------

MCMCKernel
^^^^^^^^^^
.. autoclass:: numpyro.infer.mcmc.MCMCKernel
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

BarkerMH
^^^^^^^^
.. autoclass:: numpyro.infer.barker.BarkerMH
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

HMC
^^^
.. autoclass:: numpyro.infer.hmc.HMC
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

NUTS
^^^^
.. autoclass:: numpyro.infer.hmc.NUTS
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

HMCGibbs
^^^^^^^^
.. autoclass:: numpyro.infer.hmc_gibbs.HMCGibbs
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

DiscreteHMCGibbs
^^^^^^^^^^^^^^^^
.. autoclass:: numpyro.infer.hmc_gibbs.DiscreteHMCGibbs
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

MixedHMC
^^^^^^^^
.. autoclass:: numpyro.infer.mixed_hmc.MixedHMC
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

HMCECS
^^^^^^
.. autoclass:: numpyro.infer.hmc_gibbs.HMCECS
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

SA
^^
.. autoclass:: numpyro.infer.sa.SA
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. autofunction:: numpyro.infer.hmc.hmc

.. autofunction:: numpyro.infer.hmc.hmc.init_kernel

.. autofunction:: numpyro.infer.hmc.hmc.sample_kernel

.. autofunction:: numpyro.infer.hmc_gibbs.taylor_proxy

.. autodata:: numpyro.infer.barker.BarkerMHState

.. autodata:: numpyro.infer.hmc.HMCState

.. autodata:: numpyro.infer.hmc_gibbs.HMCGibbsState

.. autodata:: numpyro.infer.sa.SAState


TensorFlow Kernels
------------------

Thin wrappers around TensorFlow Probability (TFP) distributions. For details on the TFP distribution interface,
see `its TransitionKernel docs <https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/mcmc/TransitionKernel>`_.

.. automodule:: numpyro.contrib.tfp.mcmc


MCMC Utilities
--------------

.. autofunction:: numpyro.infer.util.initialize_model

.. autofunction:: numpyro.util.fori_collect

.. autofunction:: numpyro.infer.hmc_util.consensus

.. autofunction:: numpyro.infer.hmc_util.parametric

.. autofunction:: numpyro.infer.hmc_util.parametric_draws
