Markov Chain Monte Carlo (MCMC)
===============================

We provide a high-level overview of the MCMC algorithms in NumPyro:

* `NUTS <https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.hmc.NUTS>`_, which is an adaptive variant of `HMC <https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.hmc.HMC>`_, is probably the most commonly used MCMC algorithm in NumPyro. Note that NUTS and HMC are not directly applicable to models with discrete latent variables, but in cases where the discrete variables have finite support and summing them out (i.e. enumeration) is tractable, NumPyro will automatically sum out discrete latent variables and perform NUTS/HMC on the remaining continuous latent variables. As discussed above, model `reparameterization <https://num.pyro.ai/en/latest/reparam.html#module-numpyro.infer.reparam>`_ may be important in some cases to get good performance. Note that, generally speaking, we expect inference to be harder as the dimension of the latent space increases. See the `bad geometry <https://num.pyro.ai/en/latest/tutorials/bad_posterior_geometry.html>`_ tutorial for additional tips and tricks.
* `MixedHMC <http://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.mixed_hmc.MixedHMC>`_ can be an effective inference strategy for models that contain both continuous and discrete latent variables.
* `HMCECS <http://num.pyro.ai/en/latest/mcmc.html#hmcecs>`_ can be an effective inference strategy for models with a large number of data points. It is applicable to models with continuous latent variables. See `this example <https://num.pyro.ai/en/latest/examples/covtype.html>`_ for detailed usage.
* `BarkerMH <https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.barker.BarkerMH>`_ is a gradient-based MCMC method that may be competitive with HMC and NUTS for some models. It is applicable to models with continuous latent variables.
* `HMCGibbs <https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.hmc_gibbs.HMCGibbs>`_ combines HMC/NUTS steps with custom Gibbs updates. Gibbs updates must be specified by the user.
* `DiscreteHMCGibbs <https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.hmc_gibbs.DiscreteHMCGibbs>`_ combines HMC/NUTS steps with Gibbs updates for discrete latent variables. The corresponding Gibbs updates are computed automatically.
* `SA <https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.sa.SA>`_ is the only MCMC method in NumPyro that does not leverage gradients. It is only applicable to models with continuous latent variables. It is expected to perform best for models whose latent dimension is low to moderate. It may be a good choice for models with non-differentiable log densities. Note that SA generally requires a *very* large number of samples, as mixing tends to be slow. On the plus side individual steps can be fast.
* `NestedSampler <https://num.pyro.ai/en/latest/contrib.html#nested-sampling>`_ offers a wrapper for `jaxns <https://github.com/Joshuaalbert/jaxns>`_. See `here <https://github.com/pyro-ppl/numpyro/blob/master/examples/gaussian_shells.py>`_ for an example.

Like HMC/NUTS, all remaining MCMC algorithms support enumeration over discrete latent variables if possible (see `restrictions <https://pyro.ai/examples/enumeration.html#Restriction-1:-conditional-independence>`_). Enumerated sites need to be marked with `infer={'enumerate': 'parallel'}` like in the `annotation example <https://num.pyro.ai/en/stable/examples/annotation.html>`_.

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

Thin wrappers around TensorFlow Probability (TFP) MCMC kernels. For details on the TFP MCMC kernel interface,
see `its TransitionKernel docs <https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/mcmc/TransitionKernel>`_.

.. automodule:: numpyro.contrib.tfp.mcmc


MCMC Utilities
--------------

.. autofunction:: numpyro.infer.util.initialize_model

.. autofunction:: numpyro.util.fori_collect

.. autofunction:: numpyro.infer.hmc_util.consensus

.. autofunction:: numpyro.infer.hmc_util.parametric

.. autofunction:: numpyro.infer.hmc_util.parametric_draws
