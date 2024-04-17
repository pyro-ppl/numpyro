Contributed Code
================

Nested Sampling
~~~~~~~~~~~~~~~
Nested Sampling is a non-MCMC approach that works for arbitrary probability models, and is particularly well suited to complex posteriors:

* `NestedSampler <https://num.pyro.ai/en/latest/contrib.html#nested-sampling>`_ offers a wrapper for `jaxns <https://github.com/Joshuaalbert/jaxns>`_. See `JAXNS's readthedocs <https://jaxns.readthedocs.io/en/latest/>`_ for examples and `Nested Sampling for Gaussian Shells <https://num.pyro.ai/en/stable/examples/gaussian_shells.html>`_ example for how to apply the sampler on numpyro models. Can handle arbitrary models, including ones with discrete RVs, and non-invertible transformations.

.. autoclass:: numpyro.contrib.nested_sampling.NestedSampler
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource


Stein Variational Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Stein Variational Inference (SteinVI) is a family of VI techniques for approximate Bayesian inference based on
Stein’s method (see [1] for an overview). It is gaining popularity as it combines
the scalability of traditional VI with the flexibility of non-parametric particle-based methods.

Stein variational gradient descent (SVGD) [2] is a recent SteinVI technique which uses iteratively moves a set of
particles :math:`\{z_i\}_{i=1}^N` to approximate a distribution p(z).
SVGD is well suited for capturing correlations between latent variables as a particle-based method.
The technique preserves the scalability of traditional VI approaches while offering the flexibility and modeling scope
of methods such as Markov chain Monte Carlo (MCMC). SVGD is good at capturing multi-modality [3][4].

``numpyro.contrib.einstein`` is a framework for particle-based inference using the Stein mixture algorithm.
The framework works on Stein mixtures, a restricted mixture of guide programs parameterized by Stein particles.
Similarly to how SVGD works, Stein mixtures can approximate model posteriors by moving the Stein particles according
to the Stein forces. Because the Stein particles parameterize a guide, they capture a neighborhood rather than a
single point.

``numpyro.contrib.einstein`` mimics the interface from ``numpyro.infer.svi``, so trying SteinVI requires minimal
change to the code for existing models inferred with SVI. For primary usage, see the
`Bayesian neural network example <https://num.pyro.ai/en/latest/examples/stein_bnn.html>`_.

The framework currently supports several kernels, including:

- `RBFKernel`
- `LinearKernel`
- `RandomFeatureKernel`
- `MixtureKernel`
- `GraphicalKernel`
- `ProbabilityProductKernel`

For example, usage see:

- The `Bayesian neural network example <https://num.pyro.ai/en/latest/examples/stein_bnn.html>`_

**References**

1. *Stein's Method Meets Statistics: A Review of Some Recent Developments* (2021)
Andreas Anastasiou, Alessandro Barp, François-Xavier Briol, Bruno Ebner,
Robert E. Gaunt, Fatemeh Ghaderinezhad, Jackson Gorham, Arthur Gretton,
Christophe Ley, Qiang Liu, Lester Mackey, Chris. J. Oates, Gesine Reinert,
Yvik Swan. https://arxiv.org/abs/2105.03481

2. *Stein variational gradient descent: A general-purpose Bayesian inference algorithm* (2016)
Qiang Liu, Dilin Wang. NeurIPS

3. *Nonlinear Stein Variational Gradient Descent for Learning Diversified Mixture Models* (2019)
Dilin Wang, Qiang Liu. PMLR

SteinVI Interface
-----------------
.. autoclass:: numpyro.contrib.einstein.steinvi.SteinVI

SteinVI Kernels
---------------
.. autoclass:: numpyro.contrib.einstein.stein_kernels.RBFKernel
.. autoclass:: numpyro.contrib.einstein.stein_kernels.LinearKernel
.. autoclass:: numpyro.contrib.einstein.stein_kernels.RandomFeatureKernel
.. autoclass:: numpyro.contrib.einstein.stein_kernels.MixtureKernel
.. autoclass:: numpyro.contrib.einstein.stein_kernels.GraphicalKernel
.. autoclass:: numpyro.contrib.einstein.stein_kernels.ProbabilityProductKernel


Stochastic Support
~~~~~~~~~~~~~~~~~~

.. autoclass:: numpyro.contrib.stochastic_support.dcc.StochasticSupportInference
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. autoclass:: numpyro.contrib.stochastic_support.dcc.DCC
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. autoclass:: numpyro.contrib.stochastic_support.sdvi.SDVI
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
