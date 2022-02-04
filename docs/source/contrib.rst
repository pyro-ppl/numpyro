Contributed Code
================

Nested Sampling
~~~~~~~~~~~~~~~

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

``numpyro.contrib.einstein`` is a framework for particle-based inference using the ELBO-within-Stein algorithm.
The framework works on Stein mixtures, a restricted mixture of guide programs parameterized by Stein particles.
Similarly to how SVGD works, Stein mixtures can approximate model posteriors by moving the Stein particles according
to the Stein forces. Because the Stein particles parameterize a guide, they capture a neighborhood rather than a
single point. This property means Stein mixtures significantly reduce the number of particles needed to represent
high dimensional models.

``numpyro.contrib.einstein`` mimics the interface from ``numpyro.infer.svi``, so trying SteinVI requires minimal
change to the code for existing models inferred with SVI. For primary usage, see the
`Bayesian neural network example <https://num.pyro.ai/en/latest/examples/stein_bnn.html>`_.

The framework currently supports several kernels, including:

- `RBFKernel`
- `LinearKernel`
- `RandomFeatureKernel`
- `MixtureKernel`
- `PrecondMatrixKernel`
- `HessianPrecondMatrix`
- `GraphicalKernel`

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
.. autoclass:: numpyro.contrib.einstein.kernels.RBFKernel
.. autoclass:: numpyro.contrib.einstein.kernels.LinearKernel
.. autoclass:: numpyro.contrib.einstein.kernels.RandomFeatureKernel
.. autoclass:: numpyro.contrib.einstein.kernels.MixtureKernel
.. autoclass:: numpyro.contrib.einstein.kernels.PrecondMatrixKernel
.. autoclass:: numpyro.contrib.einstein.kernels.GraphicalKernel

