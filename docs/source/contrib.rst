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
Stein variational inference (SteinVI) is a family of VI techniques for approximate Bayesian inference based on
Stein’s method (see [1] for an overview). It is gaining popularity as it combines
the scalability of traditional VI with the flexibility of non-parametric particle-based methods.

Stein variational gradient descent (SVGD) [2] is a recent SteinVI technique which uses iteratively moves a set of
particles :math:`\{z_i\}_{i=1}^N` to approximate a distribution p(z).
SVGD is well suited for capturing correlations between latent variables as a particle-based method.
The technique preserves the scalability of traditional VI approaches while offering the flexibility and modeling scope
of methods such as Markov chain Monte Carlo (MCMC). SVGD is good at capturing multi-modality [3].

``numpyro.contrib.einstein`` is a framework for particle-based inference using the Stein mixture inference algorithm [4].
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
- `RadialGaussNewtonKernel`


SteinVI based examples include:

- The `Bayesian neural network example <https://num.pyro.ai/en/latest/examples/stein_bnn.html>`_.
- The `deep Markov example <https://num.pyro.ai/en/latest/examples/stein_dmm.html>`_.

**References**

1. *Stein's Method Meets Statistics: A Review of Some Recent Developments.* 2021.
    Andreas Anastasiou, Alessandro Barp, François-Xavier Briol, Bruno Ebner,
    Robert E. Gaunt, Fatemeh Ghaderinezhad, Jackson Gorham, Arthur Gretton,
    Christophe Ley, Qiang Liu, Lester Mackey, Chris. J. Oates, Gesine Reinert,
    Yvik Swan.

2. *Stein Variational Gradient Descent: A General-Purpose Bayesian Inference Algorithm.* 2016.
    Qiang Liu, Dilin Wang. NeurIPS

3. *Nonlinear Stein Variational Gradient Descent for Learning Diversified Mixture Models.* 2019.
    Dilin Wang, Qiang Liu. PMLR

4. *ELBOing Stein: Variational Bayes with Stein Mixture Inference.* 2024.
    Ola Rønning, Eric Nalisnick, Christophe Ley, Padhraic Smyth, and Thomas Hamelryck. arXiv:2410.22948.

SteinVI Interface
-----------------
.. autoclass:: numpyro.contrib.einstein.steinvi.SteinVI
.. autoclass:: numpyro.contrib.einstein.steinvi.SVGD
.. autoclass:: numpyro.contrib.einstein.steinvi.ASVGD

SteinVI Kernels
---------------
.. autoclass:: numpyro.contrib.einstein.stein_kernels.RBFKernel
.. autoclass:: numpyro.contrib.einstein.stein_kernels.LinearKernel
.. autoclass:: numpyro.contrib.einstein.stein_kernels.RandomFeatureKernel
.. autoclass:: numpyro.contrib.einstein.stein_kernels.MixtureKernel
.. autoclass:: numpyro.contrib.einstein.stein_kernels.GraphicalKernel
.. autoclass:: numpyro.contrib.einstein.stein_kernels.ProbabilityProductKernel
.. autoclass:: numpyro.contrib.einstein.stein_kernels.RadialGaussNewtonKernel

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


Hilbert Space Gaussian Processes Approximation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module contains helper functions for use in the Hilbert Space Gaussian Process (HSGP) approximation method
described in [1] and [2].

.. warning::
    This module is experimental.

**Why do we need an approximation?** 

Gaussian processes do not scale well with the number of data points. Recall we had to invert the kernel matrix!
The computational complexity of the Gaussian process model is :math:`\mathcal{O}(n^3)`, where :math:`n` is the number of data
points. The HSGP approximation method is a way to reduce the computational complexity of the Gaussian process model
to :math:`\mathcal{O}(mn + m)`, where :math:`m` is the number of basis functions used in the approximation.

**Approximation Strategy Steps:**

We strongly recommend reading [1] and [2] for a detailed explanation of the approximation method. In [3] you can find
a practical approach using NumPyro and PyMC.

Here we provide the main steps and ingredients of the approximation method:

    1. Each stationary kernel :math:`k` has an associated spectral density :math:`S(\omega)`. There are closed formulas for the most common kernels. These formulas depend on the hyperparameters of the kernel (e.g. amplitudes and length scales).

    2. We can approximate the spectral density :math:`S(\omega)` as a polynomial series in :math:`||\omega||`. We call :math:`\omega` the frequency.

    3. We can interpret these polynomial terms as "powers" of the Laplacian operator. The key observation is that the Fourier transform of the Laplacian operator is :math:`||\omega||^2`.

    4. Next, we impose Dirichlet boundary conditions on the Laplacian operator which makes it self-adjoint and with discrete spectrum.

    5. We identify the expansion in (2) with the sum of powers of the Laplacian operator in the eigenbasis of (4).

Let :math:`m^\star = \prod_{d=1}^D m_d` be the total number of terms of the approximation, where :math:`m_d` is the number of basis functions used in the approximation for the :math:`d`-th dimension. Then, the approximation formula, in the non-centered parameterization, is:

.. math::

    f(x) \approx \sum_{j = 1}^{m^\star} 
    \overbrace{\color{red}{\left(S(\sqrt{\boldsymbol{\lambda}_j})\right)^{1/2}}}^{\text{all hyperparameters are here!}} 
    \times
    \underbrace{\color{blue}{\phi_{j}(\boldsymbol{x})}}_{\text{easy to compute!}}
    \times
    \overbrace{\color{green}{\beta_{j}}}^{\sim \: \text{Normal}(0,1)}

where :math:`\boldsymbol{x}` is a :math:`D` vector of inputs, :math:`\boldsymbol{\lambda}_j` are the eigenvalues of the Laplacian operator, :math:`\phi_{j}(\boldsymbol{x})` are the eigenfunctions of the
Laplacian operator, and :math:`\beta_{j}` are the coefficients of the expansion (see Eq. (8) in [2]). We expect this
to be a good approximation for a finite number of :math:`m^\star` terms in the series as long as the inputs values :math:`x`
are not too close to the boundaries :math:`-L_d` and :math:`L_d`.

.. note::
    Even though the periodic kernel is not stationary, one can still adapt and find a similar approximation formula. However, these kernels are not supported for multidimensional inputs.
    See Appendix B in [2] for more details.

**Example:**

Here is an example of how to use the HSGP approximation method with NumPyro. We will use the squared exponential kernel.
Other kernels can be used similarly.

    .. code-block:: python

        >>> from jax import random
        >>> import jax.numpy as jnp

        >>> import numpyro
        >>> from numpyro.contrib.hsgp.approximation import hsgp_squared_exponential
        >>> import numpyro.distributions as dist
        >>> from numpyro.infer import MCMC, NUTS


        >>> def generate_synthetic_data(rng_key, start, stop: float, num, scale):
        ...     """Generate synthetic data."""
        ...     x = jnp.linspace(start=start, stop=stop, num=num)
        ...     y = jnp.sin(4 * jnp.pi * x) + jnp.sin(7 * jnp.pi * x)
        ...     y_obs = y + scale * random.normal(rng_key, shape=(num,))
        ...     return x, y_obs


        >>> rng_key = random.PRNGKey(seed=42)
        >>> rng_key, rng_subkey = random.split(rng_key)
        >>> x, y_obs = generate_synthetic_data(
        ...     rng_key=rng_subkey, start=0, stop=1, num=80, scale=0.3
        >>> )


        >>> def model(x, ell, m, non_centered, y=None):
        ...     # --- Priors ---
        ...     alpha = numpyro.sample("alpha", dist.InverseGamma(concentration=12, rate=10))
        ...     length = numpyro.sample("length", dist.InverseGamma(concentration=6, rate=1))
        ...     noise = numpyro.sample("noise", dist.InverseGamma(concentration=12, rate=10))
        ...     # --- Parametrization ---
        ...     f = hsgp_squared_exponential(
        ...         x=x, alpha=alpha, length=length, ell=ell, m=m, non_centered=non_centered
        ...     )
        ...     # --- Likelihood ---
        ...     with numpyro.plate("data", x.shape[0]):
        ...         numpyro.sample("likelihood", dist.Normal(loc=f, scale=noise), obs=y)


        >>> sampler = NUTS(model)
        >>> mcmc = MCMC(sampler=sampler, num_warmup=500, num_samples=1_000, num_chains=2)

        >>> rng_key, rng_subkey = random.split(rng_key)

        >>> ell = 1.3
        >>> m = 20
        >>> non_centered = True

        >>> mcmc.run(rng_subkey, x, ell, m, non_centered, y_obs)

        >>> mcmc.print_summary()

                      mean       std    median      5.0%     95.0%     n_eff     r_hat
           alpha      1.24      0.34      1.18      0.72      1.74   1804.01      1.00
         beta[0]     -0.10      0.66     -0.10     -1.24      0.92   1819.91      1.00
         beta[1]      0.00      0.71     -0.01     -1.09      1.26   1872.82      1.00
         beta[2]     -0.05      0.69     -0.03     -1.09      1.16   2105.88      1.00
         beta[3]      0.25      0.74      0.26     -0.98      1.42   2281.30      1.00
         beta[4]     -0.17      0.69     -0.17     -1.21      1.00   2551.39      1.00
         beta[5]      0.09      0.75      0.10     -1.13      1.30   3232.13      1.00
         beta[6]     -0.49      0.75     -0.49     -1.65      0.82   3042.31      1.00
         beta[7]      0.42      0.75      0.44     -0.78      1.65   2885.42      1.00
         beta[8]      0.69      0.71      0.71     -0.48      1.82   2811.68      1.00
         beta[9]     -1.43      0.75     -1.40     -2.63     -0.21   2858.68      1.00
        beta[10]      0.33      0.71      0.33     -0.77      1.51   2198.65      1.00
        beta[11]      1.09      0.73      1.11     -0.23      2.18   2765.99      1.00
        beta[12]     -0.91      0.72     -0.91     -2.06      0.31   2586.53      1.00
        beta[13]      0.05      0.70      0.04     -1.16      1.12   2569.59      1.00
        beta[14]     -0.44      0.71     -0.44     -1.58      0.73   2626.09      1.00
        beta[15]      0.69      0.73      0.70     -0.45      1.88   2626.32      1.00
        beta[16]      0.98      0.74      0.98     -0.15      2.28   2282.86      1.00
        beta[17]     -2.54      0.77     -2.52     -3.82     -1.29   3347.56      1.00
        beta[18]      1.35      0.66      1.35      0.30      2.46   2638.17      1.00
        beta[19]      1.10      0.54      1.09      0.25      2.01   2428.37      1.00
          length      0.07      0.01      0.07      0.06      0.09   2321.67      1.00
           noise      0.33      0.03      0.33      0.29      0.38   2472.83      1.00

        Number of divergences: 0


.. note::
    Additional examples with code can be found in [3], [4] and [5].

**References:**

    1. Solin, A., Särkkä, S. Hilbert space methods for reduced-rank Gaussian process regression.
       Stat Comput 30, 419-446 (2020).

    2. Riutort-Mayol, G., Bürkner, PC., Andersen, M.R. et al. Practical Hilbert space
       approximate Bayesian Gaussian processes for probabilistic programming. Stat Comput 33, 17 (2023).
    
    3. `Orduz, J., A Conceptual and Practical Introduction to Hilbert Space GPs Approximation Methods <https://juanitorduz.github.io/hsgp_intro>`_.
    
    4. `Example: Hilbert space approximation for Gaussian processes <https://num.pyro.ai/en/stable/examples/hsgp.html>`_.
    
    5. `Gelman, Vehtari, Simpson, et al., Bayesian workflow book - Birthdays <https://avehtari.github.io/casestudies/Birthdays/birthdays.html>`_.

.. note::
    The code of this module is based on the code of the example
    `Example: Hilbert space approximation for Gaussian processes <https://num.pyro.ai/en/stable/examples/hsgp.html>`_ by `Omar Sosa Rodríguez <https://github.com/omarfsosa>`_.

eigenindices
----------------
.. autofunction:: numpyro.contrib.hsgp.laplacian.eigenindices

sqrt_eigenvalues
----------------
.. autofunction:: numpyro.contrib.hsgp.laplacian.sqrt_eigenvalues

eigenfunctions
--------------
.. autofunction:: numpyro.contrib.hsgp.laplacian.eigenfunctions

eigenfunctions_periodic
-----------------------
.. autofunction:: numpyro.contrib.hsgp.laplacian.eigenfunctions_periodic

spectral_density_squared_exponential
------------------------------------
.. autofunction:: numpyro.contrib.hsgp.spectral_densities.spectral_density_squared_exponential

spectral_density_matern
-----------------------
.. autofunction:: numpyro.contrib.hsgp.spectral_densities.spectral_density_matern

diag_spectral_density_squared_exponential
-----------------------------------------
.. autofunction:: numpyro.contrib.hsgp.spectral_densities.diag_spectral_density_squared_exponential

diag_spectral_density_matern
----------------------------
.. autofunction:: numpyro.contrib.hsgp.spectral_densities.diag_spectral_density_matern

diag_spectral_density_periodic
------------------------------
.. autofunction:: numpyro.contrib.hsgp.spectral_densities.diag_spectral_density_periodic

hsgp_squared_exponential
------------------------
.. autofunction:: numpyro.contrib.hsgp.approximation.hsgp_squared_exponential

hsgp_matern
-----------
.. autofunction:: numpyro.contrib.hsgp.approximation.hsgp_matern

hsgp_periodic_non_centered
--------------------------
.. autofunction:: numpyro.contrib.hsgp.approximation.hsgp_periodic_non_centered
