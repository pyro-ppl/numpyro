[![Build Status](https://github.com/pyro-ppl/numpyro/workflows/CI/badge.svg)](https://github.com/pyro-ppl/numpyro/actions)
[![Documentation Status](https://readthedocs.org/projects/numpyro/badge/?version=latest)](https://numpyro.readthedocs.io/en/latest/?badge=latest)
[![Latest Version](https://badge.fury.io/py/numpyro.svg)](https://pypi.python.org/pypi/numpyro)

# NumPyro

Probabilistic programming powered by [JAX](https://github.com/google/jax) for autograd and JIT compilation to GPU/TPU/CPU.

[Docs and Examples](https://num.pyro.ai) | [Forum](https://forum.pyro.ai/)

----------------------------------------------------------------------------------------------------

## What is NumPyro?

NumPyro is a lightweight probabilistic programming library that provides a NumPy backend for [Pyro](https://github.com/pyro-ppl/pyro). We rely on [JAX](https://github.com/google/jax) for automatic differentiation and JIT compilation to GPU / CPU. NumPyro is under active development, so beware of brittleness, bugs, and changes to the API as the design evolves.

NumPyro is designed to be *lightweight* and focuses on providing a flexible substrate that users can build on:

- **Pyro Primitives:** NumPyro programs can contain regular Python and NumPy code, in addition to [Pyro primitives](https://pyro.ai/examples/intro_part_i.html) like `sample` and `param`. The model code should look very similar to Pyro except for some minor differences between PyTorch and Numpy's API. See the [example](https://github.com/pyro-ppl/numpyro#a-simple-example---8-schools) below.
- **Inference algorithms:** NumPyro supports a number of inference algorithms, with a particular focus on MCMC algorithms like Hamiltonian Monte Carlo, including an implementation of the No U-Turn Sampler. Additional MCMC algorithms include [MixedHMC](https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.mixed_hmc.MixedHMC) (which can accommodate discrete latent variables) as well as [HMCECS](https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.hmc_gibbs.HMCECS) (which only computes the likelihood for subsets of the data in each iteration). One of the motivations for NumPyro was to speed up Hamiltonian Monte Carlo by JIT compiling the verlet integrator that includes multiple gradient computations. With JAX, we can compose `jit` and `grad` to compile the entire integration step into an XLA optimized kernel. We also eliminate Python overhead by JIT compiling the entire tree building stage in NUTS (this is possible using [Iterative NUTS](https://github.com/pyro-ppl/numpyro/wiki/Iterative-NUTS)). There is also a basic Variational Inference implementation together with many flexible (auto)guides for Automatic Differentiation Variational Inference (ADVI). The variational inference implementation supports a number of features, including support for models with discrete latent variables (see [TraceGraph_ELBO](https://num.pyro.ai/en/latest/svi.html#numpyro.infer.elbo.TraceGraph_ELBO) and [TraceEnum_ELBO](https://num.pyro.ai/en/latest/svi.html#numpyro.infer.elbo.TraceEnum_ELBO)).
- **Distributions:** The [numpyro.distributions](https://numpyro.readthedocs.io/en/latest/distributions.html) module provides distribution classes, constraints and bijective transforms. The distribution classes wrap over samplers implemented to work with JAX's [functional pseudo-random number generator](https://github.com/google/jax#random-numbers-are-different). The design of the distributions module largely follows from [PyTorch](https://pytorch.org/docs/stable/distributions.html). A major subset of the API is implemented, and it contains most of the common distributions that exist in PyTorch. As a result, Pyro and PyTorch users can rely on the same API and batching semantics as in `torch.distributions`. In addition to distributions, `constraints` and `transforms` are very useful when operating on distribution classes with bounded support. Finally, distributions from TensorFlow Probability ([TFP](https://num.pyro.ai/en/latest/distributions.html?highlight=tfp#numpyro.contrib.tfp.distributions.TFPDistribution)) can directly be used in NumPyro models.
- **Effect handlers:** Like Pyro, primitives like `sample` and `param` can be provided nonstandard interpretations using effect-handlers from the [numpyro.handlers](https://numpyro.readthedocs.io/en/latest/handlers.html) module, and these can be easily extended to implement custom inference algorithms and inference utilities.

## A Simple Example - 8 Schools

Let us explore NumPyro using a simple example. We will use the eight schools example from Gelman et al., Bayesian Data Analysis: Sec. 5.5, 2003, which studies the effect of coaching on SAT performance in eight schools.

The data is given by:

```python
>>> import numpy as np

>>> J = 8
>>> y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
>>> sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

```

, where `y` are the treatment effects and `sigma` the standard error. We build a hierarchical model for the study where we assume that the group-level parameters `theta` for each school are sampled from a Normal distribution with unknown mean `mu` and standard deviation `tau`, while the observed data are in turn generated from a Normal distribution with mean and standard deviation given by `theta` (true effect) and `sigma`, respectively. This allows us to estimate the population-level parameters `mu` and `tau` by pooling from all the observations, while still allowing for individual variation amongst the schools using the group-level `theta` parameters.

```python
>>> import numpyro
>>> import numpyro.distributions as dist

>>> # Eight Schools example
... def eight_schools(J, sigma, y=None):
...     mu = numpyro.sample('mu', dist.Normal(0, 5))
...     tau = numpyro.sample('tau', dist.HalfCauchy(5))
...     with numpyro.plate('J', J):
...         theta = numpyro.sample('theta', dist.Normal(mu, tau))
...         numpyro.sample('obs', dist.Normal(theta, sigma), obs=y)

```

Let us infer the values of the unknown parameters in our model by running MCMC using the No-U-Turn Sampler (NUTS). Note the usage of the `extra_fields` argument in [MCMC.run](https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.mcmc.MCMC.run). By default, we only collect samples from the target (posterior) distribution when we run inference using `MCMC`. However, collecting additional fields like potential energy or the acceptance probability of a sample can be easily achieved by using the `extra_fields` argument. For a list of possible fields that can be collected, see the [HMCState](https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.hmc.HMCState) object. In this example, we will additionally collect the `potential_energy` for each sample.

```python
>>> from jax import random
>>> from numpyro.infer import MCMC, NUTS

>>> nuts_kernel = NUTS(eight_schools)
>>> mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
>>> rng_key = random.PRNGKey(0)
>>> mcmc.run(rng_key, J, sigma, y=y, extra_fields=('potential_energy',))

```

We can print the summary of the MCMC run, and examine if we observed any divergences during inference. Additionally, since we collected the potential energy for each of the samples, we can easily compute the expected log joint density.

```python
>>> mcmc.print_summary()  # doctest: +SKIP

                mean       std    median      5.0%     95.0%     n_eff     r_hat
        mu      4.14      3.18      3.87     -0.76      9.50    115.42      1.01
       tau      4.12      3.58      3.12      0.51      8.56     90.64      1.02
  theta[0]      6.40      6.22      5.36     -2.54     15.27    176.75      1.00
  theta[1]      4.96      5.04      4.49     -1.98     14.22    217.12      1.00
  theta[2]      3.65      5.41      3.31     -3.47     13.77    247.64      1.00
  theta[3]      4.47      5.29      4.00     -3.22     12.92    213.36      1.01
  theta[4]      3.22      4.61      3.28     -3.72     10.93    242.14      1.01
  theta[5]      3.89      4.99      3.71     -3.39     12.54    206.27      1.00
  theta[6]      6.55      5.72      5.66     -1.43     15.78    124.57      1.00
  theta[7]      4.81      5.95      4.19     -3.90     13.40    299.66      1.00

Number of divergences: 19

>>> pe = mcmc.get_extra_fields()['potential_energy']
>>> print('Expected log joint density: {:.2f}'.format(np.mean(-pe)))  # doctest: +SKIP
Expected log joint density: -54.55

```

The values above 1 for the split Gelman Rubin diagnostic (`r_hat`) indicates that the chain has not fully converged. The low value for the effective sample size (`n_eff`), particularly for `tau`, and the number of divergent transitions looks problematic. Fortunately, this is a common pathology that can be rectified by using a [non-centered parameterization](https://mc-stan.org/docs/2_18/stan-users-guide/reparameterization-section.html) for `tau` in our model. This is straightforward to do in NumPyro by using a [TransformedDistribution](https://num.pyro.ai/en/latest/distributions.html#transformeddistribution) instance together with a [reparameterization](https://num.pyro.ai/en/latest/handlers.html#reparam) effect handler. Let us rewrite the same model but instead of sampling `theta` from a `Normal(mu, tau)`, we will instead sample it from a base `Normal(0, 1)` distribution that is transformed using an [AffineTransform](https://num.pyro.ai/en/latest/distributions.html#affinetransform). Note that by doing so, NumPyro runs HMC by generating samples `theta_base` for the base `Normal(0, 1)` distribution instead. We see that the resulting chain does not suffer from the same pathology — the Gelman Rubin diagnostic is 1 for all the parameters and the effective sample size looks quite good!

```python
>>> from numpyro.infer.reparam import TransformReparam

>>> # Eight Schools example - Non-centered Reparametrization
... def eight_schools_noncentered(J, sigma, y=None):
...     mu = numpyro.sample('mu', dist.Normal(0, 5))
...     tau = numpyro.sample('tau', dist.HalfCauchy(5))
...     with numpyro.plate('J', J):
...         with numpyro.handlers.reparam(config={'theta': TransformReparam()}):
...             theta = numpyro.sample(
...                 'theta',
...                 dist.TransformedDistribution(dist.Normal(0., 1.),
...                                              dist.transforms.AffineTransform(mu, tau)))
...         numpyro.sample('obs', dist.Normal(theta, sigma), obs=y)

>>> nuts_kernel = NUTS(eight_schools_noncentered)
>>> mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
>>> rng_key = random.PRNGKey(0)
>>> mcmc.run(rng_key, J, sigma, y=y, extra_fields=('potential_energy',))
>>> mcmc.print_summary(exclude_deterministic=False)  # doctest: +SKIP

                   mean       std    median      5.0%     95.0%     n_eff     r_hat
           mu      4.08      3.51      4.14     -1.69      9.71    720.43      1.00
          tau      3.96      3.31      3.09      0.01      8.34    488.63      1.00
     theta[0]      6.48      5.72      6.08     -2.53     14.96    801.59      1.00
     theta[1]      4.95      5.10      4.91     -3.70     12.82   1183.06      1.00
     theta[2]      3.65      5.58      3.72     -5.71     12.13    581.31      1.00
     theta[3]      4.56      5.04      4.32     -3.14     12.92   1282.60      1.00
     theta[4]      3.41      4.79      3.47     -4.16     10.79    801.25      1.00
     theta[5]      3.58      4.80      3.78     -3.95     11.55   1101.33      1.00
     theta[6]      6.31      5.17      5.75     -2.93     13.87   1081.11      1.00
     theta[7]      4.81      5.38      4.61     -3.29     14.05    954.14      1.00
theta_base[0]      0.41      0.95      0.40     -1.09      1.95    851.45      1.00
theta_base[1]      0.15      0.95      0.20     -1.42      1.66   1568.11      1.00
theta_base[2]     -0.08      0.98     -0.10     -1.68      1.54   1037.16      1.00
theta_base[3]      0.06      0.89      0.05     -1.42      1.47   1745.02      1.00
theta_base[4]     -0.14      0.94     -0.16     -1.65      1.45    719.85      1.00
theta_base[5]     -0.10      0.96     -0.14     -1.57      1.51   1128.45      1.00
theta_base[6]      0.38      0.95      0.42     -1.32      1.82   1026.50      1.00
theta_base[7]      0.10      0.97      0.10     -1.51      1.65   1190.98      1.00

Number of divergences: 0

>>> pe = mcmc.get_extra_fields()['potential_energy']
>>> # Compare with the earlier value
>>> print('Expected log joint density: {:.2f}'.format(np.mean(-pe)))  # doctest: +SKIP
Expected log joint density: -46.09

```

Note that for the class of distributions with `loc,scale` parameters such as `Normal`, `Cauchy`, `StudentT`, we also provide a [LocScaleReparam](https://num.pyro.ai/en/latest/reparam.html#loc-scale-decentering) reparameterizer to achieve the same purpose. The corresponding code will be

    with numpyro.handlers.reparam(config={'theta': LocScaleReparam(centered=0)}):
        theta = numpyro.sample('theta', dist.Normal(mu, tau))

Now, let us assume that we have a new school for which we have not observed any test scores, but we would like to generate predictions. NumPyro provides a [Predictive](https://num.pyro.ai/en/latest/utilities.html#numpyro.infer.util.Predictive) class for such a purpose. Note that in the absence of any observed data, we simply use the population-level parameters to generate predictions. The `Predictive` utility conditions the unobserved `mu` and `tau` sites to values drawn from the posterior distribution from our last MCMC run, and runs the model forward to generate predictions.

```python
>>> from numpyro.infer import Predictive

>>> # New School
... def new_school():
...     mu = numpyro.sample('mu', dist.Normal(0, 5))
...     tau = numpyro.sample('tau', dist.HalfCauchy(5))
...     return numpyro.sample('obs', dist.Normal(mu, tau))

>>> predictive = Predictive(new_school, mcmc.get_samples())
>>> samples_predictive = predictive(random.PRNGKey(1))
>>> print(np.mean(samples_predictive['obs']))  # doctest: +SKIP
3.9886456

```

## More Examples

For some more examples on specifying models and doing inference in NumPyro:

- [Bayesian Regression in NumPyro](https://nbviewer.jupyter.org/github/pyro-ppl/numpyro/blob/master/notebooks/source/bayesian_regression.ipynb) - Start here to get acquainted with writing a simple model in NumPyro, MCMC inference API, effect handlers and writing custom inference utilities.
- [Time Series Forecasting](https://nbviewer.jupyter.org/github/pyro-ppl/numpyro/blob/master/notebooks/source/time_series_forecasting.ipynb) - Illustrates how to convert for loops in the model to JAX's `lax.scan` primitive for fast inference.
- [Annotation examples](https://num.pyro.ai/en/stable/examples/annotation.html) - Illustrates how to utilize the enumeration mechanism to perform inference for models with discrete latent variables.
- [Baseball example](https://github.com/pyro-ppl/numpyro/blob/master/examples/baseball.py) - Using NUTS for a simple hierarchical model. Compare this with the baseball example in [Pyro](https://github.com/pyro-ppl/pyro/blob/dev/examples/baseball.py).
- [Hidden Markov Model](https://github.com/pyro-ppl/numpyro/blob/master/examples/hmm.py) in NumPyro as compared to [Stan](https://mc-stan.org/docs/2_19/stan-users-guide/hmms-section.html).
- [Variational Autoencoder](https://github.com/pyro-ppl/numpyro/blob/master/examples/vae.py) - As a simple example that uses Variational Inference with neural networks. [Pyro implementation](https://github.com/pyro-ppl/pyro/blob/dev/examples/vae/vae.py) for comparison.
- [Gaussian Process](https://github.com/pyro-ppl/numpyro/blob/master/examples/gp.py) - Provides a simple example to use NUTS to sample from the posterior over the hyper-parameters of a Gaussian Process.
- [Horseshoe Regression](https://github.com/pyro-ppl/numpyro/blob/master/examples/horseshoe_regression.py) - Shows how to implement generalized linear models equipped with a Horseshoe prior for both binary-valued and real-valued outputs.
- [Statistical Rethinking with NumPyro](https://github.com/fehiepsi/rethinking-numpyro) - [Notebooks](https://nbviewer.jupyter.org/github/fehiepsi/rethinking-numpyro/tree/master/notebooks/) containing translation of the code in Richard McElreath's [Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/) book second version, to NumPyro.
- Other model examples can be found in the [examples](https://num.pyro.ai/en/stable/) site.

Pyro users will note that the API for model specification and inference is largely the same as Pyro, including the distributions API, by design. However, there are some important core differences (reflected in the internals) that users should be aware of. e.g. in NumPyro, there is no global parameter store or random state, to make it possible for us to leverage JAX's JIT compilation. Also, users may need to write their models in a more *functional* style that works better with JAX. Refer to [FAQs](#frequently-asked-questions) for a list of differences.

## Overview of inference algorithms

We provide an overview of most of the inference algorithms supported by NumPyro and offer some guidelines about which inference algorithms may be appropriate for different classes of models.

### MCMC

- [NUTS](https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.hmc.NUTS), which is an adaptive variant of [HMC](https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.hmc.HMC), is probably the most commonly used inference algorithm in NumPyro. Note that NUTS and HMC are not directly applicable to models with discrete latent variables, but in cases where the discrete variables have finite support and summing them out (i.e. enumeration) is tractable, NumPyro will automatically sum out discrete latent variables and perform NUTS/HMC on the remaining continuous latent variables.
As discussed above, model [reparameterization](https://num.pyro.ai/en/latest/reparam.html#module-numpyro.infer.reparam) may be important in some cases to get good performance. Note that, generally speaking, we expect inference to be harder as the dimension of the latent space increases. See the [bad geometry](https://num.pyro.ai/en/latest/tutorials/bad_posterior_geometry.html) tutorial for additional tips and tricks.
- [MixedHMC](https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.mixed_hmc.MixedHMC) can be an effective inference strategy for models that contain both continuous and discrete latent variables.
- [HMCECS](https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.hmc_gibbs.HMCECS) can be an effective inference strategy for models with a large number of data points. It is applicable to models with continuous latent variables. See [here](https://num.pyro.ai/en/latest/examples/covtype.html) for an example.
- [BarkerMH](https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.barker.BarkerMH) is a gradient-based MCMC method that may be competitive with HMC and NUTS for some models. It is applicable to models with continuous latent variables.
- [HMCGibbs](https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.hmc_gibbs.HMCGibbs) combines HMC/NUTS steps with custom Gibbs updates. Gibbs updates must be specified by the user.
- [DiscreteHMCGibbs](https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.hmc_gibbs.DiscreteHMCGibbs) combines HMC/NUTS steps with Gibbs updates for discrete latent variables. The corresponding Gibbs updates are computed automatically.
- [SA](https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.sa.SA) is the only MCMC method in NumPyro that does not leverage gradients. It is only applicable to models with continuous latent variables. It is expected to perform best for models whose latent dimension is low to moderate. It may be a good choice for models with non-differentiable log densities. Note that SA generally requires a *very* large number of samples, as mixing tends to be slow. On the plus side individual steps can be fast.

Like HMC/NUTS, all remaining MCMC algorithms support enumeration over discrete latent variables if possible (see [restrictions](https://pyro.ai/examples/enumeration.html#Restriction-1:-conditional-independence)). Enumerated sites need to be marked with `infer={'enumerate': 'parallel'}` like in the [annotation example](https://num.pyro.ai/en/stable/examples/annotation.html).

### Nested Sampling

- [NestedSampler](https://num.pyro.ai/en/latest/contrib.html#nested-sampling) offers a wrapper for [jaxns](https://github.com/Joshuaalbert/jaxns). See [JAXNS's readthedocs](https://jaxns.readthedocs.io/en/latest/) for examples and [Nested Sampling for Gaussian Shells](https://num.pyro.ai/en/stable/examples/gaussian_shells.html) example for how to apply the sampler on numpyro models. Can handle arbitrary models, including ones with discrete RVs, and non-invertible transformations.

### Stochastic variational inference

- Variational objectives
  - [Trace_ELBO](https://num.pyro.ai/en/latest/svi.html#numpyro.infer.elbo.Trace_ELBO) is our basic ELBO implementation.
  - [TraceMeanField_ELBO](https://num.pyro.ai/en/latest/svi.html#numpyro.infer.elbo.TraceMeanField_ELBO) is like `Trace_ELBO` but computes part of the ELBO analytically if doing so is possible.
  - [TraceGraph_ELBO](https://num.pyro.ai/en/latest/svi.html#numpyro.infer.elbo.TraceGraph_ELBO) offers variance reduction strategies for models with discrete latent variables. Generally speaking, this ELBO should always be used for models with discrete latent variables.
  - [TraceEnum_ELBO](https://num.pyro.ai/en/latest/svi.html#numpyro.infer.elbo.TraceEnum_ELBO) offers variable enumeration strategies for models with discrete latent variables. Generally speaking, this ELBO should always be used for models with discrete latent variables when enumeration is possible.
- Automatic guides (appropriate for models with continuous latent variables)
  - [AutoNormal](https://num.pyro.ai/en/latest/autoguide.html#numpyro.infer.autoguide.AutoNormal) and [AutoDiagonalNormal](https://num.pyro.ai/en/latest/autoguide.html#numpyro.infer.autoguide.AutoDiagonalNormal) are our basic mean-field guides. If the latent space is non-euclidean (due to e.g. a positivity constraint on one of the sample sites) an appropriate bijective transformation is automatically used under the hood to map between the unconstrained space (where the Normal variational distribution is defined) to the corresponding constrained space (note this is true for all automatic guides). These guides are a great place to start when trying to get variational inference to work on a model you are developing.
  - [AutoMultivariateNormal](https://num.pyro.ai/en/latest/autoguide.html#numpyro.infer.autoguide.AutoMultivariateNormal) and [AutoLowRankMultivariateNormal](https://num.pyro.ai/en/latest/autoguide.html#numpyro.infer.autoguide.AutoLowRankMultivariateNormal) also construct Normal variational distributions but offer more flexibility, as they can capture correlations in the posterior. Note that these guides may be difficult to fit in the high-dimensional setting.
  - [AutoDelta](https://num.pyro.ai/en/latest/autoguide.html#numpyro.infer.autoguide.AutoDelta) is used for computing point estimates via MAP (maximum a posteriori estimation). See [here](https://github.com/pyro-ppl/numpyro/blob/bbe1f879eede79eebfdd16dfc49c77c4d1fc727c/examples/zero_inflated_poisson.py#L101) for example usage.
  - [AutoBNAFNormal](https://num.pyro.ai/en/latest/autoguide.html#numpyro.infer.autoguide.AutoBNAFNormal) and [AutoIAFNormal](https://num.pyro.ai/en/latest/autoguide.html#numpyro.infer.autoguide.AutoIAFNormal) offer flexible variational distributions parameterized by normalizing flows.
  - [AutoDAIS](https://num.pyro.ai/en/latest/autoguide.html#numpyro.infer.autoguide.AutoDAIS) is a powerful variational inference algorithm that leverages HMC. It can be a good choice for dealing with highly correlated posteriors but may be computationally expensive depending on the nature of the model.
  - [AutoSurrogateLikelihoodDAIS](https://num.pyro.ai/en/latest/autoguide.html#numpyro.infer.autoguide.AutoSurrogateLikelihoodDAIS) is a powerful variational inference algorithm that leverages HMC and that supports data subsampling.
  - [AutoSemiDAIS](https://num.pyro.ai/en/latest/autoguide.html#numpyro.infer.autoguide.AutoSemiDAIS) constructs a posterior approximation like [AutoDAIS](https://num.pyro.ai/en/latest/autoguide.html#numpyro.infer.autoguide.AutoDAIS) for local latent variables but provides support for data subsampling during ELBO training by utilizing a parametric guide for global latent variables.
  - [AutoLaplaceApproximation](https://num.pyro.ai/en/latest/autoguide.html#numpyro.infer.autoguide.AutoLaplaceApproximation) can be used to compute a Laplace approximation.

### Stein Variational Inference

See the [docs](https://num.pyro.ai/en/latest/contrib.html#stein-variational-inference) for more details.

## Installation

> **Limited Windows Support:** Note that NumPyro is untested on Windows, and might require building jaxlib from source. See this [JAX issue](https://github.com/google/jax/issues/438) for more details. Alternatively, you can install [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/) and use NumPyro on it as on a Linux system. See also [CUDA on Windows Subsystem for Linux](https://developer.nvidia.com/cuda/wsl) and [this forum post](https://forum.pyro.ai/t/numpyro-with-gpu-works-on-windows/2690) if you want to use GPUs on Windows.

To install NumPyro with the latest CPU version of JAX, you can use pip:

```
pip install numpyro
```

In case of compatibility issues arise during execution of the above command, you can instead force the installation of a known
compatible CPU version of JAX with

```
pip install numpyro[cpu]
```

To use **NumPyro on the GPU**, you need to install CUDA first and then use the following pip command:

```
pip install numpyro[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

If you need further guidance, please have a look at the [JAX GPU installation instructions](https://github.com/google/jax#pip-installation-gpu-cuda).

To run **NumPyro on Cloud TPUs**, you can look at some [JAX on Cloud TPU examples](https://github.com/google/jax/tree/master/cloud_tpu_colabs).

For Cloud TPU VM, you need to setup the TPU backend as detailed in the [Cloud TPU VM JAX Quickstart Guide](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm).
After you have verified that the TPU backend is properly set up,
you can install NumPyro using the `pip install numpyro` command.

> **Default Platform:** JAX will use GPU by default if CUDA-supported `jaxlib` package is installed. You can use [set_platform](https://num.pyro.ai/en/stable/utilities.html#set-platform) utility `numpyro.set_platform("cpu")` to switch to CPU at the beginning of your program.

You can also install NumPyro from source:

```
git clone https://github.com/pyro-ppl/numpyro.git
cd numpyro
# install jax/jaxlib first for CUDA support
pip install -e .[dev]  # contains additional dependencies for NumPyro development
```

You can also install NumPyro with conda:

```
conda install -c conda-forge numpyro
```

## Frequently Asked Questions

1. Unlike in Pyro, `numpyro.sample('x', dist.Normal(0, 1))` does not work. Why?

   You are most likely using a `numpyro.sample` statement outside an inference context. JAX does not have a global random state, and as such, distribution samplers need an explicit random number generator key ([PRNGKey](https://jax.readthedocs.io/en/latest/jax.random.html#jax.random.PRNGKey)) to generate samples from. NumPyro's inference algorithms use the [seed](https://num.pyro.ai/en/latest/handlers.html#seed) handler to thread in a random number generator key, behind the scenes.

   Your options are:

   - Call the distribution directly and provide a `PRNGKey`, e.g. `dist.Normal(0, 1).sample(PRNGKey(0))`
   - Provide the `rng_key` argument to `numpyro.sample`. e.g. `numpyro.sample('x', dist.Normal(0, 1), rng_key=PRNGKey(0))`.
   - Wrap the code in a `seed` handler, used either as a context manager or as a function that wraps over the original callable. e.g.

        ```python
        with handlers.seed(rng_seed=0):  # random.PRNGKey(0) is used
            x = numpyro.sample('x', dist.Beta(1, 1))    # uses a PRNGKey split from random.PRNGKey(0)
            y = numpyro.sample('y', dist.Bernoulli(x))  # uses different PRNGKey split from the last one
        ```

     , or as a higher order function:

        ```python
        def fn():
            x = numpyro.sample('x', dist.Beta(1, 1))
            y = numpyro.sample('y', dist.Bernoulli(x))
            return y

        print(handlers.seed(fn, rng_seed=0)())
        ```

2. Can I use the same Pyro model for doing inference in NumPyro?

   As you may have noticed from the examples, NumPyro supports all Pyro primitives like `sample`, `param`, `plate` and `module`, and effect handlers. Additionally, we have ensured that the [distributions](https://numpyro.readthedocs.io/en/latest/distributions.html) API is based on `torch.distributions`, and the inference classes like `SVI` and `MCMC` have the same interface. This along with the similarity in the API for NumPy and PyTorch operations ensures that models containing Pyro primitive statements can be used with either backend with some minor changes. Example of some differences along with the changes needed, are noted below:

   - Any `torch` operation in your model will need to be written in terms of the corresponding `jax.numpy` operation. Additionally, not all `torch` operations have a `numpy` counterpart (and vice-versa), and sometimes there are minor differences in the API.
   - `pyro.sample` statements outside an inference context will need to be wrapped in a `seed` handler, as mentioned above.
   - There is no global parameter store, and as such using `numpyro.param` outside an inference context will have no effect. To retrieve the optimized parameter values from SVI, use the [SVI.get_params](https://num.pyro.ai/en/latest/svi.html#numpyro.infer.svi.SVI.get_params) method. Note that you can still use `param` statements inside a model and NumPyro will use the [substitute](https://num.pyro.ai/en/latest/handlers.html#substitute) effect handler internally to substitute values from the optimizer when running the model in SVI.
   - PyTorch neural network modules will need to rewritten as [stax](https://github.com/google/jax#neural-net-building-with-stax), [flax](https://flax.readthedocs.io/en/latest/), or [haiku](https://dm-haiku.readthedocs.io/en/latest/) neural networks. See the [VAE](https://num.pyro.ai/en/latest/examples/vae.html) and [ProdLDA](https://num.pyro.ai/en/stable/examples/prodlda.html) examples for differences in syntax between the two backends.
   - JAX works best with functional code, particularly if we would like to leverage JIT compilation, which NumPyro does internally for many inference subroutines. As such, if your model has side-effects that are not visible to the JAX tracer, it may need to rewritten in a more functional style.

   For most small models, changes required to run inference in NumPyro should be minor. Additionally, we are working on [pyro-api](https://github.com/pyro-ppl/pyro-api) which allows you to write the same code and dispatch it to multiple backends, including NumPyro. This will necessarily be more restrictive, but has the advantage of being backend agnostic. See the [documentation](https://pyro-api.readthedocs.io/en/latest/dispatch.html#module-pyroapi.dispatch) for an example, and let us know your feedback.

3. How can I contribute to the project?

   Thanks for your interest in the project! You can take a look at beginner friendly issues that are marked with the [good first issue](https://github.com/pyro-ppl/numpyro/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) tag on Github. Also, please feel to reach out to us on the [forum](https://forum.pyro.ai/).

## Future / Ongoing Work

In the near term, we plan to work on the following. Please open new issues for feature requests and enhancements:

- Improving robustness of inference on different models, profiling and performance tuning.
- Supporting more functionality as part of the [pyro-api](https://github.com/pyro-ppl/pyro-api) generic modeling interface.
- More inference algorithms, particularly those that require second order derivatives or use HMC.
- Integration with [Funsor](https://github.com/pyro-ppl/funsor) to support inference algorithms with delayed sampling.
- Other areas motivated by Pyro's research goals and application focus, and interest from the community.

## Citing NumPyro

The motivating ideas behind NumPyro and a description of Iterative NUTS can be found in this [paper](https://arxiv.org/abs/1912.11554) that appeared in NeurIPS 2019 Program Transformations for Machine Learning Workshop.

If you use NumPyro, please consider citing:

```
@article{phan2019composable,
  title={Composable Effects for Flexible and Accelerated Probabilistic Programming in NumPyro},
  author={Phan, Du and Pradhan, Neeraj and Jankowiak, Martin},
  journal={arXiv preprint arXiv:1912.11554},
  year={2019}
}
```

as well as

```
@article{bingham2019pyro,
  author    = {Eli Bingham and
               Jonathan P. Chen and
               Martin Jankowiak and
               Fritz Obermeyer and
               Neeraj Pradhan and
               Theofanis Karaletsos and
               Rohit Singh and
               Paul A. Szerlip and
               Paul Horsfall and
               Noah D. Goodman},
  title     = {Pyro: Deep Universal Probabilistic Programming},
  journal   = {J. Mach. Learn. Res.},
  volume    = {20},
  pages     = {28:1--28:6},
  year      = {2019},
  url       = {http://jmlr.org/papers/v20/18-403.html}
}
```
