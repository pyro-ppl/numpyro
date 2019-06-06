[![Build Status](https://travis-ci.com/pyro-ppl/numpyro.svg?branch=master)](https://travis-ci.com/pyro-ppl/numpyro)
[![Documentation Status](https://readthedocs.org/projects/numpyro/badge/?version=latest)](https://numpyro.readthedocs.io/en/latest/?badge=latest)
[![Latest Version](https://badge.fury.io/py/numpyro.svg)](https://pypi.python.org/pypi/numpyro)
# NumPyro

Probabilistic programming with NumPy powered by [JAX](https://github.com/google/jax) for autograd and JIT compilation to GPU/CPU.

## What is NumPyro?

NumPyro is a small probabilistic programming library built on [JAX](https://github.com/google/jax). It essentially provides a NumPy backend for [Pyro](https://github.com/pyro-ppl/pyro), with some minor changes to the inference API and syntax. Since we use JAX, we get autograd and JIT compilation to GPU / CPU for free. This is an alpha release under active development, so beware of brittleness, bugs, and changes to the API as the design evolves.

NumPyro is designed to be *lightweight* and focuses on providing a flexible substrate that users can build on:

 - **Pyro Primitives:** NumPyro programs can contain regular Python and NumPy code, in addition to [Pyro primitives](http://pyro.ai/examples/intro_part_i.html) like `sample` and `param`. The model code should look very similar to Pyro except for some minor differences between PyTorch and Numpy's API. See [Examples](https://github.com/pyro-ppl/numpyro/#Examples).
 - **Inference algorithms:** NumPyro currently supports Hamiltonian Monte Carlo, including an implementation of the No U-Turn Sampler. One of the motivations for NumPyro was to speed up Hamiltonian Monte Carlo by JIT compiling the verlet integration step that includes multiple gradient computations. With JAX, we can compose `jit` and `grad` to compile the entire integration step into an XLA optimized kernel. We also eliminate Python overhead by JIT compiling the entire tree building stage in NUTS (this is possible using [Iterative NUTS](https://github.com/pyro-ppl/numpyro/wiki/Iterative-NUTS)). There is also a basic Variational Inference implementation for reparameterized distributions.
 - **Distributions:** The [numpyro.distributions](https://numpyro.readthedocs.io/en/latest/distributions.html) module provides distribution classes, constraints and bijective transforms. The distribution classes wrap over samplers implemented to work with JAX's [functional pseudo-random number generator](https://github.com/google/jax#random-numbers-are-different). The design of the distributions module largely follows from [PyTorch](https://pytorch.org/docs/stable/distributions.html). A major subset of the API is implemented, and it contains most of the common distributions that exist in PyTorch. As a result, Pyro and PyTorch users can rely on the same API and batching semantics as in `torch.distributions`. In addition to distributions, `constraints` and `transforms` are very useful when operating on distribution classes with bounded support.
 - **Effect handlers:** Like Pyro, primitives like `sample` and `param` can be interpreted with side-effects using effect-handlers from the [numpyro.handlers](https://numpyro.readthedocs.io/en/latest/handlers.html) module, and these can be easily extended to implement custom inference algorithms and inference utilities.


## Installation

> **Limited Windows Support:** Note that NumPyro is untested on Windows, and will require building jaxlib from source. See https://github.com/google/jax/issues/438 for more details. 

To install NumPyro with a CPU version of JAX, you can use pip:

```
pip install numpyro
```

To use NumPyro on the GPU, you will need to first [install](https://github.com/google/jax#installation) `jax` and `jaxlib` with CUDA support.

You can also install NumPyro from source:

```
git clone https://github.com/pyro-ppl/numpyro.git
# install jax/jaxlib first for CUDA support
pip install -e .[dev]
```

## Examples


For some examples on specifying models and doing inference in NumPyro:

 - [Bayesian Regression in NumPyro](https://nbviewer.jupyter.org/github/pyro-ppl/numpyro/blob/master/notebooks/bayesian_regression.ipynb) - Start here to get acquainted with writing a simple model in NumPyro, MCMC inference API, effect handlers and writing custom inference utilities.
 - [Time Series Forecasting](https://nbviewer.jupyter.org/github/pyro-ppl/numpyro/blob/master/notebooks/time_series_forecasting.ipynb) - Illustrates how to convert for loops in the model to JAX's `lax.scan` primitive for fast inference.
 - [Baseball example](https://github.com/pyro-ppl/numpyro/blob/master/examples/baseball.py) - Using NUTS for a simple hierarchical model. Compare this with the baseball example in [Pyro](https://github.com/pyro-ppl/pyro/blob/dev/examples/baseball.py).
 - [Hidden Markov Model](https://github.com/pyro-ppl/numpyro/blob/master/examples/hmm.py) in NumPyro as compared to [Stan](https://mc-stan.org/docs/2_19/stan-users-guide/hmms-section.html).
 - [Variational Autoencoder](https://github.com/pyro-ppl/numpyro/blob/master/examples/vae.py) - As a simple example that uses Variational Inference. [Pyro implementation](https://github.com/pyro-ppl/pyro/blob/dev/examples/vae/vae.py) for comparison.
 - Other model examples can be found in the [examples](https://github.com/pyro-ppl/numpyro/tree/master/examples) folder.

Users will note that the API for model specification is largely the same as Pyro including the distributions API, by design. The interface for inference algorithms and other utility functions might deviate from Pyro in favor of a more *functional* style that works better with JAX. e.g. there is no global parameter store or random state.

## Future Work

In the near term, we plan to work on the following. Please open new issues for feature requests and enhancements:

 - Improving robustness of inference on different models, profiling and performance tuning.
 - More inference algorithms, particularly those that require second order derivaties or use HMC.
 - Integration with [Funsor](https://github.com/pyro-ppl/funsor) to support inference algorithms with delayed sampling.
 - Supporting more distributions, extending the distributions API, and adding more samplers to JAX.
 - Other areas motivated by Pyro's research goals and application focus, and interest from the community.
