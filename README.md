# NumPyro 

[![Build Status](https://travis-ci.com/pyro-ppl/numpyro.svg?branch=master)](https://travis-ci.com/pyro-ppl/numpyro)

[Pyro](https://github.com/pyro-ppl/pyro) on Numpy. This uses 
[JAX](https://github.com/google/jax) for autograd and JIT support. This is an 
early stage experimental library that is under active development, and there are 
likely to be  many changes to the API and internal classes, as the design evolves. 

 ## Design Goals
 
 - **Lightweight** - We do not intend to reimplement any heavy inference machinery 
   from Pyro, but would like to provide a flexible substrate that can be built 
   upon. We will provide support for Pyro primitives like `sample` and `param` 
   which can be interpreted with side-effects using effect handlers. Users should 
   be able to extend this to implement custom inference algorithms, and write 
   their models using the familiar Numpy API.
 - **Functional** - The API for the inference algorithms and other utility functions 
   may deviate from Pyro in favor of a more *functional* style that works better 
   with JAX. e.g. no global param store or random state.
 - **Fast** - Using JAX, we aim to aggressively JIT compile intermediate computations 
   to XLA optimized kernels. We will evaluate JIT compilation, and benchmark runtime 
   for Hamiltonian Monte Carlo.
    
 ## Longer-term Plans
 
It is possible that much of this code will end up being absorbed into the Pyro 
project itself as an alternate Numpy backend.
