"""
Examples to show how to use numpyro with multithreading.
"""
from collections import defaultdict
import threading

import jax
import jax.numpy as jnp

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.primitives import set_pyro_stack


class _StackThreadDict(defaultdict):
    def current_stack(self):
        thread_id = threading.get_native_id()
        return self[thread_id]


_PYRO_THREAD_STACK = _StackThreadDict(list)
set_pyro_stack(_PYRO_THREAD_STACK)


def model():
    numpyro.sample("a", dist.Normal(0, 1))


rng_keys = jax.random.split(jax.random.PRNGKey(0), 2)
for rng_key in rng_keys:
    # creat a thread and trace
    pass
