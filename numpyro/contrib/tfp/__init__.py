# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

try:
    import tensorflow_probability.substrates.jax as tfp  # noqa: F401
except ImportError as e:
    raise ImportError("Looking like your installed tensorflow_probability does not"
                      " support JAX backend. You might try to install the nightly"
                      " version with: `pip install tfp-nightly`") from e
