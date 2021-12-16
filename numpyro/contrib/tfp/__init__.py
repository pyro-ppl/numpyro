# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

try:
    import tensorflow_probability.substrates.jax as tfp  # noqa: F401
except ImportError as e:
    raise ImportError(
        "To use this module, please install TensorFlow Probability. It can be"
        " installed with `pip install tensorflow_probability`"
    ) from e
