# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from numpyro.distributions.discrete import PRNGIdentity
import numpyro
from jax import numpy as jnp
from functools import partial


def flax_module(name, nn, input_shape=None):
    """
    Declare a :mod:`~flax` style neural network inside a
    model so that its parameters are registered for optimization via
    :func:`~numpyro.primitives.param` statements.

    :param str name: name of the module to be registered.
    :param flax.nn.Module nn: a `flax` Module which has .init and .apply methods
    :param tuple input_shape: shape of the input taken by the
        neural network.
    :return: a callable with bound parameters that takes an array
        as an input and returns the neural network transformed output
        array.
    """
    try:
        import flax  # noqa: F401
    except ImportError:
        raise ImportError("Looking like you want to use flax and/or haiku to declare "
                          "nn modules. This is an experimental feature. "
                          "You need to install `haiku` to be able to use this feature. "
                          "It can be installed with `pip install git+https://github.com/deepmind/dm-haiku`.")
    module_key = name + '$params'
    nn_params = numpyro.param(module_key)
    if nn_params is None:
        if input_shape is None:
            raise ValueError('Valid value for `input_shape` needed to initialize.')
        # feed in dummy data to init params
        rng_key = numpyro.sample(name + '$rng_key', PRNGIdentity())
        _, nn_params = nn.init(rng_key, jnp.ones(input_shape))
        numpyro.param(module_key, nn_params)
    return partial(nn.call, nn_params)


def haiku_module(name, nn, input_shape=None):
    """
    Declare a :mod:`~haiku` style neural network inside a
    model so that its parameters are registered for optimization via
    :func:`~numpyro.primitives.param` statements.

    :param str name: name of the module to be registered.
    :param haiku.Module nn: a `haiku` Module which has .init and .apply methods
    :param tuple input_shape: shape of the input taken by the
        neural network.
    :return: a callable with bound parameters that takes an array
        as an input and returns the neural network transformed output
        array.
    """
    try:
        import haiku  # noqa: F401
    except ImportError:
        raise ImportError("Looking like you want to use haiku to declare "
                          "nn modules. This is an experimental feature. "
                          "You need to install `haiku` to be able to use this feature. "
                          "It can be installed with `pip install git+https://github.com/deepmind/dm-haiku`.")

    module_key = name + '$params'
    nn_params = numpyro.param(module_key)
    if nn_params is None:
        if input_shape is None:
            raise ValueError('Valid value for `input_shape` needed to initialize.')
        # feed in dummy data to init params
        rng_key = numpyro.sample(name + '$rng_key', PRNGIdentity())
        nn_params = nn.init(rng_key, jnp.ones(input_shape))
        numpyro.param(module_key, nn_params)
    return partial(nn.apply, nn_params, None)
