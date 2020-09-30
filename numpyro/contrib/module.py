# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from copy import deepcopy
from functools import partial

from jax import numpy as jnp
from jax.tree_util import register_pytree_node, tree_flatten, tree_unflatten

import numpyro
from numpyro.distributions.discrete import PRNGIdentity

__all__ = [
    'flax_module',
    'haiku_module',
    'random_flax_module',
    'random_haiku_module',
]


def flax_module(name, nn_module, *, input_shape=None):
    """
    Declare a :mod:`~flax` style neural network inside a
    model so that its parameters are registered for optimization via
    :func:`~numpyro.primitives.param` statements.

    :param str name: name of the module to be registered.
    :param flax.nn.Module nn_module: a `flax` Module which has .init and .apply methods
    :param tuple input_shape: shape of the input taken by the
        neural network.
    :return: a callable with bound parameters that takes an array
        as an input and returns the neural network transformed output
        array.
    """
    try:
        import flax  # noqa: F401
    except ImportError as e:
        raise ImportError("Looking like you want to use flax to declare "
                          "nn modules. This is an experimental feature. "
                          "You need to install `flax` to be able to use this feature. "
                          "It can be installed with `pip install flax`.") from e
    module_key = name + '$params'
    nn_params = numpyro.param(module_key)
    if nn_params is None:
        if input_shape is None:
            raise ValueError('Valid value for `input_shape` needed to initialize.')
        # feed in dummy data to init params
        rng_key = numpyro.sample(name + '$rng_key', PRNGIdentity())
        _, nn_params = nn_module.init(rng_key, jnp.ones(input_shape))
        # make sure that nn_params keep the same order after unflatten
        params_flat, tree_def = tree_flatten(nn_params)
        nn_params = tree_unflatten(tree_def, params_flat)
        numpyro.param(module_key, nn_params)
    return partial(nn_module.call, nn_params)


def haiku_module(name, nn_module, *, input_shape=None):
    """
    Declare a :mod:`~haiku` style neural network inside a
    model so that its parameters are registered for optimization via
    :func:`~numpyro.primitives.param` statements.

    :param str name: name of the module to be registered.
    :param haiku.Module nn_module: a `haiku` Module which has .init and .apply methods
    :param tuple input_shape: shape of the input taken by the
        neural network.
    :return: a callable with bound parameters that takes an array
        as an input and returns the neural network transformed output
        array.
    """
    try:
        import haiku  # noqa: F401
    except ImportError as e:
        raise ImportError("Looking like you want to use haiku to declare "
                          "nn modules. This is an experimental feature. "
                          "You need to install `haiku` to be able to use this feature. "
                          "It can be installed with `pip install dm-haiku`.") from e

    module_key = name + '$params'
    nn_params = numpyro.param(module_key)
    if nn_params is None:
        if input_shape is None:
            raise ValueError('Valid value for `input_shape` needed to initialize.')
        # feed in dummy data to init params
        rng_key = numpyro.sample(name + '$rng_key', PRNGIdentity())
        nn_params = nn_module.init(rng_key, jnp.ones(input_shape))
        # haiku init returns an immutable dict
        nn_params = haiku.data_structures.to_mutable_dict(nn_params)
        # we cast it to a mutable one to be able to set priors for parameters
        # make sure that nn_params keep the same order after unflatten
        params_flat, tree_def = tree_flatten(nn_params)
        nn_params = tree_unflatten(tree_def, params_flat)
        numpyro.param(module_key, nn_params)
    return partial(nn_module.apply, nn_params, None)


# register an "empty" parameter which only stores its shape
# so that the optimizer can skip optimize this parameter, while
# it still provides shape information for priors
ParamShape = namedtuple("ParamShape", ["shape"])
register_pytree_node(ParamShape, lambda x: ((None,), x.shape), lambda shape, x: ParamShape(shape))


def _update_params(params, new_params, prior, prefix=''):
    """
    A helper to recursively set prior to new_params.
    """
    for name, item in params.items():
        flatten_name = ".".join([prefix, name]) if prefix else name
        if isinstance(item, dict):
            assert not isinstance(prior, dict) or flatten_name not in prior
            new_item = new_params[name]
            _update_params(item, new_item, prior, prefix=flatten_name)
        elif (not isinstance(prior, dict)) or flatten_name in prior:
            d = prior[flatten_name] if isinstance(prior, dict) else prior
            if isinstance(params[name], ParamShape):
                param_shape = params[name].shape
            else:
                param_shape = jnp.shape(params[name])
                params[name] = ParamShape(param_shape)
            param_batch_shape = param_shape[:len(param_shape) - d.event_dim]
            # XXX: here we set all dimensions of prior to event dimensions.
            new_params[name] = numpyro.sample(flatten_name, d.expand(param_batch_shape).to_event())


def random_flax_module(name, nn_module, prior, *, input_shape=None):
    """
    A primitive to place a prior over the parameters of the Flax module `nn_module`.

    .. note::
        Parameters of a Flax module are stored in a nested dict. For example,
        the module `B` defined as follows::

            class A(nn.Module):
                def apply(self, x):
                    return nn.Dense(x, 1, bias=False, name='dense')

            class B(nn.Module):
                def apply(self, x):
                    return A(x, name='inner')

        has parameters `{'inner': {'dense': {'kernel': param_value}}}`. In the argument
        `prior`, to specify `kernel` parameter, we join the path to it using dots:
        `prior={"inner.dense.kernel": param_prior}`.

    :param str name: name of NumPyro module
    :param flax.nn.Module: the module to be registered with NumPyro
    :param prior: a NumPyro distribution or a Python dict with parameter names as keys and
        respective distributions as values. For example::

            net = random_flax_module("net",
                                     flax.nn.Dense.partial(features=1),
                                     prior={"bias": dist.Cauchy(), "kernel": dist.Normal()},
                                     input_shape=(4,))

    :type param: dict or ~numpyro.distributions.Distribution
    :param tuple input_shape: shape of the input taken by the neural network.
    :returns: a sampled module

    **Example**

    .. doctest::

        # NB: this example is ported from https://github.com/ctallec/pyvarinf/blob/master/main_regression.ipynb
        >>> import numpy as np; np.random.seed(0)
        >>> import tqdm
        >>> from flax import nn
        >>> from jax import jit, random
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.contrib.module import random_flax_module
        >>> from numpyro.infer import Predictive, SVI, TraceMeanField_ELBO, autoguide, init_to_feasible
        >>>
        >>> class Net(nn.Module):
        ...     def apply(self, x, n_units):
        ...         x = nn.Dense(x[..., None], features=n_units)
        ...         x = nn.relu(x)
        ...         x = nn.Dense(x, features=n_units)
        ...         x = nn.relu(x)
        ...         mean = nn.Dense(x, features=1)
        ...         rho = nn.Dense(x, features=1)
        ...         return mean.squeeze(), rho.squeeze()
        >>>
        >>> def generate_data(n_samples):
        ...     x = np.random.normal(size=n_samples)
        ...     y = np.cos(x * 3) + np.random.normal(size=n_samples) * np.abs(x) / 2
        ...     return x, y
        >>>
        >>> def model(x, y=None, batch_size=None):
        ...     module = Net.partial(n_units=32)
        ...     net = random_flax_module("nn", module, dist.Normal(0, 0.1), input_shape=())
        ...     with numpyro.plate("batch", x.shape[0], subsample_size=batch_size):
        ...         batch_x = numpyro.subsample(x, event_dim=0)
        ...         batch_y = numpyro.subsample(y, event_dim=0) if y is not None else None
        ...         mean, rho = net(batch_x)
        ...         sigma = nn.softplus(rho)
        ...         numpyro.sample("obs", dist.Normal(mean, sigma), obs=batch_y)
        >>>
        >>> n_train_data = 5000
        >>> x_train, y_train = generate_data(n_train_data)
        >>> guide = autoguide.AutoNormal(model, init_loc_fn=init_to_feasible)
        >>> svi = SVI(model, guide, numpyro.optim.Adam(5e-3), TraceMeanField_ELBO())
        >>>
        >>> batch_size = 256
        >>> n_iterations = 3000
        >>> svi_state = svi.init(random.PRNGKey(0), x_train, y_train, batch_size=batch_size)
        >>> update_fn = jit(svi.update, static_argnums=(3,))
        >>> for i in tqdm.trange(n_iterations):
        ...     svi_state, loss = update_fn(svi_state, x_train, y_train, batch_size)
        >>>
        >>> params = svi.get_params(svi_state)
        >>> n_test_data = 100
        >>> x_test, y_test = generate_data(n_test_data)
        >>> predictive = Predictive(model, guide=guide, params=params, num_samples=1000)
        >>> y_pred = predictive(random.PRNGKey(1), x_test[:100])["obs"].copy()
        >>> assert loss < 3000
        >>> assert np.sqrt(np.mean(np.square(y_test - y_pred))) < 1
    """
    nn = flax_module(name, nn_module, input_shape=input_shape)
    params = nn.args[0]
    new_params = deepcopy(params)
    with numpyro.handlers.scope(prefix=name):
        _update_params(params, new_params, prior)
    nn_new = partial(nn.func, new_params, *nn.args[1:], **nn.keywords)
    return nn_new


def random_haiku_module(name, nn_module, prior, *, input_shape=None):
    """
    A primitive to place a prior over the parameters of the Haiku module `nn_module`.

    :param str name: name of NumPyro module
    :param haiku.Module: the module to be registered with NumPyro
    :param prior: a NumPyro distribution or a Python dict with parameter names as keys and
        respective distributions as values. For example::

            net = random_haiku_module("net",
                                      haiku.transform(lambda x: hk.Linear(1)(x)),
                                      prior={"linear.b": dist.Cauchy(), "linear.w": dist.Normal()},
                                      input_shape=(4,))

    :type param: dict or ~numpyro.distributions.Distribution
    :param tuple input_shape: shape of the input taken by the neural network.
    :returns: a sampled module
    """
    nn = haiku_module(name, nn_module, input_shape=input_shape)
    params = nn.args[0]
    new_params = deepcopy(params)
    with numpyro.handlers.scope(prefix=name):
        _update_params(params, new_params, prior)
    nn_new = partial(nn.func, new_params, *nn.args[1:], **nn.keywords)
    return nn_new
