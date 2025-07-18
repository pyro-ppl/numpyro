# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from copy import deepcopy
from functools import partial

import jax
from jax import random
import jax.numpy as jnp
import jax.tree_util as jtu

import numpyro
import numpyro.distributions as dist
from numpyro.primitives import mutable as numpyro_mutable

__all__ = [
    "flax_module",
    "haiku_module",
    "random_flax_module",
    "random_haiku_module",
    "nnx_module",
    "random_nnx_module",
]


def flax_module(
    name, nn_module, *args, input_shape=None, apply_rng=None, mutable=None, **kwargs
):
    """
    Declare a :mod:`~flax` style neural network inside a
    model so that its parameters are registered for optimization via
    :func:`~numpyro.primitives.param` statements.

    Given a flax ``nn_module``, in flax to evaluate the module with
    a given set of parameters, we use: ``nn_module.apply(params, x)``.
    In a NumPyro model, the pattern will be::

        net = flax_module("net", nn_module)
        y = net(x)

    or with dropout layers::

        net = flax_module("net", nn_module, apply_rng=["dropout"])
        rng_key = numpyro.prng_key()
        y = net(x, rngs={"dropout": rng_key})

    :param str name: name of the module to be registered.
    :param flax.linen.Module nn_module: a `flax` Module which has .init and .apply methods
    :param args: optional arguments to initialize flax neural network
        as an alternative to `input_shape`
    :param tuple input_shape: shape of the input taken by the
        neural network.
    :param list apply_rng: A list to indicate which extra rng _kinds_ are needed for
        ``nn_module``. For example, when ``nn_module`` includes dropout layers, we
        need to set ``apply_rng=["dropout"]``. Defaults to None, which means no extra
        rng key is needed. Please see
        `Flax Linen Intro <https://flax.readthedocs.io/en/latest/notebooks/linen_intro.html#Invoking-Modules>`_
        for more information in how Flax deals with stochastic layers like dropout.
    :param list mutable: A list to indicate mutable states of ``nn_module``. For example,
        if your module has BatchNorm layer, we will need to define ``mutable=["batch_stats"]``.
        See the above `Flax Linen Intro` tutorial for more information.
    :param kwargs: optional keyword arguments to initialize flax neural network
        as an alternative to `input_shape`
    :return: a callable with bound parameters that takes an array
        as an input and returns the neural network transformed output
        array.
    """
    try:
        import flax  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Looking like you want to use flax to declare "
            "nn modules. This is an experimental feature. "
            "You need to install `flax` to be able to use this feature. "
            "It can be installed with `pip install flax`."
        ) from e
    module_key = name + "$params"
    nn_params = numpyro.param(module_key)

    if mutable:
        nn_state = numpyro_mutable(name + "$state")
        assert nn_state is None or isinstance(nn_state, dict)
        assert (nn_state is None) == (nn_params is None)

    if nn_params is None:
        # feed in dummy data to init params
        args = (jnp.ones(input_shape),) if input_shape is not None else args
        rng_key = numpyro.prng_key()
        if rng_key is None:
            rng_key = random.key(0)
        # split rng_key into a dict of rng_kind: rng_key
        rngs = {}
        if apply_rng:
            assert isinstance(apply_rng, list)
            for kind in apply_rng:
                rng_key, subkey = random.split(rng_key)
                rngs[kind] = subkey
        rngs["params"] = rng_key

        nn_vars = flax.core.unfreeze(nn_module.init(rngs, *args, **kwargs))
        if "params" not in nn_vars:
            raise ValueError(
                "Your nn_module does not have any parameter. Currently, it is not"
                " supported in NumPyro. Please make a github issue if you need"
                " that feature."
            )
        nn_params = nn_vars["params"]
        if mutable:
            nn_state = {k: v for k, v in nn_vars.items() if k != "params"}
            assert set(mutable) == set(nn_state)
            numpyro_mutable(name + "$state", nn_state)
        # make sure that nn_params keep the same order after unflatten
        params_flat, tree_def = jax.tree.flatten(nn_params)
        nn_params = jax.tree.unflatten(tree_def, params_flat)
        numpyro.param(module_key, nn_params)

    def apply_with_state(params, *args, **kwargs):
        params = {"params": params, **nn_state}
        out, new_state = nn_module.apply(params, mutable=mutable, *args, **kwargs)
        new_state = jax.lax.stop_gradient(new_state)
        nn_state.update(**new_state)
        return out

    def apply_without_state(params, *args, **kwargs):
        return nn_module.apply({"params": params}, *args, **kwargs)

    apply_fn = apply_with_state if mutable else apply_without_state
    return partial(apply_fn, nn_params)


def haiku_module(name, nn_module, *args, input_shape=None, apply_rng=False, **kwargs):
    """
    Declare a :mod:`~haiku` style neural network inside a
    model so that its parameters are registered for optimization via
    :func:`~numpyro.primitives.param` statements.

    Given a haiku ``nn_module``, in haiku to evaluate the module with
    a given set of parameters, we use: ``nn_module.apply(params, None, x)``.
    In a NumPyro model, the pattern will be::

        net = haiku_module("net", nn_module)
        y = net(x)  # or y = net(rng_key, x)

    or with dropout layers::

        net = haiku_module("net", nn_module, apply_rng=True)
        rng_key = numpyro.prng_key()
        y = net(rng_key, x)

    :param str name: name of the module to be registered.
    :param nn_module: a `haiku` Module which has .init and .apply methods
    :type nn_module: haiku.Transformed or haiku.TransformedWithState
    :param args: optional arguments to initialize flax neural network
        as an alternative to `input_shape`
    :param tuple input_shape: shape of the input taken by the
        neural network.
    :param bool apply_rng: A flag to indicate if the returned callable requires
        an rng argument (e.g. when ``nn_module`` includes dropout layers). Defaults
        to False, which means no rng argument is needed. If this is True, the signature
        of the returned callable ``nn = haiku_module(..., apply_rng=True)`` will be
        ``nn(rng_key, x)`` (rather than ``nn(x)``).
    :param kwargs: optional keyword arguments to initialize flax neural network
        as an alternative to `input_shape`
    :return: a callable with bound parameters that takes an array
        as an input and returns the neural network transformed output
        array.
    """
    try:
        import haiku as hk  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Looking like you want to use haiku to declare "
            "nn modules. This is an experimental feature. "
            "You need to install `haiku` to be able to use this feature. "
            "It can be installed with `pip install dm-haiku`."
        ) from e

    if not apply_rng:
        nn_module = hk.without_apply_rng(nn_module)

    module_key = name + "$params"
    nn_params = numpyro.param(module_key)
    with_state = isinstance(nn_module, hk.TransformedWithState)
    if with_state:
        nn_state = numpyro_mutable(name + "$state")
        assert nn_state is None or isinstance(nn_state, dict)
        assert (nn_state is None) == (nn_params is None)

    if nn_params is None:
        args = (jnp.ones(input_shape),) if input_shape is not None else args
        # feed in dummy data to init params
        rng_key = numpyro.prng_key()
        if rng_key is None:
            rng_key = random.key(0)
        if with_state:
            nn_params, nn_state = nn_module.init(rng_key, *args, **kwargs)
            nn_state = dict(nn_state)
            numpyro_mutable(name + "$state", nn_state)
        else:
            nn_params = nn_module.init(rng_key, *args, **kwargs)
        # haiku init returns an immutable dict
        nn_params = hk.data_structures.to_mutable_dict(nn_params)
        # we cast it to a mutable one to be able to set priors for parameters
        # make sure that nn_params keep the same order after unflatten
        params_flat, tree_def = jax.tree.flatten(nn_params)
        nn_params = jax.tree.unflatten(tree_def, params_flat)
        numpyro.param(module_key, nn_params)

    def apply_with_state(params, *args, **kwargs):
        out, new_state = nn_module.apply(params, nn_state, *args, **kwargs)
        new_state = jax.lax.stop_gradient(new_state)
        nn_state.update(**new_state)
        return out

    apply_fn = apply_with_state if with_state else nn_module.apply
    return partial(apply_fn, nn_params)


# register an "empty" parameter which only stores its shape
# so that the optimizer can skip optimize this parameter, while
# it still provides shape information for priors
ParamShape = namedtuple("ParamShape", ["shape"])
jtu.register_pytree_node(
    ParamShape, lambda x: ((None,), x.shape), lambda shape, x: ParamShape(shape)
)


def _update_params(params, new_params, prior, prefix=""):
    """
    A helper to recursively set prior to new_params.
    """
    for name, item in params.items():
        flatten_name = ".".join([str(prefix), str(name)]) if prefix else str(name)
        if isinstance(item, dict):
            assert not isinstance(prior, dict) or flatten_name not in prior
            new_item = new_params[name]
            _update_params(item, new_item, prior, prefix=flatten_name)
        elif (not isinstance(prior, dict)) or flatten_name in prior:
            if isinstance(params[name], ParamShape):
                param_shape = params[name].shape
            else:
                param_shape = jnp.shape(params[name])
                params[name] = ParamShape(param_shape)
            if isinstance(prior, dict):
                d = prior[flatten_name]
            elif callable(prior) and not isinstance(prior, dist.Distribution):
                d = prior(flatten_name, param_shape)
            else:
                d = prior
            param_batch_shape = param_shape[: len(param_shape) - d.event_dim]
            # XXX: here we set all dimensions of prior to event dimensions.
            new_params[name] = numpyro.sample(
                flatten_name, d.expand(param_batch_shape).to_event()
            )


def random_flax_module(
    name,
    nn_module,
    prior,
    *args,
    input_shape=None,
    apply_rng=None,
    mutable=None,
    **kwargs,
):
    """
    A primitive to place a prior over the parameters of the Flax module `nn_module`.

    .. note::
        Parameters of a Flax module are stored in a nested dict. For example,
        the module `B` defined as follows::

            class A(flax.linen.Module):
                @flax.linen.compact
                def __call__(self, x):
                    return nn.Dense(1, use_bias=False, name='dense')(x)

            class B(flax.linen.Module):
                @flax.linen.compact
                def __call__(self, x):
                    return A(name='inner')(x)

        has parameters `{'inner': {'dense': {'kernel': param_value}}}`. In the argument
        `prior`, to specify `kernel` parameter, we join the path to it using dots:
        `prior={"inner.dense.kernel": param_prior}`.

    :param str name: name of NumPyro module
    :param flax.linen.Module: the module to be registered with NumPyro
    :param prior: a NumPyro distribution or a Python dict with parameter names as keys and
        respective distributions as values. For example::

            net = random_flax_module("net",
                                     flax.linen.Dense(features=1),
                                     prior={"bias": dist.Cauchy(), "kernel": dist.Normal()},
                                     input_shape=(4,))

        Alternatively, we can use a callable. For example the following are equivalent::

            prior=(lambda name, shape: dist.Cauchy() if name == "bias" else dist.Normal())
            prior={"bias": dist.Cauchy(), "kernel": dist.Normal()}

    :type prior: dict, ~numpyro.distributions.Distribution or callable
    :param args: optional arguments to initialize flax neural network
        as an alternative to `input_shape`
    :param tuple input_shape: shape of the input taken by the neural network.
    :param list apply_rng: A list to indicate which extra rng _kinds_ are needed for
        ``nn_module``. For example, when ``nn_module`` includes dropout layers, we
        need to set ``apply_rng=["dropout"]``. Defaults to None, which means no extra
        rng key is needed. Please see
        `Flax Linen Intro <https://flax.readthedocs.io/en/latest/notebooks/linen_intro.html#Invoking-Modules>`_
        for more information in how Flax deals with stochastic layers like dropout.
    :param list mutable: A list to indicate mutable states of ``nn_module``. For example,
        if your module has BatchNorm layer, we will need to define ``mutable=["batch_stats"]``.
        See the above `Flax Linen Intro` tutorial for more information.
    :param kwargs: optional keyword arguments to initialize flax neural network
        as an alternative to `input_shape`
    :returns: a sampled module

    **Example**

    .. doctest::

        # NB: this example is ported from https://github.com/ctallec/pyvarinf/blob/master/main_regression.ipynb
        >>> import numpy as np; np.random.seed(0)
        >>> import tqdm
        >>> from flax import linen as nn
        >>> from jax import jit, random
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.contrib.module import random_flax_module
        >>> from numpyro.infer import Predictive, SVI, TraceMeanField_ELBO, autoguide, init_to_feasible
        ...
        >>> class Net(nn.Module):
        ...     n_units: int
        ...
        ...     @nn.compact
        ...     def __call__(self, x):
        ...         x = nn.Dense(self.n_units)(x[..., None])
        ...         x = nn.relu(x)
        ...         x = nn.Dense(self.n_units)(x)
        ...         x = nn.relu(x)
        ...         mean = nn.Dense(1)(x)
        ...         rho = nn.Dense(1)(x)
        ...         return mean.squeeze(), rho.squeeze()
        ...
        >>> def generate_data(n_samples):
        ...     x = np.random.normal(size=n_samples)
        ...     y = np.cos(x * 3) + np.random.normal(size=n_samples) * np.abs(x) / 2
        ...     return x, y
        ...
        >>> def model(x, y=None, batch_size=None):
        ...     module = Net(n_units=32)
        ...     net = random_flax_module("nn", module, dist.Normal(0, 0.1), input_shape=())
        ...     with numpyro.plate("batch", x.shape[0], subsample_size=batch_size):
        ...         batch_x = numpyro.subsample(x, event_dim=0)
        ...         batch_y = numpyro.subsample(y, event_dim=0) if y is not None else None
        ...         mean, rho = net(batch_x)
        ...         sigma = nn.softplus(rho)
        ...         numpyro.sample("obs", dist.Normal(mean, sigma), obs=batch_y)
        ...
        >>> n_train_data = 5000
        >>> x_train, y_train = generate_data(n_train_data)
        >>> guide = autoguide.AutoNormal(model, init_loc_fn=init_to_feasible)
        >>> svi = SVI(model, guide, numpyro.optim.Adam(5e-3), TraceMeanField_ELBO())
        >>> n_iterations = 3000
        >>> svi_result = svi.run(random.PRNGKey(0), n_iterations, x_train, y_train, batch_size=256)
        >>> params, losses = svi_result.params, svi_result.losses
        >>> n_test_data = 100
        >>> x_test, y_test = generate_data(n_test_data)
        >>> predictive = Predictive(model, guide=guide, params=params, num_samples=1000)
        >>> y_pred = predictive(random.PRNGKey(1), x_test[:100])["obs"].copy()
        >>> assert losses[-1] < 3000
        >>> assert np.sqrt(np.mean(np.square(y_test - y_pred))) < 1
    """
    nn = flax_module(
        name,
        nn_module,
        *args,
        input_shape=input_shape,
        apply_rng=apply_rng,
        mutable=mutable,
        **kwargs,
    )
    params = nn.args[0]
    new_params = deepcopy(params)
    with numpyro.handlers.scope(prefix=name):
        _update_params(params, new_params, prior)
    nn_new = partial(nn.func, new_params, *nn.args[1:], **nn.keywords)
    return nn_new


def random_haiku_module(
    name, nn_module, prior, *args, input_shape=None, apply_rng=False, **kwargs
):
    """
    A primitive to place a prior over the parameters of the Haiku module `nn_module`.

    :param str name: name of NumPyro module
    :param nn_module: the module to be registered with NumPyro
    :type nn_module: haiku.Transformed or haiku.TransformedWithState
    :param prior: a NumPyro distribution or a Python dict with parameter names as keys and
        respective distributions as values. For example::

            net = random_haiku_module("net",
                                      haiku.transform(lambda x: hk.Linear(1)(x)),
                                      prior={"linear.b": dist.Cauchy(), "linear.w": dist.Normal()},
                                      input_shape=(4,))

        Alternatively, we can use a callable. For example the following are equivalent::

            prior=(lambda name, shape: dist.Cauchy() if name.startswith("b") else dist.Normal())
            prior={"bias": dist.Cauchy(), "kernel": dist.Normal()}

    :type prior: dict, ~numpyro.distributions.Distribution or callable
    :param args: optional arguments to initialize flax neural network
        as an alternative to `input_shape`
    :param tuple input_shape: shape of the input taken by the neural network.
    :param bool apply_rng: A flag to indicate if the returned callable requires
        an rng argument (e.g. when ``nn_module`` includes dropout layers). Defaults
        to False, which means no rng argument is needed. If this is True, the signature
        of the returned callable ``nn = haiku_module(..., apply_rng=True)`` will be
        ``nn(rng_key, x)`` (rather than ``nn(x)``).
    :param kwargs: optional keyword arguments to initialize flax neural network
        as an alternative to `input_shape`
    :returns: a sampled module
    """
    nn = haiku_module(
        name, nn_module, *args, input_shape=input_shape, apply_rng=apply_rng, **kwargs
    )
    params = nn.args[0]
    new_params = deepcopy(params)
    with numpyro.handlers.scope(prefix=name):
        _update_params(params, new_params, prior)
    nn_new = partial(nn.func, new_params, *nn.args[1:], **nn.keywords)
    return nn_new


def nnx_module(name, nn_module):
    """
    Declare a :mod:`~flax.nnx` style neural network inside a
    model so that its parameters are registered for optimization via
    :func:`~numpyro.primitives.param` statements.

    Given a flax NNX ``nn_module``, to evaluate the module, we directly call it.
    In a NumPyro model, the pattern will be::

        # Eager initialization outside the model
        module = nn_module(...)

        # Inside the model
        net = nnx_module("net", module)
        y = net(x)

    :param str name: name of the module to be registered.
    :param flax.nnx.Module nn_module: a pre-initialized `flax nnx` Module instance.
    :return: a callable that takes an array as an input and returns
        the neural network transformed output array.
    """
    try:
        from flax import nnx
    except ImportError as e:
        raise ImportError(
            "Looking like you want to use flax.nnx to declare "
            "nn modules. This is an experimental feature. "
            "You need to install the latest version of `flax` to use this feature. "
            "It can be installed with `pip install git+https://github.com/google/flax.git`."
        ) from e

    graph_def, eager_params_state, eager_other_state = nnx.split(
        nn_module, nnx.Param, nnx.Not(nnx.Param)
    )

    eager_params_state_dict = nnx.to_pure_dict(eager_params_state)

    module_params = None
    if eager_params_state:
        module_params = numpyro.param(name + "$params")
    if module_params is None:
        module_params = numpyro.param(name + "$params", eager_params_state_dict)

    eager_other_state_dict = nnx.to_pure_dict(eager_other_state)

    mutable_holder = None
    if eager_other_state_dict:
        mutable_holder = numpyro_mutable(name + "$state")
    if mutable_holder is None:
        mutable_holder = numpyro_mutable(
            name + "$state", {"state": eager_other_state_dict}
        )

    def apply_fn(params, *call_args, **call_kwargs):
        params_state = eager_params_state

        if params:
            nnx.replace_by_pure_dict(params_state, params)

        mutable_state = eager_other_state
        if mutable_holder:
            nnx.replace_by_pure_dict(mutable_state, mutable_holder["state"])

        model = nnx.merge(graph_def, params_state, mutable_state)

        model_call = model(*call_args, **call_kwargs)

        if mutable_holder:
            _, _, new_mutable_state = nnx.split(model, nnx.Param, nnx.Not(nnx.Param))
            new_mutable_state = jax.lax.stop_gradient(new_mutable_state)
            mutable_holder["state"] = nnx.to_pure_dict(new_mutable_state)

        return model_call

    return partial(apply_fn, module_params)


def random_nnx_module(
    name,
    nn_module,
    prior,
    scope_divider="/",
):
    """
    A primitive to create a random :mod:`~flax.nnx` style neural network
    which can be used in MCMC samplers. The parameters of the neural network
    will be sampled from ``prior``.

    :param str name: name of the module to be registered.
    :param flax.nnx.Module nn_module: a pre-initialized `flax nnx` Module instance.
    :param prior: a distribution or a dict of distributions or a callable.
        If it is a distribution, all parameters will be sampled from the same
        distribution. If it is a dict, it maps parameter names to distributions.
        If it is a callable, it takes parameter name and parameter shape as
        inputs and returns a distribution. For example::

            class Linear(nnx.Module):
                def __init__(self, din, dout, *, rngs):
                    self.w = nnx.Param(jax.random.uniform(rngs.params(), (din, dout)))
                    self.b = nnx.Param(jnp.zeros((dout,)))

                def __call__(self, x):
                    return x @ self.w + self.b

            # Eager initialization
            linear = Linear(din=4, dout=1, rngs=nnx.Rngs(params=random.PRNGKey(0)))
            net = random_nnx_module("net", linear, prior={"w": dist.Normal(), "b": dist.Cauchy()})

        Alternatively, we can use a callable. For example the following are equivalent::

            prior=(lambda name, shape: dist.Cauchy() if name.endswith("b") else dist.Normal())
            prior={"w": dist.Normal(), "b": dist.Cauchy()}

    :param str scope_divider: the divider to use for the nnx name in the scope effect handler. Defaults to "/".

    :return: a callable that takes an array as an input and returns
        the neural network transformed output array.
    """

    nn = nnx_module(name, nn_module)

    apply_fn = nn.func
    params = nn.args[0]
    other_args = nn.args[1:]
    keywords = nn.keywords

    new_params = deepcopy(params)

    with numpyro.handlers.scope(prefix=name, divider=scope_divider):
        _update_params(params, new_params, prior)

    return partial(apply_fn, new_params, *other_args, **keywords)


def eqx_module(name, nn_module):
    """
    Declare an :mod:`equinox` style neural network inside a
    model so that its parameters are registered for optimization via
    :func:`~numpyro.primitives.param` statements.

    Given an equinox ``nn_module``, to evaluate the module, we directly call it.
    In a NumPyro model, the pattern will be::

        # Eager initialization outside the model
        module = nn_module(...)

        # Inside the model
        net = eqx_module("net", module)
        y = jax.vmap(net)(x)

    In the case of stateful computation, the pattern is the following::

        # Eager initialization outside the model
        module, eager_state = eqx.nn.make_with_state(nn_module)(...)

        # Inside the model
        net = eqx_module("net", module)
        mutable_holder = numpyro_mutable("net$state", {"state": eager_state})
        batched_net = jax.vmap(net, in_axes=(0,None), out_axes=(0,None), axis_name='batch')
        y, new_state = batched_net(x, mutable_holder['state'])
        mutable_holder['state'] = new_state

    :param str name: name of the module to be registered.
    :param eqx.Module nn_module: a pre-initialized `equinox` Module instance.
    :return: a callable that takes an array as an input and returns
        the neural network transformed output array.
    """
    try:
        import equinox as eqx
    except ImportError as e:
        raise ImportError(
            "Looking like you want to use equinox to declare "
            "nn modules. This is an experimental feature. "
            "You need to install the latest version of `equinox` to use this feature. "
            "It can be installed with `pip install git+https://github.com/patrick-kidger/equinox.git`."
        ) from e

    params, static = eqx.partition(nn_module, filter_spec=eqx.is_inexact_array)
    params = numpyro.param(name + "$params", lambda _: params)
    nn_module = eqx.combine(params, static)
    return nn_module


def random_eqx_module(name, nn_module, prior, scope_divider="/"):
    """
    A primitive to create a random :mod:`equinox` style neural network
    which can be used in MCMC samplers. The parameters of the neural network
    will be sampled from ``prior``.

    For supplying a prior dictionary, the dictionary keys are based on their jax key path.
    To see the jax key paths for all of the leaves in your pytree model, you can run:

        key_paths = [jtu.keystr(path)[1:] for path, _ in jtu.tree_leaves_with_path(model_instance)]

    :param str name: name of the module to be registered.
    :param eqx.Module nn_module: a pre-initialized `equinox` Module instance.
    :param prior: a distribution or a dict of distributions or a callable.
        If it is a distribution, all parameters will be sampled from the same
        distribution. If it is a dict, it maps parameter names to distributions.
        If it is a callable, it takes parameter name and parameter shape as
        inputs and returns a distribution. For example::

            class Linear(eqx.Module):
                weight: jax.Array
                bias: jax.Array

                def __init__(self, in_size, out_size, key):
                    wkey, bkey = jax.random.split(key)
                    self.weight = jax.random.normal(wkey, (out_size, in_size))
                    self.bias = jax.random.normal(bkey, (out_size,))

                def __call__(self, x):
                    return self.weight @ x + self.bias

            # Eager initialization
            linear = Linear(in_features=3, out_features=1, key=random.PRNGKey(0))
            nn_priors = {"weight": dist.Normal(), "bias": dist.Cauchy()}
            net = random_eqx_module("net", linear, prior=nn_priors)

        Alternatively, we can use a callable. For example the following are equivalent::

            prior=(lambda name, shape: dist.Cauchy() if name == 'bias' else dist.Normal())
            prior={"weight": dist.Normal(), "bias": dist.Cauchy()}

    :param str scope_divider: the divider to use for the nnx name in the scope effect handler. Defaults to "/".

    :return: a callable that takes an array as an input and returns
        the neural network transformed output array.
    """
    try:
        import equinox as eqx
    except ImportError as e:
        raise ImportError(
            "Looking like you want to use equinox to declare "
            "nn modules. This is an experimental feature. "
            "You need to install the latest version of `equinox` to use this feature. "
            "It can be installed with `pip install git+https://github.com/patrick-kidger/equinox.git`."
        ) from e

    nn = eqx_module(name, nn_module)
    params, static = eqx.partition(nn, filter_spec=eqx.is_inexact_array)
    params_dict = eqx_to_dict(params)
    new_params = deepcopy(params_dict)
    with numpyro.handlers.scope(prefix=name, divider=scope_divider):
        _update_params(params_dict, new_params, prior)

    return eqx.combine(eqx_from_dict(new_params, tree=params), static)


def eqx_to_dict(tree):
    out = {}

    def to_dict_impl(path, leaf):
        out[jtu.keystr(path)[1:]] = leaf

    jax.tree.map_with_path(to_dict_impl, tree)
    return out


def eqx_from_dict(data: dict, tree):
    def from_dict_impl(path, _):
        return data[jtu.keystr(path)[1:]]

    return jax.tree.map_with_path(from_dict_impl, tree)
