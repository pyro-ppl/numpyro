# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Optimizer classes defined here are light wrappers over the corresponding optimizers
sourced from :mod:`jax.experimental.optimizers` with an interface that is better
suited for working with NumPyro inference algorithms.
"""

from collections import namedtuple
from typing import Callable, Tuple, TypeVar

from jax import value_and_grad
from jax.experimental import optimizers
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
from jax.scipy.optimize import minimize
from jax.tree_util import register_pytree_node, tree_map

__all__ = [
    'Adam',
    'Adagrad',
    'ClippedAdam',
    'Minimize',
    'Momentum',
    'RMSProp',
    'RMSPropMomentum',
    'SGD',
    'SM3',
]

_Params = TypeVar('_Params')
_OptState = TypeVar('_OptState')
_IterOptState = Tuple[int, _OptState]


class _NumPyroOptim(object):
    def __init__(self, optim_fn: Callable, *args, **kwargs) -> None:
        self.init_fn, self.update_fn, self.get_params_fn = optim_fn(*args, **kwargs)

    def init(self, params: _Params) -> _IterOptState:
        """
        Initialize the optimizer with parameters designated to be optimized.

        :param params: a collection of numpy arrays.
        :return: initial optimizer state.
        """
        opt_state = self.init_fn(params)
        return jnp.array(0), opt_state

    def update(self, g: _Params, state: _IterOptState) -> _IterOptState:
        """
        Gradient update for the optimizer.

        :param g: gradient information for parameters.
        :param state: current optimizer state.
        :return: new optimizer state after the update.
        """
        i, opt_state = state
        opt_state = self.update_fn(i, g, opt_state)
        return i + 1, opt_state

    def eval_and_update(self, fn: Callable, state: _IterOptState) -> _IterOptState:
        """
        Performs an optimization step for the objective function `fn`.
        For most optimizers, the update is performed based on the gradient
        of the objective function w.r.t. the current state. However, for
        some optimizers such as :class:`Minimize`, the update is performed
        by reevaluating the function multiple times to get optimal
        parameters.

        :param fn: objective function.
        :param state: current optimizer state.
        :return: a pair of the output of objective function and the new optimizer state.
        """
        params = self.get_params(state)
        out, grads = value_and_grad(fn)(params)
        return out, self.update(grads, state)

    def get_params(self, state: _IterOptState) -> _Params:
        """
        Get current parameter values.

        :param state: current optimizer state.
        :return: collection with current value for parameters.
        """
        _, opt_state = state
        return self.get_params_fn(opt_state)


def _add_doc(fn):
    def _wrapped(cls):
        cls.__doc__ = 'Wrapper class for the JAX optimizer: :func:`~jax.experimental.optimizers.{}`'\
            .format(fn.__name__)
        return cls

    return _wrapped


@_add_doc(optimizers.adam)
class Adam(_NumPyroOptim):
    def __init__(self, *args, **kwargs):
        super(Adam, self).__init__(optimizers.adam, *args, **kwargs)


class ClippedAdam(_NumPyroOptim):
    """
    :class:`~numpyro.optim.Adam` optimizer with gradient clipping.

    :param float clip_norm: All gradient values will be clipped between
        `[-clip_norm, clip_norm]`.

    **Reference:**

    `A Method for Stochastic Optimization`, Diederik P. Kingma, Jimmy Ba
    https://arxiv.org/abs/1412.6980
    """
    def __init__(self, *args, clip_norm=10., **kwargs):
        self.clip_norm = clip_norm
        super(ClippedAdam, self).__init__(optimizers.adam, *args, **kwargs)

    def update(self, g, state):
        i, opt_state = state
        # clip norm
        g = tree_map(lambda g_: jnp.clip(g_, a_min=-self.clip_norm, a_max=self.clip_norm), g)
        opt_state = self.update_fn(i, g, opt_state)
        return i + 1, opt_state


@_add_doc(optimizers.adagrad)
class Adagrad(_NumPyroOptim):
    def __init__(self, *args, **kwargs):
        super(Adagrad, self).__init__(optimizers.adagrad, *args, **kwargs)


@_add_doc(optimizers.momentum)
class Momentum(_NumPyroOptim):
    def __init__(self, *args, **kwargs):
        super(Momentum, self).__init__(optimizers.momentum, *args, **kwargs)


@_add_doc(optimizers.rmsprop)
class RMSProp(_NumPyroOptim):
    def __init__(self, *args, **kwargs):
        super(RMSProp, self).__init__(optimizers.rmsprop, *args, **kwargs)


@_add_doc(optimizers.rmsprop_momentum)
class RMSPropMomentum(_NumPyroOptim):
    def __init__(self, *args, **kwargs):
        super(RMSPropMomentum, self).__init__(optimizers.rmsprop_momentum, *args, **kwargs)


@_add_doc(optimizers.sgd)
class SGD(_NumPyroOptim):
    def __init__(self, *args, **kwargs):
        super(SGD, self).__init__(optimizers.sgd, *args, **kwargs)


@_add_doc(optimizers.sm3)
class SM3(_NumPyroOptim):
    def __init__(self, *args, **kwargs):
        super(SM3, self).__init__(optimizers.sm3, *args, **kwargs)


# TODO: currently, jax.scipy.optimize.minimize only supports 1D input,
# so we need to add the following mechanism to transform params to flat_params
# and pass `unravel_fn` arround.
# When arbitrary pytree is supported in JAX, we can just simply use
# identity functions for `init_fn` and `get_params`.
_MinimizeState = namedtuple("MinimizeState", ["flat_params", "unravel_fn"])
register_pytree_node(
    _MinimizeState,
    lambda state: ((state.flat_params,), (state.unravel_fn,)),
    lambda data, xs: _MinimizeState(xs[0], data[0]))


def _minimize_wrapper():
    def init_fn(params):
        flat_params, unravel_fn = ravel_pytree(params)
        return _MinimizeState(flat_params, unravel_fn)

    def update_fn(i, grad_tree, opt_state):
        # we don't use update_fn in Minimize, so let it do nothing
        return opt_state

    def get_params(opt_state):
        flat_params, unravel_fn = opt_state
        return unravel_fn(flat_params)

    return init_fn, update_fn, get_params


class Minimize(_NumPyroOptim):
    """
    Wrapper class for the JAX minimizer: :func:`~jax.scipy.optimize.minimize`.

    .. warnings: This optimizer is intended to be used with static guides such
        as empty guides (maximum likelihood estimate), delta guides (MAP estimate),
        or :class:`~numpyro.infer.autoguide.AutoLaplaceApproximation`.
        Using this in stochastic setting is either expensive or hard to converge.

    **Example:**

    .. doctest::

        >>> from numpy.testing import assert_allclose
        >>> from jax import random
        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.infer import SVI, Trace_ELBO
        >>> from numpyro.infer.autoguide import AutoLaplaceApproximation

        >>> def model(x, y):
        ...     a = numpyro.sample("a", dist.Normal(0, 1))
        ...     b = numpyro.sample("b", dist.Normal(0, 1))
        ...     with numpyro.plate("N", y.shape[0]):
        ...         numpyro.sample("obs", dist.Normal(a + b * x, 0.1), obs=y)

        >>> x = jnp.linspace(0, 10, 100)
        >>> y = 3 * x + 2
        >>> optimizer = numpyro.optim.Minimize()
        >>> guide = AutoLaplaceApproximation(model)
        >>> svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
        >>> init_state = svi.init(random.PRNGKey(0), x, y)
        >>> optimal_state, loss = svi.update(init_state, x, y)
        >>> params = svi.get_params(optimal_state)  # get guide's parameters
        >>> quantiles = guide.quantiles(params, 0.5)  # get means of posterior samples
        >>> assert_allclose(quantiles["a"], 2., atol=1e-3)
        >>> assert_allclose(quantiles["b"], 3., atol=1e-3)
    """
    def __init__(self, method="BFGS", **kwargs):
        super().__init__(_minimize_wrapper)
        self._method = method
        self._kwargs = kwargs

    def eval_and_update(self, fn: Callable, state: _IterOptState) -> _IterOptState:
        i, (flat_params, unravel_fn) = state
        results = minimize(lambda x: fn(unravel_fn(x)), flat_params, (),
                           method=self._method, **self._kwargs)
        flat_params, out = results.x, results.fun
        state = (i + 1, _MinimizeState(flat_params, unravel_fn))
        return out, state
