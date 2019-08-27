"""
Optimizer classes defined here are light wrappers over the corresponding optimizers
sourced from :mod:`jax.experimental.optimizers` with an interface that is better
suited for working with NumPyro inference algorithms.
"""

from typing import Callable, Tuple, TypeVar

from jax.experimental import optimizers

__all__ = [
    'Adam',
    'Adagrad',
    'Momentum',
    'RMSProp',
    'RMSPropMomentum',
    'SGD',
    'SM3',
]

_Params = TypeVar('_Params')
_OptState = TypeVar('_OptState')
_IterOptState = Tuple[int, _OptState]


class _NumpyroOptim(object):
    def __init__(self, optim_fn: Callable, *args, **kwargs) -> None:
        self.init_fn, self.update_fn, self.get_params_fn = optim_fn(*args, **kwargs)

    def init(self, params: _Params) -> _IterOptState:
        """
        Initialize the optimizer with parameters designated to be optimized.

        :param params: a collection of numpy arrays.
        :return: initial optimizer state.
        """
        opt_state = self.init_fn(params)
        return 0, opt_state

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
class Adam(_NumpyroOptim):
    def __init__(self, *args, **kwargs):
        super(Adam, self).__init__(optimizers.adam, *args, **kwargs)


@_add_doc(optimizers.adagrad)
class Adagrad(_NumpyroOptim):
    def __init__(self, *args, **kwargs):
        super(Adagrad, self).__init__(optimizers.adagrad, *args, **kwargs)


@_add_doc(optimizers.momentum)
class Momentum(_NumpyroOptim):
    def __init__(self, *args, **kwargs):
        super(Momentum, self).__init__(optimizers.momentum, *args, **kwargs)


@_add_doc(optimizers.rmsprop)
class RMSProp(_NumpyroOptim):
    def __init__(self, *args, **kwargs):
        super(RMSProp, self).__init__(optimizers.rmsprop, *args, **kwargs)


@_add_doc(optimizers.rmsprop_momentum)
class RMSPropMomentum(_NumpyroOptim):
    def __init__(self, *args, **kwargs):
        super(RMSPropMomentum, self).__init__(optimizers.rmsprop_momentum, *args, **kwargs)


@_add_doc(optimizers.sgd)
class SGD(_NumpyroOptim):
    def __init__(self, *args, **kwargs):
        super(SGD, self).__init__(optimizers.sgd, *args, **kwargs)


@_add_doc(optimizers.sm3)
class SM3(_NumpyroOptim):
    def __init__(self, *args, **kwargs):
        super(SM3, self).__init__(optimizers.sm3, *args, **kwargs)
