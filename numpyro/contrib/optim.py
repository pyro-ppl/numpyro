# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
This module provides a wrapper for Optax optimizers so that they can be used with
NumPyro inference algorithms.
"""

from typing import Tuple, TypeVar

import optax

from numpyro.optim import _NumPyroOptim

_Params = TypeVar("_Params")
_State = Tuple[_Params, optax.OptState]


def optax_to_numpyro(transformation: optax.GradientTransformation) -> _NumPyroOptim:
    """
    This function produces a ``numpyro.optim._NumPyroOptim`` instance from an
    ``optax.GradientTransformation`` so that it can be used with
    ``numpyro.infer.svi.SVI``. It is a lightweight wrapper that recreates the
    ``(init_fn, update_fn, get_params_fn)`` interface defined by
    :mod:`jax.experimental.optimizers`.

    :param transformation: An ``optax.GradientTransformation`` instance to wrap.
    :return: An instance of ``numpyro.optim._NumPyroOptim`` wrapping the supplied
        Optax optimizer.
    """

    def init_fn(params: _Params) -> _State:
        opt_state = transformation.init(params)
        return params, opt_state

    def update_fn(step, grads: _Params, state: _State) -> _State:
        params, opt_state = state
        updates, opt_state = transformation.update(grads, opt_state, params)
        updated_params = optax.apply_updates(params, updates)
        return updated_params, opt_state

    def get_params_fn(state: _State) -> _Params:
        params, _ = state
        return params

    return _NumPyroOptim(lambda x, y, z: (x, y, z), init_fn, update_fn, get_params_fn)
