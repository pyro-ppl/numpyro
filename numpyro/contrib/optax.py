# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Optimizer classes defined here are light wrappers over the corresponding optimizers
sourced from :mod:`optax` with an interface that is better suited for working with
NumPyro inference algorithms.
"""
from typing import Callable, Tuple, TypeVar

import jax.numpy as jnp
import optax
from jax import lax, value_and_grad
from jax.flatten_util import ravel_pytree

_Params = TypeVar("_Params")
_OptState = TypeVar("_OptState")
_IterOptState = Tuple[int, Tuple[_Params, _OptState]]


class _OptaxWrapper:
    """
    Wrapper class for Optax transforms / optimisers.
    """

    def __init__(self, transformation: optax.GradientTransformation) -> None:
        self.transformation = transformation

    def init(self, params: _Params) -> _IterOptState:
        """
        Initialise the optimizer with the parameters to be optimized.

        :param params: A PyTree of JAX arrays.
        :return: Initial optimizer state.
        """
        opt_state = self.transformation.init(params)
        return jnp.array(0), (params, opt_state)

    def update(self, g: _Params, state: _IterOptState) -> _IterOptState:
        """
        Gradient update for the optimizer.

        :param g: Gradients information for the parameters. Should have the same
            structure as the parameters.
        :param state: The current optimizer state.
        :return: The new optimizer state after the update.
        """
        i, (params, opt_state) = state
        updates, opt_state = self.transformation.update(g, opt_state, params)
        updated_params = optax.apply_updates(params, updates)
        return i + 1, (updated_params, opt_state)

    def eval_and_update(self, fn: Callable, state: _IterOptState) -> _IterOptState:
        """
        Performs an optimization step for the objective function ``fn``.

        :param fn: The objective function.
        :param state: Current optimizer state.
        :return: A pair of the current output of the objective function and the new
            optimizer state.
        """
        params = self.get_params(state)
        out, grads = value_and_grad(fn)(params)
        return out, self.update(grads, state)

    def eval_and_stable_update(self, fn, state):
        """
        Like :meth:`eval_and_update` but when the value of the objective function or
        the gradients are not finite, we will not update the input ``state`` and will
        set the objective output to ``nan``.

        :param fn: The objective function.
        :param state: Current optimizer state.
        :return: A pair of the current output of the objective function and the new
            optimizer state.
        """
        params = self.get_params(state)
        out, grads = value_and_grad(fn)(params)
        return lax.cond(
            jnp.isfinite(out) & jnp.isfinite(ravel_pytree(grads)[0]).all(),
            lambda _: (out, self.update(grads, state)),
            lambda _: (jnp.nan, state),
            None,
        )

    def get_params(self, state):
        """
        Helper function to extract parameter values from the current optimizer state.

        :param state: The current optimizer state.
        :return: PyTree with current parameter values.
        """
        _, (params, _) = state
        return params
