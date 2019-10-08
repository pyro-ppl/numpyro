from functools import namedtuple
import os
import warnings

import jax
from jax import random, value_and_grad

from numpyro.distributions import constraints
from numpyro.distributions.transforms import biject_to
from numpyro.handlers import seed, trace
from numpyro.infer.util import transform_fn

SVIState = namedtuple('SVIState', ['optim_state', 'rng'])
"""
A :func:`~collections.namedtuple` consisting of the following fields:
 - **optim_state** - current optimizer's state.
 - **rng** - random number generator seed used for the iteration.
"""


def svi(model, guide, loss, optim, **static_kwargs):
    """
    Stochastic Variational Inference given an ELBO loss objective.

    :param model: Python callable with Pyro primitives for the model.
    :param guide: Python callable with Pyro primitives for the guide
        (recognition network).
    :param loss: ELBO loss, i.e. negative Evidence Lower Bound, to minimize.
    :param optim: an instance of :class:`~numpyro.optim._NumpyroOptim`.
    :param **`static_kwargs`: static arguments for the model / guide, i.e. arguments
        that remain constant during fitting.
    :return: tuple of `(init_fn, update_fn, evaluate)`.
    """
    warnings.warn("This interface to SVI is deprecated and will be removed in the "
                  "next version. Please use `numpyro.infer.svi.SVI` instead.",
                  DeprecationWarning)
    constrain_fn = None

    def init_fn(rng, *args, **kwargs):
        """

        :param jax.random.PRNGKey rng: random number generator seed.
        :param args: arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: keyword arguments to the model / guide (these can possibly vary
            during the course of fitting).
        :return: tuple containing initial :data:`SVIState`, and `get_params`, a callable
            that transforms unconstrained parameter values from the optimizer to the
            specified constrained domain
        """
        nonlocal constrain_fn

        rng, model_seed, guide_seed = random.split(rng, 3)
        model_init = seed(model, model_seed)
        guide_init = seed(guide, guide_seed)
        guide_trace = trace(guide_init).get_trace(*args, **kwargs, **static_kwargs)
        model_trace = trace(model_init).get_trace(*args, **kwargs, **static_kwargs)
        params = {}
        inv_transforms = {}
        # NB: params in model_trace will be overwritten by params in guide_trace
        for site in list(model_trace.values()) + list(guide_trace.values()):
            if site['type'] == 'param':
                constraint = site['kwargs'].pop('constraint', constraints.real)
                transform = biject_to(constraint)
                inv_transforms[site['name']] = transform
                params[site['name']] = transform.inv(site['value'])

        constrain_fn = jax.partial(transform_fn, inv_transforms)
        return SVIState(optim.init(params), rng), get_params

    def get_params(svi_state):
        """
        Gets values at `param` sites of the `model` and `guide`.

        :param svi_state: current state of the optimizer.
        """
        params = constrain_fn(optim.get_params(svi_state.optim_state))
        return params

    def update_fn(svi_state, *args, **kwargs):
        """
        Take a single step of SVI (possibly on a batch / minibatch of data),
        using the optimizer.

        :param svi_state: current state of SVI.
        :param args: arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: keyword arguments to the model / guide (these can possibly vary
            during the course of fitting).
        :return: tuple of `(svi_state, loss)`.
        """
        rng, rng_seed = random.split(svi_state.rng)
        params = optim.get_params(svi_state.optim_state)
        loss_val, grads = value_and_grad(
            lambda x: loss.loss(rng_seed, constrain_fn(x), model, guide,
                                *args, **kwargs, **static_kwargs))(params)
        optim_state = optim.update(grads, svi_state.optim_state)
        return SVIState(optim_state, rng), loss_val

    def evaluate(svi_state, *args, **kwargs):
        """
        Take a single step of SVI (possibly on a batch / minibatch of data).

        :param svi_state: current state of SVI.
        :param args: arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: keyword arguments to the model / guide (these can possibly vary
            during the course of fitting).
        :return: evaluate ELBO loss given the current parameter values
            (held within `svi_state.optim_state`).
        """
        # we split to have the same seed as `update_fn` given an svi_state
        _, rng_seed = random.split(svi_state.rng)
        params = get_params(svi_state)
        return loss.loss(rng_seed, params, model, guide, *args, **kwargs, **static_kwargs)

    # Make local functions visible from the global scope once
    # `svi` is called for sphinx doc generation.
    if 'SPHINX_BUILD' in os.environ:
        svi.init_fn = init_fn
        svi.update_fn = update_fn
        svi.evaluate = evaluate

    return init_fn, update_fn, evaluate


# ========= Higher Level API ========= #


class SVI(object):
    """
    Stochastic Variational Inference given an ELBO loss objective.

    :param model: Python callable with Pyro primitives for the model.
    :param guide: Python callable with Pyro primitives for the guide
        (recognition network).
    :param loss: ELBO loss, i.e. negative Evidence Lower Bound, to minimize.
    :param optim: an instance of :class:`~numpyro.optim._NumpyroOptim`.
    :param static_kwargs: static arguments for the model / guide, i.e. arguments
        that remain constant during fitting.
    :return: tuple of `(init_fn, update_fn, evaluate)`.
    """
    def __init__(self, model, guide, loss, optim, **static_kwargs):
        self.model = model
        self.guide = guide
        self.loss = loss
        self.optim = optim
        self.static_kwargs = static_kwargs
        self.constrain_fn = None

    def init(self, rng, *args, **kwargs):
        """

        :param jax.random.PRNGKey rng: random number generator seed.
        :param args: arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: keyword arguments to the model / guide (these can possibly vary
            during the course of fitting).
        :return: tuple containing initial :data:`SVIState`, and `get_params`, a callable
            that transforms unconstrained parameter values from the optimizer to the
            specified constrained domain
        """
        rng, model_seed, guide_seed = random.split(rng, 3)
        model_init = seed(self.model, model_seed)
        guide_init = seed(self.guide, guide_seed)
        guide_trace = trace(guide_init).get_trace(*args, **kwargs, **self.static_kwargs)
        model_trace = trace(model_init).get_trace(*args, **kwargs, **self.static_kwargs)
        params = {}
        inv_transforms = {}
        # NB: params in model_trace will be overwritten by params in guide_trace
        for site in list(model_trace.values()) + list(guide_trace.values()):
            if site['type'] == 'param':
                constraint = site['kwargs'].pop('constraint', constraints.real)
                transform = biject_to(constraint)
                inv_transforms[site['name']] = transform
                params[site['name']] = transform.inv(site['value'])

        self.constrain_fn = jax.partial(transform_fn, inv_transforms)
        return SVIState(self.optim.init(params), rng)

    def get_params(self, svi_state):
        """
        Gets values at `param` sites of the `model` and `guide`.

        :param svi_state: current state of the optimizer.
        """
        params = self.constrain_fn(self.optim.get_params(svi_state.optim_state))
        return params

    def update(self, svi_state, *args, **kwargs):
        """
        Take a single step of SVI (possibly on a batch / minibatch of data),
        using the optimizer.

        :param svi_state: current state of SVI.
        :param args: arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: keyword arguments to the model / guide (these can possibly vary
            during the course of fitting).
        :return: tuple of `(svi_state, loss)`.
        """
        rng, rng_seed = random.split(svi_state.rng)
        params = self.optim.get_params(svi_state.optim_state)
        loss_val, grads = value_and_grad(
            lambda x: self.loss.loss(rng_seed, self.constrain_fn(x), self.model, self.guide,
                                     *args, **kwargs, **self.static_kwargs))(params)
        optim_state = self.optim.update(grads, svi_state.optim_state)
        return SVIState(optim_state, rng), loss_val

    def evaluate(self, svi_state, *args, **kwargs):
        """
        Take a single step of SVI (possibly on a batch / minibatch of data).

        :param svi_state: current state of SVI.
        :param args: arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: keyword arguments to the model / guide.
        :return: evaluate ELBO loss given the current parameter values
            (held within `svi_state.optim_state`).
        """
        # we split to have the same seed as `update_fn` given an svi_state
        _, rng_seed = random.split(svi_state.rng)
        params = self.get_params(svi_state)
        return self.loss.loss(rng_seed, params, self.model, self.guide,
                              *args, **kwargs, **self.static_kwargs)
