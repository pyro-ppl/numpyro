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


def _seed(model, guide, rng):
    model_seed, guide_seed = random.split(rng, 2)
    model_init = seed(model, model_seed)
    guide_init = seed(guide, guide_seed)
    return model_init, guide_init


def svi(model, guide, loss, optim, **kwargs):
    """
    Stochastic Variational Inference given an ELBo loss objective.

    :param model: Python callable with Pyro primitives for the model.
    :param guide: Python callable with Pyro primitives for the guide
        (recognition network).
    :param loss: ELBo loss, i.e. negative Evidence Lower Bound, to minimize.
    :param optim: an instance of :class:`~numpyro.optim._NumpyroOptim`.
    :param `**kwargs`: static arguments for the model / guide, i.e. arguments
        that remain constant during fitting.
    :return: tuple of `(init_fn, update_fn, evaluate)`.
    """
    warnings.warn("This interface to SVI is deprecated and will be removed in the "
                  "next version. Please use `numpyro.svi.SVI` instead.",
                  DeprecationWarning)
    constrain_fn = None

    def init_fn(rng, model_args=(), guide_args=()):
        """

        :param jax.random.PRNGKey rng: random number generator seed.
        :param tuple model_args: arguments to the model (these can possibly vary during
            the course of fitting).
        :param tuple guide_args: arguments to the guide (these can possibly vary during
            the course of fitting).
        :return: tuple containing initial :data:`SVIState`, and `get_params`, a callable
            that transforms unconstrained parameter values from the optimizer to the
            specified constrained domain
        """
        nonlocal constrain_fn

        assert isinstance(model_args, tuple)
        assert isinstance(guide_args, tuple)
        rng, rng_seed = random.split(rng)
        model_init, guide_init = _seed(model, guide, rng_seed)
        guide_trace = trace(guide_init).get_trace(*guide_args, **kwargs)
        model_trace = trace(model_init).get_trace(*model_args, **kwargs)
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

    def update_fn(svi_state, model_args=(), guide_args=()):
        """
        Take a single step of SVI (possibly on a batch / minibatch of data),
        using the optimizer.

        :param svi_state: current state of SVI.
        :param tuple model_args: dynamic arguments to the model.
        :param tuple guide_args: dynamic arguments to the guide.
        :return: tuple of `(svi_state, loss)`.
        """
        rng, rng_seed = random.split(svi_state.rng)
        params = optim.get_params(svi_state.optim_state)
        loss_val, grads = value_and_grad(
            lambda x: loss(rng_seed, constrain_fn(x), model, guide, model_args, guide_args, kwargs))(params)
        optim_state = optim.update(grads, svi_state.optim_state)
        return SVIState(optim_state, rng), loss_val

    def evaluate(svi_state, model_args=(), guide_args=()):
        """
        Take a single step of SVI (possibly on a batch / minibatch of data).

        :param svi_state: current state of SVI.
        :param tuple model_args: arguments to the model (these can possibly vary during
            the course of fitting).
        :param tuple guide_args: arguments to the guide (these can possibly vary during
            the course of fitting).
        :return: evaluate ELBo loss given the current parameter values
            (held within `svi_state.optim_state`).
        """
        # we split to have the same seed as `update_fn` given an svi_state
        _, rng_seed = random.split(svi_state.rng)
        params = get_params(svi_state)
        return loss(rng_seed, params, model, guide, model_args, guide_args, kwargs)

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
    Stochastic Variational Inference given an ELBo loss objective.

    :param model: Python callable with Pyro primitives for the model.
    :param guide: Python callable with Pyro primitives for the guide
        (recognition network).
    :param loss: ELBo loss, i.e. negative Evidence Lower Bound, to minimize.
    :param optim: an instance of :class:`~numpyro.optim._NumpyroOptim`.
    :param `**kwargs`: static arguments for the model / guide, i.e. arguments
        that remain constant during fitting.
    :return: tuple of `(init_fn, update_fn, evaluate)`.
    """
    def __init__(self, model, guide, loss, optim, **kwargs):
        self.model = model
        self.guide = guide
        self.loss = loss
        self.optim = optim
        self.kwargs = kwargs
        self.constrain_fn = None

    def init(self, rng, model_args=(), guide_args=()):
        """

        :param jax.random.PRNGKey rng: random number generator seed.
        :param tuple model_args: arguments to the model (these can possibly vary during
            the course of fitting).
        :param tuple guide_args: arguments to the guide (these can possibly vary during
            the course of fitting).
        :return: tuple containing initial :data:`SVIState`, and `get_params`, a callable
            that transforms unconstrained parameter values from the optimizer to the
            specified constrained domain
        """
        assert isinstance(model_args, tuple)
        assert isinstance(guide_args, tuple)
        rng, rng_seed = random.split(rng)
        model_init, guide_init = _seed(self.model, self.guide, rng_seed)
        guide_trace = trace(guide_init).get_trace(*guide_args, **self.kwargs)
        model_trace = trace(model_init).get_trace(*model_args, **self.kwargs)
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

    def update(self, svi_state, model_args=(), guide_args=()):
        """
        Take a single step of SVI (possibly on a batch / minibatch of data),
        using the optimizer.

        :param svi_state: current state of SVI.
        :param tuple model_args: dynamic arguments to the model.
        :param tuple guide_args: dynamic arguments to the guide.
        :return: tuple of `(svi_state, loss)`.
        """
        rng, rng_seed = random.split(svi_state.rng)
        params = self.optim.get_params(svi_state.optim_state)
        loss_val, grads = value_and_grad(
            lambda x: self.loss(rng_seed, self.constrain_fn(x), self.model, self.guide,
                                model_args, guide_args, self.kwargs))(params)
        optim_state = self.optim.update(grads, svi_state.optim_state)
        return SVIState(optim_state, rng), loss_val

    def evaluate(self, svi_state, model_args=(), guide_args=()):
        """
        Take a single step of SVI (possibly on a batch / minibatch of data).

        :param svi_state: current state of SVI.
        :param tuple model_args: arguments to the model (these can possibly vary during
            the course of fitting).
        :param tuple guide_args: arguments to the guide (these can possibly vary during
            the course of fitting).
        :return: evaluate ELBo loss given the current parameter values
            (held within `svi_state.optim_state`).
        """
        # we split to have the same seed as `update_fn` given an svi_state
        _, rng_seed = random.split(svi_state.rng)
        params = self.get_params(svi_state)
        return self.loss(rng_seed, params, self.model, self.guide, model_args, guide_args, self.kwargs)
