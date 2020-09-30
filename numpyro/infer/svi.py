# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import namedtuple, partial

from jax import random

from numpyro.distributions import constraints
from numpyro.distributions.transforms import biject_to
from numpyro.handlers import replay, seed, trace
from numpyro.infer.util import transform_fn

SVIState = namedtuple('SVIState', ['optim_state', 'rng_key'])
"""
A :func:`~collections.namedtuple` consisting of the following fields:
 - **optim_state** - current optimizer's state.
 - **rng_key** - random number generator seed used for the iteration.
"""


def _apply_loss_fn(loss_fn, rng_key, constrain_fn, model, guide,
                   args, kwargs, static_kwargs, params):
    return loss_fn(rng_key, constrain_fn(params), model, guide, *args, **kwargs, **static_kwargs)


class SVI(object):
    """
    Stochastic Variational Inference given an ELBO loss objective.

    **References**

    1. *SVI Part I: An Introduction to Stochastic Variational Inference in Pyro*,
       (http://pyro.ai/examples/svi_part_i.html)

    **Example:**

    .. doctest::

        >>> from jax import lax, random
        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.distributions import constraints
        >>> from numpyro.infer import SVI, Trace_ELBO

        >>> def model(data):
        ...     f = numpyro.sample("latent_fairness", dist.Beta(10, 10))
        ...     with numpyro.plate("N", data.shape[0]):
        ...         numpyro.sample("obs", dist.Bernoulli(f), obs=data)

        >>> def guide(data):
        ...     alpha_q = numpyro.param("alpha_q", 15., constraint=constraints.positive)
        ...     beta_q = numpyro.param("beta_q", 15., constraint=constraints.positive)
        ...     numpyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q))

        >>> data = jnp.concatenate([jnp.ones(6), jnp.zeros(4)])
        >>> optimizer = numpyro.optim.Adam(step_size=0.0005)
        >>> svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
        >>> init_state = svi.init(random.PRNGKey(0), data)
        >>> state = lax.fori_loop(0, 2000, lambda i, state: svi.update(state, data)[0], init_state)
        >>> # or to collect losses during the loop
        >>> # state, losses = lax.scan(lambda state, i: svi.update(state, data), init_state, jnp.arange(2000))
        >>> params = svi.get_params(state)
        >>> inferred_mean = params["alpha_q"] / (params["alpha_q"] + params["beta_q"])

    :param model: Python callable with Pyro primitives for the model.
    :param guide: Python callable with Pyro primitives for the guide
        (recognition network).
    :param optim: an instance of :class:`~numpyro.optim._NumpyroOptim`.
    :param loss: ELBO loss, i.e. negative Evidence Lower Bound, to minimize.
    :param static_kwargs: static arguments for the model / guide, i.e. arguments
        that remain constant during fitting.
    :return: tuple of `(init_fn, update_fn, evaluate)`.
    """
    def __init__(self, model, guide, optim, loss, **static_kwargs):
        self.model = model
        self.guide = guide
        self.loss = loss
        self.optim = optim
        self.static_kwargs = static_kwargs
        self.constrain_fn = None

    def init(self, rng_key, *args, **kwargs):
        """

        :param jax.random.PRNGKey rng_key: random number generator seed.
        :param args: arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: keyword arguments to the model / guide (these can possibly vary
            during the course of fitting).
        :return: tuple containing initial :data:`SVIState`, and `get_params`, a callable
            that transforms unconstrained parameter values from the optimizer to the
            specified constrained domain
        """
        rng_key, model_seed, guide_seed = random.split(rng_key, 3)
        model_init = seed(self.model, model_seed)
        guide_init = seed(self.guide, guide_seed)
        guide_trace = trace(guide_init).get_trace(*args, **kwargs, **self.static_kwargs)
        model_trace = trace(replay(model_init, guide_trace)).get_trace(*args, **kwargs, **self.static_kwargs)
        params = {}
        inv_transforms = {}
        # NB: params in model_trace will be overwritten by params in guide_trace
        for site in list(model_trace.values()) + list(guide_trace.values()):
            if site['type'] == 'param':
                constraint = site['kwargs'].pop('constraint', constraints.real)
                transform = biject_to(constraint)
                inv_transforms[site['name']] = transform
                params[site['name']] = transform.inv(site['value'])

        self.constrain_fn = partial(transform_fn, inv_transforms)
        return SVIState(self.optim.init(params), rng_key)

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
        rng_key, rng_key_step = random.split(svi_state.rng_key)
        loss_fn = partial(_apply_loss_fn, self.loss.loss, rng_key_step, self.constrain_fn, self.model,
                          self.guide, args, kwargs, self.static_kwargs)
        loss_val, optim_state = self.optim.eval_and_update(loss_fn, svi_state.optim_state)
        return SVIState(optim_state, rng_key), loss_val

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
        _, rng_key_eval = random.split(svi_state.rng_key)
        params = self.get_params(svi_state)
        return self.loss.loss(rng_key_eval, params, self.model, self.guide,
                              *args, **kwargs, **self.static_kwargs)
