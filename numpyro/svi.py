import os

import jax
from jax import random, value_and_grad

from numpyro.contrib.autoguide import AutoContinuous
from numpyro.distributions import constraints
from numpyro.distributions.constraints import biject_to
from numpyro.handlers import replay, seed, substitute, trace
from numpyro.infer_util import log_density, transform_fn


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
    :param optim_init: initialization function returned by a JAX optimizer.
        see: :mod:`jax.experimental.optimizers`.
    :param optim_update: update function for the optimizer
    :param get_optim_params: function to get current parameters values given the
        optimizer state.
    :param `**kwargs`: static arguments for the model / guide, i.e. arguments
        that remain constant during fitting.
    :return: tuple of `(init_fn, update_fn, evaluate)`.
    """
    constrain_fn = None

    def init_fn(rng, model_args=(), guide_args=(), params=None):
        """

        :param jax.random.PRNGKey rng: random number generator seed.
        :param tuple model_args: arguments to the model (these can possibly vary during
            the course of fitting).
        :param tuple guide_args: arguments to the guide (these can possibly vary during
            the course of fitting).
        :param dict params: initial parameter values to condition on. This can be
            useful for initializing neural networks using more specialized methods
            rather than sampling from the prior.
        :return: tuple containing initial optimizer state, and `constrain_fn`, a callable
            that transforms unconstrained parameter values from the optimizer to the
            specified constrained domain
        """
        nonlocal constrain_fn

        assert isinstance(model_args, tuple)
        assert isinstance(guide_args, tuple)
        model_init, guide_init = _seed(model, guide, rng)
        if params is None:
            params = {}
        else:
            model_init = substitute(model_init, params)
            guide_init = substitute(guide_init, params)
        guide_trace = trace(guide_init).get_trace(*guide_args, **kwargs)
        model_trace = trace(model_init).get_trace(*model_args, **kwargs)
        inv_transforms = {}
        # NB: params in model_trace will be overwritten by params in guide_trace
        for site in list(model_trace.values()) + list(guide_trace.values()):
            if site['type'] == 'param':
                constraint = site['kwargs'].pop('constraint', constraints.real)
                transform = biject_to(constraint)
                inv_transforms[site['name']] = transform
                params[site['name']] = transform.inv(site['value'])

        constrain_fn = jax.partial(transform_fn, inv_transforms)
        return optim.init(params), get_params

    def get_params(opt_state):
        """
        Gets values at `param` sites of the `model` and `guide`.

        :param dict opt_state: current state of the optimizer.
        """
        params = constrain_fn(optim.get_params(opt_state))
        return params

    def update_fn(rng, opt_state, model_args=(), guide_args=()):
        """
        Take a single step of SVI (possibly on a batch / minibatch of data),
        using the optimizer.

        :param int i: represents the i'th iteration over the epoch, passed as an
            argument to the optimizer's update function.
        :param jax.random.PRNGKey rng: random number generator seed.
        :param opt_state: current optimizer state.
        :param tuple model_args: dynamic arguments to the model.
        :param tuple guide_args: dynamic arguments to the guide.
        :return: tuple of `(loss_val, opt_state, rng)`.
        """
        rng, rng_seed = random.split(rng)
        model_init, guide_init = _seed(model, guide, rng_seed)
        params = optim.get_params(opt_state)
        loss_val, grads = value_and_grad(
            lambda x: loss(constrain_fn(x), model_init, guide_init, model_args, guide_args, kwargs))(params)
        opt_state = optim.update(grads, opt_state)
        return loss_val, opt_state, rng

    def evaluate(rng, opt_state, model_args=(), guide_args=()):
        """
        Take a single step of SVI (possibly on a batch / minibatch of data).

        :param jax.random.PRNGKey rng: random number generator seed.
        :param opt_state: current optimizer state.
        :param tuple model_args: arguments to the model (these can possibly vary during
            the course of fitting).
        :param tuple guide_args: arguments to the guide (these can possibly vary during
            the course of fitting).
        :return: evaluate ELBo loss given the current parameter values
            (held within `opt_state`).
        """
        model_init, guide_init = _seed(model, guide, rng)
        params = get_params(opt_state)
        return loss(params, model_init, guide_init, model_args, guide_args, kwargs)

    # Make local functions visible from the global scope once
    # `svi` is called for sphinx doc generation.
    if 'SPHINX_BUILD' in os.environ:
        svi.init_fn = init_fn
        svi.update_fn = update_fn
        svi.evaluate = evaluate

    return init_fn, update_fn, evaluate


def elbo(param_map, model, guide, model_args, guide_args, kwargs):
    """
    This is the most basic implementation of the Evidence Lower Bound, which is the
    fundamental objective in Variational Inference. This implementation has various
    limitations (for example it only supports random variablbes with reparameterized
    samplers) but can be used as a template to build more sophisticated loss
    objectives.

    For more details, refer to http://pyro.ai/examples/svi_part_i.html.

    :param dict param_map: dictionary of current parameter values keyed by site
        name.
    :param model: Python callable with Pyro primitives for the model.
    :param guide: Python callable with Pyro primitives for the guide
        (recognition network).
    :param tuple model_args: arguments to the model (these can possibly vary during
        the course of fitting).
    :param tuple guide_args: arguments to the guide (these can possibly vary during
        the course of fitting).
    :param dict kwargs: static keyword arguments to the model / guide.
    :return: negative of the Evidence Lower Bound (ELBo) to be minimized.
    """
    guide_log_density, guide_trace = log_density(guide, guide_args, kwargs, param_map)
    is_autoguide = False
    if isinstance(guide.__wrapped__, AutoContinuous):
        is_autoguide = True
    if is_autoguide:
        # in autoguide, a site's value holds intermediate value
        for name, site in guide_trace.items():
            if site['type'] == 'sample':
                param_map[name] = site['value']
    else:
        # NB: we only want to substitute params not available in guide_trace
        param_map = {k: v for k, v in param_map.items() if k not in guide_trace}
        model = replay(model, guide_trace)
    model_log_density, _ = log_density(model, model_args, kwargs, param_map,
                                       skip_dist_transforms=is_autoguide)
    # log p(z) - log q(z)
    elbo = model_log_density - guide_log_density
    # Return (-elbo) since by convention we do gradient descent on a loss and
    # the ELBO is a lower bound that needs to be maximized.
    return -elbo
