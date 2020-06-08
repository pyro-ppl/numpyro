# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from functools import partial
import warnings

from jax import device_get, lax, random, value_and_grad, vmap
from jax.flatten_util import ravel_pytree
import jax.numpy as np

import numpyro
from numpyro.distributions.transforms import biject_to
from numpyro.distributions.util import sum_rightmost
from numpyro.handlers import seed, substitute, trace
from numpyro.infer.initialization import init_to_uniform, init_to_value
from numpyro.util import not_jax_tracer, while_loop

__all__ = [
    'find_valid_initial_params',
    'get_potential_fn',
    'log_density',
    'log_likelihood',
    'potential_energy',
    'initialize_model',
    'Predictive',
]

ModelInfo = namedtuple('ModelInfo', ['param_info', 'potential_fn', 'postprocess_fn', 'model_trace'])
ParamInfo = namedtuple('ParamInfo', ['z', 'potential_energy', 'z_grad'])


def log_density(model, model_args, model_kwargs, params):
    """
    (EXPERIMENTAL INTERFACE) Computes log of joint density for the model given
    latent values ``params``.

    :param model: Python callable containing NumPyro primitives.
    :param tuple model_args: args provided to the model.
    :param dict model_kwargs: kwargs provided to the model.
    :param dict params: dictionary of current parameter values keyed by site
        name.
    :return: log of joint density and a corresponding model trace
    """
    model = substitute(model, param_map=params)
    model_trace = trace(model).get_trace(*model_args, **model_kwargs)
    log_joint = np.array(0.)
    for site in model_trace.values():
        if site['type'] == 'sample':
            value = site['value']
            intermediates = site['intermediates']
            mask = site['mask']
            scale = site['scale']
            # Early exit when all elements are masked
            if not_jax_tracer(mask) and mask is not None and not np.any(mask):
                continue
            if intermediates:
                log_prob = site['fn'].log_prob(value, intermediates)
            else:
                log_prob = site['fn'].log_prob(value)

            # Minor optimizations
            # XXX: note that this may not work correctly for dynamic masks, provide
            # explicit jax.DeviceArray for masking.
            if mask is not None:
                if scale is not None:
                    log_prob = np.where(mask, scale * log_prob, 0.)
                else:
                    log_prob = np.where(mask, log_prob, 0.)
            else:
                if scale is not None:
                    log_prob = scale * log_prob
            log_prob = np.sum(log_prob)
            log_joint = log_joint + log_prob
    return log_joint, model_trace


def transform_fn(transforms, params, invert=False):
    """
    (EXPERIMENTAL INTERFACE) Callable that applies a transformation from the `transforms`
    dict to values in the `params` dict and returns the transformed values keyed on
    the same names.

    :param transforms: Dictionary of transforms keyed by names. Names in
        `transforms` and `params` should align.
    :param params: Dictionary of arrays keyed by names.
    :param invert: Whether to apply the inverse of the transforms.
    :return: `dict` of transformed params.
    """
    if invert:
        transforms = {k: v.inv for k, v in transforms.items()}
    return {k: transforms[k](v) if k in transforms else v
            for k, v in params.items()}


def constrain_fn(model, model_args, model_kwargs, params, return_deterministic=False):
    """
    (EXPERIMENTAL INTERFACE) Gets value at each latent site in `model` given
    unconstrained parameters `params`. The `transforms` is used to transform these
    unconstrained parameters to base values of the corresponding priors in `model`.
    If a prior is a transformed distribution, the corresponding base value lies in
    the support of base distribution. Otherwise, the base value lies in the support
    of the distribution.

    :param model: a callable containing NumPyro primitives.
    :param tuple model_args: args provided to the model.
    :param dict model_kwargs: kwargs provided to the model.
    :param dict params: dictionary of unconstrained values keyed by site
        names.
    :param bool return_deterministic: whether to return the value of `deterministic`
        sites from the model. Defaults to `False`.
    :return: `dict` of transformed params.
    """
    def substitute_fn(site):
        if site['name'] in params:
            return biject_to(site['fn'].support)(params[site['name']])

    substituted_model = substitute(model, substitute_fn=substitute_fn)
    model_trace = trace(substituted_model).get_trace(*model_args, **model_kwargs)
    return {k: v['value'] for k, v in model_trace.items() if (k in params) or
            (return_deterministic and (v['type'] == 'deterministic'))}


def _unconstrain_reparam(params, site):
    name = site['name']
    if name in params:
        p = params[name]
        t = biject_to(site['fn'].support)
        value = t(p)

        log_det = t.log_abs_det_jacobian(p, value)
        log_det = sum_rightmost(log_det, np.ndim(log_det) - np.ndim(value) + len(site['fn'].event_shape))
        if site['scale'] is not None:
            log_det = site['scale'] * log_det
        numpyro.factor('_{}_log_det'.format(name), log_det)
        return value


def potential_energy(model, model_args, model_kwargs, params):
    """
    (EXPERIMENTAL INTERFACE) Computes potential energy of a model given unconstrained params.
    The `inv_transforms` is used to transform these unconstrained parameters to base values
    of the corresponding priors in `model`. If a prior is a transformed distribution,
    the corresponding base value lies in the support of base distribution. Otherwise,
    the base value lies in the support of the distribution.

    :param model: a callable containing NumPyro primitives.
    :param tuple model_args: args provided to the model.
    :param dict model_kwargs: kwargs provided to the model.
    :param dict params: unconstrained parameters of `model`.
    :return: potential energy given unconstrained parameters.
    """
    substituted_model = substitute(model, substitute_fn=partial(_unconstrain_reparam, params))
    # no param is needed for log_density computation because we already substitute
    log_joint, model_trace = log_density(substituted_model, model_args, model_kwargs, {})
    return - log_joint


def _init_to_unconstrained_value(site=None, values={}):
    if site is None:
        return partial(init_to_value, values=values)


def find_valid_initial_params(rng_key, model,
                              init_strategy=init_to_uniform,
                              model_args=(),
                              model_kwargs=None,
                              prototype_params=None):
    """
    (EXPERIMENTAL INTERFACE) Given a model with Pyro primitives, returns an initial
    valid unconstrained value for all the parameters. This function also returns
    the corresponding potential energy, the gradients, and an
    `is_valid` flag to say whether the initial parameters are valid. Parameter values
    are considered valid if the values and the gradients for the log density have
    finite values.

    :param jax.random.PRNGKey rng_key: random number generator seed to
        sample from the prior. The returned `init_params` will have the
        batch shape ``rng_key.shape[:-1]``.
    :param model: Python callable containing Pyro primitives.
    :param callable init_strategy: a per-site initialization function.
    :param tuple model_args: args provided to the model.
    :param dict model_kwargs: kwargs provided to the model.
    :param dict prototype_params: an optional prototype parameters, which is used
        to define the shape for initial parameters.
    :return: tuple of `init_params_info` and `is_valid`, where `init_params_info` is the tuple
        containing the initial params, their potential energy, and their gradients.
    """
    model_kwargs = {} if model_kwargs is None else model_kwargs
    init_strategy = init_strategy if isinstance(init_strategy, partial) else init_strategy()
    # handle those init strategies differently to save computation
    if init_strategy.func is init_to_uniform:
        radius = init_strategy.keywords.get("radius")
        init_values = {}
    elif init_strategy.func is _init_to_unconstrained_value:
        radius = 2
        init_values = init_strategy.keywords.get("values")
    else:
        radius = None

    def cond_fn(state):
        i, _, _, is_valid = state
        return (i < 100) & (~is_valid)

    def body_fn(state):
        i, key, _, _ = state
        key, subkey = random.split(key)

        if radius is None or prototype_params is None:
            # Wrap model in a `substitute` handler to initialize from `init_loc_fn`.
            seeded_model = substitute(seed(model, subkey), substitute_fn=init_strategy)
            model_trace = trace(seeded_model).get_trace(*model_args, **model_kwargs)
            constrained_values, inv_transforms = {}, {}
            for k, v in model_trace.items():
                if v['type'] == 'sample' and not v['is_observed'] and not v['fn'].is_discrete:
                    constrained_values[k] = v['value']
                    inv_transforms[k] = biject_to(v['fn'].support)
            params = transform_fn(inv_transforms,
                                  {k: v for k, v in constrained_values.items()},
                                  invert=True)
        else:  # this branch doesn't require tracing the model
            params = {}
            for k, v in prototype_params.items():
                if k in init_values:
                    params[k] = init_values[k]
                else:
                    params[k] = random.uniform(subkey, np.shape(v), minval=-radius, maxval=radius)
                    key, subkey = random.split(key)

        potential_fn = partial(potential_energy, model, model_args, model_kwargs)
        pe, z_grad = value_and_grad(potential_fn)(params)
        z_grad_flat = ravel_pytree(z_grad)[0]
        is_valid = np.isfinite(pe) & np.all(np.isfinite(z_grad_flat))
        return i + 1, key, (params, pe, z_grad), is_valid

    def _find_valid_params(rng_key, exit_early=False):
        init_state = (0, rng_key, (prototype_params, 0., prototype_params), False)
        if exit_early and not_jax_tracer(rng_key):
            # Early return if valid params found. This is only helpful for single chain,
            # where we can avoid compiling body_fn in while_loop.
            _, _, (init_params, pe, z_grad), is_valid = init_state = body_fn(init_state)
            if not_jax_tracer(is_valid):
                if device_get(is_valid):
                    return (init_params, pe, z_grad), is_valid

        # XXX: this requires compiling the model, so for multi-chain, we trace the model 2-times
        # even if the init_state is a valid result
        _, _, (init_params, pe, z_grad), is_valid = while_loop(cond_fn, body_fn, init_state)
        return (init_params, pe, z_grad), is_valid

    # Handle possible vectorization
    if rng_key.ndim == 1:
        (init_params, pe, z_grad), is_valid = _find_valid_params(rng_key, exit_early=True)
    else:
        (init_params, pe, z_grad), is_valid = lax.map(_find_valid_params, rng_key)
    return (init_params, pe, z_grad), is_valid


def _get_model_transforms(model, model_args=(), model_kwargs=None):
    model_kwargs = {} if model_kwargs is None else model_kwargs
    model_trace = trace(model).get_trace(*model_args, **model_kwargs)
    inv_transforms = {}
    # model code may need to be replayed in the presence of deterministic sites
    replay_model = False
    for k, v in model_trace.items():
        if v['type'] == 'sample' and not v['is_observed'] and not v['fn'].is_discrete:
            inv_transforms[k] = biject_to(v['fn'].support)
            # TODO: check if the support is dynamic, then we set replay_model = True
        elif v['type'] == 'deterministic':
            replay_model = True
    return inv_transforms, replay_model, model_trace


def get_potential_fn(model, inv_transforms, replay_model=False, dynamic_args=False, model_args=(), model_kwargs=None):
    """
    (EXPERIMENTAL INTERFACE) Given a model with Pyro primitives, returns a
    function which, given unconstrained parameters, evaluates the potential
    energy (negative log joint density). In addition, this returns a
    function to transform unconstrained values at sample sites to constrained
    values within their respective support.

    :param model: Python callable containing Pyro primitives.
    :param dict inv_transforms: dictionary of transforms keyed by names.
    :param bool replay_model: whether we need to replay model in
        `postprocess_fn` to obtain `deterministic` sites.
    :param bool dynamic_args: if `True`, the `potential_fn` and
        `constraints_fn` are themselves dependent on model arguments.
        When provided a `*model_args, **model_kwargs`, they return
        `potential_fn` and `constraints_fn` callables, respectively.
    :param tuple model_args: args provided to the model.
    :param dict model_kwargs: kwargs provided to the model.
    :return: tuple of (`potential_fn`, `postprocess_fn`). The latter is used
        to constrain unconstrained samples (e.g. those returned by HMC)
        to values that lie within the site's support, and return values at
        `deterministic` sites in the model.
    """
    if dynamic_args:
        def potential_fn(*args, **kwargs):
            return partial(potential_energy, model, args, kwargs)

        def postprocess_fn(*args, **kwargs):
            if replay_model:
                return partial(constrain_fn, model, args, kwargs, return_deterministic=True)
            else:
                return partial(transform_fn, inv_transforms)
    else:
        model_kwargs = {} if model_kwargs is None else model_kwargs
        potential_fn = partial(potential_energy, model, model_args, model_kwargs)
        if replay_model:
            postprocess_fn = partial(constrain_fn, model, model_args, model_kwargs,
                                     return_deterministic=True)
        else:
            postprocess_fn = partial(transform_fn, inv_transforms)

    return potential_fn, postprocess_fn


def initialize_model(rng_key, model,
                     init_strategy=init_to_uniform,
                     dynamic_args=False,
                     model_args=(),
                     model_kwargs=None):
    """
    (EXPERIMENTAL INTERFACE) Helper function that calls :func:`~numpyro.infer.util.get_potential_fn`
    and :func:`~numpyro.infer.util.find_valid_initial_params` under the hood
    to return a tuple of (`init_params_info`, `potential_fn`, `postprocess_fn`, `model_trace`).

    :param jax.random.PRNGKey rng_key: random number generator seed to
        sample from the prior. The returned `init_params` will have the
        batch shape ``rng_key.shape[:-1]``.
    :param model: Python callable containing Pyro primitives.
    :param callable init_strategy: a per-site initialization function.
        See :ref:`init_strategy` section for available functions.
    :param bool dynamic_args: if `True`, the `potential_fn` and
        `constraints_fn` are themselves dependent on model arguments.
        When provided a `*model_args, **model_kwargs`, they return
        `potential_fn` and `constraints_fn` callables, respectively.
    :param tuple model_args: args provided to the model.
    :param dict model_kwargs: kwargs provided to the model.
    :return: a namedtupe `ModelInfo` which contains the fields
        (`param_info`, `potential_fn`, `postprocess_fn`, `model_trace`), where
        `param_info` is a namedtuple `ParamInfo` containing values from the prior
        used to initiate MCMC, their corresponding potential energy, and their gradients;
        `postprocess_fn` is a callable that uses inverse transforms
        to convert unconstrained HMC samples to constrained values that
        lie within the site's support, in addition to returning values
        at `deterministic` sites in the model.
    """
    model_kwargs = {} if model_kwargs is None else model_kwargs
    substituted_model = substitute(seed(model, rng_key if np.ndim(rng_key) == 1 else rng_key[0]),
                                   substitute_fn=init_strategy)
    inv_transforms, replay_model, model_trace = _get_model_transforms(
        substituted_model, model_args, model_kwargs)
    constrained_values = {k: v['value'] for k, v in model_trace.items()
                          if v['type'] == 'sample' and not v['is_observed']
                          and not v['fn'].is_discrete}

    potential_fn, postprocess_fn = get_potential_fn(model,
                                                    inv_transforms,
                                                    replay_model=replay_model,
                                                    dynamic_args=dynamic_args,
                                                    model_args=model_args,
                                                    model_kwargs=model_kwargs)

    init_strategy = init_strategy if isinstance(init_strategy, partial) else init_strategy()
    if init_strategy.func is init_to_value:
        init_values = init_strategy.keywords.get("values")
        unconstrained_values = transform_fn(inv_transforms, init_values, invert=True)
        init_strategy = _init_to_unconstrained_value(values=unconstrained_values)
    prototype_params = transform_fn(inv_transforms, constrained_values, invert=True)
    (init_params, pe, grad), is_valid = find_valid_initial_params(rng_key, model,
                                                                  init_strategy=init_strategy,
                                                                  model_args=model_args,
                                                                  model_kwargs=model_kwargs,
                                                                  prototype_params=prototype_params)

    if not_jax_tracer(is_valid):
        if device_get(~np.all(is_valid)):
            raise RuntimeError("Cannot find valid initial parameters. Please check your model again.")
    return ModelInfo(ParamInfo(init_params, pe, grad), potential_fn, postprocess_fn, model_trace)


def _predictive(rng_key, model, posterior_samples, num_samples, return_sites=None,
                parallel=True, model_args=(), model_kwargs={}):
    rng_keys = random.split(rng_key, num_samples)

    def single_prediction(val):
        rng_key, samples = val
        model_trace = trace(seed(substitute(model, samples), rng_key)).get_trace(
            *model_args, **model_kwargs)
        if return_sites is not None:
            if return_sites == '':
                sites = {k for k, site in model_trace.items() if site['type'] != 'plate'}
            else:
                sites = return_sites
        else:
            sites = {k for k, site in model_trace.items()
                     if (site['type'] == 'sample' and k not in samples) or (site['type'] == 'deterministic')}
        return {name: site['value'] for name, site in model_trace.items() if name in sites}

    if parallel:
        return vmap(single_prediction)((rng_keys, posterior_samples))
    else:
        return lax.map(single_prediction, (rng_keys, posterior_samples))


class Predictive(object):
    """
    This class is used to construct predictive distribution. The predictive distribution is obtained
    by running model conditioned on latent samples from `posterior_samples`.

    .. warning::
        The interface for the `Predictive` class is experimental, and
        might change in the future.

    :param model: Python callable containing Pyro primitives.
    :param dict posterior_samples: dictionary of samples from the posterior.
    :param callable guide: optional guide to get posterior samples of sites not present
        in `posterior_samples`.
    :param dict params: dictionary of values for param sites of model/guide.
    :param int num_samples: number of samples
    :param list return_sites: sites to return; by default only sample sites not present
        in `posterior_samples` are returned.
    :param bool parallel: whether to predict in parallel using JAX vectorized map :func:`jax.vmap`.
        Defaults to False.

    :return: dict of samples from the predictive distribution.
    """
    def __init__(self, model, posterior_samples=None, guide=None, params=None, num_samples=None,
                 return_sites=None, parallel=False):
        if posterior_samples is None and num_samples is None:
            raise ValueError("Either posterior_samples or num_samples must be specified.")

        posterior_samples = {} if posterior_samples is None else posterior_samples

        for name, sample in posterior_samples.items():
            batch_size = sample.shape[0]
            if (num_samples is not None) and (num_samples != batch_size):
                warnings.warn("Sample's leading dimension size {} is different from the "
                              "provided {} num_samples argument. Defaulting to {}."
                              .format(batch_size, num_samples, batch_size), UserWarning)
            num_samples = batch_size

        if num_samples is None:
            raise ValueError("No sample sites in posterior samples to infer `num_samples`.")

        if return_sites is not None:
            assert isinstance(return_sites, (list, tuple, set))

        self.model = model
        self.posterior_samples = {} if posterior_samples is None else posterior_samples
        self.num_samples = num_samples
        self.guide = guide
        self.params = {} if params is None else params
        self.return_sites = return_sites
        self.parallel = parallel

    def __call__(self, rng_key, *args, **kwargs):
        """
        Returns dict of samples from the predictive distribution. By default, only sample sites not
        contained in `posterior_samples` are returned. This can be modified by changing the
        `return_sites` keyword argument of this :class:`Predictive` instance.

        :param jax.random.PRNGKey rng_key: random key to draw samples.
        :param args: model arguments.
        :param kwargs: model kwargs.
        """
        posterior_samples = self.posterior_samples
        if self.guide is not None:
            rng_key, guide_rng_key = random.split(rng_key)
            # use return_sites='' as a special signal to return all sites
            guide = substitute(self.guide, self.params)
            posterior_samples = _predictive(guide_rng_key, guide, posterior_samples,
                                            self.num_samples, return_sites='', parallel=self.parallel,
                                            model_args=args, model_kwargs=kwargs)
        model = substitute(self.model, self.params)
        return _predictive(rng_key, model, posterior_samples, self.num_samples,
                           return_sites=self.return_sites, parallel=self.parallel,
                           model_args=args, model_kwargs=kwargs)

    def get_samples(self, rng_key, *args, **kwargs):
        warnings.warn("The method `.get_samples` has been deprecated in favor of `.__call__`.",
                      DeprecationWarning)
        return self.__call__(rng_key, *args, **kwargs)


def log_likelihood(model, posterior_samples, *args, **kwargs):
    """
    (EXPERIMENTAL INTERFACE) Returns log likelihood at observation nodes of model,
    given samples of all latent variables.

    :param model: Python callable containing Pyro primitives.
    :param dict posterior_samples: dictionary of samples from the posterior.
    :param args: model arguments.
    :param kwargs: model kwargs.
    :return: dict of log likelihoods at observation sites.
    """
    def single_loglik(samples):
        model_trace = trace(substitute(model, samples)).get_trace(*args, **kwargs)
        return {name: site['fn'].log_prob(site['value']) for name, site in model_trace.items()
                if site['type'] == 'sample' and site['is_observed']}

    return vmap(single_loglik)(posterior_samples)
