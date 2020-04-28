# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import partial
import warnings

import jax
from jax import device_get, lax, random, value_and_grad, vmap
from jax.flatten_util import ravel_pytree
import jax.numpy as np

import funsor
funsor.set_backend("jax")

import numpyro
import numpyro.distributions as dist
from numpyro.distributions.constraints import real
from numpyro.distributions.transforms import ComposeTransform, biject_to
from numpyro.handlers import block, seed, substitute, trace
from numpyro.util import not_jax_tracer, while_loop

from .enum_messenger import trace as packed_trace


__all__ = [
    'find_valid_initial_params',
    'get_potential_fn',
    'log_density',
    'log_likelihood',
    'init_to_feasible',
    'init_to_median',
    'init_to_prior',
    'init_to_uniform',
    'init_to_value',
    'potential_energy',
    'initialize_model',
    'Predictive',
    'transformed_potential_energy',
]


def log_density(model, model_args, model_kwargs, params, skip_dist_transforms=False):
    """
    (EXPERIMENTAL INTERFACE) Computes log of joint density for the model given
    latent values ``params``.

    :param model: Python callable containing NumPyro primitives.
    :param tuple model_args: args provided to the model.
    :param dict model_kwargs: kwargs provided to the model.
    :param dict params: dictionary of current parameter values keyed by site
        name.
    :param bool skip_dist_transforms: whether to compute log probability of a site
        (if its prior is a transformed distribution) in its base distribution
        domain.
    :return: log of joint density and a corresponding model trace
    """
    # We skip transforms in
    #   + autoguide's model
    #   + hmc's model
    # We apply transforms in
    #   + autoguide's guide
    #   + svi's model + guide
    if skip_dist_transforms:
        model = substitute(model, base_param_map=params)
    else:
        model = substitute(model, param_map=params)
    model_trace = trace(model).get_trace(*model_args, **model_kwargs)
    log_joint = jax.device_put(0.)
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
                if skip_dist_transforms:
                    log_prob = site['fn'].base_dist.log_prob(intermediates[0][0])
                else:
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


def constrain_fn(model, transforms, model_args, model_kwargs, params, return_deterministic=False):
    """
    (EXPERIMENTAL INTERFACE) Gets value at each latent site in `model` given
    unconstrained parameters `params`. The `transforms` is used to transform these
    unconstrained parameters to base values of the corresponding priors in `model`.
    If a prior is a transformed distribution, the corresponding base value lies in
    the support of base distribution. Otherwise, the base value lies in the support
    of the distribution.

    :param model: a callable containing NumPyro primitives.
    :param dict transforms: dictionary of transforms keyed by names. Names in
        `transforms` and `params` should align.
    :param tuple model_args: args provided to the model.
    :param dict model_kwargs: kwargs provided to the model.
    :param dict params: dictionary of unconstrained values keyed by site
        names.
    :param bool return_deterministic: whether to return the value of `deterministic`
        sites from the model. Defaults to `False`.
    :return: `dict` of transformed params.
    """
    params_constrained = transform_fn(transforms, params)
    substituted_model = substitute(model, base_param_map=params_constrained)
    model_trace = trace(substituted_model).get_trace(*model_args, **model_kwargs)
    return {k: v['value'] for k, v in model_trace.items() if (k in params) or
            (return_deterministic and v['type'] == 'deterministic')}


def potential_energy(model, inv_transforms, model_args, model_kwargs, params):
    """
    (EXPERIMENTAL INTERFACE) Computes potential energy of a model given unconstrained params.
    The `inv_transforms` is used to transform these unconstrained parameters to base values
    of the corresponding priors in `model`. If a prior is a transformed distribution,
    the corresponding base value lies in the support of base distribution. Otherwise,
    the base value lies in the support of the distribution.

    :param model: a callable containing NumPyro primitives.
    :param dict inv_transforms: dictionary of transforms keyed by names.
    :param tuple model_args: args provided to the model.
    :param dict model_kwargs: kwargs provided to the model.
    :param dict params: unconstrained parameters of `model`.
    :return: potential energy given unconstrained parameters.
    """
    params_constrained = transform_fn(inv_transforms, params)
    log_joint, model_trace = log_density(model, model_args, model_kwargs, params_constrained,
                                         skip_dist_transforms=True)
    for name, t in inv_transforms.items():
        t_log_det = np.sum(t.log_abs_det_jacobian(params[name], params_constrained[name]))
        if model_trace[name]['scale'] is not None:
            t_log_det = model_trace[name]['scale'] * t_log_det
        log_joint = log_joint + t_log_det
    return - log_joint


def enum_potential_energy(model, inv_transforms, model_args, model_kwargs, params):
    params_constrained = transform_fn(inv_transforms, params)
    model = substitute(model, base_param_map=params_constrained)
    model_trace = packed_trace(model).get_trace(*model_args, **model_kwargs)
    log_factors = []
    sum_vars, prod_vars = frozenset(), frozenset()
    for site in model_trace.values():
        if site['type'] == 'sample':
            value = site['value']
            intermediates = site['intermediates']
            mask = site['mask']
            # scale = site['scale']  # TODO handle scaling
            # Early exit when all elements are masked
            if not_jax_tracer(mask) and mask is not None and not np.any(mask):
                continue
            if intermediates:
                log_prob = site['fn'].base_dist.log_prob(intermediates[0][0])
            else:
                log_prob = site['fn'].log_prob(value)

            # TODO handle masking and scaling together
            if mask is not None:
                log_prob = np.where(mask, log_prob, 0.)

            log_prob = funsor.to_funsor(log_prob, output=funsor.reals(), dim_to_name=site['infer']['dim_to_name'])
            log_factors.append(log_prob)
            sum_vars |= frozenset({site['name']})
            prod_vars |= frozenset(f.name for f in site['cond_indep_stack'] if f.dim is not None)

    for name, t in inv_transforms.items():
        t_log_det = t.log_abs_det_jacobian(params[name], params_constrained[name])
        if model_trace[name]['scale'] is not None:
            t_log_det = model_trace[name]['scale'] * t_log_det
        keepdims = False
        for d in range(-len(t_log_det.shape), 0, 1):
            if d not in model_trace[name]['infer']['dim_to_name']:
                t_log_det = t_log_det.sum(d, keepdims=keepdims)
            else:
                keepdims = True
        log_factors.append(funsor.to_funsor(
            t_log_det, output=funsor.reals(),
            dim_to_name=model_trace[name]['infer']['dim_to_name']))

    with funsor.interpreter.interpretation(funsor.terms.lazy):
        lazy_result = funsor.sum_product.sum_product(
            funsor.ops.logaddexp, funsor.ops.add, log_factors,
            eliminate=sum_vars | prod_vars, plates=prod_vars)
    result = funsor.optimizer.apply_optimizer(lazy_result)
    return -result.data


def transformed_potential_energy(potential_energy, inv_transform, z):
    """
    Given a potential energy `p(x)`, compute potential energy of `p(z)`
    with `z = transform(x)` (i.e. `x = inv_transform(z)`).

    :param potential_energy: a callable to compute potential energy of original
        variable `x`.
    :param ~numpyro.distributions.constraints.Transform inv_transform: a
        transform from the new variable `z` to `x`.
    :param z: new variable to compute potential energy
    :return: potential energy of `z`.
    """
    x, intermediates = inv_transform.call_with_intermediates(z)
    logdet = inv_transform.log_abs_det_jacobian(z, x, intermediates=intermediates)
    return potential_energy(x) - logdet


def _init_to_median(site, num_samples=15, skip_param=False):
    if site['type'] == 'sample' and not site['is_observed']:
        if isinstance(site['fn'], dist.TransformedDistribution):
            fn = site['fn'].base_dist
        else:
            fn = site['fn']
        samples = numpyro.sample('_init', fn,
                                 sample_shape=(num_samples,) + site['kwargs']['sample_shape'])
        return np.median(samples, axis=0)

    if site['type'] == 'param' and not skip_param:
        # return base value of param site
        constraint = site['kwargs'].pop('constraint', real)
        transform = biject_to(constraint)
        value = site['args'][0]
        if isinstance(transform, ComposeTransform):
            base_transform = transform.parts[0]
            value = base_transform(transform.inv(value))
        return value


def init_to_median(num_samples=15):
    """
    Initialize to the prior median.

    :param int num_samples: number of prior points to calculate median.
    """
    return partial(_init_to_median, num_samples=num_samples)


def init_to_prior():
    """
    Initialize to a prior sample.
    """
    return init_to_median(num_samples=1)


def _init_to_uniform(site, radius=2, skip_param=False):
    if site['type'] == 'sample' and not site['is_observed']:
        if site['fn'].has_enumerate_support:
            return site['value']
        if isinstance(site['fn'], dist.TransformedDistribution):
            fn = site['fn'].base_dist
        else:
            fn = site['fn']
        value = numpyro.sample('_init', fn, sample_shape=site['kwargs']['sample_shape'])
        base_transform = biject_to(fn.support)
        unconstrained_value = numpyro.sample('_unconstrained_init', dist.Uniform(-radius, radius),
                                             sample_shape=np.shape(base_transform.inv(value)))
        return base_transform(unconstrained_value)

    if site['type'] == 'param' and not skip_param:
        # return base value of param site
        constraint = site['kwargs'].pop('constraint', real)
        transform = biject_to(constraint)
        value = site['args'][0]
        unconstrained_value = numpyro.sample('_unconstrained_init', dist.Uniform(-radius, radius),
                                             sample_shape=np.shape(transform.inv(value)))
        if isinstance(transform, ComposeTransform):
            base_transform = transform.parts[0]
        else:
            base_transform = transform
        return base_transform(unconstrained_value)


def init_to_uniform(radius=2):
    """
    Initialize to a random point in the area `(-radius, radius)` of unconstrained domain.

    :param float radius: specifies the range to draw an initial point in the unconstrained domain.
    """
    return partial(_init_to_uniform, radius=radius)


def init_to_feasible():
    """
    Initialize to an arbitrary feasible point, ignoring distribution
    parameters.
    """
    return init_to_uniform(radius=0)


def _init_to_value(site, values={}, skip_param=False):
    if site['type'] == 'sample' and not site['is_observed']:
        if site['name'] not in values:
            return _init_to_uniform(site, skip_param=skip_param)

        value = values[site['name']]
        if isinstance(site['fn'], dist.TransformedDistribution):
            value = ComposeTransform(site['fn'].transforms).inv(value)
        return value

    if site['type'] == 'param' and not skip_param:
        # return base value of param site
        constraint = site['kwargs'].pop('constraint', real)
        transform = biject_to(constraint)
        value = site['args'][0]
        if isinstance(transform, ComposeTransform):
            base_transform = transform.parts[0]
            value = base_transform(transform.inv(value))
        return value


def init_to_value(values):
    """
    Initialize to the value specified in `values`. We defer to
    :func:`init_to_uniform` strategy for sites which do not appear in `values`.

    :param dict values: dictionary of initial values keyed by site name.
    """
    return partial(_init_to_value, values=values)


def find_valid_initial_params(rng_key, model,
                              init_strategy=init_to_uniform(),
                              param_as_improper=False,
                              model_args=(),
                              model_kwargs=None):
    """
    (EXPERIMENTAL INTERFACE) Given a model with Pyro primitives, returns an initial
    valid unconstrained value for all the parameters. This function also returns an
    `is_valid` flag to say whether the initial parameters are valid. Parameter values
    are considered valid if the values and the gradients for the log density have
    finite values.

    :param jax.random.PRNGKey rng_key: random number generator seed to
        sample from the prior. The returned `init_params` will have the
        batch shape ``rng_key.shape[:-1]``.
    :param model: Python callable containing Pyro primitives.
    :param callable init_strategy: a per-site initialization function.
    :param bool param_as_improper: a flag to decide whether to consider sites with
        `param` statement as sites with improper priors.
    :param tuple model_args: args provided to the model.
    :param dict model_kwargs: kwargs provided to the model.
    :return: tuple of (`init_params`, `is_valid`).
    """
    init_strategy = jax.partial(init_strategy, skip_param=not param_as_improper)

    def cond_fn(state):
        i, _, _, is_valid = state
        return (i < 100) & (~is_valid)

    def body_fn(state):
        i, key, _, _ = state
        key, subkey = random.split(key)

        # Wrap model in a `substitute` handler to initialize from `init_loc_fn`.
        # Use `block` to not record sample primitives in `init_loc_fn`.
        seeded_model = substitute(model, substitute_fn=block(seed(init_strategy, subkey)))
        model_trace = trace(seeded_model).get_trace(*model_args, **model_kwargs)
        constrained_values, inv_transforms = {}, {}
        for k, v in model_trace.items():
            if v['type'] == 'sample' and not v['is_observed']:
                if v['fn'].has_enumerate_support:
                    continue
                elif v['intermediates']:
                    constrained_values[k] = v['intermediates'][0][0]
                    inv_transforms[k] = biject_to(v['fn'].base_dist.support)
                else:
                    constrained_values[k] = v['value']
                    inv_transforms[k] = biject_to(v['fn'].support)
            elif v['type'] == 'param' and param_as_improper:
                constraint = v['kwargs'].pop('constraint', real)
                transform = biject_to(constraint)
                if isinstance(transform, ComposeTransform):
                    base_transform = transform.parts[0]
                    inv_transforms[k] = base_transform
                    constrained_values[k] = base_transform(transform.inv(v['value']))
                else:
                    inv_transforms[k] = transform
                    constrained_values[k] = v['value']
        params = transform_fn(inv_transforms,
                              {k: v for k, v in constrained_values.items()},
                              invert=True)
        potential_fn = jax.partial(potential_energy, model, inv_transforms, model_args, model_kwargs)
        pe, param_grads = value_and_grad(potential_fn)(params)
        z_grad = ravel_pytree(param_grads)[0]
        is_valid = np.isfinite(pe) & np.all(np.isfinite(z_grad))
        return i + 1, key, params, is_valid

    def _find_valid_params(rng_key_):
        _, _, prototype_params, is_valid = init_state = body_fn((0, rng_key_, None, None))
        # Early return if valid params found.
        if not_jax_tracer(is_valid):
            if device_get(is_valid):
                return prototype_params, is_valid

        _, _, init_params, is_valid = while_loop(cond_fn, body_fn, init_state)
        return init_params, is_valid

    # Handle possible vectorization
    if rng_key.ndim == 1:
        init_params, is_valid = _find_valid_params(rng_key)
    else:
        init_params, is_valid = lax.map(_find_valid_params, rng_key)
    return init_params, is_valid


def get_model_transforms(rng_key, model, model_args=(), model_kwargs=None):
    model_kwargs = {} if model_kwargs is None else model_kwargs
    seeded_model = seed(model, rng_key if rng_key.ndim == 1 else rng_key[0])
    model_trace = trace(seeded_model).get_trace(*model_args, **model_kwargs)
    inv_transforms = {}
    # model code may need to be replayed in the presence of dynamic constraints
    # or deterministic sites
    replay_model = False
    for k, v in model_trace.items():
        if v['type'] == 'sample' and not v['is_observed']:
            if v['fn'].has_enumerate_support:
                continue
            elif v['intermediates']:
                inv_transforms[k] = biject_to(v['fn'].base_dist.support)
                replay_model = True
            else:
                inv_transforms[k] = biject_to(v['fn'].support)
        elif v['type'] == 'param':
            constraint = v['kwargs'].pop('constraint', real)
            transform = biject_to(constraint)
            if isinstance(transform, ComposeTransform):
                inv_transforms[k] = transform.parts[0]
                replay_model = True
            else:
                inv_transforms[k] = transform
        elif v['type'] == 'deterministic':
            replay_model = True
    return inv_transforms, replay_model


def get_potential_fn(rng_key, model, dynamic_args=False, model_args=(), model_kwargs=None, enum=True):
    """
    (EXPERIMENTAL INTERFACE) Given a model with Pyro primitives, returns a
    function which, given unconstrained parameters, evaluates the potential
    energy (negative log joint density). In addition, this returns a
    function to transform unconstrained values at sample sites to constrained
    values within their respective support.

    :param jax.random.PRNGKey rng_key: random number generator seed to
        sample from the prior. The returned `init_params` will have the
        batch shape ``rng_key.shape[:-1]``.
    :param model: Python callable containing Pyro primitives.
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
    _potential_energy = enum_potential_energy if enum else potential_energy
    if dynamic_args:
        def potential_fn(*args, **kwargs):
            inv_transforms, replay_model = get_model_transforms(rng_key, model, args, kwargs)
            return jax.partial(_potential_energy, model, inv_transforms, args, kwargs)

        def postprocess_fn(*args, **kwargs):
            inv_transforms, replay_model = get_model_transforms(rng_key, model, args, kwargs)
            if replay_model:
                return jax.partial(constrain_fn, model, inv_transforms, args, kwargs,
                                   return_deterministic=True)
            else:
                return jax.partial(transform_fn, inv_transforms)
    else:
        inv_transforms, replay_model = get_model_transforms(rng_key, model, model_args, model_kwargs)
        potential_fn = jax.partial(_potential_energy, model, inv_transforms, model_args, model_kwargs)
        if replay_model:
            postprocess_fn = jax.partial(constrain_fn, model, inv_transforms, model_args, model_kwargs,
                                         return_deterministic=True)
        else:
            postprocess_fn = jax.partial(transform_fn, inv_transforms)

    return potential_fn, postprocess_fn


def initialize_model(rng_key, model,
                     init_strategy=init_to_uniform(),
                     dynamic_args=False,
                     model_args=(),
                     model_kwargs=None):
    """
    (EXPERIMENTAL INTERFACE) Helper function that calls :func:`~numpyro.infer.util.get_potential_fn`
    and :func:`~numpyro.infer.util.find_valid_initial_params` under the hood
    to return a tuple of (`init_params`, `potential_fn`, `constrain_fn`).

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
    :return: tuple of (`init_params`, `potential_fn`, `postprocess_fn`),
        `init_params` are values from the prior used to initiate MCMC,
        `postprocess_fn` is a callable that uses inverse transforms
        to convert unconstrained HMC samples to constrained values that
        lie within the site's support, in addition to returning values
        at `deterministic` sites in the model.
    """
    if model_kwargs is None:
        model_kwargs = {}
    potential_fn, postprocess_fn = get_potential_fn(rng_key if rng_key.ndim == 1 else rng_key[0],
                                                    model,
                                                    dynamic_args=dynamic_args,
                                                    model_args=model_args,
                                                    model_kwargs=model_kwargs)

    init_params, is_valid = find_valid_initial_params(rng_key, model,
                                                      init_strategy=init_strategy,
                                                      param_as_improper=True,
                                                      model_args=model_args,
                                                      model_kwargs=model_kwargs)

    if not_jax_tracer(is_valid):
        if device_get(~np.all(is_valid)):
            raise RuntimeError("Cannot find valid initial parameters. Please check your model again.")
    return init_params, potential_fn, postprocess_fn


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
