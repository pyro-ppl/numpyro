import jax
from jax import random, value_and_grad
from jax.flatten_util import ravel_pytree
import jax.numpy as np

import numpyro.distributions as dist
from numpyro.distributions.constraints import ComposeTransform, biject_to, real
from numpyro.handlers import block, sample, seed, substitute, trace
from numpyro.util import while_loop


def log_density(model, model_args, model_kwargs, params, skip_dist_transforms=True):
    """
    Computes log of joint density for the model given latent values ``params``.

    :param model: Python callable containing NumPyro primitives.
    :param tuple model_args: args provided to the model.
    :param dict model_kwargs`: kwargs provided to the model.
    :param dict params: dictionary of current parameter values keyed by site
        name.
    :param bool skip_dist_transforms: whether to compute log probability of a site
        (if its prior is a transformed distribution) in its base distribution
        domain.
    :return: log of joint density and a corresponding model trace
    """
    model = substitute(model, base_param_map=params)
    model_trace = trace(model).get_trace(*model_args, **model_kwargs)
    log_joint = 0.
    for site in model_trace.values():
        if site['type'] == 'sample':
            value = site['value']
            intermediates = site['intermediates']
            if intermediates:
                if skip_dist_transforms:
                    log_prob = site['fn'].base_dist.log_prob(intermediates[0][0])
                else:
                    log_prob = site['fn'].log_prob(value, intermediates)
            else:
                log_prob = site['fn'].log_prob(value)
            log_prob = np.sum(log_prob)
            if 'scale' in site:
                log_prob = site['scale'] * log_prob
            log_joint = log_joint + log_prob
    return log_joint, model_trace


def transform_fn(transforms, params, invert=False):
    """
    Callable that applies a transformation from the `transforms` dict to values in the
    `params` dict and returns the transformed values keyed on the same names.

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


def constrain_fn(model, model_args, model_kwargs, transforms, params):
    """
    Gets value at each latent site in `model` given unconstrained parameters `params`.
    The `transforms` is used to transform these unconstrained parameters to base values
    of the corresponding priors in `model`. If a prior is a transformed distribution,
    the corresponding base value lies in the support of base distribution. Otherwise,
    the base value lies in the support of the distribution.

    :param model: a callable containing NumPyro primitives.
    :param tuple model_args: args provided to the model.
    :param dict model_kwargs`: kwargs provided to the model.
    :param dict transforms: dictionary of transforms keyed by names. Names in
        `transforms` and `params` should align.
    :param dict params: dictionary of unconstrained values keyed by site
        names.
    :return: `dict` of transformed params.
    """
    params_constrained = transform_fn(transforms, params)
    substituted_model = substitute(model, base_param_map=params_constrained)
    model_trace = trace(substituted_model).get_trace(*model_args, **model_kwargs)
    return {k: model_trace[k]['value'] for k, v in params.items() if k in model_trace}


def potential_energy(model, model_args, model_kwargs, inv_transforms, params):
    """
    Makes a callable which computes potential energy of a model given unconstrained params.
    The `inv_transforms` is used to transform these unconstrained parameters to base values
    of the corresponding priors in `model`. If a prior is a transformed distribution,
    the corresponding base value lies in the support of base distribution. Otherwise,
    the base value lies in the support of the distribution.

    :param model: a callable containing NumPyro primitives.
    :param tuple model_args: args provided to the model.
    :param dict model_kwargs`: kwargs provided to the model.
    :param dict inv_transforms: dictionary of transforms keyed by names.
    :return: a callable that computes potential energy given unconstrained parameters.
    """
    params_constrained = transform_fn(inv_transforms, params)
    log_joint, model_trace = log_density(model, model_args, model_kwargs, params_constrained)
    for name, t in inv_transforms.items():
        t_log_det = np.sum(t.log_abs_det_jacobian(params[name], params_constrained[name]))
        if 'scale' in model_trace[name]:
            t_log_det = model_trace[name]['scale'] * t_log_det
        log_joint = log_joint + t_log_det
    return - log_joint


def init_to_median(site, num_samples=15, skip_param=False):
    """
    Initialize to the prior median.
    """
    if site['type'] == 'sample' and not site['is_observed']:
        if isinstance(site['fn'], dist.TransformedDistribution):
            fn = site['fn'].base_dist
        else:
            fn = site['fn']
        samples = sample('_init', fn, sample_shape=(num_samples,))
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


def init_to_prior(site, skip_param=False):
    """
    Initialize to a prior sample.
    """
    return init_to_median(site, num_samples=1, skip_param=skip_param)


def init_to_uniform(site, radius=2, skip_param=False):
    """
    Initialize to an arbitrary feasible point, ignoring distribution
    parameters.
    """
    if site['type'] == 'sample' and not site['is_observed']:
        if isinstance(site['fn'], dist.TransformedDistribution):
            fn = site['fn'].base_dist
        else:
            fn = site['fn']
        value = sample('_init', fn)
        base_transform = biject_to(fn.support)
        unconstrained_value = sample('_unconstrained_init', dist.Uniform(-radius, radius),
                                     sample_shape=np.shape(base_transform.inv(value)))
        return base_transform(unconstrained_value)

    if site['type'] == 'param' and not skip_param:
        # return base value of param site
        constraint = site['kwargs'].pop('constraint', real)
        transform = biject_to(constraint)
        value = site['args'][0]
        unconstrained_value = sample('_unconstrained_init', dist.Uniform(-radius, radius),
                                     sample_shape=np.shape(transform.inv(value)))
        if isinstance(transform, ComposeTransform):
            base_transform = transform.parts[0]
        else:
            base_transform = transform
        return base_transform(unconstrained_value)


def init_to_feasible(site, skip_param=False):
    """
    Initialize to an arbitrary feasible point, ignoring distribution
    parameters.
    """
    return init_to_uniform(site, radius=0, skip_param=skip_param)


def find_valid_initial_params(rng, model, *model_args, init_strategy=init_to_uniform,
                              param_as_improper=False, prototype_params=None, **model_kwargs):
    """
    Given a model with Pyro primitives, returns an initial valid unconstrained
    parameters. This function also returns an `is_valid` flag to say whether the
    initial parameters are valid.

    :param jax.random.PRNGKey rng: random number generator seed to
        sample from the prior. The returned `init_params` will have the
        batch shape ``rng.shape[:-1]``.
    :param model: Python callable containing Pyro primitives.
    :param `*model_args`: args provided to the model.
    :param callable init_strategy: a per-site initialization function.
    :param bool param_as_improper: a flag to decide whether to consider sites with
        `param` statement as sites with improper priors.
    :param `**model_kwargs`: kwargs provided to the model.
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
                if v['intermediates']:
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
        potential_fn = jax.partial(potential_energy, model, model_args, model_kwargs, inv_transforms)
        pe, param_grads = value_and_grad(potential_fn)(params)
        z_grad = ravel_pytree(param_grads)[0]
        is_valid = np.isfinite(pe) & np.all(np.isfinite(z_grad))
        return i + 1, key, params, is_valid

    if prototype_params is not None:
        init_state = (0, rng, prototype_params, False)
    else:
        init_state = body_fn((0, rng, None, None))
    _, _, init_params, is_valid = while_loop(cond_fn, body_fn, init_state)
    return init_params, is_valid
