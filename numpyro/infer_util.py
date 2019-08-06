import jax.numpy as np

from numpyro.handlers import substitute, trace


def log_density(model, model_args, model_kwargs, params, skip_dist_transforms=False):
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
