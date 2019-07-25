import jax.numpy as np

from numpyro.handlers import substitute, trace


def log_density(model, model_args, model_kwargs, params):
    """
    Computes log of joint density for the model given latent values ``params``.

    :param model: Python callable containing Pyro primitives.
    :param tuple model_args: args provided to the model.
    :param dict model_kwargs`: kwargs provided to the model.
    :param dict params: dictionary of current parameter values keyed by site
        name.
    :return: log of joint density and a corresponding model trace
    """
    model = substitute(model, params)
    model_trace = trace(model).get_trace(*model_args, **model_kwargs)
    log_joint = 0.
    for site in model_trace.values():
        if site['type'] == 'sample':
            value = site['value']
            intermediates = site['intermediates']
            log_prob = np.sum(site['fn'].log_prob(value, intermediates) if intermediates
                              else site['fn'].log_prob(value))
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
