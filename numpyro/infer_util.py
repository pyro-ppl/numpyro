from functools import partial

from jax import grad, random
from jax.flatten_util import ravel_pytree
import jax.numpy as np

import numpyro.distributions as dist
from numpyro.distributions.constraints import biject_to, real
from numpyro.handlers import block, sample, seed, substitute, trace
from numpyro.util import while_loop


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
            log_prob = np.sum(site['fn'].log_prob(site['value']))
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
    return {k: transforms[k](v) if not invert else transforms[k].inv(v)
            for k, v in params.items()}


def potential_energy(model, model_args, model_kwargs, inv_transforms):
    def _potential_energy(params):
        params_constrained = transform_fn(inv_transforms, params)
        log_joint, model_trace = log_density(model, model_args, model_kwargs, params_constrained)
        for name, t in inv_transforms.items():
            t_log_det = np.sum(t.log_abs_det_jacobian(params[name], params_constrained[name]))
            if 'scale' in model_trace[name]:
                t_log_det = model_trace[name]['scale'] * t_log_det
            log_joint = log_joint + t_log_det
        return - log_joint

    return _potential_energy


def init_to_uniform(site, radius=2):
    """
    Initialize to an arbitrary feasible point, ignoring distribution
    parameters.
    """
    if site['is_observed']:
        return None
    value = sample('_init', site['fn'])
    t = biject_to(site['fn'].support)
    unconstrained_value = sample('_unconstrained_init', dist.Uniform(-radius, radius),
                                 sample_shape=np.shape(t.inv(value)))
    return t(unconstrained_value)


def init_to_median(site, num_samples=15):
    """
    Initialize to the prior median.
    """
    if site['is_observed']:
        return None
    samples = sample('_init', site['fn'], sample_shape=(num_samples,))
    value = np.quantile(samples, 0.5, axis=0)
    return value


init_to_feasible = lambda site: init_to_uniform(site, radius=0)
init_to_prior = lambda site: init_to_median(site, num_samples=1)


def find_valid_initial_params(rng, model, *model_args, init_strategy=init_to_uniform,
                              **model_kwargs):
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
    :param `**model_kwargs`: kwargs provided to the model.
    :return: tuple of (`init_params`, `is_valid`).
    """
    def cond_fn(state):
        i, _, _, is_valid = state
        return (i < 100) & (~is_valid)

    def body_fn(state):
        i, key, _, _ = state
        key, subkey = random.split(key)

        seeded_model = substitute(model, substitute_fn=block(seed(init_strategy, subkey)))
        model_trace = trace(seeded_model).get_trace(*model_args, **model_kwargs)
        constrained_values, inv_transforms = {}, {}
        for k, v in model_trace.items():
            if v['type'] == 'sample' and not v['is_observed']:
                constrained_values[k] = v['value']
                inv_transforms[k] = biject_to(v['fn'].support)
            elif v['type'] == 'param':
                constrained_values[k] = v['value']
                constraint = v['kwargs'].pop('constraint', real)
                inv_transforms[k] = biject_to(constraint)
        params = transform_fn(inv_transforms,
                              {k: v for k, v in constrained_values.items()}, invert=True)

        potential_fn = potential_energy(model, model_args, model_kwargs, inv_transforms)
        param_grads = grad(potential_fn)(params)
        z = ravel_pytree(params)[0]
        z_grad = ravel_pytree(param_grads)[0]
        is_valid = np.all(np.isfinite(z)) & np.all(np.isfinite(z_grad))
        return i + 1, key, params, is_valid

    # NB: the logic here is kind of do-while instead of while-do
    init_state = body_fn((0, rng, None, None))
    _, _, init_params, is_valid = while_loop(cond_fn, body_fn, init_state)
    return init_params, is_valid
