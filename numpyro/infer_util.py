from jax import grad, random
from jax.flatten_util import ravel_pytree
import jax.numpy as np

from numpyro.distributions.constraints import biject_to, real
from numpyro.handlers import seed, substitute, trace
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


def find_valid_initial_params(rng, model, *model_args, init_strategy='uniform', **model_kwargs):
    """
    Given a model with Pyro primitives, returns an initial valid unconstrained
    parameters. This function also returns an `is_valid` flag to say whether the
    initial parameters are valid.

    :param jax.random.PRNGKey rng: random number generator seed to
        sample from the prior. The returned `init_params` will have the
        batch shape ``rng.shape[:-1]``.
    :param model: Python callable containing Pyro primitives.
    :param `*model_args`: args provided to the model.
    :param str init_strategy: initialization strategy - `uniform`
        initializes the unconstrained parameters by drawing from
        a `Uniform(-2, 2)` distribution (as used by Stan), whereas
        `prior` initializes the parameters by sampling from the prior
        for each of the sample sites.
    :param `**model_kwargs`: kwargs provided to the model.
    :return: tuple of (`init_params`, `is_valid`).
    """
    def cond_fn(state):
        i, _, _, is_valid = state
        return (i < 100) & (~is_valid)

    def body_fn(state):
        i, key, _, _ = state
        key, subkey = random.split(key)
        # TODO: incorporate init_to_median here
        seeded_model = seed(model, subkey)
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
        prior_params = transform_fn(inv_transforms,
                                    {k: v for k, v in constrained_values.items()}, invert=True)
        if init_strategy == 'uniform':
            params = {}
            for k, v in prior_params.items():
                key, = random.split(key, 1)
                params[k] = random.uniform(key, shape=np.shape(v), minval=-2, maxval=2)
        elif init_strategy == 'prior':
            params = prior_params
        else:
            raise ValueError('initialize={} is not a valid initialization strategy.'.format(init_strategy))

        potential_fn = potential_energy(seeded_model, model_args, model_kwargs, inv_transforms)
        param_grads = grad(potential_fn)(params)
        z = ravel_pytree(params)[0]
        z_grad = ravel_pytree(param_grads)[0]
        is_valid = np.all(np.isfinite(z)) & np.all(np.isfinite(z_grad))
        return i + 1, key, params, is_valid

    # NB: the logic here is kind of do-while instead of while-do
    init_state = body_fn((0, rng, None, None))
    _, _, init_params, is_valid = while_loop(cond_fn, body_fn, init_state)
    return init_params, is_valid
