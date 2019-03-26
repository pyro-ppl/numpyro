import jax.numpy as np
from jax import random, value_and_grad
from jax.experimental import optimizers

from numpyro.distributions.distribution import jax_continuous
from numpyro.distributions.util import validation_disabled
from numpyro.handlers import replay, seed, substitute, trace


def _seed(model, guide, rng):
    model_seed, guide_seed = random.split(rng, 2)
    model_init = seed(model, model_seed)
    guide_init = seed(guide, guide_seed)
    return model_init, guide_init


def svi(model, guide, loss, optim_init, optim_update, **kwargs):
    def init_fn(rng, model_args=(), guide_args=(), params=None):
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
        for site in list(guide_trace.values()) + list(model_trace.values()):
            if site['type'] == 'param':
                params[site['name']] = site['value']
        return optim_init(params)

    def update_fn(i, opt_state, rng, model_args=(), guide_args=()):
        model_init, guide_init = _seed(model, guide, rng)
        params = optimizers.get_params(opt_state)
        loss_val, grads = value_and_grad(loss)(params, model_init, guide_init, model_args, guide_args, kwargs)
        opt_state = optim_update(i, grads, opt_state)
        rng, = random.split(rng, 1)
        return loss_val, opt_state, rng

    def evaluate(opt_state, rng, model_args=(), guide_args=()):
        model_init, guide_init = _seed(model, guide, rng)
        params = optimizers.get_params(opt_state)
        return loss(params, model_init, guide_init, model_args, guide_args, kwargs)

    return init_fn, update_fn, evaluate


# This is a basic implementation of the Evidence Lower Bound, which is the
# fundamental objective in Variational Inference.
# See http://pyro.ai/examples/svi_part_i.html for details.
# This implementation has various limitations (for example it only supports
# random variablbes with reparameterized samplers), but all the ELBO
# implementations in Pyro share the same basic logic.
def elbo(param_map, model, guide, model_args, guide_args, kwargs):
    model = substitute(model, param_map)
    guide = substitute(guide, param_map)
    guide_trace = trace(guide).get_trace(*guide_args, **kwargs)
    model_trace = trace(replay(model, guide_trace)).get_trace(*model_args, **kwargs)
    elbo = 0.
    # Loop over all the sample sites in the model and add the corresponding
    # log p(z) term to the ELBO. Note that this will also include any observed
    # data, i.e. sample sites with the keyword `obs=...`.

    def logp(d, val):
        # TODO: Find alternatives to this anti-pattern.
        with validation_disabled():
            return d.logpdf(val) if isinstance(d.dist, jax_continuous) else d.logpmf(val)

    for site in model_trace.values():
        if site["type"] == "sample":
            elbo = elbo + np.sum(logp(site["fn"], site["value"]))
    # Loop over all the sample sites in the guide and add the corresponding
    # -log q(z) term to the ELBO.
    for site in guide_trace.values():
        if site["type"] == "sample":
            elbo = elbo - np.sum(logp(site["fn"], site["value"]))
    # Return (-elbo) since by convention we do gradient descent on a loss and
    # the ELBO is a lower bound that needs to be maximized.
    return -elbo
