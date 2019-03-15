from jax import grad, random, jit, partial
from jax.experimental import optimizers
import jax.numpy as np

from numpyro.handlers import replay, trace, substitute, seed


def _seed(model, guide, rng):
    model_seed, guide_seed = random.split(rng, 2)
    model_init = seed(model, model_seed)
    guide_init = seed(guide, guide_seed)
    return model_init, guide_init


@partial(jit, static_argnums=(1, 2, 3, 5))
def _svi_update(i, model, guide, loss, opt_state, opt_update, rng, args, kwargs):
    model_init, guide_init = _seed(model, guide, rng)
    params = optimizers.get_params(opt_state)
    loss_val = loss(params, model_init, guide_init, args, kwargs)
    grads = grad(loss)(params, model_init, guide_init, args, kwargs)
    opt_state = opt_update(i, grads, opt_state)
    return loss_val, opt_state


def svi(model, guide, loss, optim_init, optim_update):
    def init_fn(rng, *args, params=None, **kwargs):
        if params is None:
            params = {}
        model_init, guide_init = _seed(model, guide, rng)
        guide_trace = trace(guide_init).get_trace(*args, **kwargs)
        model_trace = trace(model_init).get_trace(*args, **kwargs)
        for site in list(guide_trace.values()) + list(model_trace.values()):
            if site['type'] == 'param':
                params[site['name']] = site['value']
        return optim_init(params)

    def update_fn(i, opt_state, rng, *args, **kwargs):
        loss_val, opt_state = _svi_update(i, model, guide, loss, opt_state, optim_update, rng, args, kwargs)
        rng, = random.split(rng, 1)
        return loss_val, opt_state, rng

    return init_fn, update_fn


# This is a basic implementation of the Evidence Lower Bound, which is the
# fundamental objective in Variational Inference.
# See http://pyro.ai/examples/svi_part_i.html for details.
# This implementation has various limitations (for example it only supports
# random variablbes with reparameterized samplers), but all the ELBO
# implementations in Pyro share the same basic logic.
def elbo(param_map, model, guide, args, kwargs):
    model = substitute(model, param_map)
    guide = substitute(guide, param_map)
    guide_trace = trace(guide).get_trace(*args, **kwargs)
    model_trace = trace(replay(model, guide_trace)).get_trace(*args, **kwargs)
    elbo = 0.
    # Loop over all the sample sites in the model and add the corresponding
    # log p(z) term to the ELBO. Note that this will also include any observed
    # data, i.e. sample sites with the keyword `obs=...`.
    for site in model_trace.values():
        if site["type"] == "sample":
            elbo = elbo + np.sum(site["fn"].logpdf(site["value"]))
    # Loop over all the sample sites in the guide and add the corresponding
    # -log q(z) term to the ELBO.
    for site in guide_trace.values():
        if site["type"] == "sample":
            elbo = elbo - np.sum(site["fn"].logpdf(site["value"]))
    # Return (-elbo) since by convention we do gradient descent on a loss and
    # the ELBO is a lower bound that needs to be maximized.
    return -elbo
