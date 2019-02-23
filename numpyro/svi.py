from jax import grad, random, jit
from jax.experimental import optimizers
import jax.numpy as np
from jax.random import PRNGKey

from numpyro.handlers import replay, trace, substitute, seed


# This is a unified interface for stochastic variational inference in Pyro.
# The actual construction of the loss is taken care of by `loss`.
# See http://docs.pyro.ai/en/0.3.0-release/inference_algos.html
class SVI(object):
    def __init__(self, model, guide, opt_init, opt_update, loss, rng=PRNGKey(0)):
        self.model = model
        self.guide = guide
        self.opt_init = opt_init
        self.opt_update = opt_update
        self.loss = loss
        self.rng = rng
        self._jitted_fn = None

    def init_state(self, *args, **kwargs):
        model, guide = self._seed(self.rng)
        guide_trace = trace(guide).get_trace(*args, **kwargs)
        model_trace = trace(model).get_trace(*args, **kwargs)
        params = {}
        for site in list(guide_trace.values()) + list(model_trace.values()):
            if site['type'] == 'param':
                params[site['name']] = site['value']
        return self.opt_init(params)

    def _seed(self, rng):
        model = seed(self.model, rng)
        _, subkey = random.split(rng)
        guide = seed(self.guide, subkey)
        return model, guide

    def step(self, i, *args, **kwargs):
        def jit_fn(i, opt_state, rng, args, kwargs):
            model, guide = self._seed(rng)
            params = optimizers.get_params(opt_state)
            loss = self.loss(params, model, guide, args, kwargs)
            grads = grad(self.loss)(params, model, guide, args, kwargs)
            opt_state = self.opt_update(i, grads, opt_state)
            return loss, opt_state
        opt_state = kwargs.pop('opt_state')
        if opt_state is None:
            opt_state = self.init_state(*args, **kwargs)
        if not self._jitted_fn:
            self._jitted_fn = jit(jit_fn)
        _, self.rng = random.split(self.rng)
        return self._jitted_fn(i, opt_state, self.rng, args, kwargs)


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
