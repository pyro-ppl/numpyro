from collections import namedtuple

import jax
from jax import random
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
from jax.scipy.special import expit

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, init_to_uniform
from numpyro.infer.hmc_util import warmup_adapter
from numpyro.infer.util import initialize_model
from numpyro.util import identity

BarkerState = namedtuple("BarkerState", [
    "i", "z", "potential_energy", "z_grad", "accept_prob", "mean_accept_prob", "adapt_state", "rng_key"])


class Barker(numpyro.infer.mcmc.MCMCKernel):

    def __init__(self, model=None, potential_fn=None, step_size=1.0,
                 adapt_step_size=True, adapt_mass_matrix=True, dense_mass=False,
                 target_accept_prob=0.8, init_strategy=init_to_uniform):
        # TODO: probably the default target accept prob is not high like HMC
        if not (model is None) ^ (potential_fn is None):
            raise ValueError('Only one of `model` or `potential_fn` must be specified.')
        self._model = model
        self._potential_fn = potential_fn
        self._step_size = step_size
        self._adapt_step_size = adapt_step_size
        self._adapt_mass_matrix = adapt_mass_matrix
        self._dense_mass = dense_mass
        self._target_accept_prob = target_accept_prob
        self._init_strategy = init_strategy

    @property
    def model(self):
        return self._model

    @property
    def sample_field(self):
        return 'z'

    def get_diagnostics_str(self, state):
        return 'step size {:.2e}. acc. prob={:.2f}'.format(state.adapt_state.step_size,
                                                           state.mean_accept_prob)

    def _init_state(self, rng_key, model_args, model_kwargs, init_params):
        if self._model is not None:
            params_info, potential_fn_gen, self._postprocess_fn, model_trace = initialize_model(
                rng_key,
                self._model,
                dynamic_args=True,
                init_strategy=self._init_strategy,
                model_args=model_args,
                model_kwargs=model_kwargs)
            init_params = params_info[0]
            model_kwargs = {} if model_kwargs is None else model_kwargs
            self._potential_fn = potential_fn_gen(*model_args, **model_kwargs)
        return init_params

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        self._num_warmup = num_warmup
        # TODO (low-priority): support chain_method="vectorized", i.e. rng_key is a batch of keys
        rng_key, rng_key_init_model, rng_key_wa = random.split(rng_key, 3)
        init_params = self._init_state(rng_key_init_model, model_args, model_kwargs, init_params)
        if self._potential_fn and init_params is None:
            raise ValueError('Valid value of `init_params` must be provided with'
                             ' `potential_fn`.')

        pe, grad = jax.value_and_grad(self._potential_fn)(init_params)

        wa_init, self._wa_update = warmup_adapter(
            num_warmup,
            adapt_step_size=self._adapt_step_size,
            adapt_mass_matrix=self._adapt_mass_matrix,
            dense_mass=self._dense_mass,
            target_accept_prob=self._target_accept_prob)
        size = len(ravel_pytree(init_params)[0])
        wa_state = wa_init(None, rng_key_wa, self._step_size, mass_matrix_size=size)
        wa_state = wa_state._replace(rng_key=None)
        return jax.device_put(BarkerState(0, init_params, pe, grad, 0., 0., wa_state, rng_key))

    def sample(self, state, model_args, model_kwargs):
        i, x, x_pe, x_grad, _, mean_accept_prob, adapt_state, rng_key = state
        x_flat, unravel_fn = ravel_pytree(x)
        x_grad_flat, _ = ravel_pytree(x_grad)
        shape = jnp.shape(x_flat)
        rng_key, key_normal, key_bernoulli, key_accept = random.split(rng_key, 4)

        # get proposal y
        # TODO: if we use dense mass, then we need to resort this *
        # TODO: double check if using step_size and mass_matrix is consistent with the paper
        z_proposal = adapt_state.step_size * random.normal(key_normal, shape) * adapt_state.mass_matrix_sqrt
        p = expit(-z_proposal * x_grad_flat)
        b = jnp.where(random.uniform(key_bernoulli, shape) < p, 1., -1.)
        bz = b * z_proposal
        y_flat = x_flat + bz

        y = unravel_fn(y_flat)
        y_pe, y_grad = jax.value_and_grad(self._potential_fn)(y)
        y_grad_flat, _ = ravel_pytree(y_grad)
        log_accept_ratio = x_pe - y_pe + jnp.sum(
            jax.nn.softplus(bz * x_grad_flat) - jax.nn.softplus(-bz * y_grad_flat))
        accept_prob = jnp.clip(jnp.exp(log_accept_ratio), a_max=1.)

        x, x_flat, pe, x_grad = jax.lax.cond(random.bernoulli(key_accept, accept_prob),
                                             (y, y_flat, y_pe, y_grad), identity,
                                             (x, x_flat, x_pe, x_grad), identity)

        # not update adapt_state after warmup phase
        adapt_state = jax.lax.cond(i < self._num_warmup,
                                   (i, accept_prob, (x,), adapt_state),
                                   lambda args: self._wa_update(*args),
                                   adapt_state,
                                   identity)

        itr = i + 1
        n = jnp.where(i < self._num_warmup, itr, itr - self._num_warmup)
        mean_accept_prob = mean_accept_prob + (accept_prob - mean_accept_prob) / n

        return BarkerState(itr, x, pe, x_grad, accept_prob, mean_accept_prob, adapt_state, rng_key)


def model():
    numpyro.sample("x", dist.Normal().expand([100]))


kernel = Barker(model)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=10000, progress_bar=True)
mcmc.run(random.PRNGKey(0))
samples = mcmc.get_samples()
mcmc.print_summary()
print(mcmc.last_state.adapt_state.mass_matrix_sqrt)
