# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple

import jax
from jax import random
from jax.flatten_util import ravel_pytree
from jax.nn import softplus
import jax.numpy as jnp
from jax.scipy.special import expit

from numpyro.infer.hmc_util import warmup_adapter
from numpyro.infer.initialization import init_to_uniform
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import initialize_model
from numpyro.util import identity

BarkerMHState = namedtuple(
    "BarkerMHState",
    [
        "i",
        "z",
        "potential_energy",
        "z_grad",
        "accept_prob",
        "mean_accept_prob",
        "adapt_state",
        "rng_key",
    ],
)
"""
A :func:`~collections.namedtuple` consisting of the following fields:

 - **i** - iteration. This is reset to 0 after warmup.
 - **z** - Python collection representing values (unconstrained samples from
   the posterior) at latent sites.
 - **potential_energy** - Potential energy computed at the given value of ``z``.
 - **z_grad** - Gradient of potential energy w.r.t. latent sample sites.
 - **accept_prob** - Acceptance probability of the proposal. Note that ``z``
   does not correspond to the proposal if it is rejected.
 - **mean_accept_prob** - Mean acceptance probability until current iteration
   during warmup adaptation or sampling (for diagnostics).
 - **adapt_state** - A ``HMCAdaptState`` namedtuple which contains adaptation information
   during warmup:

   + **step_size** - Step size to be used by the integrator in the next iteration.
   + **inverse_mass_matrix** - The inverse mass matrix to be used for the next
     iteration.
   + **mass_matrix_sqrt** - The square root of mass matrix to be used for the next
     iteration. In case of dense mass, this is the Cholesky factorization of the
     mass matrix.

 - **rng_key** - random number generator seed used for generating proposals, etc.
"""


class BarkerMH(MCMCKernel):
    """
    This is a gradient-based MCMC algorithm of Metropolis-Hastings type that uses
    a skew-symmetric proposal distribution that depends on the gradient of the
    potential (the Barker proposal; see reference [1]). In particular the proposal
    distribution is skewed in the direction of the gradient at the current sample.

    We expect this algorithm to be particularly effective for low to moderate dimensional
    models, where it may be competitive with HMC and NUTS.

    .. note:: We recommend to use this kernel with `progress_bar=False` in :class:`MCMC`
        to reduce JAX's dispatch overhead.

    **References:**

    1. The Barker proposal: combining robustness and efficiency in gradient-based MCMC.
       Samuel Livingstone, Giacomo Zanella.

    :param model: Python callable containing Pyro :mod:`~numpyro.primitives`.
        If model is provided, `potential_fn` will be inferred using the model.
    :param potential_fn: Python callable that computes the potential energy
        given input parameters. The input parameters to `potential_fn` can be
        any python collection type, provided that `init_params` argument to
        :meth:`init` has the same type.
    :param float step_size: (Initial) step size to use in the Barker proposal.
    :param bool adapt_step_size: Whether to adapt the step size during warm-up.
        Defaults to ``adapt_step_size==True``.
    :param bool adapt_mass_matrix: Whether to adapt the mass matrix during warm-up.
        Defaults to ``adapt_mass_matrix==True``.
    :param bool dense_mass: Whether to use a dense (i.e. full-rank) or diagonal mass matrix.
        (defaults to ``dense_mass=False``).
    :param float target_accept_prob: The target acceptance probability that is used to guide
        step size adapation. Defaults to ``target_accept_prob=0.4``.
    :param callable init_strategy: a per-site initialization function.
        See :ref:`init_strategy` section for available functions.

    **Example**

    .. doctest::

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.infer import MCMC, BarkerMH

        >>> def model():
        ...     x = numpyro.sample("x", dist.Normal().expand([10]))
        ...     numpyro.sample("obs", dist.Normal(x, 1.0), obs=jnp.ones(10))
        >>>
        >>> kernel = BarkerMH(model)
        >>> mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, progress_bar=True)
        >>> mcmc.run(jax.random.PRNGKey(0))
        >>> mcmc.print_summary()  # doctest: +SKIP
    """

    def __init__(
        self,
        model=None,
        potential_fn=None,
        step_size=1.0,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        dense_mass=False,
        target_accept_prob=0.4,
        init_strategy=init_to_uniform,
    ):
        if not (model is None) ^ (potential_fn is None):
            raise ValueError("Only one of `model` or `potential_fn` must be specified.")
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
        return "z"

    def get_diagnostics_str(self, state):
        return "step size {:.2e}. acc. prob={:.2f}".format(
            state.adapt_state.step_size, state.mean_accept_prob
        )

    def _init_state(self, rng_key, model_args, model_kwargs, init_params):
        if self._model is not None:
            (
                params_info,
                potential_fn_gen,
                self._postprocess_fn,
                model_trace,
            ) = initialize_model(
                rng_key,
                self._model,
                dynamic_args=True,
                init_strategy=self._init_strategy,
                model_args=model_args,
                model_kwargs=model_kwargs,
            )
            init_params = params_info[0]
            model_kwargs = {} if model_kwargs is None else model_kwargs
            self._potential_fn = potential_fn_gen(*model_args, **model_kwargs)
        return init_params

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        self._num_warmup = num_warmup
        # TODO (low-priority): support chain_method="vectorized", i.e. rng_key is a batch of keys
        assert rng_key.shape == (2,), (
            "BarkerMH only supports chain_method='parallel' or chain_method='sequential'."
            " Please put in a feature request if you think it would be useful to be able "
            "to use BarkerMH in vectorized mode."
        )
        rng_key, rng_key_init_model, rng_key_wa = random.split(rng_key, 3)
        init_params = self._init_state(
            rng_key_init_model, model_args, model_kwargs, init_params
        )
        if self._potential_fn and init_params is None:
            raise ValueError(
                "Valid value of `init_params` must be provided with" " `potential_fn`."
            )

        pe, grad = jax.value_and_grad(self._potential_fn)(init_params)

        wa_init, self._wa_update = warmup_adapter(
            num_warmup,
            adapt_step_size=self._adapt_step_size,
            adapt_mass_matrix=self._adapt_mass_matrix,
            dense_mass=self._dense_mass,
            target_accept_prob=self._target_accept_prob,
        )
        size = len(ravel_pytree(init_params)[0])
        wa_state = wa_init(
            (init_params,), rng_key_wa, self._step_size, mass_matrix_size=size
        )
        wa_state = wa_state._replace(rng_key=None)
        init_state = BarkerMHState(
            jnp.array(0),
            init_params,
            pe,
            grad,
            jnp.zeros(()),
            jnp.zeros(()),
            wa_state,
            rng_key,
        )
        return jax.device_put(init_state)

    def postprocess_fn(self, args, kwargs):
        if self._postprocess_fn is None:
            return identity
        return self._postprocess_fn(*args, **kwargs)

    def sample(self, state, model_args, model_kwargs):
        i, x, x_pe, x_grad, _, mean_accept_prob, adapt_state, rng_key = state
        x_flat, unravel_fn = ravel_pytree(x)
        x_grad_flat, _ = ravel_pytree(x_grad)
        shape = jnp.shape(x_flat)
        rng_key, key_normal, key_bernoulli, key_accept = random.split(rng_key, 4)

        mass_sqrt_inv = adapt_state.mass_matrix_sqrt_inv

        x_grad_flat_scaled = (
            mass_sqrt_inv @ x_grad_flat
            if self._dense_mass
            else mass_sqrt_inv * x_grad_flat
        )

        # Generate proposal y.
        z = adapt_state.step_size * random.normal(key_normal, shape)

        p = expit(-z * x_grad_flat_scaled)
        b = jnp.where(random.uniform(key_bernoulli, shape) < p, 1.0, -1.0)

        dx_flat = b * z
        dx_flat_scaled = (
            mass_sqrt_inv.T @ dx_flat if self._dense_mass else mass_sqrt_inv * dx_flat
        )

        y_flat = x_flat + dx_flat_scaled

        y = unravel_fn(y_flat)
        y_pe, y_grad = jax.value_and_grad(self._potential_fn)(y)
        y_grad_flat, _ = ravel_pytree(y_grad)
        y_grad_flat_scaled = (
            mass_sqrt_inv @ y_grad_flat
            if self._dense_mass
            else mass_sqrt_inv * y_grad_flat
        )

        log_accept_ratio = (
            x_pe
            - y_pe
            + jnp.sum(
                softplus(dx_flat * x_grad_flat_scaled)
                - softplus(-dx_flat * y_grad_flat_scaled)
            )
        )
        accept_prob = jnp.clip(jnp.exp(log_accept_ratio), a_max=1.0)

        x, x_flat, pe, x_grad = jax.lax.cond(
            random.bernoulli(key_accept, accept_prob),
            (y, y_flat, y_pe, y_grad),
            identity,
            (x, x_flat, x_pe, x_grad),
            identity,
        )

        # do not update adapt_state after warmup phase
        adapt_state = jax.lax.cond(
            i < self._num_warmup,
            (i, accept_prob, (x,), adapt_state),
            lambda args: self._wa_update(*args),
            adapt_state,
            identity,
        )

        itr = i + 1
        n = jnp.where(i < self._num_warmup, itr, itr - self._num_warmup)
        mean_accept_prob = mean_accept_prob + (accept_prob - mean_accept_prob) / n

        return BarkerMHState(
            itr, x, pe, x_grad, accept_prob, mean_accept_prob, adapt_state, rng_key
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_postprocess_fn"] = None
        state["_wa_update"] = None
        return state
