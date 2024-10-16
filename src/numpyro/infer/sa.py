# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple

from jax import device_put, lax, random, vmap
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import numpyro.distributions as dist
from numpyro.distributions.util import cholesky_update
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import init_to_uniform, initialize_model
from numpyro.util import identity, is_prng_key


def _get_proposal_loc_and_scale(samples, loc, scale, new_sample):
    # get loc/scale of q_{-n} (Algorithm 1, line 5 of ref [1]) for n from 1 -> N
    # these loc/scale will be stacked to the first dim; so
    #   proposal_loc.shape[0] = proposal_loc.shape[0] = N
    # Here, we use the numerical stability procedure in Appendix 6 of [1].
    weight = 1 / samples.shape[0]
    if scale.ndim > loc.ndim:
        new_scale = cholesky_update(scale, new_sample - loc, weight)
        proposal_scale = cholesky_update(new_scale, samples - loc, -weight)
        proposal_scale = cholesky_update(
            proposal_scale, new_sample - samples, -(weight**2)
        )
    else:
        var = jnp.square(scale) + weight * jnp.square(new_sample - loc)
        proposal_var = var - weight * jnp.square(samples - loc)
        proposal_var = proposal_var - weight**2 * jnp.square(new_sample - samples)
        proposal_scale = jnp.sqrt(proposal_var)

    proposal_loc = loc + weight * (new_sample - samples)
    return proposal_loc, proposal_scale


def _sample_proposal(inv_mass_matrix_sqrt, rng_key, batch_shape=()):
    eps = random.normal(rng_key, batch_shape + jnp.shape(inv_mass_matrix_sqrt)[:1])
    if inv_mass_matrix_sqrt.ndim == 1:
        r = jnp.multiply(inv_mass_matrix_sqrt, eps)
    elif inv_mass_matrix_sqrt.ndim == 2:
        r = jnp.matmul(inv_mass_matrix_sqrt, eps[..., None])[..., 0]
    else:
        raise ValueError("Mass matrix has incorrect number of dims.")
    return r


SAAdaptState = namedtuple("SAAdaptState", ["zs", "pes", "loc", "inv_mass_matrix_sqrt"])
SAState = namedtuple(
    "SAState",
    [
        "i",
        "z",
        "potential_energy",
        "accept_prob",
        "mean_accept_prob",
        "diverging",
        "adapt_state",
        "rng_key",
    ],
)
"""
A :func:`~collections.namedtuple` used in Sample Adaptive MCMC.
This consists of the following fields:

 - **i** - iteration. This is reset to 0 after warmup.
 - **z** - Python collection representing values (unconstrained samples from
   the posterior) at latent sites.
 - **potential_energy** - Potential energy computed at the given value of ``z``.
 - **accept_prob** - Acceptance probability of the proposal. Note that ``z``
   does not correspond to the proposal if it is rejected.
 - **mean_accept_prob** - Mean acceptance probability until current iteration
   during warmup or sampling (for diagnostics).
 - **diverging** - A boolean value to indicate whether the new sample potential energy
   is diverging from the current one.
 - **adapt_state** - A ``SAAdaptState`` namedtuple which contains adaptation information:

   + **zs** - Step size to be used by the integrator in the next iteration.
   + **pes** - Potential energies of `zs`.
   + **loc** - Mean of those `zs`.
   + **inv_mass_matrix_sqrt** - If using dense mass matrix, this is Cholesky of the
     covariance of `zs`. Otherwise, this is standard deviation of those `zs`.

 - **rng_key** - random number generator seed used for the iteration.
"""


def _numpy_delete(x, idx):
    """
    Gets the subarray from `x` where data from index `idx` on the first axis is removed.
    """
    # NB: numpy.delete is not yet available in JAX
    mask = jnp.arange(x.shape[0] - 1) < idx
    return jnp.where(mask.reshape((-1,) + (1,) * (x.ndim - 1)), x[:-1], x[1:])


# TODO: consider to expose this functional style
def _sa(potential_fn=None, potential_fn_gen=None):
    wa_steps = None
    max_delta_energy = 1000.0

    def init_kernel(
        init_params,
        num_warmup,
        adapt_state_size=None,
        inverse_mass_matrix=None,
        dense_mass=False,
        model_args=(),
        model_kwargs=None,
        rng_key=None,
    ):
        rng_key = random.PRNGKey(0) if rng_key is None else rng_key
        nonlocal wa_steps
        wa_steps = num_warmup
        pe_fn = potential_fn
        if potential_fn_gen:
            if pe_fn is not None:
                raise ValueError(
                    "Only one of `potential_fn` or `potential_fn_gen` must be provided."
                )
            else:
                kwargs = {} if model_kwargs is None else model_kwargs
                pe_fn = potential_fn_gen(*model_args, **kwargs)
        rng_key_sa, rng_key_zs, rng_key_z = random.split(rng_key, 3)
        z = init_params
        z_flat, unravel_fn = ravel_pytree(z)
        if inverse_mass_matrix is None:
            inverse_mass_matrix = (
                jnp.identity(z_flat.shape[-1])
                if dense_mass
                else jnp.ones(z_flat.shape[-1])
            )
        inv_mass_matrix_sqrt = (
            jnp.linalg.cholesky(inverse_mass_matrix)
            if dense_mass
            else jnp.sqrt(inverse_mass_matrix)
        )
        if adapt_state_size is None:
            # XXX: heuristic choice
            adapt_state_size = 2 * z_flat.shape[-1]
        else:
            assert adapt_state_size > 1, "adapt_state_size should be greater than 1."
        # NB: mean is init_params
        zs = z_flat + _sample_proposal(
            inv_mass_matrix_sqrt, rng_key_zs, (adapt_state_size,)
        )
        # compute potential energies
        pes = lax.map(lambda z: pe_fn(unravel_fn(z)), zs)
        if dense_mass:
            cov = jnp.cov(zs, rowvar=False, bias=True)
            if cov.shape == ():  # JAX returns scalar for 1D input
                cov = cov.reshape((1, 1))
            cholesky = jnp.linalg.cholesky(cov)
            # if cholesky is NaN, we use the scale from `sample_proposal` here
            inv_mass_matrix_sqrt = jnp.where(
                jnp.any(jnp.isnan(cholesky)), inv_mass_matrix_sqrt, cholesky
            )
        else:
            inv_mass_matrix_sqrt = jnp.std(zs, 0)
        adapt_state = SAAdaptState(zs, pes, jnp.mean(zs, 0), inv_mass_matrix_sqrt)
        k = random.categorical(rng_key_z, jnp.zeros(zs.shape[0]))
        z = unravel_fn(zs[k])
        pe = pes[k]
        sa_state = SAState(
            jnp.array(0),
            z,
            pe,
            jnp.zeros(()),
            jnp.zeros(()),
            jnp.array(False),
            adapt_state,
            rng_key_sa,
        )
        return device_put(sa_state)

    def sample_kernel(sa_state, model_args=(), model_kwargs=None):
        pe_fn = potential_fn
        if potential_fn_gen:
            pe_fn = potential_fn_gen(*model_args, **model_kwargs)
        zs, pes, loc, scale = sa_state.adapt_state
        # we recompute loc/scale after each iteration to avoid precision loss
        # XXX: consider to expose a setting to do this job periodically
        # to save some computations
        loc = jnp.mean(zs, 0)
        if scale.ndim == 2:
            cov = jnp.cov(zs, rowvar=False, bias=True)
            if cov.shape == ():  # JAX returns scalar for 1D input
                cov = cov.reshape((1, 1))
            cholesky = jnp.linalg.cholesky(cov)
            scale = jnp.where(jnp.any(jnp.isnan(cholesky)), scale, cholesky)
        else:
            scale = jnp.std(zs, 0)

        rng_key, rng_key_z, rng_key_reject, rng_key_accept = random.split(
            sa_state.rng_key, 4
        )
        _, unravel_fn = ravel_pytree(sa_state.z)

        z = loc + _sample_proposal(scale, rng_key_z)
        pe = pe_fn(unravel_fn(z))
        pe = jnp.where(jnp.isnan(pe), jnp.inf, pe)
        diverging = (pe - sa_state.potential_energy) > max_delta_energy

        # NB: all terms having the pattern *s will have shape N x ...
        # and all terms having the pattern *s_ will have shape (N + 1) x ...
        locs, scales = _get_proposal_loc_and_scale(zs, loc, scale, z)
        zs_ = jnp.concatenate([zs, z[None, :]])
        pes_ = jnp.concatenate([pes, pe[None]])
        locs_ = jnp.concatenate([locs, loc[None, :]])
        scales_ = jnp.concatenate([scales, scale[None, ...]])
        if scale.ndim == 2:  # dense_mass
            log_weights_ = (
                dist.MultivariateNormal(locs_, scale_tril=scales_).log_prob(zs_) + pes_
            )
        else:
            log_weights_ = dist.Normal(locs_, scales_).log_prob(zs_).sum(-1) + pes_
        # mask invalid values (nan, +inf) by -inf
        log_weights_ = jnp.where(jnp.isfinite(log_weights_), log_weights_, -jnp.inf)
        # get rejecting index
        j = random.categorical(rng_key_reject, log_weights_)
        zs = _numpy_delete(zs_, j)
        pes = _numpy_delete(pes_, j)
        loc = locs_[j]
        scale = scales_[j]
        adapt_state = SAAdaptState(zs, pes, loc, scale)

        # NB: weights[-1] / sum(weights) is the probability of rejecting the new sample `z`.
        accept_prob = 1 - jnp.exp(log_weights_[-1] - logsumexp(log_weights_))
        itr = sa_state.i + 1
        n = jnp.where(sa_state.i < wa_steps, itr, itr - wa_steps)
        mean_accept_prob = (
            sa_state.mean_accept_prob + (accept_prob - sa_state.mean_accept_prob) / n
        )

        # XXX: we make a modification of SA sampler in [1]
        # in [1], each MCMC state contains N points `zs`
        # here we do resampling to pick randomly a point from those N points
        k = random.categorical(rng_key_accept, jnp.zeros(zs.shape[0]))
        z = unravel_fn(zs[k])
        pe = pes[k]
        return SAState(
            itr, z, pe, accept_prob, mean_accept_prob, diverging, adapt_state, rng_key
        )

    return init_kernel, sample_kernel


# TODO: this shares almost the same code as HMC, so we can abstract out much of the implementation
class SA(MCMCKernel):
    """
    Sample Adaptive MCMC, a gradient-free sampler.

    This is a very fast (in term of n_eff / s) sampler but requires
    many warmup (burn-in) steps. In each MCMC step, we only need to
    evaluate potential function at one point.

    Note that unlike in reference [1], we return a randomly selected (i.e. thinned)
    subset of approximate posterior samples of size num_chains x num_samples
    instead of num_chains x num_samples x adapt_state_size.

    .. note:: We recommend to use this kernel with `progress_bar=False` in
        :class:`~numpyro.infer.mcmc.MCMC` to reduce JAX's dispatch overhead.

    **References:**

    1. *Sample Adaptive MCMC* (https://papers.nips.cc/paper/9107-sample-adaptive-mcmc),
       Michael Zhu

    :param model: Python callable containing Pyro :mod:`~numpyro.primitives`.
        If model is provided, `potential_fn` will be inferred using the model.
    :param potential_fn: Python callable that computes the potential energy
        given input parameters. The input parameters to `potential_fn` can be
        any python collection type, provided that `init_params` argument to
        :meth:`init` has the same type.
    :param int adapt_state_size: The number of points to generate proposal
        distribution. Defaults to 2 times latent size.
    :param bool dense_mass:  A flag to decide if mass matrix is dense or
        diagonal (default to ``dense_mass=True``)
    :param callable init_strategy: a per-site initialization function.
        See :ref:`init_strategy` section for available functions.
    """

    def __init__(
        self,
        model=None,
        potential_fn=None,
        adapt_state_size=None,
        dense_mass=True,
        init_strategy=init_to_uniform,
    ):
        if not (model is None) ^ (potential_fn is None):
            raise ValueError("Only one of `model` or `potential_fn` must be specified.")
        self._model = model
        self._potential_fn = potential_fn
        self._adapt_state_size = adapt_state_size
        self._dense_mass = dense_mass
        self._init_strategy = init_strategy
        self._init_fn = None
        self._potential_fn_gen = None
        self._postprocess_fn = None
        self._sample_fn = None

    def _init_state(self, rng_key, model_args, model_kwargs, init_params):
        if self._model is not None:
            init_params, potential_fn, postprocess_fn, _ = initialize_model(
                rng_key,
                self._model,
                dynamic_args=True,
                init_strategy=self._init_strategy,
                model_args=model_args,
                model_kwargs=model_kwargs,
                validate_grad=False,
            )
            init_params = init_params[0]
            # NB: init args is different from HMC
            self._init_fn, sample_fn = _sa(potential_fn_gen=potential_fn)
            self._potential_fn_gen = potential_fn
            if self._postprocess_fn is None:
                self._postprocess_fn = postprocess_fn
        else:
            self._init_fn, sample_fn = _sa(potential_fn=self._potential_fn)

        if self._sample_fn is None:
            self._sample_fn = sample_fn
        return init_params

    def init(
        self, rng_key, num_warmup, init_params=None, model_args=(), model_kwargs={}
    ):
        # non-vectorized
        if is_prng_key(rng_key):
            rng_key, rng_key_init_model = random.split(rng_key)
        # vectorized
        else:
            rng_key, rng_key_init_model = jnp.swapaxes(
                vmap(random.split)(rng_key), 0, 1
            )
            # we need only a single key for initializing PE / constraints fn
            rng_key_init_model = rng_key_init_model[0]
        init_params = self._init_state(
            rng_key_init_model, model_args, model_kwargs, init_params
        )
        if self._potential_fn and init_params is None:
            raise ValueError(
                "Valid value of `init_params` must be provided with" " `potential_fn`."
            )

        # NB: init args is different from HMC
        sa_init_fn = lambda init_params, rng_key: self._init_fn(  # noqa: E731
            init_params,
            num_warmup=num_warmup,
            adapt_state_size=self._adapt_state_size,
            dense_mass=self._dense_mass,
            rng_key=rng_key,
            model_args=model_args,
            model_kwargs=model_kwargs,
        )
        if is_prng_key(rng_key):
            init_state = sa_init_fn(init_params, rng_key)
        else:
            init_state = vmap(sa_init_fn)(init_params, rng_key)
            sample_fn = vmap(self._sample_fn, in_axes=(0, None, None))
            self._sample_fn = sample_fn
        return init_state

    @property
    def model(self):
        return self._model

    @property
    def sample_field(self):
        return "z"

    @property
    def default_fields(self):
        return ("z", "diverging")

    def get_diagnostics_str(self, state):
        return "acc. prob={:.2f}".format(state.mean_accept_prob)

    def postprocess_fn(self, args, kwargs):
        if self._postprocess_fn is None:
            return identity
        return self._postprocess_fn(*args, **kwargs)

    def sample(self, state, model_args, model_kwargs):
        """
        Run SA from the given :data:`~numpyro.infer.sa.SAState` and return the resulting
        :data:`~numpyro.infer.sa.SAState`.

        :param SAState state: Represents the current state.
        :param model_args: Arguments provided to the model.
        :param model_kwargs: Keyword arguments provided to the model.
        :return: Next `state` after running SA.
        """
        return self._sample_fn(state, model_args, model_kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_sample_fn"] = None
        state["_init_fn"] = None
        return state
