# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
import copy
from functools import partial

from jax import device_put, ops, random, value_and_grad
import jax.numpy as jnp
from jax.scipy.special import expit

from numpyro.distributions import biject_to
from numpyro.handlers import condition, seed, trace, substitute
from numpyro.infer.hmc import HMC
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import log_likelihood, potential_energy, _guess_max_plate_nesting
from numpyro.util import cond, fori_loop, identity, ravel_pytree


HMCGibbsState = namedtuple("HMCGibbsState", "z, hmc_state, rng_key")
"""
 - **z** - a dict of the current latent values (both HMC and Gibbs sites)
 - **hmc_state** - current hmc_state
 - **rng_key** - random key for the current step
"""


def _wrap_model(model):
    def fn(*args, **kwargs):
        gibbs_values = kwargs.pop("_gibbs_sites", {})
        with condition(data=gibbs_values), substitute(data=gibbs_values):
            model(*args, **kwargs)
    return fn


class HMCGibbs(MCMCKernel):
    """
    [EXPERIMENTAL INTERFACE]

    HMC-within-Gibbs. This inference algorithm allows the user to combine
    general purpose gradient-based inference (HMC or NUTS) with custom
    Gibbs samplers.

    Note that it is the user's responsibility to provide a correct implementation
    of `gibbs_fn` that samples from the corresponding posterior conditional.

    :param inner_kernel: One of :class:`~numpyro.infer.hmc.HMC` or :class:`~numpyro.infer.hmc.NUTS`.
    :param gibbs_fn: A Python callable that returns a dictionary of Gibbs samples conditioned
        on the HMC sites. Must include an argument `rng_key` that should be used for all sampling.
        Must also include arguments `hmc_sites` and `gibbs_sites`, each of which is a dictionary
        with keys that are site names and values that are sample values. Note that a given `gibbs_fn`
        may not need make use of all these sample values.
    :param gibbs_sites: a list of site names for the latent variables that are covered by the Gibbs sampler.

    **Example**

    .. doctest::

        >>> from jax import random
        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.infer import MCMC, NUTS, HMCGibbs
        ...
        >>> def model():
        ...     x = numpyro.sample("x", dist.Normal(0.0, 2.0))
        ...     y = numpyro.sample("y", dist.Normal(0.0, 2.0))
        ...     numpyro.sample("obs", dist.Normal(x + y, 1.0), obs=jnp.array([1.0]))
        ...
        >>> def gibbs_fn(rng_key, gibbs_sites, hmc_sites):
        ...    y = hmc_sites['y']
        ...    new_x = dist.Normal(0.8 * (1-y), jnp.sqrt(0.8)).sample(rng_key)
        ...    return {'x': new_x}
        ...
        >>> hmc_kernel = NUTS(model)
        >>> kernel = HMCGibbs(hmc_kernel, gibbs_fn=gibbs_fn, gibbs_sites=['x'])
        >>> mcmc = MCMC(kernel, 100, 100, progress_bar=False)
        >>> mcmc.run(random.PRNGKey(0))
        >>> mcmc.print_summary()  # doctest: +SKIP

    """

    sample_field = "z"

    def __init__(self, inner_kernel, gibbs_fn, gibbs_sites):
        if not isinstance(inner_kernel, HMC):
            raise ValueError("inner_kernel must be a HMC or NUTS sampler.")
        if not callable(gibbs_fn):
            raise ValueError("gibbs_fn must be a callable")
        assert inner_kernel.model is not None, "HMCGibbs does not support models specified via a potential function."

        self.inner_kernel = copy.copy(inner_kernel)
        self.inner_kernel._model = _wrap_model(inner_kernel.model)
        self._gibbs_sites = gibbs_sites
        self._gibbs_fn = gibbs_fn

    @property
    def model(self):
        return self.inner_kernel._model

    def get_diagnostics_str(self, state):
        state = state.hmc_state
        return '{} steps of size {:.2e}. acc. prob={:.2f}'.format(state.num_steps,
                                                                  state.adapt_state.step_size,
                                                                  state.mean_accept_prob)

    def postprocess_fn(self, args, kwargs):
        def fn(z):
            model_kwargs = {} if kwargs is None else kwargs.copy()
            hmc_sites = {k: v for k, v in z.items() if k not in self._gibbs_sites}
            gibbs_sites = {k: v for k, v in z.items() if k in self._gibbs_sites}
            model_kwargs["_gibbs_sites"] = gibbs_sites
            hmc_sites = self.inner_kernel.postprocess_fn(args, model_kwargs)(hmc_sites)
            return {**gibbs_sites, **hmc_sites}

        return fn

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        model_kwargs = {} if model_kwargs is None else model_kwargs.copy()
        rng_key, key_u, key_z = random.split(rng_key, 3)
        prototype_trace = trace(seed(self.model, key_u)).get_trace(*model_args, **model_kwargs)

        gibbs_sites = {name: site["value"] for name, site in prototype_trace.items() if name in self._gibbs_sites}
        model_kwargs["_gibbs_sites"] = gibbs_sites
        hmc_state = self.inner_kernel.init(key_z, num_warmup, init_params, model_args, model_kwargs)

        z = {**gibbs_sites, **hmc_state.z}

        return device_put(HMCGibbsState(z, hmc_state, rng_key))

    def sample(self, state, model_args, model_kwargs):
        model_kwargs = {} if model_kwargs is None else model_kwargs
        rng_key, rng_gibbs = random.split(state.rng_key)

        def potential_fn(z_gibbs, z_hmc):
            return self.inner_kernel._potential_fn_gen(
                *model_args, _gibbs_sites=z_gibbs, **model_kwargs)(z_hmc)

        z_gibbs = {k: v for k, v in state.z.items() if k not in state.hmc_state.z}
        z_hmc = {k: v for k, v in state.z.items() if k in state.hmc_state.z}
        model_kwargs_ = model_kwargs.copy()
        model_kwargs_["_gibbs_sites"] = z_gibbs
        # TODO: give the user more control over which sites are transformed from unconstrained to constrained space
        z_hmc = self.inner_kernel.postprocess_fn(model_args, model_kwargs_)(z_hmc)

        z_gibbs = self._gibbs_fn(rng_key=rng_gibbs, gibbs_sites=z_gibbs, hmc_sites=z_hmc)

        pe, z_grad = value_and_grad(partial(potential_fn, z_gibbs))(state.hmc_state.z)
        hmc_state = state.hmc_state._replace(z_grad=z_grad, potential_energy=pe)

        model_kwargs_["_gibbs_sites"] = z_gibbs
        hmc_state = self.inner_kernel.sample(hmc_state, model_args, model_kwargs_)

        z = {**z_gibbs, **hmc_state.z}

        return HMCGibbsState(z, hmc_state, rng_key)


def _discrete_gibbs_proposal_body_fn(z_init_flat, unravel_fn, pe_init, potential_fn, idx, i, val):
    rng_key, z, pe, log_weight_sum = val
    rng_key, rng_transition = random.split(rng_key)
    proposal = jnp.where(i >= z_init_flat[idx], i + 1, i)
    z_new_flat = ops.index_update(z_init_flat, idx, proposal)
    z_new = unravel_fn(z_new_flat)
    pe_new = potential_fn(z_new)
    log_weight_new = pe_init - pe_new
    # Handles the NaN case...
    log_weight_new = jnp.where(jnp.isfinite(log_weight_new), log_weight_new, -jnp.inf)
    # transition_prob = e^weight_new / (e^weight_logsumexp + e^weight_new)
    transition_prob = expit(log_weight_new - log_weight_sum)
    z, pe = cond(random.bernoulli(rng_transition, transition_prob),
                 (z_new, pe_new), identity,
                 (z, pe), identity)
    log_weight_sum = jnp.logaddexp(log_weight_new, log_weight_sum)
    return rng_key, z, pe, log_weight_sum


def _discrete_gibbs_proposal(rng_key, z_discrete, pe, potential_fn, idx, support_size):
    # idx: current index of `z_discrete_flat` to update
    # support_size: support size of z_discrete at the index idx

    z_discrete_flat, unravel_fn = ravel_pytree(z_discrete)
    # Here we loop over the support of z_flat[idx] to get z_new
    # XXX: we can't vmap potential_fn over all proposals and sample from the conditional
    # categorical distribution because support_size is a traced value, i.e. its value
    # might change across different discrete variables;
    # so here we will loop over all proposals and use an online scheme to sample from
    # the conditional categorical distribution
    body_fn = partial(_discrete_gibbs_proposal_body_fn,
                      z_discrete_flat, unravel_fn, pe, potential_fn, idx)
    init_val = (rng_key, z_discrete, pe, jnp.array(0.))
    rng_key, z_new, pe_new, _ = fori_loop(0, support_size - 1, body_fn, init_val)
    log_accept_ratio = jnp.array(0.)
    return rng_key, z_new, pe_new, log_accept_ratio


def _discrete_modified_gibbs_proposal(rng_key, z_discrete, pe, potential_fn, idx, support_size,
                                      stay_prob=0.):
    assert isinstance(stay_prob, float) and stay_prob >= 0. and stay_prob < 1
    z_discrete_flat, unravel_fn = ravel_pytree(z_discrete)
    body_fn = partial(_discrete_gibbs_proposal_body_fn,
                      z_discrete_flat, unravel_fn, pe, potential_fn, idx)
    # like gibbs_step but here, weight of the current value is 0
    init_val = (rng_key, z_discrete, pe, jnp.array(-jnp.inf))
    rng_key, z_new, pe_new, log_weight_sum = fori_loop(0, support_size - 1, body_fn, init_val)
    rng_key, rng_stay = random.split(rng_key)
    z_new, pe_new = cond(random.bernoulli(rng_stay, stay_prob),
                         (z_discrete, pe), identity,
                         (z_new, pe_new), identity)
    # here we calculate the MH correction: (1 - P(z)) / (1 - P(z_new))
    # where 1 - P(z) ~ weight_sum
    # and 1 - P(z_new) ~ 1 + weight_sum - z_new_weight
    log_accept_ratio = log_weight_sum - jnp.log(jnp.exp(log_weight_sum) - jnp.expm1(pe - pe_new))
    return rng_key, z_new, pe_new, log_accept_ratio


def _discrete_rw_proposal(rng_key, z_discrete, pe, potential_fn, idx, support_size):
    rng_key, rng_proposal = random.split(rng_key, 2)
    z_discrete_flat, unravel_fn = ravel_pytree(z_discrete)

    proposal = random.randint(rng_proposal, (), minval=0, maxval=support_size)
    z_new_flat = ops.index_update(z_discrete_flat, idx, proposal)
    z_new = unravel_fn(z_new_flat)
    pe_new = potential_fn(z_new)
    log_accept_ratio = pe - pe_new
    return rng_key, z_new, pe_new, log_accept_ratio


def _discrete_modified_rw_proposal(rng_key, z_discrete, pe, potential_fn, idx, support_size,
                                   stay_prob=0.):
    assert isinstance(stay_prob, float) and stay_prob >= 0. and stay_prob < 1
    rng_key, rng_proposal, rng_stay = random.split(rng_key, 3)
    z_discrete_flat, unravel_fn = ravel_pytree(z_discrete)

    i = random.randint(rng_proposal, (), minval=0, maxval=support_size - 1)
    proposal = jnp.where(i >= z_discrete_flat[idx], i + 1, i)
    proposal = jnp.where(random.bernoulli(rng_stay, stay_prob), idx, proposal)
    z_new_flat = ops.index_update(z_discrete_flat, idx, proposal)
    z_new = unravel_fn(z_new_flat)
    pe_new = potential_fn(z_new)
    log_accept_ratio = pe - pe_new
    return rng_key, z_new, pe_new, log_accept_ratio


def discrete_gibbs_fn(model, model_args=(), model_kwargs={}, *, random_walk=False, modified=False):
    """
    [EXPERIMENTAL INTERFACE]

    Returns a gibbs_fn to be used in :class:`HMCGibbs`, which works for discrete latent sites
    with enumerate support. The site update order is randomly permuted at each step.

    Note that those discrete latent sites that are not specified in the constructor of
    :class:`HMCGibbs` will be marginalized out by default (if they have enumerate supports).

    :param callable model: a callable with NumPyro primitives. This should be the same model
        as the one used in the `inner_kernel` of :class:`HMCGibbs`.
    :param tuple model_args: Arguments provided to the model.
    :param dict model_kwargs: Keyword arguments provided to the model.
    :param bool random_walk: If False, Gibbs sampling will be used to draw a sample from the
        conditional `p(gibbs_site | remaining sites)`. Otherwise, a sample will be drawn uniformly
        from the domain of `gibbs_site`.
    :param bool modified: whether to use a modified proposal, as suggested in reference [1], which
        always proposes a new state for the current Gibbs site.
        The modified scheme appears in the literature under the name "modified Gibbs sampler" or
        "Metropolised Gibbs sampler".
    :return: a callable `gibbs_fn` to be used in :class:`HMCGibbs`

    **References:**

    1. *Peskun's theorem and a modified discrete-state Gibbs sampler*,
       Liu, J. S. (1996)

    **Example**

    .. doctest::

        >>> from jax import random
        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.infer import MCMC, NUTS, HMCGibbs, discrete_gibbs_fn
        ...
        >>> def model(probs, locs):
        ...     c = numpyro.sample("c", dist.Categorical(probs))
        ...     numpyro.sample("x", dist.Normal(locs[c], 0.5))
        ...
        >>> probs = jnp.array([0.15, 0.3, 0.3, 0.25])
        >>> locs = jnp.array([-2, 0, 2, 4])
        >>> gibbs_fn = discrete_gibbs_fn(model, (probs, locs))
        >>> kernel = HMCGibbs(NUTS(model), gibbs_fn, gibbs_sites=["c"])
        >>> mcmc = MCMC(kernel, 1000, 100000, progress_bar=False)
        >>> mcmc.run(random.PRNGKey(0), probs, locs)
        >>> mcmc.print_summary()  # doctest: +SKIP

    """
    # NB: all of the information such as `model`, `model_args`, `model_kwargs`
    # can be accessed from HMCGibbs.sample but we require them here to
    # simplify the api of `gibbs_fn`
    prototype_trace = trace(seed(model, rng_seed=0)).get_trace(*model_args, **model_kwargs)
    support_sizes = {
        name: jnp.broadcast_to(site["fn"].enumerate_support(False).shape[0], jnp.shape(site["value"]))
        for name, site in prototype_trace.items()
        if site["type"] == "sample" and site["fn"].has_enumerate_support and not site["is_observed"]
    }
    max_plate_nesting = _guess_max_plate_nesting(prototype_trace)
    if random_walk:
        if modified:
            proposal_fn = partial(_discrete_modified_rw_proposal, stay_prob=0.)
        else:
            proposal_fn = _discrete_rw_proposal
    else:
        if modified:
            proposal_fn = partial(_discrete_modified_gibbs_proposal, stay_prob=0.)
        else:
            proposal_fn = _discrete_gibbs_proposal

    def gibbs_fn(rng_key, gibbs_sites, hmc_sites):
        # convert to unconstrained values
        z_hmc = {k: biject_to(prototype_trace[k]["fn"].support).inv(v)
                 for k, v in hmc_sites.items()
                 if k in prototype_trace and prototype_trace[k]["type"] == "sample"}
        use_enum = len(set(support_sizes) - set(gibbs_sites)) > 0
        wrapped_model = _wrap_model(model)
        if use_enum:
            from numpyro.contrib.funsor import config_enumerate, enum

            wrapped_model = enum(config_enumerate(wrapped_model), -max_plate_nesting - 1)

        def potential_fn(z_discrete):
            model_kwargs_ = model_kwargs.copy()
            model_kwargs_["_gibbs_sites"] = z_discrete
            return potential_energy(wrapped_model, model_args, model_kwargs_, z_hmc, enum=use_enum)

        # get support_sizes of gibbs_sites
        support_sizes_flat, _ = ravel_pytree({k: support_sizes[k] for k in gibbs_sites})
        num_discretes = support_sizes_flat.shape[0]

        rng_key, rng_permute = random.split(rng_key)
        idxs = random.permutation(rng_key, jnp.arange(num_discretes))

        def body_fn(i, val):
            idx = idxs[i]
            support_size = support_sizes_flat[idx]
            rng_key, z, pe = val
            rng_key, z_new, pe_new, log_accept_ratio = proposal_fn(
                rng_key, z, pe, potential_fn=potential_fn, idx=idx, support_size=support_size)
            rng_key, rng_accept = random.split(rng_key)
            # u ~ Uniform(0, 1), u < accept_ratio => -log(u) > -log_accept_ratio
            # and -log(u) ~ exponential(1)
            z, pe = cond(random.exponential(rng_accept) > -log_accept_ratio,
                         (z_new, pe_new), identity,
                         (z, pe), identity)
            return rng_key, z, pe

        init_val = (rng_key, gibbs_sites, potential_fn(gibbs_sites))
        _, gibbs_sites, _ = fori_loop(0, num_discretes, body_fn, init_val)
        return gibbs_sites

    return gibbs_fn


def subsample_gibbs_fn(model, model_args=(), model_kwargs={}):
    """
    [EXPERIMENTAL INTERFACE]

    Returns a gibbs_fn to be used in :class:`HMCGibbs`, which works for subsampling
    statements using :class:`~numpyro.plate` primitive. This implements the Algorithm 1
    of reference [1] but uses a naive estimation (without control variates) of log likelihood,
    hence might incur a high variance.

    **References:**

    1. *Hamiltonian Monte Carlo with energy conserving subsampling*,
       Dang, K. D., Quiroz, M., Kohn, R., Minh-Ngoc, T., & Villani, M. (2019)
    2. *Speeding Up MCMC by Efficient Data Subsampling*,
       Quiroz, M., Kohn, R., Villani, M., & Tran, M. N. (2018)

    :param callable model: a callable with NumPyro primitives. This should be the same model
        as the one used in the `inner_kernel` of :class:`HMCGibbs`.
    :param tuple model_args: Arguments provided to the model.
    :param dict model_kwargs: Keyword arguments provided to the model.
    :return: a callable `gibbs_fn` to be used in :class:`HMCGibbs`

    **Example**

    .. doctest::

        >>> from jax import random
        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.infer import MCMC, NUTS, HMCGibbs, subsample_gibbs_fn
        ...
        >>> def model(data):
        ...     x = numpyro.sample("x", dist.Normal(0, 1))
        ...     with numpyro.plate("N", data.shape[0], subsample_size=100):
        ...         batch = numpyro.subsample(data, event_dim=0)
        ...         numpyro.sample("obs", dist.Normal(x, 1), obs=batch)
        ...
        >>> data = random.normal(random.PRNGKey(0), (10000,)) + 1
        >>> gibbs_fn = subsample_gibbs_fn(model, (data,))
        >>> kernel = HMCGibbs(NUTS(model), gibbs_fn, gibbs_sites=["N"])
        >>> mcmc = MCMC(kernel, 1000, 1000)
        >>> mcmc.run(random.PRNGKey(0), data)
        >>> samples = mcmc.get_samples()["x"]
        >>> assert abs(jnp.mean(samples).copy() - 1.) < 0.1

    """
    prototype_trace = trace(seed(model, rng_seed=0)).get_trace(*model_args, **model_kwargs)
    plate_sizes = {
        name: site["args"]
        for name, site in prototype_trace.items()
        if site["type"] == "plate" and site["args"][0] > site["args"][1]  # i.e. size > subsample_size
    }
    enum = any(site["type"] == "sample"
               and not site["is_observed"]
               and site["fn"].has_enumerate_support
               for name, site in prototype_trace.items())
    assert not enum, "Enumeration is not supported for subsample_gibbs_fn."

    def gibbs_fn(rng_key, gibbs_sites, hmc_sites):
        assert set(gibbs_sites) == set(plate_sizes)
        u_new = {}
        for name in gibbs_sites:
            size, subsample_size = plate_sizes[name]
            rng_key, subkey = random.split(rng_key)
            u_new[name] = random.choice(subkey, size, (subsample_size,), replace=False)

        u_loglik = log_likelihood(_wrap_model(model), hmc_sites, *model_args, batch_ndims=0,
                                  **model_kwargs, _gibbs_sites=gibbs_sites)
        u_loglik = sum(v.sum() for v in u_loglik.values())
        u_new_loglik = log_likelihood(_wrap_model(model), hmc_sites, *model_args, batch_ndims=0,
                                      **model_kwargs, _gibbs_sites=u_new)
        u_new_loglik = sum(v.sum() for v in u_new_loglik.values())
        accept_prob = jnp.clip(jnp.exp(u_new_loglik - u_loglik), a_max=1.0)
        return cond(random.bernoulli(rng_key, accept_prob), u_new, identity, gibbs_sites, identity)

    return gibbs_fn
