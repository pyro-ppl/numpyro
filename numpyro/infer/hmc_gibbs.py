# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict, namedtuple
import copy
from functools import partial
import warnings

import numpy as np

from jax import (
    device_put,
    grad,
    hessian,
    jacfwd,
    jacobian,
    lax,
    ops,
    random,
    value_and_grad,
)
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
from jax.scipy.special import expit

import numpyro
from numpyro.distributions.transforms import biject_to
from numpyro.handlers import block, condition, seed, substitute, trace
from numpyro.infer.hmc import HMC
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import _unconstrain_reparam
from numpyro.util import cond, fori_loop, identity

HMCGibbsState = namedtuple("HMCGibbsState", "z, hmc_state, rng_key")
"""
 - **z** - a dict of the current latent values (both HMC and Gibbs sites)
 - **hmc_state** - current hmc_state
 - **rng_key** - random key for the current step
"""


def _wrap_model(model, *args, **kwargs):
    gibbs_values = kwargs.pop("_gibbs_sites", {})
    with condition(data=gibbs_values), substitute(data=gibbs_values):
        return model(*args, **kwargs)


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
    :param list gibbs_sites: a list of site names for the latent variables that are covered by the Gibbs sampler.

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
        >>> mcmc = MCMC(kernel, num_warmup=100, num_samples=100, progress_bar=False)
        >>> mcmc.run(random.PRNGKey(0))
        >>> mcmc.print_summary()  # doctest: +SKIP

    """

    sample_field = "z"

    def __init__(self, inner_kernel, gibbs_fn, gibbs_sites):
        if not isinstance(inner_kernel, HMC):
            raise ValueError("inner_kernel must be a HMC or NUTS sampler.")
        if not callable(gibbs_fn):
            raise ValueError("gibbs_fn must be a callable")
        assert (
            inner_kernel.model is not None
        ), "HMCGibbs does not support models specified via a potential function."

        self.inner_kernel = copy.copy(inner_kernel)
        self.inner_kernel._model = partial(_wrap_model, inner_kernel.model)
        self._gibbs_sites = gibbs_sites
        self._gibbs_fn = gibbs_fn
        self._prototype_trace = None

    @property
    def model(self):
        return self.inner_kernel._model

    def get_diagnostics_str(self, state):
        state = state.hmc_state
        return "{} steps of size {:.2e}. acc. prob={:.2f}".format(
            state.num_steps, state.adapt_state.step_size, state.mean_accept_prob
        )

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
        if self._prototype_trace is None:
            rng_key, key_u = random.split(rng_key)
            self._prototype_trace = trace(seed(self.model, key_u)).get_trace(
                *model_args, **model_kwargs
            )

        rng_key, key_z = random.split(rng_key)
        gibbs_sites = {
            name: site["value"]
            for name, site in self._prototype_trace.items()
            if name in self._gibbs_sites
        }
        model_kwargs["_gibbs_sites"] = gibbs_sites
        hmc_state = self.inner_kernel.init(
            key_z, num_warmup, init_params, model_args, model_kwargs
        )

        z = {**gibbs_sites, **hmc_state.z}

        return device_put(HMCGibbsState(z, hmc_state, rng_key))

    def sample(self, state, model_args, model_kwargs):
        model_kwargs = {} if model_kwargs is None else model_kwargs
        rng_key, rng_gibbs = random.split(state.rng_key)

        def potential_fn(z_gibbs, z_hmc):
            return self.inner_kernel._potential_fn_gen(
                *model_args, _gibbs_sites=z_gibbs, **model_kwargs
            )(z_hmc)

        z_gibbs = {k: v for k, v in state.z.items() if k not in state.hmc_state.z}
        z_hmc = {k: v for k, v in state.z.items() if k in state.hmc_state.z}
        model_kwargs_ = model_kwargs.copy()
        model_kwargs_["_gibbs_sites"] = z_gibbs
        z_hmc = self.inner_kernel.postprocess_fn(model_args, model_kwargs_)(z_hmc)

        z_gibbs = self._gibbs_fn(
            rng_key=rng_gibbs, gibbs_sites=z_gibbs, hmc_sites=z_hmc
        )

        if self.inner_kernel._forward_mode_differentiation:
            pe = potential_fn(z_gibbs, state.hmc_state.z)
            z_grad = jacfwd(partial(potential_fn, z_gibbs))(state.hmc_state.z)
        else:
            pe, z_grad = value_and_grad(partial(potential_fn, z_gibbs))(
                state.hmc_state.z
            )
        hmc_state = state.hmc_state._replace(z_grad=z_grad, potential_energy=pe)

        model_kwargs_["_gibbs_sites"] = z_gibbs
        hmc_state = self.inner_kernel.sample(hmc_state, model_args, model_kwargs_)

        z = {**z_gibbs, **hmc_state.z}

        return HMCGibbsState(z, hmc_state, rng_key)


def _discrete_gibbs_proposal_body_fn(
    z_init_flat, unravel_fn, pe_init, potential_fn, idx, i, val
):
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
    z, pe = cond(
        random.bernoulli(rng_transition, transition_prob),
        (z_new, pe_new),
        identity,
        (z, pe),
        identity,
    )
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
    body_fn = partial(
        _discrete_gibbs_proposal_body_fn,
        z_discrete_flat,
        unravel_fn,
        pe,
        potential_fn,
        idx,
    )
    init_val = (rng_key, z_discrete, pe, jnp.array(0.0))
    rng_key, z_new, pe_new, _ = fori_loop(0, support_size - 1, body_fn, init_val)
    log_accept_ratio = jnp.array(0.0)
    return rng_key, z_new, pe_new, log_accept_ratio


def _discrete_modified_gibbs_proposal(
    rng_key, z_discrete, pe, potential_fn, idx, support_size, stay_prob=0.0
):
    assert isinstance(stay_prob, float) and stay_prob >= 0.0 and stay_prob < 1
    z_discrete_flat, unravel_fn = ravel_pytree(z_discrete)
    body_fn = partial(
        _discrete_gibbs_proposal_body_fn,
        z_discrete_flat,
        unravel_fn,
        pe,
        potential_fn,
        idx,
    )
    # like gibbs_step but here, weight of the current value is 0
    init_val = (rng_key, z_discrete, pe, jnp.array(-jnp.inf))
    rng_key, z_new, pe_new, log_weight_sum = fori_loop(
        0, support_size - 1, body_fn, init_val
    )
    rng_key, rng_stay = random.split(rng_key)
    z_new, pe_new = cond(
        random.bernoulli(rng_stay, stay_prob),
        (z_discrete, pe),
        identity,
        (z_new, pe_new),
        identity,
    )
    # here we calculate the MH correction: (1 - P(z)) / (1 - P(z_new))
    # where 1 - P(z) ~ weight_sum
    # and 1 - P(z_new) ~ 1 + weight_sum - z_new_weight
    log_accept_ratio = log_weight_sum - jnp.log(
        jnp.exp(log_weight_sum) - jnp.expm1(pe - pe_new)
    )
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


def _discrete_modified_rw_proposal(
    rng_key, z_discrete, pe, potential_fn, idx, support_size, stay_prob=0.0
):
    assert isinstance(stay_prob, float) and stay_prob >= 0.0 and stay_prob < 1
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


def _discrete_gibbs_fn(potential_fn, support_sizes, proposal_fn):
    def gibbs_fn(rng_key, gibbs_sites, hmc_sites, pe):
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
                rng_key,
                z,
                pe,
                potential_fn=partial(potential_fn, z_hmc=hmc_sites),
                idx=idx,
                support_size=support_size,
            )
            rng_key, rng_accept = random.split(rng_key)
            # u ~ Uniform(0, 1), u < accept_ratio => -log(u) > -log_accept_ratio
            # and -log(u) ~ exponential(1)
            z, pe = cond(
                random.exponential(rng_accept) > -log_accept_ratio,
                (z_new, pe_new),
                identity,
                (z, pe),
                identity,
            )
            return rng_key, z, pe

        init_val = (rng_key, gibbs_sites, pe)
        _, gibbs_sites, pe = fori_loop(0, num_discretes, body_fn, init_val)
        return gibbs_sites, pe

    return gibbs_fn


class DiscreteHMCGibbs(HMCGibbs):
    """
    [EXPERIMENTAL INTERFACE]

    A subclass of :class:`HMCGibbs` which performs Metropolis updates for discrete latent sites.

    .. note:: The site update order is randomly permuted at each step.

    .. note:: This class supports enumeration of discrete latent variables. To marginalize out a
        discrete latent site, we can specify `infer={'enumerate': 'parallel'}` keyword in its
        corresponding :func:`~numpyro.primitives.sample` statement.

    :param inner_kernel: One of :class:`~numpyro.infer.hmc.HMC` or :class:`~numpyro.infer.hmc.NUTS`.
    :param bool random_walk: If False, Gibbs sampling will be used to draw a sample from the
        conditional `p(gibbs_site | remaining sites)`. Otherwise, a sample will be drawn uniformly
        from the domain of `gibbs_site`. Defaults to False.
    :param bool modified: whether to use a modified proposal, as suggested in reference [1], which
        always proposes a new state for the current Gibbs site. Defaults to False.
        The modified scheme appears in the literature under the name "modified Gibbs sampler" or
        "Metropolised Gibbs sampler".

    **References:**

    1. *Peskun's theorem and a modified discrete-state Gibbs sampler*,
       Liu, J. S. (1996)

    **Example**

    .. doctest::

        >>> from jax import random
        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.infer import DiscreteHMCGibbs, MCMC, NUTS
        ...
        >>> def model(probs, locs):
        ...     c = numpyro.sample("c", dist.Categorical(probs))
        ...     numpyro.sample("x", dist.Normal(locs[c], 0.5))
        ...
        >>> probs = jnp.array([0.15, 0.3, 0.3, 0.25])
        >>> locs = jnp.array([-2, 0, 2, 4])
        >>> kernel = DiscreteHMCGibbs(NUTS(model), modified=True)
        >>> mcmc = MCMC(kernel, num_warmup=1000, num_samples=100000, progress_bar=False)
        >>> mcmc.run(random.PRNGKey(0), probs, locs)
        >>> mcmc.print_summary()  # doctest: +SKIP
        >>> samples = mcmc.get_samples()["x"]
        >>> assert abs(jnp.mean(samples) - 1.3) < 0.1
        >>> assert abs(jnp.var(samples) - 4.36) < 0.5

    """

    def __init__(self, inner_kernel, *, random_walk=False, modified=False):
        super().__init__(inner_kernel, identity, None)
        self._random_walk = random_walk
        self._modified = modified
        if random_walk:
            if modified:
                self._discrete_proposal_fn = partial(
                    _discrete_modified_rw_proposal, stay_prob=0.0
                )
            else:
                self._discrete_proposal_fn = _discrete_rw_proposal
        else:
            if modified:
                self._discrete_proposal_fn = partial(
                    _discrete_modified_gibbs_proposal, stay_prob=0.0
                )
            else:
                self._discrete_proposal_fn = _discrete_gibbs_proposal

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        model_kwargs = {} if model_kwargs is None else model_kwargs.copy()
        rng_key, key_u = random.split(rng_key)
        self._prototype_trace = trace(seed(self.model, key_u)).get_trace(
            *model_args, **model_kwargs
        )

        self._support_sizes = {
            name: np.broadcast_to(
                site["fn"].enumerate_support(False).shape[0], jnp.shape(site["value"])
            )
            for name, site in self._prototype_trace.items()
            if site["type"] == "sample"
            and site["fn"].has_enumerate_support
            and not site["is_observed"]
        }
        self._gibbs_sites = [
            name
            for name, site in self._prototype_trace.items()
            if site["type"] == "sample"
            and site["fn"].has_enumerate_support
            and not site["is_observed"]
            and site["infer"].get("enumerate", "") != "parallel"
        ]
        assert (
            self._gibbs_sites
        ), "Cannot detect any discrete latent variables in the model."
        return super().init(rng_key, num_warmup, init_params, model_args, model_kwargs)

    def sample(self, state, model_args, model_kwargs):
        model_kwargs = {} if model_kwargs is None else model_kwargs
        rng_key, rng_gibbs = random.split(state.rng_key)

        def potential_fn(z_gibbs, z_hmc):
            return self.inner_kernel._potential_fn_gen(
                *model_args, _gibbs_sites=z_gibbs, **model_kwargs
            )(z_hmc)

        z_gibbs = {k: v for k, v in state.z.items() if k not in state.hmc_state.z}
        z_hmc = {k: v for k, v in state.z.items() if k in state.hmc_state.z}
        model_kwargs_ = model_kwargs.copy()
        model_kwargs_["_gibbs_sites"] = z_gibbs

        # different from the implementation in HMCGibbs.sample, we feed the current potential energy
        # and get new potential energy from gibbs_fn
        gibbs_fn = _discrete_gibbs_fn(
            potential_fn, self._support_sizes, self._discrete_proposal_fn
        )
        z_gibbs, pe = gibbs_fn(
            rng_key=rng_gibbs,
            gibbs_sites=z_gibbs,
            hmc_sites=z_hmc,
            pe=state.hmc_state.potential_energy,
        )

        if self.inner_kernel._forward_mode_differentiation:
            z_grad = jacfwd(partial(potential_fn, z_gibbs))(state.hmc_state.z)
        else:
            z_grad = grad(partial(potential_fn, z_gibbs))(state.hmc_state.z)
        hmc_state = state.hmc_state._replace(z_grad=z_grad, potential_energy=pe)

        model_kwargs_["_gibbs_sites"] = z_gibbs
        hmc_state = self.inner_kernel.sample(hmc_state, model_args, model_kwargs_)

        z = {**z_gibbs, **hmc_state.z}

        return HMCGibbsState(z, hmc_state, rng_key)


def _update_block(rng_key, num_blocks, subsample_idx, plate_size):
    size, subsample_size = plate_size
    rng_key, subkey, block_key = random.split(rng_key, 3)
    block_size = (subsample_size - 1) // num_blocks + 1
    pad = block_size - (subsample_size - 1) % block_size - 1

    chosen_block = random.randint(block_key, shape=(), minval=0, maxval=num_blocks)
    new_idx = random.randint(subkey, minval=0, maxval=size, shape=(block_size,))
    subsample_idx_padded = jnp.pad(subsample_idx, (0, pad))
    start = chosen_block * block_size
    subsample_idx_padded = lax.dynamic_update_slice_in_dim(
        subsample_idx_padded, new_idx, start, 0
    )
    return rng_key, subsample_idx_padded[:subsample_size], pad, new_idx, start


def _block_update(plate_sizes, num_blocks, rng_key, gibbs_sites, gibbs_state):
    u_new = {}
    for name, subsample_idx in gibbs_sites.items():
        rng_key, u_new[name], *_ = _update_block(
            rng_key, num_blocks, subsample_idx, plate_sizes[name]
        )
    return u_new, gibbs_state


def _block_update_proxy(num_blocks, rng_key, gibbs_sites, plate_sizes):
    u_new = {}
    pads = {}
    new_idxs = {}
    starts = {}
    for name, subsample_idx in gibbs_sites.items():
        rng_key, u_new[name], pads[name], new_idxs[name], starts[name] = _update_block(
            rng_key, num_blocks, subsample_idx, plate_sizes[name]
        )
    return u_new, pads, new_idxs, starts


HMCECSState = namedtuple(
    "HMCECSState", "z, hmc_state, rng_key, gibbs_state, accept_prob"
)
TaylorProxyState = namedtuple(
    "TaylorProxyState",
    "ref_subsample_log_liks, "
    "ref_subsample_log_lik_grads, ref_subsample_log_lik_hessians",
)


def _wrap_gibbs_state(model, *args, **kwargs):
    # this is to let estimate_likelihood handler knows what is the current gibbs_state
    msg = {"type": "_gibbs_state", "value": kwargs.pop("_gibbs_state", ())}
    numpyro.primitives.apply_stack(msg)
    return model(*args, **kwargs)


class HMCECS(HMCGibbs):
    """
    [EXPERIMENTAL INTERFACE]

    HMC with Energy Conserving Subsampling.

    A subclass of :class:`HMCGibbs` for performing HMC-within-Gibbs for models with subsample
    statements using the :class:`~numpyro.plate` primitive. This implements Algorithm 1
    of reference [1] but uses a naive estimation (without control variates) of log likelihood,
    hence might incur a high variance.

    The function can divide subsample indices into blocks and update only one block at each
    MCMC step to improve the acceptance rate of proposed subsamples as detailed in [3].

    .. note:: New subsample indices are proposed randomly with replacement at each MCMC step.

    **References:**

    1. *Hamiltonian Monte Carlo with energy conserving subsampling*,
       Dang, K. D., Quiroz, M., Kohn, R., Minh-Ngoc, T., & Villani, M. (2019)
    2. *Speeding Up MCMC by Efficient Data Subsampling*,
       Quiroz, M., Kohn, R., Villani, M., & Tran, M. N. (2018)
    3. *The Block Pseudo-Margional Sampler*,
       Tran, M.-N., Kohn, R., Quiroz, M. Villani, M. (2017)
    4. *The Fundamental Incompatibility of Scalable Hamiltonian Monte Carlo and Naive Data Subsampling*
       Betancourt, M. (2015)

    :param inner_kernel: One of :class:`~numpyro.infer.hmc.HMC` or :class:`~numpyro.infer.hmc.NUTS`.
    :param int num_blocks: Number of blocks to partition subsample into.
    :param proxy: Either :func:`~numpyro.infer.hmc_gibbs.taylor_proxy` for likelihood estimation,
                  or, None for naive (in-between trajectory) subsampling as outlined in [4].

    **Example**

    .. doctest::

        >>> from jax import random
        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.infer import HMCECS, MCMC, NUTS
        ...
        >>> def model(data):
        ...     x = numpyro.sample("x", dist.Normal(0, 1))
        ...     with numpyro.plate("N", data.shape[0], subsample_size=100):
        ...         batch = numpyro.subsample(data, event_dim=0)
        ...         numpyro.sample("obs", dist.Normal(x, 1), obs=batch)
        ...
        >>> data = random.normal(random.PRNGKey(0), (10000,)) + 1
        >>> kernel = HMCECS(NUTS(model), num_blocks=10)
        >>> mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000)
        >>> mcmc.run(random.PRNGKey(0), data)
        >>> samples = mcmc.get_samples()["x"]
        >>> assert abs(jnp.mean(samples) - 1.) < 0.1

    """

    def __init__(self, inner_kernel, *, num_blocks=1, proxy=None):
        super().__init__(inner_kernel, identity, None)

        self.inner_kernel._model = partial(_wrap_gibbs_state, self.inner_kernel._model)
        self._num_blocks = num_blocks
        self._proxy = proxy

    def postprocess_fn(self, args, kwargs):
        def fn(z):
            model_kwargs = {} if kwargs is None else kwargs.copy()
            hmc_sites = {k: v for k, v in z.items() if k not in self._gibbs_sites}
            gibbs_sites = {k: v for k, v in z.items() if k in self._gibbs_sites}
            model_kwargs["_gibbs_sites"] = gibbs_sites
            hmc_sites = self.inner_kernel.postprocess_fn(args, model_kwargs)(hmc_sites)
            return hmc_sites

        return fn

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        model_kwargs = {} if model_kwargs is None else model_kwargs.copy()
        rng_key, key_u = random.split(rng_key)
        self._prototype_trace = trace(seed(self.model, key_u)).get_trace(
            *model_args, **model_kwargs
        )
        self._subsample_plate_sizes = {
            name: site["args"]
            for name, site in self._prototype_trace.items()
            if site["type"] == "plate" and site["args"][0] > site["args"][1]
        }  # i.e. size > subsample_size
        self._gibbs_sites = list(self._subsample_plate_sizes.keys())
        assert self._gibbs_sites, "Cannot detect any subsample statements in the model."
        if self._proxy is not None:
            proxy_fn, gibbs_init, self._gibbs_update = self._proxy(
                self._prototype_trace,
                self._subsample_plate_sizes,
                self.model,
                model_args,
                model_kwargs.copy(),
                num_blocks=self._num_blocks,
            )
            method = perturbed_method(self._subsample_plate_sizes, proxy_fn)
            self.inner_kernel._model = estimate_likelihood(
                self.inner_kernel._model, method
            )

            z_gibbs = {
                name: site["value"]
                for name, site in self._prototype_trace.items()
                if name in self._gibbs_sites
            }
            rng_key, rng_state = random.split(rng_key)
            gibbs_state = gibbs_init(rng_state, z_gibbs)
        else:
            self._gibbs_update = partial(
                _block_update, self._subsample_plate_sizes, self._num_blocks
            )
            gibbs_state = ()

        model_kwargs["_gibbs_state"] = gibbs_state
        state = super().init(rng_key, num_warmup, init_params, model_args, model_kwargs)
        return HMCECSState(
            state.z, state.hmc_state, state.rng_key, gibbs_state, jnp.zeros(())
        )

    def sample(self, state, model_args, model_kwargs):
        model_kwargs = {} if model_kwargs is None else model_kwargs.copy()
        rng_key, rng_gibbs = random.split(state.rng_key)

        def potential_fn(z_gibbs, gibbs_state, z_hmc):
            return self.inner_kernel._potential_fn_gen(
                *model_args,
                _gibbs_sites=z_gibbs,
                _gibbs_state=gibbs_state,
                **model_kwargs,
            )(z_hmc)

        z_gibbs = {k: v for k, v in state.z.items() if k not in state.hmc_state.z}
        z_gibbs_new, gibbs_state_new = self._gibbs_update(
            rng_key, z_gibbs, state.gibbs_state
        )

        # given a fixed hmc_sites, pe_new - pe_curr = loglik_new - loglik_curr
        pe = state.hmc_state.potential_energy
        pe_new = potential_fn(z_gibbs_new, gibbs_state_new, state.hmc_state.z)
        accept_prob = jnp.clip(jnp.exp(pe - pe_new), a_max=1.0)
        transition = random.bernoulli(rng_key, accept_prob)
        grad_ = jacfwd if self.inner_kernel._forward_mode_differentiation else grad
        z_gibbs, gibbs_state, pe, z_grad = cond(
            transition,
            (z_gibbs_new, gibbs_state_new, pe_new),
            lambda vals: vals
            + (grad_(partial(potential_fn, vals[0], vals[1]))(state.hmc_state.z),),
            (z_gibbs, state.gibbs_state, pe, state.hmc_state.z_grad),
            identity,
        )

        hmc_state = state.hmc_state._replace(z_grad=z_grad, potential_energy=pe)

        model_kwargs["_gibbs_sites"] = z_gibbs
        model_kwargs["_gibbs_state"] = gibbs_state
        hmc_state = self.inner_kernel.sample(hmc_state, model_args, model_kwargs)

        z = {**z_gibbs, **hmc_state.z}
        return HMCECSState(z, hmc_state, rng_key, gibbs_state, accept_prob)

    @staticmethod
    def taylor_proxy(reference_params):
        return taylor_proxy(reference_params)


def perturbed_method(subsample_plate_sizes, proxy_fn):
    def estimator(likelihoods, params, gibbs_state):
        subsample_log_liks = defaultdict(float)
        for (fn, value, name, subsample_dim) in likelihoods.values():
            subsample_log_liks[name] += _sum_all_except_at_dim(
                fn.log_prob(value), subsample_dim
            )

        log_lik_sum = 0.0

        proxy_value_all, proxy_value_subsample = proxy_fn(
            params, subsample_log_liks.keys(), gibbs_state
        )

        for (
            name,
            subsample_log_lik,
        ) in subsample_log_liks.items():  # loop over all subsample sites
            n, m = subsample_plate_sizes[name]

            diff = subsample_log_lik - proxy_value_subsample[name]

            unbiased_log_lik = proxy_value_all[name] + n * jnp.mean(diff)
            variance = n ** 2 / m * jnp.var(diff)
            log_lik_sum += unbiased_log_lik - 0.5 * variance
        return log_lik_sum

    return estimator


def taylor_proxy(reference_params):
    """Control variate for unbiased log likelihood estimation using a Taylor expansion around a reference
    parameter. Suggest for subsampling in [1].

    :param dict reference_params: Model parameterization at MLE or MAP-estimate.

    ** References: **

    [1] Towards scaling up Markov chainMonte Carlo: an adaptive subsampling approach
        Bardenet., R., Doucet, A., Holmes, C. (2014)
    """

    def construct_proxy_fn(
        prototype_trace,
        subsample_plate_sizes,
        model,
        model_args,
        model_kwargs,
        num_blocks=1,
    ):
        ref_params = {
            name: biject_to(prototype_trace[name]["fn"].support).inv(value)
            for name, value in reference_params.items()
        }

        ref_params_flat, unravel_fn = ravel_pytree(ref_params)

        def log_likelihood(params_flat, subsample_indices=None):
            if subsample_indices is None:
                subsample_indices = {
                    k: jnp.arange(v[0]) for k, v in subsample_plate_sizes.items()
                }
            params = unravel_fn(params_flat)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = {
                    name: biject_to(prototype_trace[name]["fn"].support)(value)
                    for name, value in params.items()
                }
                with block(), trace() as tr, substitute(
                    data=subsample_indices
                ), substitute(data=params):
                    model(*model_args, **model_kwargs)

            log_lik = {}
            for site in tr.values():
                if site["type"] == "sample" and site["is_observed"]:
                    for frame in site["cond_indep_stack"]:
                        if frame.name in log_lik:
                            log_lik[frame.name] += _sum_all_except_at_dim(
                                site["fn"].log_prob(site["value"]), frame.dim
                            )
                        else:
                            log_lik[frame.name] = _sum_all_except_at_dim(
                                site["fn"].log_prob(site["value"]), frame.dim
                            )
            return log_lik

        def log_likelihood_sum(params_flat, subsample_indices=None):
            return {
                k: v.sum()
                for k, v in log_likelihood(params_flat, subsample_indices).items()
            }

        # those stats are dict keyed by subsample names
        ref_log_likelihoods_sum = log_likelihood_sum(ref_params_flat)
        ref_log_likelihood_grads_sum = jacobian(log_likelihood_sum)(ref_params_flat)
        ref_log_likelihood_hessians_sum = hessian(log_likelihood_sum)(ref_params_flat)

        def gibbs_init(rng_key, gibbs_sites):
            ref_subsample_log_liks = log_likelihood(ref_params_flat, gibbs_sites)
            ref_subsample_log_lik_grads = jacfwd(log_likelihood)(
                ref_params_flat, gibbs_sites
            )
            ref_subsample_log_lik_hessians = jacfwd(jacfwd(log_likelihood))(
                ref_params_flat, gibbs_sites
            )
            return TaylorProxyState(
                ref_subsample_log_liks,
                ref_subsample_log_lik_grads,
                ref_subsample_log_lik_hessians,
            )

        def gibbs_update(rng_key, gibbs_sites, gibbs_state):
            u_new, pads, new_idxs, starts = _block_update_proxy(
                num_blocks, rng_key, gibbs_sites, subsample_plate_sizes
            )

            new_states = defaultdict(dict)
            ref_subsample_log_liks = log_likelihood(ref_params_flat, new_idxs)
            ref_subsample_log_lik_grads = jacfwd(log_likelihood)(
                ref_params_flat, new_idxs
            )
            ref_subsample_log_lik_hessians = jacfwd(jacfwd(log_likelihood))(
                ref_params_flat, new_idxs
            )
            for stat, new_block_values, last_values in zip(
                ["log_liks", "grads", "hessians"],
                [
                    ref_subsample_log_liks,
                    ref_subsample_log_lik_grads,
                    ref_subsample_log_lik_hessians,
                ],
                [
                    gibbs_state.ref_subsample_log_liks,
                    gibbs_state.ref_subsample_log_lik_grads,
                    gibbs_state.ref_subsample_log_lik_hessians,
                ],
            ):
                for name, subsample_idx in gibbs_sites.items():
                    size, subsample_size = subsample_plate_sizes[name]
                    pad, start = pads[name], starts[name]
                    new_value = jnp.pad(
                        last_values[name],
                        [(0, pad)] + [(0, 0)] * (jnp.ndim(last_values[name]) - 1),
                    )
                    new_value = lax.dynamic_update_slice_in_dim(
                        new_value, new_block_values[name], start, 0
                    )
                    new_states[stat][name] = new_value[:subsample_size]
            gibbs_state = TaylorProxyState(
                new_states["log_liks"], new_states["grads"], new_states["hessians"]
            )
            return u_new, gibbs_state

        def proxy_fn(params, subsample_lik_sites, gibbs_state):
            params_flat, _ = ravel_pytree(params)
            params_diff = params_flat - ref_params_flat

            ref_subsample_log_liks = gibbs_state.ref_subsample_log_liks
            ref_subsample_log_lik_grads = gibbs_state.ref_subsample_log_lik_grads
            ref_subsample_log_lik_hessians = gibbs_state.ref_subsample_log_lik_hessians

            proxy_sum = defaultdict(float)
            proxy_subsample = defaultdict(float)
            for name in subsample_lik_sites:
                proxy_subsample[name] = (
                    ref_subsample_log_liks[name]
                    + jnp.dot(ref_subsample_log_lik_grads[name], params_diff)
                    + 0.5
                    * jnp.dot(
                        jnp.dot(ref_subsample_log_lik_hessians[name], params_diff),
                        params_diff,
                    )
                )

                proxy_sum[name] = (
                    ref_log_likelihoods_sum[name]
                    + jnp.dot(ref_log_likelihood_grads_sum[name], params_diff)
                    + 0.5
                    * jnp.dot(
                        jnp.dot(ref_log_likelihood_hessians_sum[name], params_diff),
                        params_diff,
                    )
                )
            return proxy_sum, proxy_subsample

        return proxy_fn, gibbs_init, gibbs_update

    return construct_proxy_fn


def _sum_all_except_at_dim(x, dim):
    x = x.reshape((-1,) + x.shape[dim:]).sum(0)
    return x.reshape(x.shape[:1] + (-1,)).sum(-1)


class estimate_likelihood(numpyro.primitives.Messenger):
    def __init__(self, fn=None, method=None):
        # estimate_likelihood: accept likelihood tuple (fn, value, subsample_name, subsample_dim)
        # and current unconstrained params
        # and returns log of the bias-corrected likelihood
        assert method is not None
        super().__init__(fn)
        self.method = method
        self.params = None
        self.likelihoods = {}
        self.subsample_plates = {}
        self.gibbs_state = None

    def __enter__(self):
        for handler in numpyro.primitives._PYRO_STACK[::-1]:
            # the potential_fn in HMC makes the PYRO_STACK nested like trace(...); so we can extract the
            # unconstrained_params from the _unconstrain_reparam substitute_fn
            if (
                isinstance(handler, substitute)
                and isinstance(handler.substitute_fn, partial)
                and handler.substitute_fn.func is _unconstrain_reparam
            ):
                self.params = handler.substitute_fn.args[0]
                break
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        # make sure exit trackback is nice if an error happens
        super().__exit__(exc_type, exc_value, traceback)
        if exc_type is not None:
            return

        if self.params is None:
            return

        if numpyro.get_mask() is not False:
            numpyro.factor(
                "_biased_corrected_log_likelihood",
                self.method(self.likelihoods, self.params, self.gibbs_state),
            )

        # clean up
        self.params = None
        self.likelihoods = {}
        self.subsample_plates = {}
        self.gibbs_state = None

    def process_message(self, msg):
        if self.params is None:
            return

        if msg["type"] == "_gibbs_state":
            self.gibbs_state = msg["value"]
            return

        if msg["type"] == "sample" and msg["is_observed"]:
            assert msg["name"] not in self.params
            # store the likelihood for the estimator
            for frame in msg["cond_indep_stack"]:
                if frame.name in self.subsample_plates:
                    if msg["name"] in self.likelihoods:
                        raise RuntimeError(
                            f"Multiple subsample plates at site {msg['name']} "
                            "are not allowed. Please reshape your data."
                        )
                    self.likelihoods[msg["name"]] = (
                        msg["fn"],
                        msg["value"],
                        frame.name,
                        frame.dim,
                    )
                    # mask the current likelihood
                    msg["fn"] = msg["fn"].mask(False)
        elif msg["type"] == "plate" and msg["args"][0] > msg["args"][1]:
            self.subsample_plates[msg["name"]] = msg["value"]
