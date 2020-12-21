# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from functools import partial
from operator import attrgetter
import os
import warnings

from jax import device_put, jit, lax, pmap, random, vmap
from jax.core import Tracer
from jax.interpreters.xla import DeviceArray
from jax.lib import xla_bridge
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_map, tree_multimap

from numpyro.diagnostics import print_summary
from numpyro.util import cached_by, fori_collect, identity

__all__ = [
    'MCMCKernel',
    'MCMC',
]


class MCMCKernel(ABC):
    """
    Defines the interface for the Markov transition kernel that is
    used for :class:`~numpyro.infer.MCMC` inference.

    **Example:**

    .. doctest::

        >>> from collections import namedtuple
        >>> from jax import random
        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.infer import MCMC

        >>> MHState = namedtuple("MHState", ["u", "rng_key"])

        >>> class MetropolisHastings(numpyro.infer.mcmc.MCMCKernel):
        ...     sample_field = "u"
        ...
        ...     def __init__(self, potential_fn, step_size=0.1):
        ...         self.potential_fn = potential_fn
        ...         self.step_size = step_size
        ...
        ...     def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        ...         return MHState(init_params, rng_key)
        ...
        ...     def sample(self, state, model_args, model_kwargs):
        ...         u, rng_key = state
        ...         rng_key, key_proposal, key_accept = random.split(rng_key, 3)
        ...         u_proposal = dist.Normal(u, self.step_size).sample(key_proposal)
        ...         accept_prob = jnp.exp(self.potential_fn(u) - self.potential_fn(u_proposal))
        ...         u_new = jnp.where(dist.Uniform().sample(key_accept) < accept_prob, u_proposal, u)
        ...         return MHState(u_new, rng_key)

        >>> def f(x):
        ...     return ((x - 2) ** 2).sum()

        >>> kernel = MetropolisHastings(f)
        >>> mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000)
        >>> mcmc.run(random.PRNGKey(0), init_params=jnp.array([1., 2.]))
        >>> samples = mcmc.get_samples()
        >>> mcmc.print_summary()  # doctest: +SKIP
    """
    def postprocess_fn(self, model_args, model_kwargs):
        """
        Get a function that transforms unconstrained values at sample sites to values
        constrained to the site's support, in addition to returning deterministic
        sites in the model.

        :param model_args: Arguments to the model.
        :param model_kwargs: Keyword arguments to the model.
        """
        return identity

    @abstractmethod
    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        """
        Initialize the `MCMCKernel` and return an initial state to begin sampling
        from.

        :param random.PRNGKey rng_key: Random number generator key to initialize
            the kernel.
        :param int num_warmup: Number of warmup steps. This can be useful
            when doing adaptation during warmup.
        :param tuple init_params: Initial parameters to begin sampling. The type must
            be consistent with the input type to `potential_fn`.
        :param model_args: Arguments provided to the model.
        :param model_kwargs: Keyword arguments provided to the model.
        :return: The initial state representing the state of the kernel. This can be
            any class that is registered as a
            `pytree <https://jax.readthedocs.io/en/latest/pytrees.html>`_.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, state, model_args, model_kwargs):
        """
        Given the current `state`, return the next `state` using the given
        transition kernel.

        :param state: A `pytree <https://jax.readthedocs.io/en/latest/pytrees.html>`_
            class representing the state for the kernel. For HMC, this is given
            by :data:`~numpyro.infer.hmc.HMCState`. In general, this could be any
            class that supports `getattr`.
        :param model_args: Arguments provided to the model.
        :param model_kwargs: Keyword arguments provided to the model.
        :return: Next `state`.
        """
        raise NotImplementedError

    @property
    def sample_field(self):
        """
        The attribute of the `state` object passed to :meth:`sample` that denotes
        the MCMC sample. This is used by :meth:`postprocess_fn` and for reporting
        results in :meth:`MCMC.print_summary()
        <numpyro.infer.mcmc.MCMC.print_summary>`.
        """
        raise NotImplementedError

    @property
    def default_fields(self):
        """
        The attributes of the `state` object to be collected by default during
        the MCMC run (when :meth:`MCMC.run() <numpyro.infer.MCMC.run>` is called).
        """
        return (self.sample_field,)

    def get_diagnostics_str(self, state):
        """
        Given the current `state`, returns the diagnostics string to
        be added to progress bar for diagnostics purpose.
        """
        return ''


def _get_progbar_desc_str(num_warmup, phase, i):
    if phase is not None:
        return phase
    return 'warmup' if i < num_warmup else 'sample'


def _get_value_from_index(xs, i):
    return tree_map(lambda x: x[i], xs)


def _laxmap(f, xs):
    n = tree_flatten(xs)[0][0].shape[0]

    ys = []
    for i in range(n):
        x = jit(_get_value_from_index)(xs, i)
        ys.append(f(x))

    return tree_multimap(lambda *args: jnp.stack(args), *ys)


def _sample_fn_jit_args(state, sampler):
    hmc_state, args, kwargs = state
    return sampler.sample(hmc_state, args, kwargs), args, kwargs


def _sample_fn_nojit_args(state, sampler, args, kwargs):
    # state is a tuple of size 1 - containing HMCState
    return sampler.sample(state[0], args, kwargs),


def _collect_fn(collect_fields):
    @cached_by(_collect_fn, collect_fields)
    def collect(x):
        if collect_fields:
            return attrgetter(*collect_fields)(x[0])
        else:
            return x[0]

    return collect


# XXX: Is there a better hash key that we can use?
def _hashable(x):
    # When the arguments are JITed, ShapedArray is hashable.
    if isinstance(x, Tracer):
        return x
    elif isinstance(x, DeviceArray):
        return x.copy().tobytes()
    elif isinstance(x, jnp.ndarray):
        return x.tobytes()
    return x


class MCMC(object):
    """
    Provides access to Markov Chain Monte Carlo inference algorithms in NumPyro.

    .. note:: `chain_method` is an experimental arg, which might be removed in a future version.

    .. note:: Setting `progress_bar=False` will improve the speed for many cases.

    :param MCMCKernel sampler: an instance of :class:`~numpyro.infer.mcmc.MCMCKernel` that
        determines the sampler for running MCMC. Currently, only :class:`~numpyro.infer.mcmc.HMC`
        and :class:`~numpyro.infer.mcmc.NUTS` are available.
    :param int num_warmup: Number of warmup steps.
    :param int num_samples: Number of samples to generate from the Markov chain.
    :param int num_chains: Number of Number of MCMC chains to run. By default,
        chains will be run in parallel using :func:`jax.pmap`, failing which,
        chains will be run in sequence.
    :param postprocess_fn: Post-processing callable - used to convert a collection of unconstrained
        sample values returned from the sampler to constrained values that lie within the support
        of the sample sites. Additionally, this is used to return values at deterministic sites in
        the model.
    :param str chain_method: One of 'parallel' (default), 'sequential', 'vectorized'. The method
        'parallel' is used to execute the drawing process in parallel on XLA devices (CPUs/GPUs/TPUs),
        If there are not enough devices for 'parallel', we fall back to 'sequential' method to draw
        chains sequentially. 'vectorized' method is an experimental feature which vectorizes the
        drawing method, hence allowing us to collect samples in parallel on a single device.
    :param bool progress_bar: Whether to enable progress bar updates. Defaults to
        ``True``.
    :param bool jit_model_args: If set to `True`, this will compile the potential energy
        computation as a function of model arguments. As such, calling `MCMC.run` again
        on a same sized but different dataset will not result in additional compilation cost.
    """
    def __init__(self,
                 sampler,
                 num_warmup,
                 num_samples,
                 num_chains=1,
                 thinning=1,
                 postprocess_fn=None,
                 chain_method='parallel',
                 progress_bar=True,
                 jit_model_args=False):
        self.sampler = sampler
        self._sample_field = sampler.sample_field
        self._default_fields = sampler.default_fields
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        print("thin", thinning)
        self.thinning = thinning
        self.postprocess_fn = postprocess_fn
        if chain_method not in ['parallel', 'vectorized', 'sequential']:
            raise ValueError('Only supporting the following methods to draw chains:'
                             ' "sequential", "parallel", or "vectorized"')
        self.chain_method = chain_method
        self.progress_bar = progress_bar
        # TODO: We should have progress bars (maybe without diagnostics) for num_chains > 1
        if (chain_method == 'parallel' and num_chains > 1) or (
                "CI" in os.environ or "PYTEST_XDIST_WORKER" in os.environ):
            self.progress_bar = False
        self._jit_model_args = jit_model_args
        self._states = None
        self._states_flat = None
        # HMCState returned by last run
        self._last_state = None
        # HMCState returned by last warmup
        self._warmup_state = None
        # HMCState returned by hmc.init_kernel
        self._init_state_cache = {}
        self._cache = {}
        self._collection_params = {}
        self._set_collection_params(thinning=self.thinning)

    def _get_cached_fn(self):
        if self._jit_model_args:
            args, kwargs = (None,), (None,)
        else:
            args = tree_map(lambda x: _hashable(x), self._args)
            kwargs = tree_map(lambda x: _hashable(x), tuple(sorted(self._kwargs.items())))
        key = args + kwargs
        try:
            fn = self._cache.get(key, None)
        # If unhashable arguments are provided, proceed normally
        # without caching
        except TypeError:
            fn, key = None, None
        if fn is None:
            if self._jit_model_args:
                fn = partial(_sample_fn_jit_args, sampler=self.sampler)
            else:
                fn = partial(_sample_fn_nojit_args, sampler=self.sampler,
                             args=self._args, kwargs=self._kwargs)
            if key is not None:
                self._cache[key] = fn
        return fn

    def _get_cached_init_state(self, rng_key, args, kwargs):
        rng_key = (_hashable(rng_key),)
        args = tree_map(lambda x: _hashable(x), args)
        kwargs = tree_map(lambda x: _hashable(x), tuple(sorted(kwargs.items())))
        key = rng_key + args + kwargs
        try:
            return self._init_state_cache.get(key, None)
        # If unhashable arguments are provided, return None
        except TypeError:
            return None

    def _single_chain_mcmc(self, init, args, kwargs, collect_fields):
        rng_key, init_state, init_params = init
        if init_state is None:
            init_state = self.sampler.init(rng_key, self.num_warmup, init_params,
                                           model_args=args, model_kwargs=kwargs)
        if self.postprocess_fn is None:
            postprocess_fn = self.sampler.postprocess_fn(args, kwargs)
        else:
            postprocess_fn = self.postprocess_fn
        diagnostics = lambda x: self.sampler.get_diagnostics_str(x[0]) if rng_key.ndim == 1 else ''   # noqa: E731
        init_val = (init_state, args, kwargs) if self._jit_model_args else (init_state,)
        lower_idx = self._collection_params["lower"]
        upper_idx = self._collection_params["upper"]
        phase = self._collection_params["phase"]

        collect_vals = fori_collect(lower_idx,
                                    upper_idx,
                                    self._get_cached_fn(),
                                    init_val,
                                    transform=_collect_fn(collect_fields),
                                    progbar=self.progress_bar,
                                    return_last_val=True,
                                    thinning=self._collection_params["thinning"],
                                    collection_size=self._collection_params["collection_size"],
                                    progbar_desc=partial(_get_progbar_desc_str, lower_idx, phase),
                                    diagnostics_fn=diagnostics)
        states, last_val = collect_vals
        # Get first argument of type `HMCState`
        last_state = last_val[0]
        if len(collect_fields) == 1:
            states = (states,)
        states = dict(zip(collect_fields, states))
        # Apply constraints if number of samples is non-zero
        site_values = tree_flatten(states[self._sample_field])[0]
        # XXX: lax.map still works if some arrays have 0 size
        # so we only need to filter out the case site_value.shape[0] == 0
        # (which happens when lower_idx==upper_idx)
        if len(site_values) > 0 and jnp.shape(site_values[0])[0] > 0:
            if self.chain_method == "vectorized" and self.num_chains > 1:
                postprocess_fn = vmap(postprocess_fn)
            states[self._sample_field] = lax.map(postprocess_fn, states[self._sample_field])
        return states, last_state

    def _set_collection_params(self, lower=None, upper=None, collection_size=None, phase=None, thinning=1):
        self._collection_params["lower"] = self.num_warmup if lower is None else lower
        self._collection_params["upper"] = self.num_warmup + self.num_samples if upper is None else upper
        self._collection_params["collection_size"] = collection_size
        self._collection_params["phase"] = phase
        self._collection_params["thinning"] = thinning

    def _compile(self, rng_key, *args, extra_fields=(), init_params=None, **kwargs):
        self._set_collection_params(0, 0, self.num_samples, thinning=self.thinning)
        self.run(rng_key, *args, extra_fields=extra_fields, init_params=init_params, **kwargs)
        rng_key = (_hashable(rng_key),)
        args = tree_map(lambda x: _hashable(x), args)
        kwargs = tree_map(lambda x: _hashable(x), tuple(sorted(kwargs.items())))
        key = rng_key + args + kwargs
        try:
            self._init_state_cache[key] = self._last_state
        # If unhashable arguments are provided, return None
        except TypeError:
            pass

    def warmup(self, rng_key, *args, extra_fields=(), collect_warmup=False, init_params=None, **kwargs):
        """
        Run the MCMC warmup adaptation phase. After this call, the :meth:`run` method
        will skip the warmup adaptation phase. To run `warmup` again for the new data,
        it is required to run :meth:`warmup` again.

        :param random.PRNGKey rng_key: Random number generator key to be used for the sampling.
        :param args: Arguments to be provided to the :meth:`numpyro.infer.mcmc.MCMCKernel.init` method.
            These are typically the arguments needed by the `model`.
        :param extra_fields: Extra fields (aside from :meth:`~numpyro.infer.MCMCKernel.default_fields`)
            from the state object (e.g. :data:`numpyro.infer.mcmc.HMCState` for HMC) to collect during
            the MCMC run.
        :type extra_fields: tuple or list
        :param bool collect_warmup: Whether to collect samples from the warmup phase. Defaults
            to `False`.
        :param init_params: Initial parameters to begin sampling. The type must be consistent
            with the input type to `potential_fn`.
        :param kwargs: Keyword arguments to be provided to the :meth:`numpyro.infer.mcmc.MCMCKernel.init`
            method. These are typically the keyword arguments needed by the `model`.
        """
        self._warmup_state = None
        if collect_warmup:
            self._set_collection_params(0, self.num_warmup, self.num_warmup, "warmup")
        else:
            self._set_collection_params(self.num_warmup, self.num_warmup, self.num_samples, "warmup")
        self.run(rng_key, *args, extra_fields=extra_fields, init_params=init_params, **kwargs)
        self._warmup_state = self._last_state

    def run(self, rng_key, *args, extra_fields=(), init_params=None, **kwargs):
        """
        Run the MCMC samplers and collect samples.

        :param random.PRNGKey rng_key: Random number generator key to be used for the sampling.
            For multi-chains, a batch of `num_chains` keys can be supplied. If `rng_key`
            does not have batch_size, it will be split in to a batch of `num_chains` keys.
        :param args: Arguments to be provided to the :meth:`numpyro.infer.mcmc.MCMCKernel.init` method.
            These are typically the arguments needed by the `model`.
        :param extra_fields: Extra fields (aside from `z`, `diverging`) from :data:`numpyro.infer.mcmc.HMCState`
            to collect during the MCMC run.
        :type extra_fields: tuple or list
        :param init_params: Initial parameters to begin sampling. The type must be consistent
            with the input type to `potential_fn`.
        :param kwargs: Keyword arguments to be provided to the :meth:`numpyro.infer.mcmc.MCMCKernel.init`
            method. These are typically the keyword arguments needed by the `model`.

        .. note:: jax allows python code to continue even when the compiled code has not finished yet.
            This can cause troubles when trying to profile the code for speed.
            See https://jax.readthedocs.io/en/latest/async_dispatch.html and
            https://jax.readthedocs.io/en/latest/profiling.html for pointers on profiling jax programs.
        """
        self._args = args
        self._kwargs = kwargs
        init_state = self._get_cached_init_state(rng_key, args, kwargs)
        if self.num_chains > 1 and rng_key.ndim == 1:
            rng_key = random.split(rng_key, self.num_chains)

        if self._warmup_state is not None:
            self._set_collection_params(0, self.num_samples, self.num_samples, "sample", thinning=self.thinning)
            init_state = self._warmup_state._replace(rng_key=rng_key)

        chain_method = self.chain_method
        if chain_method == 'parallel' and xla_bridge.device_count() < self.num_chains:
            chain_method = 'sequential'
            warnings.warn('There are not enough devices to run parallel chains: expected {} but got {}.'
                          ' Chains will be drawn sequentially. If you are running MCMC in CPU,'
                          ' consider to use `numpyro.set_host_device_count({})` at the beginning'
                          ' of your program.'
                          .format(self.num_chains, xla_bridge.device_count(), self.num_chains))

        if init_params is not None and self.num_chains > 1:
            prototype_init_val = tree_flatten(init_params)[0][0]
            if jnp.shape(prototype_init_val)[0] != self.num_chains:
                raise ValueError('`init_params` must have the same leading dimension'
                                 ' as `num_chains`.')
        assert isinstance(extra_fields, (tuple, list))
        collect_fields = tuple(set((self._sample_field,) + tuple(self._default_fields) +
                                   tuple(extra_fields)))
        partial_map_fn = partial(self._single_chain_mcmc,
                                 args=args,
                                 kwargs=kwargs,
                                 collect_fields=collect_fields)
        map_args = (rng_key, init_state, init_params)
        if self.num_chains == 1:
            states_flat, last_state = partial_map_fn(map_args)
            states = tree_map(lambda x: x[jnp.newaxis, ...], states_flat)
        else:
            if chain_method == 'sequential':
                if self.progress_bar:
                    states, last_state = _laxmap(partial_map_fn, map_args)
                else:
                    states, last_state = lax.map(partial_map_fn, map_args)
            elif chain_method == 'parallel':
                states, last_state = pmap(partial_map_fn)(map_args)
                # TODO: remove when https://github.com/google/jax/issues/3597 is resolved
                states = device_put(states)
            else:
                assert chain_method == 'vectorized'
                states, last_state = partial_map_fn(map_args)
                # swap num_samples x num_chains to num_chains x num_samples
                states = tree_map(lambda x: jnp.swapaxes(x, 0, 1), states)
            states_flat = tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), states)
        self._last_state = last_state
        self._states = states
        self._states_flat = states_flat
        self._set_collection_params()

    def get_samples(self, group_by_chain=False):
        """
        Get samples from the MCMC run.

        :param bool group_by_chain: Whether to preserve the chain dimension. If True,
            all samples will have num_chains as the size of their leading dimension.
        :return: Samples having the same data type as `init_params`. The data type is a
            `dict` keyed on site names if a model containing Pyro primitives is used,
            but can be any :func:`jaxlib.pytree`, more generally (e.g. when defining a
            `potential_fn` for HMC that takes `list` args).
        """
        return self._states[self._sample_field] if group_by_chain \
            else self._states_flat[self._sample_field]

    def get_extra_fields(self, group_by_chain=False):
        """
        Get extra fields from the MCMC run.

        :param bool group_by_chain: Whether to preserve the chain dimension. If True,
            all samples will have num_chains as the size of their leading dimension.
        :return: Extra fields keyed by field names which are specified in the
            `extra_fields` keyword of :meth:`run`.
        """
        states = self._states if group_by_chain else self._states_flat
        return {k: v for k, v in states.items() if k != self._sample_field}

    def print_summary(self, prob=0.9, exclude_deterministic=True):
        """
        Print the statistics of posterior samples collected during running this MCMC instance.

        :param float prob: the probability mass of samples within the credible interval.
        :param bool exclude_deterministic: whether or not print out the statistics
            at deterministic sites.
        """
        # Exclude deterministic sites by default
        sites = self._states[self._sample_field]
        if isinstance(sites, dict) and exclude_deterministic:
            state_sample_field = attrgetter(self._sample_field)(self._last_state)
            # XXX: there might be the case that state.z is not a dictionary but
            # its postprocessed value `sites` is a dictionary.
            # TODO: in general, when both `sites` and `state.z` are dictionaries,
            # they can have different key names, not necessary due to deterministic
            # behavior. We might revise this logic if needed in the future.
            if isinstance(state_sample_field, dict):
                sites = {k: v for k, v in self._states[self._sample_field].items()
                         if k in state_sample_field}
        print_summary(sites, prob=prob)
        extra_fields = self.get_extra_fields()
        if 'diverging' in extra_fields:
            print("Number of divergences: {}".format(jnp.sum(extra_fields['diverging'])))
