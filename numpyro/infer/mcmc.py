# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from functools import partial
from operator import attrgetter
import os
import warnings

import numpy as np

import jax
from jax import device_get, jit, lax, local_device_count, pmap, random, vmap
import jax.numpy as jnp

from numpyro.diagnostics import print_summary
from numpyro.util import (
    cached_by,
    find_stack_level,
    fori_collect,
    identity,
    is_prng_key,
)

__all__ = [
    "MCMCKernel",
    "MCMC",
]


class MCMCKernel(ABC):
    """
    Defines the interface for the Markov transition kernel that is
    used for :class:`~numpyro.infer.mcmc.MCMC` inference.

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
        >>> posterior_samples = mcmc.get_samples()
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

    @property
    def is_ensemble_kernel(self):
        """
        Denotes whether the kernel is an ensemble kernel. If True,
        diagnostics_str will be displayed during the MCMC run
        (when :meth:`MCMC.run() <numpyro.infer.MCMC.run>` is called)
        if `chain_method` = "vectorized".
        """
        return False

    def get_diagnostics_str(self, state):
        """
        Given the current `state`, returns the diagnostics string to
        be added to progress bar for diagnostics purpose.
        """
        return ""


def _get_progbar_desc_str(num_warmup, phase, i):
    if phase is not None:
        return phase
    return "warmup" if i < num_warmup else "sample"


def _get_value_from_index(xs, i):
    return jax.tree.map(lambda x: x[i], xs)


def _laxmap(f, xs):
    n = jax.tree.flatten(xs)[0][0].shape[0]

    ys = []
    for i in range(n):
        x = jit(_get_value_from_index)(xs, i)
        ys.append(f(x))

    return jax.tree.map(lambda *args: jnp.stack(args), *ys)


def _sample_fn_jit_args(state, sampler):
    hmc_state, args, kwargs = state
    return sampler.sample(hmc_state, args, kwargs), args, kwargs


def _sample_fn_nojit_args(state, sampler, args, kwargs):
    # state is a tuple of size 1 - containing HMCState
    return (sampler.sample(state[0], args, kwargs),)


def _collect_fn(collect_fields, remove_sites):
    @cached_by(_collect_fn, collect_fields, remove_sites)
    def collect(x):
        if collect_fields:
            fields = attrgetter(*collect_fields)(x[0])

            if remove_sites != ():
                fields = [fields] if len(collect_fields) == 1 else list(fields)
                assert isinstance(fields[0], dict)

                sample_sites = fields[0].copy()
                for site in remove_sites:
                    sample_sites.pop(site)
                fields[0] = sample_sites
                fields = fields[0] if len(collect_fields) == 1 else fields

            return fields
        else:
            return x[0]

    return collect


# XXX: Is there a better hash key that we can use?
def _hashable(x):
    # NOTE: When the arguments are JITed, ShapedArray is hashable.
    if isinstance(x, (np.ndarray, jnp.ndarray)):
        return id(x)
    return x


class MCMC(object):
    """
    Provides access to Markov Chain Monte Carlo inference algorithms in NumPyro.

    .. note:: `chain_method` is an experimental arg, which might be removed in a future version.

    .. note:: Setting `progress_bar=False` will improve the speed for many cases. But it might
        require more memory than the other option.

    .. note:: If setting `num_chains` greater than `1` in a Jupyter Notebook, then you will need to
        have installed `ipywidgets <https://ipywidgets.readthedocs.io/en/latest/user_install.html>`_
        in the environment from which you launced Jupyter in order for the progress bars to render
        correctly. If you are using Jupyter Notebook or Jupyter Lab, please also install the
        corresponding extension package like `widgetsnbextension` or `jupyterlab_widgets`.

    .. note:: If your dataset is large and you have access to multiple acceleration devices,
        you can distribute the computation across multiple devices. Make sure that your jax version
        is v0.4.4 or newer. For example,

        .. code-block:: python

            import jax
            from jax.experimental import mesh_utils
            from jax.sharding import PositionalSharding
            import numpy as np
            import numpyro
            import numpyro.distributions as dist
            from numpyro.infer import MCMC, NUTS

            X = np.random.randn(128, 3)
            y = np.random.randn(128)

            def model(X, y):
                beta = numpyro.sample("beta", dist.Normal(0, 1).expand([3]))
                numpyro.sample("obs", dist.Normal(X @ beta, 1), obs=y)

            mcmc = MCMC(NUTS(model), num_warmup=10, num_samples=10)
            # See https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
            sharding = PositionalSharding(mesh_utils.create_device_mesh((8,)))
            X_shard = jax.device_put(X, sharding.reshape(8, 1))
            y_shard = jax.device_put(y, sharding.reshape(8))
            mcmc.run(jax.random.PRNGKey(0), X_shard, y_shard)

    :param MCMCKernel sampler: an instance of :class:`~numpyro.infer.mcmc.MCMCKernel` that
        determines the sampler for running MCMC. Currently, only :class:`~numpyro.infer.hmc.HMC`
        and :class:`~numpyro.infer.hmc.NUTS` are available.
    :param int num_warmup: Number of warmup steps.
    :param int num_samples: Number of samples to generate from the Markov chain.
    :param int thinning: Positive integer that controls the fraction of post-warmup samples that are
        retained. For example if thinning is 2 then every other sample is retained.
        Defaults to 1, i.e. no thinning.
    :param int num_chains: Number of MCMC chains to run. By default, chains will be
        run in parallel using :func:`jax.pmap`. If there are not enough devices
        available, chains will be run in sequence.
    :param postprocess_fn: Post-processing callable - used to convert a collection of unconstrained
        sample values returned from the sampler to constrained values that lie within the support
        of the sample sites. Additionally, this is used to return values at deterministic sites in
        the model.
    :param str chain_method: A callable jax transform like `jax.vmap` or one of
        'parallel' (default), 'sequential', 'vectorized'. The method
        'parallel' is used to execute the drawing process in parallel on XLA devices (CPUs/GPUs/TPUs),
        If there are not enough devices for 'parallel', we fall back to 'sequential' method to draw
        chains sequentially. 'vectorized' method is an experimental feature which vectorizes the
        drawing method, hence allowing us to collect samples in parallel on a single device.
    :param bool progress_bar: Whether to enable progress bar updates. Defaults to
        ``True``.
    :param bool jit_model_args: If set to `True`, this will compile the potential energy
        computation as a function of model arguments. As such, calling `MCMC.run` again
        on a same sized but different dataset will not result in additional compilation cost.
        Note that currently, this does not take effect for the case ``num_chains > 1``
        and ``chain_method == 'parallel'``.

    .. note:: It is possible to mix parallel and vectorized sampling, i.e., run vectorized chains
        on multiple devices using explicit `pmap`. Currently, doing so requires disabling the
        progress bar. For example,

        .. code-block:: python

            def do_mcmc(rng_key, n_vectorized=8):
                nuts_kernel = NUTS(model)
                mcmc = MCMC(
                    nuts_kernel,
                    progress_bar=False,
                    num_chains=n_vectorized,
                    chain_method='vectorized'
                )
                mcmc.run(
                    rng_key,
                    extra_fields=("potential_energy",),
                )
                return {**mcmc.get_samples(), **mcmc.get_extra_fields()}
            # Number of devices to pmap over
            n_parallel = jax.local_device_count()
            rng_keys = jax.random.split(PRNGKey(rng_seed), n_parallel)
            traces = pmap(do_mcmc)(rng_keys)
            # concatenate traces along pmap'ed axis
            trace = {k: np.concatenate(v) for k, v in traces.items()}
    """

    def __init__(
        self,
        sampler,
        *,
        num_warmup,
        num_samples,
        num_chains=1,
        thinning=1,
        postprocess_fn=None,
        chain_method="parallel",
        progress_bar=True,
        jit_model_args=False,
    ):
        self.sampler = sampler
        self._sample_field = sampler.sample_field
        self._default_fields = sampler.default_fields
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        if not isinstance(thinning, int) or thinning < 1:
            raise ValueError("thinning must be a positive integer")
        self.thinning = thinning
        self.postprocess_fn = postprocess_fn
        if not callable(chain_method) and chain_method not in [
            "parallel",
            "vectorized",
            "sequential",
        ]:
            raise ValueError(
                "Only supporting the following methods to draw chains:"
                ' "sequential", "parallel", or "vectorized"'
            )
        if chain_method == "parallel" and local_device_count() < self.num_chains:
            chain_method = "sequential"
            warnings.warn(
                "There are not enough devices to run parallel chains: expected {} but got {}."
                " Chains will be drawn sequentially. If you are running MCMC in CPU,"
                " consider using `numpyro.set_host_device_count({})` at the beginning"
                " of your program. You can double-check how many devices are available in"
                " your system using `jax.local_device_count()`.".format(
                    self.num_chains, local_device_count(), self.num_chains
                ),
                stacklevel=find_stack_level(),
            )
        self.chain_method = chain_method
        self.progress_bar = progress_bar
        if "CI" in os.environ or "PYTEST_XDIST_WORKER" in os.environ:
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
        self._set_collection_params()

    def _get_cached_fns(self):
        if self._jit_model_args:
            args, kwargs = (None,), (None,)
        else:
            args = jax.tree.map(lambda x: _hashable(x), self._args)
            kwargs = jax.tree.map(
                lambda x: _hashable(x), tuple(sorted(self._kwargs.items()))
            )
        key = args + kwargs
        try:
            fns = self._cache.get(key, None)
        # If unhashable arguments are provided, proceed normally
        # without caching
        except TypeError:
            fns, key = None, None
        if fns is None:

            def laxmap_postprocess_fn(states, args, kwargs):
                if self.postprocess_fn is None:
                    body_fn = self.sampler.postprocess_fn(args, kwargs)
                else:
                    body_fn = self.postprocess_fn
                if self.chain_method == "vectorized" and self.num_chains > 1:
                    body_fn = vmap(body_fn)

                return lax.map(body_fn, states)

            if self._jit_model_args:
                sample_fn = partial(_sample_fn_jit_args, sampler=self.sampler)
                postprocess_fn = jit(laxmap_postprocess_fn)
            else:
                sample_fn = partial(
                    _sample_fn_nojit_args,
                    sampler=self.sampler,
                    args=self._args,
                    kwargs=self._kwargs,
                )
                postprocess_fn = jit(
                    partial(laxmap_postprocess_fn, args=self._args, kwargs=self._kwargs)
                )

            fns = sample_fn, postprocess_fn
            if key is not None:
                self._cache[key] = fns
        return fns

    def _get_cached_init_state(self, rng_key, args, kwargs):
        rng_key = (_hashable(rng_key),)
        args = jax.tree.map(lambda x: _hashable(x), args)
        kwargs = jax.tree.map(lambda x: _hashable(x), tuple(sorted(kwargs.items())))
        key = rng_key + args + kwargs
        try:
            return self._init_state_cache.get(key, None)
        # If unhashable arguments are provided, return None
        except TypeError:
            return None

    def _single_chain_mcmc(self, init, args, kwargs, collect_fields, remove_sites):
        rng_key, init_state, init_params = init
        # Check if _sample_fn is None, then we need to initialize the sampler.
        if init_state is None or (getattr(self.sampler, "_sample_fn", None) is None):
            new_init_state = self.sampler.init(
                rng_key,
                self.num_warmup,
                init_params,
                model_args=args,
                model_kwargs=kwargs,
            )
            init_state = new_init_state if init_state is None else init_state
        sample_fn, postprocess_fn = self._get_cached_fns()
        diagnostics = (  # noqa: E731
            lambda x: self.sampler.get_diagnostics_str(x[0])
            if is_prng_key(rng_key) or self.sampler.is_ensemble_kernel
            else ""
        )
        init_val = (init_state, args, kwargs) if self._jit_model_args else (init_state,)
        lower_idx = self._collection_params["lower"]
        upper_idx = self._collection_params["upper"]
        phase = self._collection_params["phase"]
        collection_size = self._collection_params["collection_size"]
        collection_size = (
            collection_size
            if collection_size is None
            else collection_size // self.thinning
        )
        collect_vals = fori_collect(
            lower_idx,
            upper_idx,
            sample_fn,
            init_val,
            transform=_collect_fn(collect_fields, remove_sites),
            progbar=self.progress_bar,
            return_last_val=True,
            thinning=self.thinning,
            collection_size=collection_size,
            progbar_desc=partial(_get_progbar_desc_str, lower_idx, phase),
            diagnostics_fn=diagnostics,
            num_chains=self.num_chains
            if (callable(self.chain_method) or self.chain_method == "parallel")
            else 1,
        )
        states, last_val = collect_vals
        # Get first argument of type `HMCState`
        last_state = last_val[0]
        if len(collect_fields) == 1:
            states = (states,)
        states = dict(zip(collect_fields, states))
        # Apply constraints if number of samples is non-zero
        site_values = jax.tree.flatten(states[self._sample_field])[0]
        # XXX: lax.map still works if some arrays have 0 size
        # so we only need to filter out the case site_value.shape[0] == 0
        # (which happens when lower_idx==upper_idx)
        if len(site_values) > 0 and jnp.shape(site_values[0])[0] > 0:
            if self._jit_model_args:
                states[self._sample_field] = postprocess_fn(
                    states[self._sample_field], args, kwargs
                )
            else:
                states[self._sample_field] = postprocess_fn(states[self._sample_field])
        return states, last_state

    def _set_collection_params(
        self, lower=None, upper=None, collection_size=None, phase=None
    ):
        self._collection_params["lower"] = self.num_warmup if lower is None else lower
        self._collection_params["upper"] = (
            self.num_warmup + self.num_samples if upper is None else upper
        )
        self._collection_params["collection_size"] = collection_size
        self._collection_params["phase"] = phase

    def _compile(self, rng_key, *args, extra_fields=(), init_params=None, **kwargs):
        self._set_collection_params(0, 0, self.num_samples)
        self.run(
            rng_key, *args, extra_fields=extra_fields, init_params=init_params, **kwargs
        )
        rng_key = (_hashable(rng_key),)
        args = jax.tree.map(lambda x: _hashable(x), args)
        kwargs = jax.tree.map(lambda x: _hashable(x), tuple(sorted(kwargs.items())))
        key = rng_key + args + kwargs
        try:
            self._init_state_cache[key] = self._last_state
        # If unhashable arguments are provided, return None
        except TypeError:
            pass

    def _get_states_flat(self):
        if self._states_flat is None:
            self._states_flat = jax.tree.map(
                # need to calculate first dimension manually; see issue #1328
                lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1],) + x.shape[2:]),
                self._states,
            )
        return self._states_flat

    @property
    def post_warmup_state(self):
        """
        The state before the sampling phase. If this attribute is not None,
        :meth:`run` will skip the warmup phase and start with the state
        specified in this attribute.

        .. note:: This attribute can be used to sequentially draw MCMC samples. For example,

            .. code-block:: python

                mcmc = MCMC(NUTS(model), num_warmup=100, num_samples=100)
                mcmc.run(random.PRNGKey(0))
                first_100_samples = mcmc.get_samples()
                mcmc.post_warmup_state = mcmc.last_state
                mcmc.run(mcmc.post_warmup_state.rng_key)  # or mcmc.run(random.PRNGKey(1))
                second_100_samples = mcmc.get_samples()
        """
        return self._warmup_state

    @post_warmup_state.setter
    def post_warmup_state(self, state):
        self._warmup_state = state

    @property
    def last_state(self):
        """
        The final MCMC state at the end of the sampling phase.
        """
        return self._last_state

    def warmup(
        self,
        rng_key,
        *args,
        extra_fields=(),
        collect_warmup=False,
        init_params=None,
        **kwargs,
    ):
        """
        Run the MCMC warmup adaptation phase. After this call, `self.post_warmup_state` will be set
        and the :meth:`run` method will skip the warmup adaptation phase. To run `warmup` again
        for the new data, it is required to run :meth:`warmup` again.

        :param random.PRNGKey rng_key: Random number generator key to be used for the sampling.
        :param args: Arguments to be provided to the :meth:`numpyro.infer.mcmc.MCMCKernel.init` method.
            These are typically the arguments needed by the `model`.
        :param extra_fields: Extra fields (aside from :meth:`~numpyro.infer.mcmc.MCMCKernel.default_fields`)
            from the state object (e.g. :data:`numpyro.infer.hmc.HMCState` for HMC) to collect during
            the MCMC run. Exclude sample sites from collection with "~`sampler.sample_field`.`sample_site`".
            e.g. "~z.a" will prevent site "a" from being collected if you're using the NUTS sampler.
        :type extra_fields: tuple or list
        :param bool collect_warmup: Whether to collect samples from the warmup phase. Defaults
            to `False`.
        :param init_params: Initial parameters to begin sampling. The type must be consistent
            with the input type to `potential_fn` provided to the kernel. If the kernel is
            instantiated by a numpyro model, the initial parameters here correspond to latent
            values in unconstrained space.
        :param kwargs: Keyword arguments to be provided to the :meth:`numpyro.infer.mcmc.MCMCKernel.init`
            method. These are typically the keyword arguments needed by the `model`.
        """
        self._warmup_state = None
        if collect_warmup:
            self._set_collection_params(0, self.num_warmup, self.num_warmup, "warmup")
        else:
            self._set_collection_params(
                self.num_warmup, self.num_warmup, self.num_samples, "warmup"
            )
        self.run(
            rng_key, *args, extra_fields=extra_fields, init_params=init_params, **kwargs
        )
        self._warmup_state = self._last_state

    def run(self, rng_key, *args, extra_fields=(), init_params=None, **kwargs):
        """
        Run the MCMC samplers and collect samples.

        :param random.PRNGKey rng_key: Random number generator key to be used for the sampling.
            For multi-chains, a batch of `num_chains` keys can be supplied. If `rng_key`
            does not have batch_size, it will be split in to a batch of `num_chains` keys.
        :param args: Arguments to be provided to the :meth:`numpyro.infer.mcmc.MCMCKernel.init` method.
            These are typically the arguments needed by the `model`.
        :param extra_fields: Extra fields (aside from `"z"`, `"diverging"`) from the
            state object (e.g. :data:`numpyro.infer.hmc.HMCState` for HMC) to be collected
            during the MCMC run. Note that subfields can be accessed using dots, e.g.
            `"adapt_state.step_size"` can be used to collect step sizes at each step. Exclude sample sites from
            collection with "~`sampler.sample_field`.`sample_site`". e.g. "~z.a" will prevent site "a" from
            being collected if you're using the NUTS sampler.
        :type extra_fields: tuple or list of str
        :param init_params: Initial parameters to begin sampling. The type must be consistent
            with the input type to `potential_fn` provided to the kernel. If the kernel is
            instantiated by a numpyro model, the initial parameters here correspond to latent
            values in unconstrained space.
        :param kwargs: Keyword arguments to be provided to the :meth:`numpyro.infer.mcmc.MCMCKernel.init`
            method. These are typically the keyword arguments needed by the `model`.

        .. note:: jax allows python code to continue even when the compiled code has not finished yet.
            This can cause troubles when trying to profile the code for speed.
            See https://jax.readthedocs.io/en/latest/async_dispatch.html and
            https://jax.readthedocs.io/en/latest/profiling.html for pointers on profiling jax programs.
        """
        init_params = jax.tree.map(
            lambda x: lax.convert_element_type(x, jnp.result_type(x)), init_params
        )
        self._args = args
        self._kwargs = kwargs
        init_state = self._get_cached_init_state(rng_key, args, kwargs)
        if self.num_chains > 1 and is_prng_key(rng_key):
            rng_key = random.split(rng_key, self.num_chains)

        if self._warmup_state is not None:
            self._set_collection_params(0, self.num_samples, self.num_samples, "sample")
            init_state = self._warmup_state._replace(rng_key=rng_key)

        if init_params is not None and self.num_chains > 1:
            prototype_init_val = jax.tree.flatten(init_params)[0][0]
            if jnp.shape(prototype_init_val)[0] != self.num_chains:
                raise ValueError(
                    "`init_params` must have the same leading dimension"
                    " as `num_chains`."
                )
        assert isinstance(extra_fields, (tuple, list))

        collect_fields = {}
        remove_sites = {}
        for field_name in (
            (self._sample_field,) + tuple(self._default_fields) + tuple(extra_fields)
        ):
            if field_name.startswith(f"~{self._sample_field}."):
                remove_sites[(field_name[len(self._sample_field) + 2 :])] = None
            else:
                collect_fields[field_name] = None
        collect_fields = tuple(collect_fields.keys())
        remove_sites = tuple(remove_sites.keys())

        partial_map_fn = partial(
            self._single_chain_mcmc,
            args=args,
            kwargs=kwargs,
            collect_fields=collect_fields,
            remove_sites=remove_sites,
        )
        map_args = (rng_key, init_state, init_params)
        if self.num_chains == 1:
            states_flat, last_state = partial_map_fn(map_args)
            states = jax.tree.map(lambda x: x[jnp.newaxis, ...], states_flat)
        else:
            if self.chain_method == "sequential":
                states, last_state = _laxmap(partial_map_fn, map_args)
            elif self.chain_method == "parallel":
                states, last_state = pmap(partial_map_fn)(map_args)
            elif callable(self.chain_method):
                states, last_state = self.chain_method(partial_map_fn)(map_args)
            else:
                assert self.chain_method == "vectorized"
                states, last_state = partial_map_fn(map_args)
                # swap num_samples x num_chains to num_chains x num_samples
                states = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), states)

        self._last_state = last_state
        self._states = states
        self._states_flat = None
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

        **Example:**

        You can then pass those samples to :class:`~numpyro.infer.util.Predictive`::

            posterior_samples = mcmc.get_samples()
            predictive = Predictive(model, posterior_samples=posterior_samples)
            samples = predictive(rng_key1, *model_args, **model_kwargs)

        """
        return (
            self._states[self._sample_field]
            if group_by_chain
            else self._get_states_flat()[self._sample_field]
        )

    def get_extra_fields(self, group_by_chain=False):
        """
        Get extra fields from the MCMC run.

        :param bool group_by_chain: Whether to preserve the chain dimension. If True,
            all samples will have num_chains as the size of their leading dimension.
        :return: Extra fields keyed by field names which are specified in the
            `extra_fields` keyword of :meth:`run`.
        """
        states = self._states if group_by_chain else self._get_states_flat()
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
                sites = {
                    k: v
                    for k, v in self._states[self._sample_field].items()
                    if k in state_sample_field
                }
        print_summary(sites, prob=prob)
        extra_fields = self.get_extra_fields()
        if "diverging" in extra_fields:
            print(
                "Number of divergences: {}".format(jnp.sum(extra_fields["diverging"]))
            )

    def transfer_states_to_host(self):
        """
        Reduce the memory footprint of collected samples by transferring them to the host device.
        """
        self._states = device_get(self._states)
        self._states_flat = device_get(self._get_states_flat())

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state
