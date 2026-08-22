# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from collections.abc import Callable, Sequence
import copy
from functools import partial
from typing import Any, TypeAlias

import jax
from jax import random
import jax.numpy as jnp

import numpyro
from numpyro._typing import (
    ConstrainFn,
    ModelArgs,
    ModelKwargs,
    ModelT,
    PyTree,
    SiteValues,
)
from numpyro.contrib.ecs_proxies import block_update, perturbed_method, taylor_proxy
from numpyro.infer.gibbs import CustomGibbs, DiscreteGibbs, Gibbs, GibbsState
from numpyro.infer.gibbs_util import (
    GIBBS_SITES_KWARG,
    GibbsUpdateFn,
    ModelWrapper,
    conditioned,
    discrete_latent_sites,
    prototype_trace,
    subsample_plate_sizes,
    with_conditioning,
)
from numpyro.infer.hmc import HMC, HMCState
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import _unconstrain_params
from numpyro.util import cond, identity


class HMCGibbsState(GibbsState):
    """
    :class:`~numpyro.infer.gibbs.GibbsState` of :class:`HMCGibbs` and :class:`DiscreteHMCGibbs`,
    constructed by their `init` method (not positionally).

    - **z** - a dict of the current latent values (both HMC and Gibbs sites)
    - **block_states** - the states of the Gibbs block and of the HMC block
    - **rng_key** - random key for the current step
    - **hmc_state** - property returning the current :data:`~numpyro.infer.hmc.HMCState`
      (the last block state), so that `extra_fields=["hmc_state.potential_energy"]` works
    """

    __slots__ = ()

    @property
    def hmc_state(self) -> HMCState:
        return self.block_states[-1]


class HMCGibbs(Gibbs):
    """
    [EXPERIMENTAL INTERFACE]

    HMC-within-Gibbs. This inference algorithm allows the user to combine
    general purpose gradient-based inference (HMC or NUTS) with custom
    Gibbs samplers. It is equivalent to
    ``Gibbs([(CustomGibbs(gibbs_fn), gibbs_sites), (inner_kernel, None)])``.

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
        ...     y = hmc_sites['y']
        ...     new_x = dist.Normal(0.8 * (1-y), jnp.sqrt(0.8)).sample(rng_key)
        ...     return {'x': new_x}
        ...
        >>> hmc_kernel = NUTS(model)
        >>> kernel = HMCGibbs(hmc_kernel, gibbs_fn=gibbs_fn, gibbs_sites=['x'])
        >>> mcmc = MCMC(kernel, num_warmup=100, num_samples=100, progress_bar=False)
        >>> mcmc.run(random.key(0))
        >>> mcmc.print_summary()  # doctest: +SKIP

    """

    _state_cls = HMCGibbsState

    def __init__(
        self,
        inner_kernel: HMC,
        gibbs_fn: GibbsUpdateFn,
        gibbs_sites: Sequence[str],
    ) -> None:
        if not isinstance(inner_kernel, HMC):
            raise ValueError("inner_kernel must be an HMC or NUTS sampler.")
        if not callable(gibbs_fn):
            raise ValueError("gibbs_fn must be a callable")
        if inner_kernel.model is None:
            raise ValueError(
                "HMCGibbs does not support models specified via a potential function."
            )
        super().__init__([(CustomGibbs(gibbs_fn), gibbs_sites), (inner_kernel, None)])
        self.inner_kernel = self._kernels[1]
        self._gibbs_fn = gibbs_fn
        self._gibbs_sites = tuple(gibbs_sites)


class DiscreteHMCGibbs(Gibbs):
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
        >>> mcmc.run(random.key(0), probs, locs)
        >>> mcmc.print_summary()  # doctest: +SKIP
        >>> samples = mcmc.get_samples()["x"]
        >>> assert abs(jnp.mean(samples) - 1.3) < 0.2
        >>> assert abs(jnp.var(samples) - 4.36) < 0.5

    """

    _state_cls = HMCGibbsState

    def __init__(
        self,
        inner_kernel: HMC,
        *,
        random_walk: bool = False,
        modified: bool = False,
    ) -> None:
        if not isinstance(inner_kernel, HMC):
            raise ValueError("inner_kernel must be an HMC or NUTS sampler.")
        if inner_kernel.model is None:
            raise ValueError(
                "DiscreteHMCGibbs does not support models specified via a potential function."
            )
        discrete_kernel = DiscreteGibbs(
            inner_kernel.model, random_walk=random_walk, modified=modified
        )
        super().__init__(
            [(discrete_kernel, discrete_latent_sites), (inner_kernel, None)]
        )
        self.inner_kernel = self._kernels[1]
        self._random_walk = random_walk
        self._modified = modified


HMCECSState = namedtuple(
    "HMCECSState", "z, hmc_state, rng_key, gibbs_state, accept_prob"
)

LikelihoodEstimator: TypeAlias = Callable[
    [dict[str, tuple], SiteValues, PyTree], jax.Array
]
"""
`(likelihoods, unconstrained_params, gibbs_state) -> log-likelihood estimate`; see
`perturbed_method` in :mod:`numpyro.contrib.ecs_proxies`.
"""

ProxyConstructor: TypeAlias = Callable[
    ..., tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]
]
"""
`(prototype_trace, subsample_plate_sizes, model, model_args, model_kwargs, num_blocks) ->
(proxy_fn, gibbs_init, gibbs_update)`; see :func:`~numpyro.contrib.ecs_proxies.taylor_proxy`.
"""


def _ecs_model(model, estimator, *args, **kwargs):
    """
    Model wrapper installed once by :class:`HMCECS`: pops `_gibbs_state` from the keyword
    arguments, hands it to `estimator`, and runs `model` under `estimator`. When the estimator
    has no `method` (no proxy), the model runs with its plain subsampled likelihood.
    """
    gibbs_state = kwargs.pop("_gibbs_state", ())
    if estimator.method is None:
        return model(*args, **kwargs)
    estimator.gibbs_state = gibbs_state
    with estimator:
        return model(*args, **kwargs)


def _wrap_ecs(model, estimator):
    return partial(_ecs_model, conditioned(model), estimator)


class HMCECS(MCMCKernel):
    """
    [EXPERIMENTAL INTERFACE]

    HMC with Energy Conserving Subsampling.

    A wrapper around an HMC kernel for performing HMC-within-Gibbs for models with subsample
    statements using the :class:`~numpyro.plate` primitive: it changes the target of the
    inner kernel (likelihood estimator) and performs the pseudo-marginal Metropolis update of
    the subsample indices. This implements Algorithm 1 of reference [1] but uses a naive
    estimation (without control variates) of log likelihood, hence might incur a high variance.

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
        >>> data = random.normal(random.key(0), (10000,)) + 1
        >>> kernel = HMCECS(NUTS(model), num_blocks=10)
        >>> mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000)
        >>> mcmc.run(random.key(0), data)
        >>> samples = mcmc.get_samples()["x"]
        >>> assert abs(jnp.mean(samples) - 1.) < 0.2

    """

    sample_field: str = "z"

    def __init__(
        self,
        inner_kernel: HMC,
        *,
        num_blocks: int = 1,
        proxy: ProxyConstructor | None = None,
    ) -> None:
        if not isinstance(inner_kernel, HMC):
            raise ValueError("inner_kernel must be an HMC or NUTS sampler.")
        if inner_kernel.model is None:
            raise ValueError(
                "HMCECS does not support models specified via a potential function."
            )
        self._model = inner_kernel.model
        self._estimator = estimate_likelihood()
        # wrap once, at construction: `init` only binds the estimator's method and state
        self.inner_kernel = inner_kernel.wrap_model(
            partial(_wrap_ecs, estimator=self._estimator)
        )
        self._num_blocks = num_blocks
        self._proxy = proxy
        # static metadata resolved at `init`
        self._subsample_plate_sizes: dict[str, tuple[int, int]] | None = None
        self._gibbs_sites: tuple[str, ...] = ()
        self._gibbs_update = None
        self._sample_fn = None

    @property
    def model(self) -> ModelT:
        return self._model

    def get_diagnostics_str(self, state: HMCECSState) -> str:
        return self.inner_kernel.get_diagnostics_str(state.hmc_state)

    def _inner_kwargs(
        self, model_kwargs: ModelKwargs | None, z_gibbs: SiteValues, gibbs_state: PyTree
    ) -> ModelKwargs:
        model_kwargs = with_conditioning(model_kwargs, z_gibbs)
        model_kwargs["_gibbs_state"] = gibbs_state
        return model_kwargs

    def _split(self, z: SiteValues) -> tuple[SiteValues, SiteValues]:
        z_gibbs = {k: v for k, v in z.items() if k in self._gibbs_sites}
        z_hmc = {k: v for k, v in z.items() if k not in self._gibbs_sites}
        return z_gibbs, z_hmc

    def postprocess_fn(
        self, model_args: ModelArgs, model_kwargs: ModelKwargs | None
    ) -> ConstrainFn:
        """Inner postprocess on the HMC sites; subsample indices are dropped."""

        def fn(z: SiteValues) -> SiteValues:
            z_gibbs, z_hmc = self._split(z)
            return self.inner_kernel.postprocess_fn(
                model_args, with_conditioning(model_kwargs, z_gibbs)
            )(z_hmc)

        return fn

    def get_constrain_fn(
        self, model_args: ModelArgs, model_kwargs: ModelKwargs | None
    ) -> ConstrainFn:
        """Inner constrain function on the HMC sites; subsample indices are dropped."""

        def fn(z: SiteValues) -> SiteValues:
            z_gibbs, z_hmc = self._split(z)
            return self.inner_kernel.get_constrain_fn(
                model_args, with_conditioning(model_kwargs, z_gibbs)
            )(z_hmc)

        return fn

    def init(
        self,
        rng_key: jax.Array,
        num_warmup: int,
        init_params: SiteValues | None,
        model_args: ModelArgs,
        model_kwargs: ModelKwargs | None,
    ) -> HMCECSState:
        model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        rng_key, key_u = random.split(rng_key)
        model = conditioned(self._model)
        model_trace = prototype_trace(model, key_u, model_args, model_kwargs)
        self._subsample_plate_sizes = subsample_plate_sizes(model_trace)
        self._gibbs_sites = tuple(self._subsample_plate_sizes)
        if not self._gibbs_sites:
            raise ValueError("Cannot detect any subsample statements in the model.")
        for name in model_kwargs.get(GIBBS_SITES_KWARG, {}):
            site = model_trace.get(name)
            if site is not None and any(
                frame.name in self._subsample_plate_sizes
                for frame in site["cond_indep_stack"]
            ):
                raise ValueError(
                    f"Site '{name}' is conditioned by an enclosing kernel but lies inside "
                    "a subsample plate; HMCECS cannot estimate its likelihood."
                )
        if self._proxy is not None:
            if any(
                site["type"] == "sample"
                and (not site["is_observed"])
                and site["fn"].support.is_discrete
                for site in model_trace.values()
            ):
                raise RuntimeError(
                    "Currently, the proxy does not support models with "
                    "discrete latent sites."
                )
            proxy_fn, gibbs_init, self._gibbs_update = self._proxy(
                model_trace,
                self._subsample_plate_sizes,
                model,
                model_args,
                model_kwargs.copy(),
                num_blocks=self._num_blocks,
            )
            self._estimator.method = perturbed_method(
                self._subsample_plate_sizes, proxy_fn
            )
        else:
            self._estimator.method = None
            self._gibbs_update = partial(
                block_update, self._subsample_plate_sizes, self._num_blocks
            )

        init_params = None if init_params is None else dict(init_params)
        z_gibbs = {}
        for name in self._gibbs_sites:
            if init_params and name in init_params:
                z_gibbs[name] = init_params.pop(name)
            else:
                z_gibbs[name] = model_trace[name]["value"]

        if self._proxy is not None:
            rng_key, rng_state = random.split(rng_key)
            gibbs_state = gibbs_init(rng_state, z_gibbs)
        else:
            gibbs_state = ()

        rng_key, key_z = random.split(rng_key)
        hmc_state = self.inner_kernel.init(
            key_z,
            num_warmup,
            init_params or None,
            model_args,
            self._inner_kwargs(model_kwargs, z_gibbs, gibbs_state),
        )
        z = {**z_gibbs, **hmc_state.z}
        self._sample_fn = self._sample_one
        return HMCECSState(z, hmc_state, rng_key, gibbs_state, jnp.zeros(()))

    def _sample_one(
        self,
        state: HMCECSState,
        model_args: ModelArgs,
        model_kwargs: ModelKwargs | None,
    ) -> HMCECSState:
        rng_key, rng_gibbs, rng_accept = random.split(state.rng_key, 3)

        z_gibbs, _ = self._split(state.z)
        z_gibbs_new, gibbs_state_new = self._gibbs_update(
            rng_gibbs, z_gibbs, state.gibbs_state
        )

        # given a fixed hmc_sites, pe_new - pe_curr = loglik_new - loglik_curr
        pe = state.hmc_state.potential_energy
        pe_new = self.inner_kernel.get_potential_fn(
            model_args, self._inner_kwargs(model_kwargs, z_gibbs_new, gibbs_state_new)
        )(state.hmc_state.z)
        accept_prob = jnp.clip(jnp.exp(pe - pe_new), None, 1.0)
        transition = random.bernoulli(rng_accept, accept_prob)

        def accept(vals):
            z_gibbs_new, gibbs_state_new, _ = vals
            refreshed = self.inner_kernel.refresh(
                state.hmc_state,
                model_args,
                self._inner_kwargs(model_kwargs, z_gibbs_new, gibbs_state_new),
            )
            return (
                z_gibbs_new,
                gibbs_state_new,
                refreshed.potential_energy,
                refreshed.z_grad,
            )

        z_gibbs, gibbs_state, pe, z_grad = cond(
            transition,
            (z_gibbs_new, gibbs_state_new, pe_new),
            accept,
            (z_gibbs, state.gibbs_state, pe, state.hmc_state.z_grad),
            identity,
        )

        hmc_state = state.hmc_state._replace(z_grad=z_grad, potential_energy=pe)
        hmc_state = self.inner_kernel.sample(
            hmc_state,
            model_args,
            self._inner_kwargs(model_kwargs, z_gibbs, gibbs_state),
        )

        z = {**z_gibbs, **hmc_state.z}
        return HMCECSState(z, hmc_state, rng_key, gibbs_state, accept_prob)

    def sample(
        self,
        state: HMCECSState,
        model_args: ModelArgs,
        model_kwargs: ModelKwargs | None,
    ) -> HMCECSState:
        return self._sample_fn(state, model_args, model_kwargs)

    def refresh(
        self,
        state: HMCECSState,
        model_args: ModelArgs,
        model_kwargs: ModelKwargs | None,
    ) -> HMCECSState:
        """Delegates to the inner kernel with the current subsample indices and proxy state."""
        z_gibbs, _ = self._split(state.z)
        hmc_state = self.inner_kernel.refresh(
            state.hmc_state,
            model_args,
            self._inner_kwargs(model_kwargs, z_gibbs, state.gibbs_state),
        )
        return state._replace(hmc_state=hmc_state)

    def wrap_model(self, wrapper: ModelWrapper) -> "HMCECS":
        kernel = copy.copy(self)
        kernel._model = wrapper(self._model)
        kernel.inner_kernel = self.inner_kernel.wrap_model(wrapper)
        kernel._sample_fn = None
        return kernel

    @staticmethod
    def taylor_proxy(reference_params: SiteValues, degree: int = 2) -> ProxyConstructor:
        """
        This is just a convenient static method which calls
        :func:`~numpyro.contrib.ecs_proxies.taylor_proxy`.
        """
        return taylor_proxy(reference_params, degree)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_sample_fn"] = None
        return state


class estimate_likelihood(numpyro.primitives.Messenger):
    """
    Handler that replaces the subsampled likelihood of a model by a bias-corrected estimate.
    `method` accepts the likelihood tuples `(fn, value, subsample_name, subsample_dim)`, the
    current unconstrained parameters and the proxy state (`gibbs_state`) and returns the log
    of the estimated likelihood. Both `method` and `gibbs_state` can be set after
    construction; the handler is inert while `method` is `None`.
    """

    def __init__(
        self, fn: ModelT | None = None, method: LikelihoodEstimator | None = None
    ) -> None:
        super().__init__(fn)
        self.method = method
        self.params = None
        self.likelihoods = {}
        self.subsample_plates = {}
        self.gibbs_state = None

    def __enter__(self):
        if self.method is not None:
            for handler in numpyro.primitives._PYRO_STACK[::-1]:
                # the potential_fn in HMC makes the PYRO_STACK nested like trace(...); so we
                # can extract the unconstrained params from the `_unconstrain_params` handler
                if isinstance(handler, _unconstrain_params):
                    self.params = handler.params
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
        elif (
            msg["type"] == "plate"
            and (msg["args"][1] is not None)
            and msg["args"][0] > msg["args"][1]
        ):
            self.subsample_plates[msg["name"]] = msg["value"]
