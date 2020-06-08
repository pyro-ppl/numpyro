# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections import namedtuple
import functools
import math
from operator import attrgetter
import os
import warnings

from jax import jit, lax, partial, pmap, random, vmap, device_put
from jax.core import Tracer
from jax.dtypes import canonicalize_dtype
from jax.flatten_util import ravel_pytree
from jax.interpreters.xla import DeviceArray
from jax.lib import xla_bridge
import jax.numpy as np
from jax.random import PRNGKey
from jax.scipy.special import logsumexp
from jax.tree_util import tree_flatten, tree_map, tree_multimap

from numpyro.diagnostics import print_summary
import numpyro.distributions as dist
from numpyro.distributions.util import categorical_logits, cholesky_update
from numpyro.infer.hmc_util import (
    IntegratorState,
    build_tree,
    euclidean_kinetic_energy,
    find_reasonable_step_size,
    velocity_verlet,
    warmup_adapter
)
from numpyro.infer.util import ParamInfo, init_to_uniform, initialize_model
from numpyro.util import cond, copy_docs_from, fori_collect, fori_loop, identity, cached_by

HMCState = namedtuple('HMCState', ['i', 'z', 'z_grad', 'potential_energy', 'energy', 'num_steps', 'accept_prob',
                                   'mean_accept_prob', 'diverging', 'adapt_state', 'rng_key'])
"""
A :func:`~collections.namedtuple` consisting of the following fields:

 - **i** - iteration. This is reset to 0 after warmup.
 - **z** - Python collection representing values (unconstrained samples from
   the posterior) at latent sites.
 - **z_grad** - Gradient of potential energy w.r.t. latent sample sites.
 - **potential_energy** - Potential energy computed at the given value of ``z``.
 - **energy** - Sum of potential energy and kinetic energy of the current state.
 - **num_steps** - Number of steps in the Hamiltonian trajectory (for diagnostics).
 - **accept_prob** - Acceptance probability of the proposal. Note that ``z``
   does not correspond to the proposal if it is rejected.
 - **mean_accept_prob** - Mean acceptance probability until current iteration
   during warmup adaptation or sampling (for diagnostics).
 - **diverging** - A boolean value to indicate whether the current trajectory is diverging.
 - **adapt_state** - A ``HMCAdaptState`` namedtuple which contains adaptation information
   during warmup:

   + **step_size** - Step size to be used by the integrator in the next iteration.
   + **inverse_mass_matrix** - The inverse mass matrix to be used for the next
     iteration.
   + **mass_matrix_sqrt** - The square root of mass matrix to be used for the next
     iteration. In case of dense mass, this is the Cholesky factorization of the
     mass matrix.

 - **rng_key** - random number generator seed used for the iteration.
"""


def _get_num_steps(step_size, trajectory_length):
    num_steps = np.clip(trajectory_length / step_size, a_min=1)
    # NB: casting to np.int64 does not take effect (returns np.int32 instead)
    # if jax_enable_x64 is False
    return num_steps.astype(canonicalize_dtype(np.int64))


def momentum_generator(prototype_r, mass_matrix_sqrt, rng_key):
    _, unpack_fn = ravel_pytree(prototype_r)
    eps = random.normal(rng_key, np.shape(mass_matrix_sqrt)[:1])
    if mass_matrix_sqrt.ndim == 1:
        r = np.multiply(mass_matrix_sqrt, eps)
        return unpack_fn(r)
    elif mass_matrix_sqrt.ndim == 2:
        r = np.dot(mass_matrix_sqrt, eps)
        return unpack_fn(r)
    else:
        raise ValueError("Mass matrix has incorrect number of dims.")


def get_diagnostics_str(mcmc_state):
    if isinstance(mcmc_state, HMCState):
        return '{} steps of size {:.2e}. acc. prob={:.2f}'.format(mcmc_state.num_steps,
                                                                  mcmc_state.adapt_state.step_size,
                                                                  mcmc_state.mean_accept_prob)
    else:
        return 'acc. prob={:.2f}'.format(mcmc_state.mean_accept_prob)


def get_progbar_desc_str(num_warmup, i):
    if i < num_warmup:
        return 'warmup'
    return 'sample'


def hmc(potential_fn=None, potential_fn_gen=None, kinetic_fn=None, algo='NUTS'):
    r"""
    Hamiltonian Monte Carlo inference, using either fixed number of
    steps or the No U-Turn Sampler (NUTS) with adaptive path length.

    **References:**

    1. *MCMC Using Hamiltonian Dynamics*,
       Radford M. Neal
    2. *The No-U-turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo*,
       Matthew D. Hoffman, and Andrew Gelman.
    3. *A Conceptual Introduction to Hamiltonian Monte Carlo`*,
       Michael Betancourt

    :param potential_fn: Python callable that computes the potential energy
        given input parameters. The input parameters to `potential_fn` can be
        any python collection type, provided that `init_params` argument to
        `init_kernel` has the same type.
    :param potential_fn_gen: Python callable that when provided with model
        arguments / keyword arguments returns `potential_fn`. This
        may be provided to do inference on the same model with changing data.
        If the data shape remains the same, we can compile `sample_kernel`
        once, and use the same for multiple inference runs.
    :param kinetic_fn: Python callable that returns the kinetic energy given
        inverse mass matrix and momentum. If not provided, the default is
        euclidean kinetic energy.
    :param str algo: Whether to run ``HMC`` with fixed number of steps or ``NUTS``
        with adaptive path length. Default is ``NUTS``.
    :return: a tuple of callables (`init_kernel`, `sample_kernel`), the first
        one to initialize the sampler, and the second one to generate samples
        given an existing one.

    .. warning::
        Instead of using this interface directly, we would highly recommend you
        to use the higher level :class:`numpyro.infer.MCMC` API instead.

    **Example**

    .. doctest::

        >>> import jax
        >>> from jax import random
        >>> import jax.numpy as np
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.infer.mcmc import hmc
        >>> from numpyro.infer.util import initialize_model
        >>> from numpyro.util import fori_collect

        >>> true_coefs = np.array([1., 2., 3.])
        >>> data = random.normal(random.PRNGKey(2), (2000, 3))
        >>> dim = 3
        >>> labels = dist.Bernoulli(logits=(true_coefs * data).sum(-1)).sample(random.PRNGKey(3))
        >>>
        >>> def model(data, labels):
        ...     coefs_mean = np.zeros(dim)
        ...     coefs = numpyro.sample('beta', dist.Normal(coefs_mean, np.ones(3)))
        ...     intercept = numpyro.sample('intercept', dist.Normal(0., 10.))
        ...     return numpyro.sample('y', dist.Bernoulli(logits=(coefs * data + intercept).sum(-1)), obs=labels)
        >>>
        >>> model_info = initialize_model(random.PRNGKey(0), model, model_args=(data, labels,))
        >>> init_kernel, sample_kernel = hmc(model_info.potential_fn, algo='NUTS')
        >>> hmc_state = init_kernel(model_info.param_info,
        ...                         trajectory_length=10,
        ...                         num_warmup=300)
        >>> samples = fori_collect(0, 500, sample_kernel, hmc_state,
        ...                        transform=lambda state: model_info.postprocess_fn(state.z))
        >>> print(np.mean(samples['beta'], axis=0))  # doctest: +SKIP
        [0.9153987 2.0754058 2.9621222]
    """
    if kinetic_fn is None:
        kinetic_fn = euclidean_kinetic_energy
    vv_update = None
    trajectory_len = None
    max_treedepth = None
    wa_update = None
    wa_steps = None
    max_delta_energy = 1000.
    if algo not in {'HMC', 'NUTS'}:
        raise ValueError('`algo` must be one of `HMC` or `NUTS`.')

    def init_kernel(init_params,
                    num_warmup,
                    step_size=1.0,
                    inverse_mass_matrix=None,
                    adapt_step_size=True,
                    adapt_mass_matrix=True,
                    dense_mass=False,
                    target_accept_prob=0.8,
                    trajectory_length=2*math.pi,
                    max_tree_depth=10,
                    find_heuristic_step_size=False,
                    model_args=(),
                    model_kwargs=None,
                    rng_key=PRNGKey(0)):
        """
        Initializes the HMC sampler.

        :param init_params: Initial parameters to begin sampling. The type must
            be consistent with the input type to `potential_fn`.
        :param int num_warmup: Number of warmup steps; samples generated
            during warmup are discarded.
        :param float step_size: Determines the size of a single step taken by the
            verlet integrator while computing the trajectory using Hamiltonian
            dynamics. If not specified, it will be set to 1.
        :param numpy.ndarray inverse_mass_matrix: Initial value for inverse mass matrix.
            This may be adapted during warmup if adapt_mass_matrix = True.
            If no value is specified, then it is initialized to the identity matrix.
        :param bool adapt_step_size: A flag to decide if we want to adapt step_size
            during warm-up phase using Dual Averaging scheme.
        :param bool adapt_mass_matrix: A flag to decide if we want to adapt mass
            matrix during warm-up phase using Welford scheme.
        :param bool dense_mass: A flag to decide if mass matrix is dense or
            diagonal (default when ``dense_mass=False``)
        :param float target_accept_prob: Target acceptance probability for step size
            adaptation using Dual Averaging. Increasing this value will lead to a smaller
            step size, hence the sampling will be slower but more robust. Default to 0.8.
        :param float trajectory_length: Length of a MCMC trajectory for HMC. Default
            value is :math:`2\\pi`.
        :param int max_tree_depth: Max depth of the binary tree created during the doubling
            scheme of NUTS sampler. Defaults to 10.
        :param bool find_heuristic_step_size: whether to a heuristic function to adjust the
            step size at the beginning of each adaptation window. Defaults to False.
        :param tuple model_args: Model arguments if `potential_fn_gen` is specified.
        :param dict model_kwargs: Model keyword arguments if `potential_fn_gen` is specified.
        :param jax.random.PRNGKey rng_key: random key to be used as the source of
            randomness.

        """
        step_size = lax.convert_element_type(step_size, canonicalize_dtype(np.float64))
        nonlocal wa_update, trajectory_len, max_treedepth, vv_update, wa_steps
        wa_steps = num_warmup
        trajectory_len = trajectory_length
        max_treedepth = max_tree_depth
        if isinstance(init_params, ParamInfo):
            z, pe, z_grad = init_params
        else:
            z, pe, z_grad = init_params, None, None
        pe_fn = potential_fn
        if potential_fn_gen:
            if pe_fn is not None:
                raise ValueError('Only one of `potential_fn` or `potential_fn_gen` must be provided.')
            else:
                kwargs = {} if model_kwargs is None else model_kwargs
                pe_fn = potential_fn_gen(*model_args, **kwargs)

        find_reasonable_ss = None
        if find_heuristic_step_size:
            find_reasonable_ss = partial(find_reasonable_step_size,
                                         pe_fn,
                                         kinetic_fn,
                                         momentum_generator)

        wa_init, wa_update = warmup_adapter(num_warmup,
                                            adapt_step_size=adapt_step_size,
                                            adapt_mass_matrix=adapt_mass_matrix,
                                            dense_mass=dense_mass,
                                            target_accept_prob=target_accept_prob,
                                            find_reasonable_step_size=find_reasonable_ss)

        rng_key_hmc, rng_key_wa, rng_key_momentum = random.split(rng_key, 3)
        wa_state = wa_init(z, rng_key_wa, step_size,
                           inverse_mass_matrix=inverse_mass_matrix,
                           mass_matrix_size=np.size(ravel_pytree(z)[0]))
        r = momentum_generator(z, wa_state.mass_matrix_sqrt, rng_key_momentum)
        vv_init, vv_update = velocity_verlet(pe_fn, kinetic_fn)
        vv_state = vv_init(z, r, potential_energy=pe, z_grad=z_grad)
        energy = kinetic_fn(wa_state.inverse_mass_matrix, vv_state.r)
        hmc_state = HMCState(0, vv_state.z, vv_state.z_grad, vv_state.potential_energy, energy,
                             0, 0., 0., False, wa_state, rng_key_hmc)
        return device_put(hmc_state)

    def _hmc_next(step_size, inverse_mass_matrix, vv_state,
                  model_args, model_kwargs, rng_key):
        if potential_fn_gen:
            nonlocal vv_update
            pe_fn = potential_fn_gen(*model_args, **model_kwargs)
            _, vv_update = velocity_verlet(pe_fn, kinetic_fn)

        num_steps = _get_num_steps(step_size, trajectory_len)
        vv_state_new = fori_loop(0, num_steps,
                                 lambda i, val: vv_update(step_size, inverse_mass_matrix, val),
                                 vv_state)
        energy_old = vv_state.potential_energy + kinetic_fn(inverse_mass_matrix, vv_state.r)
        energy_new = vv_state_new.potential_energy + kinetic_fn(inverse_mass_matrix, vv_state_new.r)
        delta_energy = energy_new - energy_old
        delta_energy = np.where(np.isnan(delta_energy), np.inf, delta_energy)
        accept_prob = np.clip(np.exp(-delta_energy), a_max=1.0)
        diverging = delta_energy > max_delta_energy
        transition = random.bernoulli(rng_key, accept_prob)
        vv_state, energy = cond(transition,
                                (vv_state_new, energy_new), identity,
                                (vv_state, energy_old), identity)
        return vv_state, energy, num_steps, accept_prob, diverging

    def _nuts_next(step_size, inverse_mass_matrix, vv_state,
                   model_args, model_kwargs, rng_key):
        if potential_fn_gen:
            nonlocal vv_update
            pe_fn = potential_fn_gen(*model_args, **model_kwargs)
            _, vv_update = velocity_verlet(pe_fn, kinetic_fn)

        binary_tree = build_tree(vv_update, kinetic_fn, vv_state,
                                 inverse_mass_matrix, step_size, rng_key,
                                 max_delta_energy=max_delta_energy,
                                 max_tree_depth=max_treedepth)
        accept_prob = binary_tree.sum_accept_probs / binary_tree.num_proposals
        num_steps = binary_tree.num_proposals
        vv_state = IntegratorState(z=binary_tree.z_proposal,
                                   r=vv_state.r,
                                   potential_energy=binary_tree.z_proposal_pe,
                                   z_grad=binary_tree.z_proposal_grad)
        return vv_state, binary_tree.z_proposal_energy, num_steps, accept_prob, binary_tree.diverging

    _next = _nuts_next if algo == 'NUTS' else _hmc_next

    def sample_kernel(hmc_state, model_args=(), model_kwargs=None):
        """
        Given an existing :data:`~numpyro.infer.mcmc.HMCState`, run HMC with fixed (possibly adapted)
        step size and return a new :data:`~numpyro.infer.mcmc.HMCState`.

        :param hmc_state: Current sample (and associated state).
        :param tuple model_args: Model arguments if `potential_fn_gen` is specified.
        :param dict model_kwargs: Model keyword arguments if `potential_fn_gen` is specified.
        :return: new proposed :data:`~numpyro.infer.mcmc.HMCState` from simulating
            Hamiltonian dynamics given existing state.

        """
        model_kwargs = {} if model_kwargs is None else model_kwargs
        rng_key, rng_key_momentum, rng_key_transition = random.split(hmc_state.rng_key, 3)
        r = momentum_generator(hmc_state.z, hmc_state.adapt_state.mass_matrix_sqrt, rng_key_momentum)
        vv_state = IntegratorState(hmc_state.z, r, hmc_state.potential_energy, hmc_state.z_grad)
        vv_state, energy, num_steps, accept_prob, diverging = _next(hmc_state.adapt_state.step_size,
                                                                    hmc_state.adapt_state.inverse_mass_matrix,
                                                                    vv_state,
                                                                    model_args,
                                                                    model_kwargs,
                                                                    rng_key_transition)
        # not update adapt_state after warmup phase
        adapt_state = cond(hmc_state.i < wa_steps,
                           (hmc_state.i, accept_prob, vv_state.z, hmc_state.adapt_state),
                           lambda args: wa_update(*args),
                           hmc_state.adapt_state,
                           identity)

        itr = hmc_state.i + 1
        n = np.where(hmc_state.i < wa_steps, itr, itr - wa_steps)
        mean_accept_prob = hmc_state.mean_accept_prob + (accept_prob - hmc_state.mean_accept_prob) / n

        return HMCState(itr, vv_state.z, vv_state.z_grad, vv_state.potential_energy, energy, num_steps,
                        accept_prob, mean_accept_prob, diverging, adapt_state, rng_key)

    # Make `init_kernel` and `sample_kernel` visible from the global scope once
    # `hmc` is called for sphinx doc generation.
    if 'SPHINX_BUILD' in os.environ:
        hmc.init_kernel = init_kernel
        hmc.sample_kernel = sample_kernel

    return init_kernel, sample_kernel


class MCMCKernel(ABC):
    """
    Defines the interface for the Markov transition kernel that is
    used for :class:`~numpyro.infer.MCMC` inference.
    """
    def postprocess_fn(self, model_args, model_kwargs):
        """
        Function that transforms unconstrained values at sample sites to values
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
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, state, model_args, model_kwargs):
        """
        Given the current `state`, return the next `state` using the given
        transition kernel.

        :param state: Arbitrary data structure representing the state for the
            kernel. For HMC, this is given by :data:`~numpyro.infer.mcmc.HMCState`.
        :param model_args: Arguments provided to the model.
        :param model_kwargs: Keyword arguments provided to the model.
        :return: Next `state`.
        """
        raise NotImplementedError


class HMC(MCMCKernel):
    """
    Hamiltonian Monte Carlo inference, using fixed trajectory length, with
    provision for step size and mass matrix adaptation.

    **References:**

    1. *MCMC Using Hamiltonian Dynamics*,
       Radford M. Neal

    :param model: Python callable containing Pyro :mod:`~numpyro.primitives`.
        If model is provided, `potential_fn` will be inferred using the model.
    :param potential_fn: Python callable that computes the potential energy
        given input parameters. The input parameters to `potential_fn` can be
        any python collection type, provided that `init_params` argument to
        `init_kernel` has the same type.
    :param kinetic_fn: Python callable that returns the kinetic energy given
        inverse mass matrix and momentum. If not provided, the default is
        euclidean kinetic energy.
    :param float step_size: Determines the size of a single step taken by the
        verlet integrator while computing the trajectory using Hamiltonian
        dynamics. If not specified, it will be set to 1.
    :param bool adapt_step_size: A flag to decide if we want to adapt step_size
        during warm-up phase using Dual Averaging scheme.
    :param bool adapt_mass_matrix: A flag to decide if we want to adapt mass
        matrix during warm-up phase using Welford scheme.
    :param bool dense_mass:  A flag to decide if mass matrix is dense or
        diagonal (default when ``dense_mass=False``)
    :param float target_accept_prob: Target acceptance probability for step size
        adaptation using Dual Averaging. Increasing this value will lead to a smaller
        step size, hence the sampling will be slower but more robust. Default to 0.8.
    :param float trajectory_length: Length of a MCMC trajectory for HMC. Default
        value is :math:`2\\pi`.
    :param callable init_strategy: a per-site initialization function.
        See :ref:`init_strategy` section for available functions.
    :param bool find_heuristic_step_size: whether to a heuristic function to adjust the
        step size at the beginning of each adaptation window. Defaults to False.
    """
    def __init__(self,
                 model=None,
                 potential_fn=None,
                 kinetic_fn=None,
                 step_size=1.0,
                 adapt_step_size=True,
                 adapt_mass_matrix=True,
                 dense_mass=False,
                 target_accept_prob=0.8,
                 trajectory_length=2 * math.pi,
                 init_strategy=init_to_uniform,
                 find_heuristic_step_size=False):
        if not (model is None) ^ (potential_fn is None):
            raise ValueError('Only one of `model` or `potential_fn` must be specified.')
        self._model = model
        self._potential_fn = potential_fn
        self._kinetic_fn = kinetic_fn if kinetic_fn is not None else euclidean_kinetic_energy
        self._step_size = step_size
        self._adapt_step_size = adapt_step_size
        self._adapt_mass_matrix = adapt_mass_matrix
        self._dense_mass = dense_mass
        self._target_accept_prob = target_accept_prob
        self._trajectory_length = trajectory_length
        self._algo = 'HMC'
        self._max_tree_depth = 10
        self._init_strategy = init_strategy
        self._find_heuristic_step_size = find_heuristic_step_size
        # Set on first call to init
        self._init_fn = None
        self._postprocess_fn = None
        self._sample_fn = None

    def _init_state(self, rng_key, model_args, model_kwargs, init_params):
        if self._model is not None:
            init_params, potential_fn, postprocess_fn, model_trace = initialize_model(
                rng_key,
                self._model,
                dynamic_args=True,
                model_args=model_args,
                model_kwargs=model_kwargs)
            if any(v['type'] == 'param' for v in model_trace.values()):
                warnings.warn("'param' sites will be treated as constants during inference. To define "
                              "an improper variable, please use a 'sample' site with log probability "
                              "masked out. For example, `sample('x', dist.LogNormal(0, 1).mask(False)` "
                              "means that `x` has improper distribution over the positive domain.")
            if self._init_fn is None:
                self._init_fn, self._sample_fn = hmc(potential_fn_gen=potential_fn,
                                                     kinetic_fn=self._kinetic_fn,
                                                     algo=self._algo)
            self._postprocess_fn = postprocess_fn
        elif self._init_fn is None:
            self._init_fn, self._sample_fn = hmc(potential_fn=self._potential_fn,
                                                 kinetic_fn=self._kinetic_fn,
                                                 algo=self._algo)

        return init_params

    @property
    def model(self):
        return self._model

    @copy_docs_from(MCMCKernel.init)
    def init(self, rng_key, num_warmup, init_params=None, model_args=(), model_kwargs={}):
        # non-vectorized
        if rng_key.ndim == 1:
            rng_key, rng_key_init_model = random.split(rng_key)
        # vectorized
        else:
            rng_key, rng_key_init_model = np.swapaxes(vmap(random.split)(rng_key), 0, 1)
        init_params = self._init_state(rng_key_init_model, model_args, model_kwargs, init_params)
        if self._potential_fn and init_params is None:
            raise ValueError('Valid value of `init_params` must be provided with'
                             ' `potential_fn`.')

        hmc_init_fn = lambda init_params, rng_key: self._init_fn(  # noqa: E731
            init_params,
            num_warmup=num_warmup,
            step_size=self._step_size,
            adapt_step_size=self._adapt_step_size,
            adapt_mass_matrix=self._adapt_mass_matrix,
            dense_mass=self._dense_mass,
            target_accept_prob=self._target_accept_prob,
            trajectory_length=self._trajectory_length,
            max_tree_depth=self._max_tree_depth,
            find_heuristic_step_size=self._find_heuristic_step_size,
            model_args=model_args,
            model_kwargs=model_kwargs,
            rng_key=rng_key,
        )
        if rng_key.ndim == 1:
            init_state = hmc_init_fn(init_params, rng_key)
        else:
            # XXX it is safe to run hmc_init_fn under vmap despite that hmc_init_fn changes some
            # nonlocal variables: momentum_generator, wa_update, trajectory_len, max_treedepth,
            # wa_steps because those variables do not depend on traced args: init_params, rng_key.
            init_state = vmap(hmc_init_fn)(init_params, rng_key)
            sample_fn = vmap(self._sample_fn, in_axes=(0, None, None))
            self._sample_fn = sample_fn
        return init_state

    @copy_docs_from(MCMCKernel.postprocess_fn)
    def postprocess_fn(self, args, kwargs):
        if self._postprocess_fn is None:
            return identity
        return self._postprocess_fn(*args, **kwargs)

    def sample(self, state, model_args, model_kwargs):
        """
        Run HMC from the given :data:`~numpyro.infer.mcmc.HMCState` and return the resulting
        :data:`~numpyro.infer.mcmc.HMCState`.

        :param HMCState state: Represents the current state.
        :param model_args: Arguments provided to the model.
        :param model_kwargs: Keyword arguments provided to the model.
        :return: Next `state` after running HMC.
        """
        return self._sample_fn(state, model_args, model_kwargs)


class NUTS(HMC):
    """
    Hamiltonian Monte Carlo inference, using the No U-Turn Sampler (NUTS)
    with adaptive path length and mass matrix adaptation.

    **References:**

    1. *MCMC Using Hamiltonian Dynamics*,
       Radford M. Neal
    2. *The No-U-turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo*,
       Matthew D. Hoffman, and Andrew Gelman.
    3. *A Conceptual Introduction to Hamiltonian Monte Carlo`*,
       Michael Betancourt

    :param model: Python callable containing Pyro :mod:`~numpyro.primitives`.
        If model is provided, `potential_fn` will be inferred using the model.
    :param potential_fn: Python callable that computes the potential energy
        given input parameters. The input parameters to `potential_fn` can be
        any python collection type, provided that `init_params` argument to
        `init_kernel` has the same type.
    :param kinetic_fn: Python callable that returns the kinetic energy given
        inverse mass matrix and momentum. If not provided, the default is
        euclidean kinetic energy.
    :param float step_size: Determines the size of a single step taken by the
        verlet integrator while computing the trajectory using Hamiltonian
        dynamics. If not specified, it will be set to 1.
    :param bool adapt_step_size: A flag to decide if we want to adapt step_size
        during warm-up phase using Dual Averaging scheme.
    :param bool adapt_mass_matrix: A flag to decide if we want to adapt mass
        matrix during warm-up phase using Welford scheme.
    :param bool dense_mass:  A flag to decide if mass matrix is dense or
        diagonal (default when ``dense_mass=False``)
    :param float target_accept_prob: Target acceptance probability for step size
        adaptation using Dual Averaging. Increasing this value will lead to a smaller
        step size, hence the sampling will be slower but more robust. Default to 0.8.
    :param float trajectory_length: Length of a MCMC trajectory for HMC. This arg has
        no effect in NUTS sampler.
    :param int max_tree_depth: Max depth of the binary tree created during the doubling
        scheme of NUTS sampler. Defaults to 10.
    :param callable init_strategy: a per-site initialization function.
        See :ref:`init_strategy` section for available functions.
    :param bool find_heuristic_step_size: whether to a heuristic function to adjust the
        step size at the beginning of each adaptation window. Defaults to False.
    """
    def __init__(self,
                 model=None,
                 potential_fn=None,
                 kinetic_fn=None,
                 step_size=1.0,
                 adapt_step_size=True,
                 adapt_mass_matrix=True,
                 dense_mass=False,
                 target_accept_prob=0.8,
                 trajectory_length=None,
                 max_tree_depth=10,
                 init_strategy=init_to_uniform,
                 find_heuristic_step_size=False):
        super(NUTS, self).__init__(potential_fn=potential_fn, model=model, kinetic_fn=kinetic_fn,
                                   step_size=step_size, adapt_step_size=adapt_step_size,
                                   adapt_mass_matrix=adapt_mass_matrix, dense_mass=dense_mass,
                                   target_accept_prob=target_accept_prob,
                                   trajectory_length=trajectory_length,
                                   init_strategy=init_strategy,
                                   find_heuristic_step_size=find_heuristic_step_size)
        self._max_tree_depth = max_tree_depth
        self._algo = 'NUTS'


def _get_proposal_loc_and_scale(samples, loc, scale, new_sample):
    # get loc/scale of q_{-n} (Algorithm 1, line 5 of ref [1]) for n from 1 -> N
    # these loc/scale will be stacked to the first dim; so
    #   proposal_loc.shape[0] = proposal_loc.shape[0] = N
    # Here, we use the numerical stability procedure in Appendix 6 of [1].
    weight = 1 / samples.shape[0]
    if scale.ndim > loc.ndim:
        new_scale = cholesky_update(scale, new_sample - loc, weight)
        proposal_scale = cholesky_update(new_scale, samples - loc, -weight)
        proposal_scale = cholesky_update(proposal_scale, new_sample - samples, - (weight ** 2))
    else:
        var = np.square(scale) + weight * np.square(new_sample - loc)
        proposal_var = var - weight * np.square(samples - loc)
        proposal_var = proposal_var - weight ** 2 * np.square(new_sample - samples)
        proposal_scale = np.sqrt(proposal_var)

    proposal_loc = loc + weight * (new_sample - samples)
    return proposal_loc, proposal_scale


def _sample_proposal(inv_mass_matrix_sqrt, rng_key, batch_shape=()):
    eps = random.normal(rng_key, batch_shape + np.shape(inv_mass_matrix_sqrt)[:1])
    if inv_mass_matrix_sqrt.ndim == 1:
        r = np.multiply(inv_mass_matrix_sqrt, eps)
    elif inv_mass_matrix_sqrt.ndim == 2:
        r = np.matmul(inv_mass_matrix_sqrt, eps[..., None])[..., 0]
    else:
        raise ValueError("Mass matrix has incorrect number of dims.")
    return r


# XXX: probably we need to recompute `loc`, `inv_mass_matrix_sqrt` from `zs`
# because we might lose precision after many iterations of using _get_proposal_loc_and_scale;
# If we recompute, we don't need to store `loc` and `inv_mass_matrix_sqrt` here.
# We may also update those values every 10D iterations...
SAAdaptState = namedtuple('SAAdaptState', ['zs', 'pes', 'loc', 'inv_mass_matrix_sqrt'])
SAState = namedtuple('SAState', ['i', 'z', 'potential_energy', 'accept_prob',
                                 'mean_accept_prob', 'diverging', 'adapt_state', 'rng_key'])
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
    mask = np.arange(x.shape[0] - 1) < idx
    return np.where(mask.reshape((-1,) + (1,) * (x.ndim - 1)), x[:-1], x[1:])


# TODO: consider to expose this functional style
def _sa(potential_fn=None, potential_fn_gen=None):
    wa_steps = None
    max_delta_energy = 1000.

    def init_kernel(init_params,
                    num_warmup,
                    adapt_state_size=None,
                    inverse_mass_matrix=None,
                    dense_mass=False,
                    model_args=(),
                    model_kwargs=None,
                    rng_key=PRNGKey(0)):
        nonlocal wa_steps
        wa_steps = num_warmup
        pe_fn = potential_fn
        if potential_fn_gen:
            if pe_fn is not None:
                raise ValueError('Only one of `potential_fn` or `potential_fn_gen` must be provided.')
            else:
                kwargs = {} if model_kwargs is None else model_kwargs
                pe_fn = potential_fn_gen(*model_args, **kwargs)
        rng_key_sa, rng_key_zs, rng_key_z = random.split(rng_key, 3)
        z = init_params
        z_flat, unravel_fn = ravel_pytree(z)
        if inverse_mass_matrix is None:
            inverse_mass_matrix = np.identity(z_flat.shape[-1]) if dense_mass else np.ones(z_flat.shape[-1])
        inv_mass_matrix_sqrt = np.linalg.cholesky(inverse_mass_matrix) if dense_mass \
            else np.sqrt(inverse_mass_matrix)
        if adapt_state_size is None:
            # XXX: heuristic choice
            adapt_state_size = 2 * z_flat.shape[-1]
        else:
            assert adapt_state_size > 1, 'adapt_state_size should be greater than 1.'
        # NB: mean is init_params
        zs = z_flat + _sample_proposal(inv_mass_matrix_sqrt, rng_key_zs, (adapt_state_size,))
        # compute potential energies
        pes = lax.map(lambda z: pe_fn(unravel_fn(z)), zs)
        if dense_mass:
            cov = np.cov(zs, rowvar=False, bias=True)
            if cov.shape == ():  # JAX returns scalar for 1D input
                cov = cov.reshape((1, 1))
            inv_mass_matrix_sqrt = np.linalg.cholesky(cov)
        else:
            inv_mass_matrix_sqrt = np.std(zs, 0)
        adapt_state = SAAdaptState(zs, pes, np.mean(zs, 0), inv_mass_matrix_sqrt)
        k = categorical_logits(rng_key_z, np.zeros(zs.shape[0]))
        z = unravel_fn(zs[k])
        pe = pes[k]
        sa_state = SAState(0, z, pe, 0., 0., False, adapt_state, rng_key_sa)
        return device_put(sa_state)

    def sample_kernel(sa_state, model_args=(), model_kwargs=None):
        pe_fn = potential_fn
        if potential_fn_gen:
            pe_fn = potential_fn_gen(*model_args, **model_kwargs)
        zs, pes, loc, scale = sa_state.adapt_state
        rng_key, rng_key_z, rng_key_reject, rng_key_accept = random.split(sa_state.rng_key, 4)
        _, unravel_fn = ravel_pytree(sa_state.z)

        z = loc + _sample_proposal(scale, rng_key_z)
        pe = pe_fn(unravel_fn(z))
        pe = np.where(np.isnan(pe), np.inf, pe)
        diverging = (pe - sa_state.potential_energy) > max_delta_energy

        # NB: all terms having the pattern *s will have shape N x ...
        # and all terms having the pattern *s_ will have shape (N + 1) x ...
        locs, scales = _get_proposal_loc_and_scale(zs, loc, scale, z)
        zs_ = np.concatenate([zs, z[None, :]])
        pes_ = np.concatenate([pes, pe[None]])
        locs_ = np.concatenate([locs, loc[None, :]])
        scales_ = np.concatenate([scales, scale[None, ...]])
        if scale.ndim == 2:  # dense_mass
            log_weights_ = dist.MultivariateNormal(locs_, scale_tril=scales_).log_prob(zs_) + pes_
        else:
            log_weights_ = dist.Normal(locs_, scales_).log_prob(zs_).sum(-1) + pes_
        log_weights_ = np.where(np.isnan(log_weights_), -np.inf, log_weights_)
        # get rejecting index
        j = categorical_logits(rng_key_reject, log_weights_)
        zs = _numpy_delete(zs_, j)
        pes = _numpy_delete(pes_, j)
        loc = locs_[j]
        scale = scales_[j]
        adapt_state = SAAdaptState(zs, pes, loc, scale)

        # NB: weights[-1] / sum(weights) is the probability of rejecting the new sample `z`.
        accept_prob = 1 - np.exp(log_weights_[-1] - logsumexp(log_weights_))
        itr = sa_state.i + 1
        n = np.where(sa_state.i < wa_steps, itr, itr - wa_steps)
        mean_accept_prob = sa_state.mean_accept_prob + (accept_prob - sa_state.mean_accept_prob) / n

        # XXX: we make a modification of SA sampler in [1]
        # in [1], each MCMC state contains N points `zs`
        # here we do resampling to pick randomly a point from those N points
        k = categorical_logits(rng_key_accept, np.zeros(zs.shape[0]))
        z = unravel_fn(zs[k])
        pe = pes[k]
        return SAState(itr, z, pe, accept_prob, mean_accept_prob, diverging, adapt_state, rng_key)

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

    .. note:: We recommend to use this kernel with `progress_bar=False` in :class:`MCMC`
        to reduce JAX's dispatch overhead.

    **References:**

    1. *Sample Adaptive MCMC* (https://papers.nips.cc/paper/9107-sample-adaptive-mcmc),
       Michael Zhu

    :param model: Python callable containing Pyro :mod:`~numpyro.primitives`.
        If model is provided, `potential_fn` will be inferred using the model.
    :param potential_fn: Python callable that computes the potential energy
        given input parameters. The input parameters to `potential_fn` can be
        any python collection type, provided that `init_params` argument to
        `init_kernel` has the same type.
    :param int adapt_state_size: The number of points to generate proposal
        distribution. Defaults to 2 times latent size.
    :param bool dense_mass:  A flag to decide if mass matrix is dense or
        diagonal (default to ``dense_mass=True``)
    :param callable init_strategy: a per-site initialization function.
        See :ref:`init_strategy` section for available functions.
    """
    def __init__(self, model=None, potential_fn=None, adapt_state_size=None,
                 dense_mass=True, init_strategy=init_to_uniform):
        if not (model is None) ^ (potential_fn is None):
            raise ValueError('Only one of `model` or `potential_fn` must be specified.')
        self._model = model
        self._potential_fn = potential_fn
        self._adapt_state_size = adapt_state_size
        self._dense_mass = dense_mass
        self._init_strategy = init_strategy
        self._init_fn = None
        self._postprocess_fn = None
        self._sample_fn = None

    def _init_state(self, rng_key, model_args, model_kwargs, init_params):
        if self._model is not None:
            init_params, potential_fn, postprocess_fn, _ = initialize_model(
                rng_key,
                self._model,
                dynamic_args=True,
                model_args=model_args,
                model_kwargs=model_kwargs)
            init_params = init_params[0]
            # NB: init args is different from HMC
            self._init_fn, sample_fn = _sa(potential_fn_gen=potential_fn)
            if self._postprocess_fn is None:
                self._postprocess_fn = postprocess_fn
        else:
            self._init_fn, sample_fn = _sa(potential_fn=self._potential_fn)

        if self._sample_fn is None:
            self._sample_fn = sample_fn
        return init_params

    @copy_docs_from(MCMCKernel.init)
    def init(self, rng_key, num_warmup, init_params=None, model_args=(), model_kwargs={}):
        # non-vectorized
        if rng_key.ndim == 1:
            rng_key, rng_key_init_model = random.split(rng_key)
        # vectorized
        else:
            rng_key, rng_key_init_model = np.swapaxes(vmap(random.split)(rng_key), 0, 1)
            # we need only a single key for initializing PE / constraints fn
            rng_key_init_model = rng_key_init_model[0]
        init_params = self._init_state(rng_key_init_model, model_args, model_kwargs, init_params)
        if self._potential_fn and init_params is None:
            raise ValueError('Valid value of `init_params` must be provided with'
                             ' `potential_fn`.')

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
        if rng_key.ndim == 1:
            init_state = sa_init_fn(init_params, rng_key)
        else:
            init_state = vmap(sa_init_fn)(init_params, rng_key)
            sample_fn = vmap(self._sample_fn, in_axes=(0, None, None))
            self._sample_fn = sample_fn
        return init_state

    @copy_docs_from(MCMCKernel.postprocess_fn)
    def postprocess_fn(self, args, kwargs):
        if self._postprocess_fn is None:
            return identity
        return self._postprocess_fn(*args, **kwargs)

    def sample(self, state, model_args, model_kwargs):
        """
        Run SA from the given :data:`~numpyro.infer.mcmc.SAState` and return the resulting
        :data:`~numpyro.infer.mcmc.SAState`.

        :param SAState state: Represents the current state.
        :param model_args: Arguments provided to the model.
        :param model_kwargs: Keyword arguments provided to the model.
        :return: Next `state` after running SA.
        """
        return self._sample_fn(state, model_args, model_kwargs)


def _get_value_from_index(xs, i):
    return tree_map(lambda x: x[i], xs)


def _laxmap(f, xs):
    n = tree_flatten(xs)[0][0].shape[0]

    ys = []
    for i in range(n):
        x = jit(_get_value_from_index)(xs, i)
        ys.append(f(x))

    return tree_multimap(lambda *args: np.stack(args), *ys)


def _sample_fn_jit_args(state, sampler):
    hmc_state, args, kwargs = state
    return sampler.sample(hmc_state, args, kwargs), args, kwargs


def _sample_fn_nojit_args(state, sampler, args, kwargs):
    # state is a tuple of size 1 - containing HMCState
    return sampler.sample(state[0], args, kwargs),


def _collect_fn(collect_fields):
    @cached_by(_collect_fn, collect_fields)
    def collect(x):
        return attrgetter(*collect_fields)(x[0])

    return collect


# XXX: Is there a better hash key that we can use?
def _hashable(x):
    # When the arguments are JITed, ShapedArray is hashable.
    if isinstance(x, Tracer):
        return x
    elif isinstance(x, DeviceArray):
        return x.copy().tobytes()
    elif isinstance(x, np.ndarray):
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
                 postprocess_fn=None,
                 chain_method='parallel',
                 progress_bar=True,
                 jit_model_args=False):
        self.sampler = sampler
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.postprocess_fn = postprocess_fn
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
        self._set_collection_params()

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

    def _single_chain_mcmc(self, rng_key, init_state, init_params, args, kwargs, collect_fields=('z',)):
        if init_state is None:
            init_state = self.sampler.init(rng_key, self.num_warmup, init_params,
                                           model_args=args, model_kwargs=kwargs)
        if self.postprocess_fn is None:
            postprocess_fn = self.sampler.postprocess_fn(args, kwargs)
        else:
            postprocess_fn = self.postprocess_fn
        diagnostics = lambda x: get_diagnostics_str(x[0]) if rng_key.ndim == 1 else None   # noqa: E731
        init_val = (init_state, args, kwargs) if self._jit_model_args else (init_state,)
        lower_idx = self._collection_params["lower"]
        upper_idx = self._collection_params["upper"]

        collect_vals = fori_collect(lower_idx,
                                    upper_idx,
                                    self._get_cached_fn(),
                                    init_val,
                                    transform=_collect_fn(collect_fields),
                                    progbar=self.progress_bar,
                                    return_last_val=True,
                                    collection_size=self._collection_params["collection_size"],
                                    progbar_desc=functools.partial(get_progbar_desc_str,
                                                                   lower_idx),
                                    diagnostics_fn=diagnostics)
        states, last_val = collect_vals
        # Get first argument of type `HMCState`
        last_state = last_val[0]
        if len(collect_fields) == 1:
            states = (states,)
        states = dict(zip(collect_fields, states))
        # Apply constraints if number of samples is non-zero
        site_values = tree_flatten(states['z'])[0]
        if len(site_values) > 0 and site_values[0].size > 0:
            states['z'] = lax.map(postprocess_fn, states['z'])
        return states, last_state

    def _single_chain_jit_args(self, init, collect_fields=('z',)):
        return self._single_chain_mcmc(*init, collect_fields=collect_fields)

    def _single_chain_nojit_args(self, init, model_args, model_kwargs, collect_fields=('z',)):
        return self._single_chain_mcmc(*init, model_args, model_kwargs, collect_fields=collect_fields)

    def _set_collection_params(self, lower=None, upper=None, collection_size=None):
        self._collection_params["lower"] = self.num_warmup if lower is None else lower
        self._collection_params["upper"] = self.num_warmup + self.num_samples if upper is None else upper
        self._collection_params["collection_size"] = collection_size

    def _compile(self, rng_key, *args, extra_fields=(), init_params=None, **kwargs):
        self._set_collection_params(0, 0, self.num_samples)
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
        :param extra_fields: Extra fields (aside from `z`, `diverging`) from :data:`numpyro.infer.mcmc.HMCState`
            to collect during the MCMC run.
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
            self._set_collection_params(0, self.num_warmup, self.num_warmup)
        else:
            self._set_collection_params(self.num_warmup, self.num_warmup, self.num_samples)
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
            self._set_collection_params(0, self.num_samples, self.num_samples)
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
            if np.shape(prototype_init_val)[0] != self.num_chains:
                raise ValueError('`init_params` must have the same leading dimension'
                                 ' as `num_chains`.')
        assert isinstance(extra_fields, (tuple, list))
        collect_fields = tuple(set(('z', 'diverging') + tuple(extra_fields)))
        if self.num_chains == 1:
            states_flat, last_state = self._single_chain_mcmc(rng_key, init_state, init_params,
                                                              args, kwargs, collect_fields)
            states = tree_map(lambda x: x[np.newaxis, ...], states_flat)
        else:
            if self._jit_model_args:
                partial_map_fn = partial(self._single_chain_jit_args,
                                         collect_fields=collect_fields)
            else:
                partial_map_fn = partial(self._single_chain_nojit_args,
                                         model_args=args,
                                         model_kwargs=kwargs,
                                         collect_fields=collect_fields)
            if chain_method == 'sequential':
                if self.progress_bar:
                    map_fn = partial(_laxmap, partial_map_fn)
                else:
                    map_fn = partial(lax.map, partial_map_fn)
            elif chain_method == 'parallel':
                map_fn = pmap(partial_map_fn)
            elif chain_method == 'vectorized':
                map_fn = partial_map_fn
            else:
                raise ValueError('Only supporting the following methods to draw chains:'
                                 ' "sequential", "parallel", or "vectorized"')
            if self._jit_model_args:
                states, last_state = map_fn((rng_key, init_state, init_params, args, kwargs))
            else:
                states, last_state = map_fn((rng_key, init_state, init_params))
            if chain_method == 'vectorized':
                # swap num_samples x num_chains to num_chains x num_samples
                states = tree_map(lambda x: np.swapaxes(x, 0, 1), states)
            states_flat = tree_map(lambda x: np.reshape(x, (-1,) + x.shape[2:]), states)
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
        return self._states['z'] if group_by_chain else self._states_flat['z']

    def get_extra_fields(self, group_by_chain=False):
        """
        Get extra fields from the MCMC run.

        :param bool group_by_chain: Whether to preserve the chain dimension. If True,
            all samples will have num_chains as the size of their leading dimension.
        :return: Extra fields keyed by field names which are specified in the
            `extra_fields` keyword of :meth:`run`.
        """
        states = self._states if group_by_chain else self._states_flat
        return {k: v for k, v in states.items() if k != 'z'}

    def print_summary(self, prob=0.9, exclude_deterministic=True):
        # Exclude deterministic sites by default
        sites = self._states['z']
        if isinstance(sites, dict) and exclude_deterministic:
            sites = {k: v for k, v in self._states['z'].items() if k in self._last_state.z}
        print_summary(sites, prob=prob)
        extra_fields = self.get_extra_fields()
        if 'diverging' in extra_fields:
            print("Number of divergences: {}".format(np.sum(extra_fields['diverging'])))
