from abc import ABC, abstractmethod
from collections import namedtuple
import functools
import math
from operator import attrgetter
import os
import warnings

from jax import jit, lax, partial, pmap, random, vmap, device_get
from jax.core import Tracer
from jax.dtypes import canonicalize_dtype
from jax.flatten_util import ravel_pytree
from jax.interpreters.xla import DeviceArray
from jax.lib import xla_bridge
import jax.numpy as np
from jax.random import PRNGKey
from jax.tree_util import tree_flatten, tree_map, tree_multimap

from numpyro.diagnostics import print_summary
from numpyro.infer.hmc_util import (
    IntegratorState,
    build_tree,
    euclidean_kinetic_energy,
    find_reasonable_step_size,
    velocity_verlet,
    warmup_adapter
)
from numpyro.infer.util import init_to_uniform, get_potential_fn, find_valid_initial_params
from numpyro.util import cond, copy_docs_from, fori_collect, fori_loop, identity, not_jax_tracer

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
 - **adapt_state** - A ``AdaptState`` namedtuple which contains adaptation information
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


def _sample_momentum(unpack_fn, mass_matrix_sqrt, rng_key):
    eps = random.normal(rng_key, np.shape(mass_matrix_sqrt)[:1])
    if mass_matrix_sqrt.ndim == 1:
        r = np.multiply(mass_matrix_sqrt, eps)
        return unpack_fn(r)
    elif mass_matrix_sqrt.ndim == 2:
        r = np.dot(mass_matrix_sqrt, eps)
        return unpack_fn(r)
    else:
        raise ValueError("Mass matrix has incorrect number of dims.")


def get_diagnostics_str(hmc_state):
    return '{} steps of size {:.2e}. acc. prob={:.2f}'.format(hmc_state.num_steps,
                                                              hmc_state.adapt_state.step_size,
                                                              hmc_state.mean_accept_prob)


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

    .. testsetup::

        import jax
        from jax import random
        import jax.numpy as np
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer.mcmc import hmc
        from numpyro.infer.util import initialize_model
        from numpyro.util import fori_collect

    .. doctest::

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
        >>> init_params, potential_fn, constrain_fn = initialize_model(random.PRNGKey(0),
        ...                                                            model, data, labels)
        >>> init_kernel, sample_kernel = hmc(potential_fn, algo='NUTS')
        >>> hmc_state = init_kernel(init_params,
        ...                         trajectory_length=10,
        ...                         num_warmup=300)
        >>> samples = fori_collect(0, 500, sample_kernel, hmc_state,
        ...                        transform=lambda state: constrain_fn(state.z))
        >>> print(np.mean(samples['beta'], axis=0))  # doctest: +SKIP
        [0.9153987 2.0754058 2.9621222]
    """
    if kinetic_fn is None:
        kinetic_fn = euclidean_kinetic_energy
    vv_update = None
    trajectory_len = None
    max_treedepth = None
    momentum_generator = None
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
        :param tuple model_args: Model arguments if `potential_fn_gen` is specified.
        :param dict model_kwargs: Model keyword arguments if `potential_fn_gen` is specified.
        :param jax.random.PRNGKey rng_key: random key to be used as the source of
            randomness.

        """
        step_size = lax.convert_element_type(step_size, canonicalize_dtype(np.float64))
        nonlocal momentum_generator, wa_update, trajectory_len, max_treedepth, vv_update, wa_steps
        wa_steps = num_warmup
        trajectory_len = trajectory_length
        max_treedepth = max_tree_depth
        z = init_params
        z_flat, unravel_fn = ravel_pytree(z)
        momentum_generator = partial(_sample_momentum, unravel_fn)
        pe_fn = potential_fn
        if potential_fn_gen:
            if pe_fn is not None:
                raise ValueError('Only one of `potential_fn` or `potential_fn_gen` must be provided.')
            else:
                kwargs = {} if model_kwargs is None else model_kwargs
                pe_fn = potential_fn_gen(*model_args, **kwargs)

        find_reasonable_ss = partial(find_reasonable_step_size,
                                     pe_fn, kinetic_fn, momentum_generator)

        wa_init, wa_update = warmup_adapter(num_warmup,
                                            adapt_step_size=adapt_step_size,
                                            adapt_mass_matrix=adapt_mass_matrix,
                                            dense_mass=dense_mass,
                                            target_accept_prob=target_accept_prob,
                                            find_reasonable_step_size=find_reasonable_ss)

        rng_key_hmc, rng_key_wa = random.split(rng_key)
        wa_state = wa_init(z, rng_key_wa, step_size,
                           inverse_mass_matrix=inverse_mass_matrix,
                           mass_matrix_size=np.size(z_flat))
        r = momentum_generator(wa_state.mass_matrix_sqrt, rng_key)
        vv_init, vv_update = velocity_verlet(pe_fn, kinetic_fn)
        vv_state = vv_init(z, r)
        energy = kinetic_fn(wa_state.inverse_mass_matrix, vv_state.r)
        hmc_state = HMCState(0, vv_state.z, vv_state.z_grad, vv_state.potential_energy, energy,
                             0, 0., 0., False, wa_state, rng_key_hmc)
        return hmc_state

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
                                (vv_state_new, energy_new), lambda args: args,
                                (vv_state, energy_old), lambda args: args)
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
        r = momentum_generator(hmc_state.adapt_state.mass_matrix_sqrt, rng_key_momentum)
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
                           lambda x: x)

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
    def constrain_fn(self, model_args, model_kwargs):
        """
        Function that transforms unconstrained values at sample sites to values
        constrained to the site's support.

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
                 init_strategy=init_to_uniform()):
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
        # Set on first call to init
        self._init_fn = None
        self._constrain_fn = None
        self._sample_fn = None

    def _init_state(self, rng_key, model_args, model_kwargs):
        if self._model is not None:
            potential_fn_gen, self._constrain_fn = get_potential_fn(
                rng_key,
                self._model,
                dynamic_args=True,
                model_args=model_args,
                model_kwargs=model_kwargs)
            self._init_fn, self._sample_fn = hmc(potential_fn_gen=potential_fn_gen,
                                                 kinetic_fn=self._kinetic_fn,
                                                 algo=self._algo)
        else:
            self._init_fn, self._sample_fn = hmc(potential_fn=self._potential_fn,
                                                 kinetic_fn=self._kinetic_fn,
                                                 algo=self._algo)

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
            # we need only a single key for initializing PE / constraints fn
            rng_key_init_model = rng_key_init_model[0]
        if not self._init_fn:
            self._init_state(rng_key_init_model, model_args, model_kwargs)
        if self._potential_fn and init_params is None:
            raise ValueError('Valid value of `init_params` must be provided with'
                             ' `potential_fn`.')
        # Find valid initial params
        if self._model and not init_params:
            init_params, is_valid = find_valid_initial_params(rng_key, self._model, *model_args,
                                                              init_strategy=self._init_strategy,
                                                              param_as_improper=True,
                                                              **model_kwargs)
            if not_jax_tracer(is_valid):
                if device_get(~np.all(is_valid)):
                    raise RuntimeError("Cannot find valid initial parameters. "
                                       "Please check your model again.")

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
            rng_key=rng_key,
            model_args=model_args,
            model_kwargs=model_kwargs,
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

    @copy_docs_from(MCMCKernel.constrain_fn)
    def constrain_fn(self, args, kwargs):
        if self._constrain_fn is None:
            return identity
        return self._constrain_fn(*args, **kwargs)

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
                 init_strategy=init_to_uniform()):
        super(NUTS, self).__init__(potential_fn=potential_fn, model=model, kinetic_fn=kinetic_fn,
                                   step_size=step_size, adapt_step_size=adapt_step_size,
                                   adapt_mass_matrix=adapt_mass_matrix, dense_mass=dense_mass,
                                   target_accept_prob=target_accept_prob,
                                   trajectory_length=trajectory_length, init_strategy=init_strategy)
        self._max_tree_depth = max_tree_depth
        self._algo = 'NUTS'


def _laxmap(f, xs):
    n = tree_flatten(xs)[0][0].shape[0]

    def get_value_from_index(i):
        return tree_map(lambda x: x[i], xs)

    ys = []
    for i in range(n):
        x = jit(get_value_from_index)(i)
        ys.append(f(x))

    return tree_multimap(lambda *args: np.stack(args), *ys)


def _sample_fn_jit_args(state, sampler):
    hmc_state, args, kwargs = state
    return sampler.sample(hmc_state, args, kwargs), args, kwargs


def _sample_fn_nojit_args(state, sampler, args, kwargs):
    # state is a tuple of size 1 - containing HMCState
    return sampler.sample(state[0], args, kwargs),


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
    :param constrain_fn: Callable that converts a collection of unconstrained
        sample values returned from the sampler to constrained values that
        lie within the support of the sample sites.
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
                 constrain_fn=None,
                 chain_method='parallel',
                 progress_bar=True,
                 jit_model_args=False):
        self.sampler = sampler
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.constrain_fn = constrain_fn
        self.chain_method = chain_method
        self.progress_bar = progress_bar
        # TODO: We should have progress bars (maybe without diagnostics) for num_chains > 1
        if (chain_method == 'parallel' and num_chains > 1) or (
                "CI" in os.environ or "PYTEST_XDIST_WORKER" in os.environ):
            self.progress_bar = False
        self._jit_model_args = jit_model_args
        self._states = None
        self._states_flat = None
        self._cache = {}

    def _get_cached_fn(self):
        if self._jit_model_args:
            args, kwargs = (None,), (None,)
        else:
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

    def _single_chain_mcmc(self, rng_key, init_params, args, kwargs, collect_fields=('z',), collect_warmup=False):
        init_state = self.sampler.init(rng_key, self.num_warmup, init_params,
                                       model_args=args, model_kwargs=kwargs)
        if self.constrain_fn is None:
            self.constrain_fn = self.sampler.constrain_fn(args, kwargs)
        collect_fn = attrgetter(*collect_fields)
        lower = 0 if collect_warmup else self.num_warmup
        diagnostics = lambda x: get_diagnostics_str(x[0]) if rng_key.ndim == 1 else None   # noqa: E731
        init_val = (init_state, args, kwargs) if self._jit_model_args else (init_state,)
        states = fori_collect(lower, self.num_warmup + self.num_samples,
                              self._get_cached_fn(),
                              init_val,
                              transform=lambda x: collect_fn(x[0]),  # noqa: E731
                              progbar=self.progress_bar,
                              progbar_desc=functools.partial(get_progbar_desc_str, self.num_warmup),
                              diagnostics_fn=diagnostics)
        if len(collect_fields) == 1:
            states = (states,)
        states = dict(zip(collect_fields, states))
        # Apply constraints if number of samples is non-zero
        if len(tree_flatten(states['z'])[0]) > 0:
            states['z'] = vmap(self.constrain_fn)(states['z'])
        return states

    def _single_chain_jit_args(self, init, collect_fields=('z',), collect_warmup=False):
        return self._single_chain_mcmc(*init, collect_fields=collect_fields, collect_warmup=collect_warmup)

    def _single_chain_nojit_args(self, init, model_args, model_kwargs, collect_fields=('z',), collect_warmup=False):
        return self._single_chain_mcmc(*init, model_args, model_kwargs,
                                       collect_fields=collect_fields,
                                       collect_warmup=collect_warmup)

    def run(self, rng_key, *args, extra_fields=(), collect_warmup=False, init_params=None, **kwargs):
        """
        Run the MCMC samplers and collect samples.

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
        self._args = args
        self._kwargs = kwargs
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
            states_flat = self._single_chain_mcmc(rng_key, init_params, args, kwargs, collect_fields, collect_warmup)
            states = tree_map(lambda x: x[np.newaxis, ...], states_flat)
        else:
            rng_keys = random.split(rng_key, self.num_chains)
            if self._jit_model_args:
                partial_map_fn = partial(self._single_chain_jit_args,
                                         collect_fields=collect_fields,
                                         collect_warmup=collect_warmup)
            else:
                partial_map_fn = partial(self._single_chain_nojit_args,
                                         model_args=args,
                                         model_kwargs=kwargs,
                                         collect_fields=collect_fields,
                                         collect_warmup=collect_warmup)
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
                states = map_fn((rng_keys, init_params, args, kwargs))
            else:
                states = map_fn((rng_keys, init_params))
            if chain_method == 'vectorized':
                # swap num_samples x num_chains to num_chains x num_samples
                states = tree_map(lambda x: np.swapaxes(x, 0, 1), states)
            states_flat = tree_map(lambda x: np.reshape(x, (-1,) + x.shape[2:]), states)
        self._states = states
        self._states_flat = states_flat

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

    def print_summary(self, prob=0.9):
        print_summary(self._states['z'], prob=prob)
        extra_fields = self.get_extra_fields()
        if 'diverging' in extra_fields:
            print("Number of divergences: {}".format(np.sum(extra_fields['diverging'])))
