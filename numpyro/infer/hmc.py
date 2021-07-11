# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict, namedtuple
import math
import os

from jax import device_put, lax, partial, random, vmap
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp

from numpyro.infer.hmc_util import (
    IntegratorState,
    build_tree,
    euclidean_kinetic_energy,
    find_reasonable_step_size,
    velocity_verlet,
    warmup_adapter,
)
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import ParamInfo, init_to_uniform, initialize_model
from numpyro.util import cond, fori_loop, identity

HMCState = namedtuple(
    "HMCState",
    [
        "i",
        "z",
        "z_grad",
        "potential_energy",
        "energy",
        "r",
        "trajectory_length",
        "num_steps",
        "accept_prob",
        "mean_accept_prob",
        "diverging",
        "adapt_state",
        "rng_key",
    ],
)
"""
A :func:`~collections.namedtuple` consisting of the following fields:

 - **i** - iteration. This is reset to 0 after warmup.
 - **z** - Python collection representing values (unconstrained samples from
   the posterior) at latent sites.
 - **z_grad** - Gradient of potential energy w.r.t. latent sample sites.
 - **potential_energy** - Potential energy computed at the given value of ``z``.
 - **energy** - Sum of potential energy and kinetic energy of the current state.
 - **r** - The current momentum variable. If this is None, a new momentum variable
   will be drawn at the beginning of each sampling step.
 - **trajectory_length** - The amount of time to run HMC dynamics in each sampling step.
   This field is not used in NUTS.
 - **num_steps** - Number of steps in the Hamiltonian trajectory (for diagnostics).
   In NUTS sampler, the tree depth of a trajectory can be computed from this field
   with `tree_depth = np.log2(num_steps).astype(int) + 1`.
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
    num_steps = jnp.ceil(trajectory_length / step_size)
    # NB: casting to jnp.int64 does not take effect (returns jnp.int32 instead)
    # if jax_enable_x64 is False
    return num_steps.astype(jnp.result_type(int))


def momentum_generator(prototype_r, mass_matrix_sqrt, rng_key):
    if isinstance(mass_matrix_sqrt, dict):
        rng_keys = random.split(rng_key, len(mass_matrix_sqrt))
        r = {}
        for (site_names, mm_sqrt), rng_key in zip(mass_matrix_sqrt.items(), rng_keys):
            r_block = OrderedDict([(k, prototype_r[k]) for k in site_names])
            r.update(momentum_generator(r_block, mm_sqrt, rng_key))
        return r

    _, unpack_fn = ravel_pytree(prototype_r)
    eps = random.normal(rng_key, jnp.shape(mass_matrix_sqrt)[:1])
    if mass_matrix_sqrt.ndim == 1:
        r = jnp.multiply(mass_matrix_sqrt, eps)
        return unpack_fn(r)
    elif mass_matrix_sqrt.ndim == 2:
        r = jnp.dot(mass_matrix_sqrt, eps)
        return unpack_fn(r)
    else:
        raise ValueError("Mass matrix has incorrect number of dims.")


def hmc(potential_fn=None, potential_fn_gen=None, kinetic_fn=None, algo="NUTS"):
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
        to use the higher level :class:`~numpyro.infer.mcmc.MCMC` API instead.

    **Example**

    .. doctest::

        >>> import jax
        >>> from jax import random
        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.infer.hmc import hmc
        >>> from numpyro.infer.util import initialize_model
        >>> from numpyro.util import fori_collect

        >>> true_coefs = jnp.array([1., 2., 3.])
        >>> data = random.normal(random.PRNGKey(2), (2000, 3))
        >>> labels = dist.Bernoulli(logits=(true_coefs * data).sum(-1)).sample(random.PRNGKey(3))
        >>>
        >>> def model(data, labels):
        ...     coefs = numpyro.sample('coefs', dist.Normal(jnp.zeros(3), jnp.ones(3)))
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
        >>> print(jnp.mean(samples['coefs'], axis=0))  # doctest: +SKIP
        [0.9153987 2.0754058 2.9621222]
    """
    if kinetic_fn is None:
        kinetic_fn = euclidean_kinetic_energy
    vv_update = None
    max_treedepth = None
    wa_update = None
    wa_steps = None
    forward_mode_ad = False
    max_delta_energy = 1000.0
    if algo not in {"HMC", "NUTS"}:
        raise ValueError("`algo` must be one of `HMC` or `NUTS`.")

    def init_kernel(
        init_params,
        num_warmup,
        *,
        step_size=1.0,
        inverse_mass_matrix=None,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        dense_mass=False,
        target_accept_prob=0.8,
        trajectory_length=2 * math.pi,
        max_tree_depth=10,
        find_heuristic_step_size=False,
        forward_mode_differentiation=False,
        regularize_mass_matrix=True,
        model_args=(),
        model_kwargs=None,
        rng_key=random.PRNGKey(0),
    ):
        """
        Initializes the HMC sampler.

        :param init_params: Initial parameters to begin sampling. The type must
            be consistent with the input type to `potential_fn`.
        :param int num_warmup: Number of warmup steps; samples generated
            during warmup are discarded.
        :param float step_size: Determines the size of a single step taken by the
            verlet integrator while computing the trajectory using Hamiltonian
            dynamics. If not specified, it will be set to 1.
        :param inverse_mass_matrix: Initial value for inverse mass matrix.
            This may be adapted during warmup if adapt_mass_matrix = True.
            If no value is specified, then it is initialized to the identity matrix.
            For a potential_fn with general JAX pytree parameters, the order of entries
            of the mass matrix is the order of the flattened version of pytree parameters
            obtained with `jax.tree_flatten`, which is a bit ambiguous (see more at
            https://jax.readthedocs.io/en/latest/pytrees.html). If `model` is not None,
            here we can specify a structured block mass matrix as a dictionary, where
            keys are tuple of site names and values are the corresponding block of the
            mass matrix.
            For more information about structured mass matrix, see `dense_mass` argument.
        :type inverse_mass_matrix: numpy.ndarray or dict
        :param bool adapt_step_size: A flag to decide if we want to adapt step_size
            during warm-up phase using Dual Averaging scheme.
        :param bool adapt_mass_matrix: A flag to decide if we want to adapt mass
            matrix during warm-up phase using Welford scheme.
        :param dense_mass:  This flag controls whether mass matrix is dense (i.e. full-rank) or
            diagonal (defaults to ``dense_mass=False``). To specify a structured mass matrix,
            users can provide a list of tuples of site names. Each tuple represents
            a block in the joint mass matrix. For example, assuming that the model
            has latent variables "x", "y", "z" (where each variable can be multi-dimensional),
            possible specifications and corresponding mass matrix structures are as follows:

                + dense_mass=[("x", "y")]: use a dense mass matrix for the joint
                  (x, y) and a diagonal mass matrix for z
                + dense_mass=[] (equivalent to dense_mass=False): use a diagonal mass
                  matrix for the joint (x, y, z)
                + dense_mass=[("x", "y", "z")] (equivalent to full_mass=True):
                  use a dense mass matrix for the joint (x, y, z)
                + dense_mass=[("x",), ("y",), ("z")]: use dense mass matrices for
                  each of x, y, and z (i.e. block-diagonal with 3 blocks)

        :type dense_mass: bool or list
        :param float target_accept_prob: Target acceptance probability for step size
            adaptation using Dual Averaging. Increasing this value will lead to a smaller
            step size, hence the sampling will be slower but more robust. Defaults to 0.8.
        :param float trajectory_length: Length of a MCMC trajectory for HMC. Default
            value is :math:`2\\pi`.
        :param int max_tree_depth: Max depth of the binary tree created during the doubling
            scheme of NUTS sampler. Defaults to 10. This argument also accepts a tuple of
            integers `(d1, d2)`, where `d1` is the max tree depth during warmup phase and
            `d2` is the max tree depth during post warmup phase.
        :param bool find_heuristic_step_size: whether to a heuristic function to adjust the
            step size at the beginning of each adaptation window. Defaults to False.
        :param bool regularize_mass_matrix: whether or not to regularize the estimated mass
            matrix for numerical stability during warmup phase. Defaults to True. This flag
            does not take effect if ``adapt_mass_matrix == False``.
        :param tuple model_args: Model arguments if `potential_fn_gen` is specified.
        :param dict model_kwargs: Model keyword arguments if `potential_fn_gen` is specified.
        :param jax.random.PRNGKey rng_key: random key to be used as the source of
            randomness.

        """
        step_size = lax.convert_element_type(step_size, jnp.result_type(float))
        if trajectory_length is not None:
            trajectory_length = lax.convert_element_type(
                trajectory_length, jnp.result_type(float)
            )
        nonlocal wa_update, max_treedepth, vv_update, wa_steps, forward_mode_ad
        forward_mode_ad = forward_mode_differentiation
        wa_steps = num_warmup
        max_treedepth = (
            max_tree_depth
            if isinstance(max_tree_depth, tuple)
            else (max_tree_depth, max_tree_depth)
        )
        if isinstance(init_params, ParamInfo):
            z, pe, z_grad = init_params
        else:
            z, pe, z_grad = init_params, None, None
        pe_fn = potential_fn
        if potential_fn_gen:
            if pe_fn is not None:
                raise ValueError(
                    "Only one of `potential_fn` or `potential_fn_gen` must be provided."
                )
            else:
                kwargs = {} if model_kwargs is None else model_kwargs
                pe_fn = potential_fn_gen(*model_args, **kwargs)

        find_reasonable_ss = None
        if find_heuristic_step_size:
            find_reasonable_ss = partial(
                find_reasonable_step_size, pe_fn, kinetic_fn, momentum_generator
            )

        wa_init, wa_update = warmup_adapter(
            num_warmup,
            adapt_step_size=adapt_step_size,
            adapt_mass_matrix=adapt_mass_matrix,
            dense_mass=dense_mass,
            target_accept_prob=target_accept_prob,
            find_reasonable_step_size=find_reasonable_ss,
            regularize_mass_matrix=regularize_mass_matrix,
        )

        rng_key_hmc, rng_key_wa, rng_key_momentum = random.split(rng_key, 3)
        z_info = IntegratorState(z=z, potential_energy=pe, z_grad=z_grad)
        wa_state = wa_init(
            z_info, rng_key_wa, step_size, inverse_mass_matrix=inverse_mass_matrix
        )
        r = momentum_generator(z, wa_state.mass_matrix_sqrt, rng_key_momentum)
        vv_init, vv_update = velocity_verlet(pe_fn, kinetic_fn, forward_mode_ad)
        vv_state = vv_init(z, r, potential_energy=pe, z_grad=z_grad)
        energy = vv_state.potential_energy + kinetic_fn(
            wa_state.inverse_mass_matrix, vv_state.r
        )
        zero_int = jnp.array(0, dtype=jnp.result_type(int))
        hmc_state = HMCState(
            zero_int,
            vv_state.z,
            vv_state.z_grad,
            vv_state.potential_energy,
            energy,
            None,
            trajectory_length,
            zero_int,
            jnp.zeros(()),
            jnp.zeros(()),
            jnp.array(False),
            wa_state,
            rng_key_hmc,
        )
        return device_put(hmc_state)

    def _hmc_next(
        step_size,
        inverse_mass_matrix,
        vv_state,
        model_args,
        model_kwargs,
        rng_key,
        trajectory_length,
    ):
        if potential_fn_gen:
            nonlocal vv_update, forward_mode_ad
            pe_fn = potential_fn_gen(*model_args, **model_kwargs)
            _, vv_update = velocity_verlet(pe_fn, kinetic_fn, forward_mode_ad)

        # no need to spend too many steps if the state z has 0 size (i.e. z is empty)
        if len(inverse_mass_matrix) == 0:
            num_steps = 1
        else:
            num_steps = _get_num_steps(step_size, trajectory_length)
        # makes sure trajectory length is constant, rather than step_size * num_steps
        step_size = trajectory_length / num_steps
        vv_state_new = fori_loop(
            0,
            num_steps,
            lambda i, val: vv_update(step_size, inverse_mass_matrix, val),
            vv_state,
        )
        energy_old = vv_state.potential_energy + kinetic_fn(
            inverse_mass_matrix, vv_state.r
        )
        energy_new = vv_state_new.potential_energy + kinetic_fn(
            inverse_mass_matrix, vv_state_new.r
        )
        delta_energy = energy_new - energy_old
        delta_energy = jnp.where(jnp.isnan(delta_energy), jnp.inf, delta_energy)
        accept_prob = jnp.clip(jnp.exp(-delta_energy), a_max=1.0)
        diverging = delta_energy > max_delta_energy
        transition = random.bernoulli(rng_key, accept_prob)
        vv_state, energy = cond(
            transition,
            (vv_state_new, energy_new),
            identity,
            (vv_state, energy_old),
            identity,
        )
        return vv_state, energy, num_steps, accept_prob, diverging

    def _nuts_next(
        step_size,
        inverse_mass_matrix,
        vv_state,
        model_args,
        model_kwargs,
        rng_key,
        max_treedepth_current,
    ):
        if potential_fn_gen:
            nonlocal vv_update, forward_mode_ad
            pe_fn = potential_fn_gen(*model_args, **model_kwargs)
            _, vv_update = velocity_verlet(pe_fn, kinetic_fn, forward_mode_ad)

        binary_tree = build_tree(
            vv_update,
            kinetic_fn,
            vv_state,
            inverse_mass_matrix,
            step_size,
            rng_key,
            max_delta_energy=max_delta_energy,
            max_tree_depth=(max_treedepth_current, max(max_treedepth)),
        )
        accept_prob = binary_tree.sum_accept_probs / binary_tree.num_proposals
        num_steps = binary_tree.num_proposals
        vv_state = IntegratorState(
            z=binary_tree.z_proposal,
            r=vv_state.r,
            potential_energy=binary_tree.z_proposal_pe,
            z_grad=binary_tree.z_proposal_grad,
        )
        return (
            vv_state,
            binary_tree.z_proposal_energy,
            num_steps,
            accept_prob,
            binary_tree.diverging,
        )

    _next = _nuts_next if algo == "NUTS" else _hmc_next

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
        rng_key, rng_key_momentum, rng_key_transition = random.split(
            hmc_state.rng_key, 3
        )
        r = (
            momentum_generator(
                hmc_state.z, hmc_state.adapt_state.mass_matrix_sqrt, rng_key_momentum
            )
            if hmc_state.r is None
            else hmc_state.r
        )
        vv_state = IntegratorState(
            hmc_state.z, r, hmc_state.potential_energy, hmc_state.z_grad
        )
        if algo == "HMC":
            hmc_length_args = (hmc_state.trajectory_length,)
        else:
            hmc_length_args = (
                jnp.where(hmc_state.i < wa_steps, max_treedepth[0], max_treedepth[1]),
            )
        vv_state, energy, num_steps, accept_prob, diverging = _next(
            hmc_state.adapt_state.step_size,
            hmc_state.adapt_state.inverse_mass_matrix,
            vv_state,
            model_args,
            model_kwargs,
            rng_key_transition,
            *hmc_length_args,
        )
        # not update adapt_state after warmup phase
        adapt_state = cond(
            hmc_state.i < wa_steps,
            (hmc_state.i, accept_prob, vv_state, hmc_state.adapt_state),
            lambda args: wa_update(*args),
            hmc_state.adapt_state,
            identity,
        )

        itr = hmc_state.i + 1
        n = jnp.where(hmc_state.i < wa_steps, itr, itr - wa_steps)
        mean_accept_prob = (
            hmc_state.mean_accept_prob + (accept_prob - hmc_state.mean_accept_prob) / n
        )

        r = vv_state.r if hmc_state.r is not None else None
        return HMCState(
            itr,
            vv_state.z,
            vv_state.z_grad,
            vv_state.potential_energy,
            energy,
            r,
            hmc_state.trajectory_length,
            num_steps,
            accept_prob,
            mean_accept_prob,
            diverging,
            adapt_state,
            rng_key,
        )

    # Make `init_kernel` and `sample_kernel` visible from the global scope once
    # `hmc` is called for sphinx doc generation.
    if "SPHINX_BUILD" in os.environ:
        hmc.init_kernel = init_kernel
        hmc.sample_kernel = sample_kernel

    return init_kernel, sample_kernel


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
        :meth:`init` has the same type.
    :param kinetic_fn: Python callable that returns the kinetic energy given
        inverse mass matrix and momentum. If not provided, the default is
        euclidean kinetic energy.
    :param float step_size: Determines the size of a single step taken by the
        verlet integrator while computing the trajectory using Hamiltonian
        dynamics. If not specified, it will be set to 1.
    :param inverse_mass_matrix: Initial value for inverse mass matrix.
        This may be adapted during warmup if adapt_mass_matrix = True.
        If no value is specified, then it is initialized to the identity matrix.
        For a potential_fn with general JAX pytree parameters, the order of entries
        of the mass matrix is the order of the flattened version of pytree parameters
        obtained with `jax.tree_flatten`, which is a bit ambiguous (see more at
        https://jax.readthedocs.io/en/latest/pytrees.html). If `model` is not None,
        here we can specify a structured block mass matrix as a dictionary, where
        keys are tuple of site names and values are the corresponding block of the
        mass matrix.
        For more information about structured mass matrix, see `dense_mass` argument.
    :type inverse_mass_matrix: numpy.ndarray or dict
    :param bool adapt_step_size: A flag to decide if we want to adapt step_size
        during warm-up phase using Dual Averaging scheme.
    :param bool adapt_mass_matrix: A flag to decide if we want to adapt mass
        matrix during warm-up phase using Welford scheme.
    :param dense_mass:  This flag controls whether mass matrix is dense (i.e. full-rank) or
        diagonal (defaults to ``dense_mass=False``). To specify a structured mass matrix,
        users can provide a list of tuples of site names. Each tuple represents
        a block in the joint mass matrix. For example, assuming that the model
        has latent variables "x", "y", "z" (where each variable can be multi-dimensional),
        possible specifications and corresponding mass matrix structures are as follows:

            + dense_mass=[("x", "y")]: use a dense mass matrix for the joint
              (x, y) and a diagonal mass matrix for z
            + dense_mass=[] (equivalent to dense_mass=False): use a diagonal mass
              matrix for the joint (x, y, z)
            + dense_mass=[("x", "y", "z")] (equivalent to full_mass=True):
              use a dense mass matrix for the joint (x, y, z)
            + dense_mass=[("x",), ("y",), ("z")]: use dense mass matrices for
              each of x, y, and z (i.e. block-diagonal with 3 blocks)

    :type dense_mass: bool or list
    :param float target_accept_prob: Target acceptance probability for step size
        adaptation using Dual Averaging. Increasing this value will lead to a smaller
        step size, hence the sampling will be slower but more robust. Defaults to 0.8.
    :param float trajectory_length: Length of a MCMC trajectory for HMC. Default
        value is :math:`2\\pi`.
    :param callable init_strategy: a per-site initialization function.
        See :ref:`init_strategy` section for available functions.
    :param bool find_heuristic_step_size: whether or not to use a heuristic function
        to adjust the step size at the beginning of each adaptation window. Defaults
        to False.
    :param bool forward_mode_differentiation: whether to use forward-mode differentiation
        or reverse-mode differentiation. By default, we use reverse mode but the forward
        mode can be useful in some cases to improve the performance. In addition, some
        control flow utility on JAX such as `jax.lax.while_loop` or `jax.lax.fori_loop`
        only supports forward-mode differentiation. See
        `JAX's The Autodiff Cookbook <https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html>`_
        for more information.
    :param bool regularize_mass_matrix: whether or not to regularize the estimated mass
        matrix for numerical stability during warmup phase. Defaults to True. This flag
        does not take effect if ``adapt_mass_matrix == False``.
    """

    def __init__(
        self,
        model=None,
        potential_fn=None,
        kinetic_fn=None,
        step_size=1.0,
        inverse_mass_matrix=None,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        dense_mass=False,
        target_accept_prob=0.8,
        trajectory_length=2 * math.pi,
        init_strategy=init_to_uniform,
        find_heuristic_step_size=False,
        forward_mode_differentiation=False,
        regularize_mass_matrix=True,
    ):
        if not (model is None) ^ (potential_fn is None):
            raise ValueError("Only one of `model` or `potential_fn` must be specified.")
        self._model = model
        self._potential_fn = potential_fn
        self._kinetic_fn = (
            kinetic_fn if kinetic_fn is not None else euclidean_kinetic_energy
        )
        self._step_size = float(step_size) if isinstance(step_size, int) else step_size
        self._inverse_mass_matrix = inverse_mass_matrix
        self._adapt_step_size = adapt_step_size
        self._adapt_mass_matrix = adapt_mass_matrix
        self._dense_mass = dense_mass
        self._target_accept_prob = target_accept_prob
        self._trajectory_length = (
            float(trajectory_length)
            if isinstance(trajectory_length, int)
            else trajectory_length
        )
        self._algo = "HMC"
        self._max_tree_depth = 10
        self._init_strategy = init_strategy
        self._find_heuristic_step_size = find_heuristic_step_size
        self._forward_mode_differentiation = forward_mode_differentiation
        self._regularize_mass_matrix = regularize_mass_matrix
        # Set on first call to init
        self._init_fn = None
        self._potential_fn_gen = None
        self._postprocess_fn = None
        self._sample_fn = None

    def _init_state(self, rng_key, model_args, model_kwargs, init_params):
        if self._model is not None:
            init_params, potential_fn, postprocess_fn, model_trace = initialize_model(
                rng_key,
                self._model,
                dynamic_args=True,
                init_strategy=self._init_strategy,
                model_args=model_args,
                model_kwargs=model_kwargs,
                forward_mode_differentiation=self._forward_mode_differentiation,
            )
            if self._init_fn is None:
                self._init_fn, self._sample_fn = hmc(
                    potential_fn_gen=potential_fn,
                    kinetic_fn=self._kinetic_fn,
                    algo=self._algo,
                )
            self._potential_fn_gen = potential_fn
            self._postprocess_fn = postprocess_fn
        elif self._init_fn is None:
            self._init_fn, self._sample_fn = hmc(
                potential_fn=self._potential_fn,
                kinetic_fn=self._kinetic_fn,
                algo=self._algo,
            )

        return init_params

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
        return "{} steps of size {:.2e}. acc. prob={:.2f}".format(
            state.num_steps, state.adapt_state.step_size, state.mean_accept_prob
        )

    def init(
        self, rng_key, num_warmup, init_params=None, model_args=(), model_kwargs={}
    ):
        # non-vectorized
        if rng_key.ndim == 1:
            rng_key, rng_key_init_model = random.split(rng_key)
        # vectorized
        else:
            rng_key, rng_key_init_model = jnp.swapaxes(
                vmap(random.split)(rng_key), 0, 1
            )
        init_params = self._init_state(
            rng_key_init_model, model_args, model_kwargs, init_params
        )
        if self._potential_fn and init_params is None:
            raise ValueError(
                "Valid value of `init_params` must be provided with" " `potential_fn`."
            )

        # change dense_mass to a structural form
        dense_mass = self._dense_mass
        inverse_mass_matrix = self._inverse_mass_matrix
        if self._model is not None:
            z = init_params[0] if isinstance(init_params, ParamInfo) else init_params
            if isinstance(dense_mass, bool):
                # XXX: by default, the order variables are sorted by their names,
                # this is to be compatible with older numpyro versions
                # and to match autoguide scale parameter and jax flatten utils
                dense_mass = [tuple(sorted(z))] if dense_mass else []
            assert isinstance(dense_mass, list)

        hmc_init_fn = lambda init_params, rng_key: self._init_fn(  # noqa: E731
            init_params,
            num_warmup=num_warmup,
            step_size=self._step_size,
            inverse_mass_matrix=inverse_mass_matrix,
            adapt_step_size=self._adapt_step_size,
            adapt_mass_matrix=self._adapt_mass_matrix,
            dense_mass=dense_mass,
            target_accept_prob=self._target_accept_prob,
            trajectory_length=self._trajectory_length,
            max_tree_depth=self._max_tree_depth,
            find_heuristic_step_size=self._find_heuristic_step_size,
            forward_mode_differentiation=self._forward_mode_differentiation,
            regularize_mass_matrix=self._regularize_mass_matrix,
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

    def postprocess_fn(self, args, kwargs):
        if self._postprocess_fn is None:
            return identity
        return self._postprocess_fn(*args, **kwargs)

    def sample(self, state, model_args, model_kwargs):
        """
        Run HMC from the given :data:`~numpyro.infer.hmc.HMCState` and return the resulting
        :data:`~numpyro.infer.hmc.HMCState`.

        :param HMCState state: Represents the current state.
        :param model_args: Arguments provided to the model.
        :param model_kwargs: Keyword arguments provided to the model.
        :return: Next `state` after running HMC.
        """
        return self._sample_fn(state, model_args, model_kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_sample_fn"] = None
        state["_init_fn"] = None
        state["_postprocess_fn"] = None
        state["_potential_fn_gen"] = None
        return state


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
    :param inverse_mass_matrix: Initial value for inverse mass matrix.
        This may be adapted during warmup if adapt_mass_matrix = True.
        If no value is specified, then it is initialized to the identity matrix.
        For a potential_fn with general JAX pytree parameters, the order of entries
        of the mass matrix is the order of the flattened version of pytree parameters
        obtained with `jax.tree_flatten`, which is a bit ambiguous (see more at
        https://jax.readthedocs.io/en/latest/pytrees.html). If `model` is not None,
        here we can specify a structured block mass matrix as a dictionary, where
        keys are tuple of site names and values are the corresponding block of the
        mass matrix.
        For more information about structured mass matrix, see `dense_mass` argument.
    :type inverse_mass_matrix: numpy.ndarray or dict
    :param bool adapt_step_size: A flag to decide if we want to adapt step_size
        during warm-up phase using Dual Averaging scheme.
    :param bool adapt_mass_matrix: A flag to decide if we want to adapt mass
        matrix during warm-up phase using Welford scheme.
    :param dense_mass:  This flag controls whether mass matrix is dense (i.e. full-rank) or
        diagonal (defaults to ``dense_mass=False``). To specify a structured mass matrix,
        users can provide a list of tuples of site names. Each tuple represents
        a block in the joint mass matrix. For example, assuming that the model
        has latent variables "x", "y", "z" (where each variable can be multi-dimensional),
        possible specifications and corresponding mass matrix structures are as follows:

            + dense_mass=[("x", "y")]: use a dense mass matrix for the joint
              (x, y) and a diagonal mass matrix for z
            + dense_mass=[] (equivalent to dense_mass=False): use a diagonal mass
              matrix for the joint (x, y, z)
            + dense_mass=[("x", "y", "z")] (equivalent to full_mass=True):
              use a dense mass matrix for the joint (x, y, z)
            + dense_mass=[("x",), ("y",), ("z")]: use dense mass matrices for
              each of x, y, and z (i.e. block-diagonal with 3 blocks)

    :type dense_mass: bool or list
    :param float target_accept_prob: Target acceptance probability for step size
        adaptation using Dual Averaging. Increasing this value will lead to a smaller
        step size, hence the sampling will be slower but more robust. Defaults to 0.8.
    :param float trajectory_length: Length of a MCMC trajectory for HMC. This arg has
        no effect in NUTS sampler.
    :param int max_tree_depth: Max depth of the binary tree created during the doubling
        scheme of NUTS sampler. Defaults to 10. This argument also accepts a tuple of
        integers `(d1, d2)`, where `d1` is the max tree depth during warmup phase and
        `d2` is the max tree depth during post warmup phase.
    :param callable init_strategy: a per-site initialization function.
        See :ref:`init_strategy` section for available functions.
    :param bool find_heuristic_step_size: whether or not to use a heuristic function
        to adjust the step size at the beginning of each adaptation window. Defaults
        to False.
    :param bool forward_mode_differentiation: whether to use forward-mode differentiation
        or reverse-mode differentiation. By default, we use reverse mode but the forward
        mode can be useful in some cases to improve the performance. In addition, some
        control flow utility on JAX such as `jax.lax.while_loop` or `jax.lax.fori_loop`
        only supports forward-mode differentiation. See
        `JAX's The Autodiff Cookbook <https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html>`_
        for more information.
    """

    def __init__(
        self,
        model=None,
        potential_fn=None,
        kinetic_fn=None,
        step_size=1.0,
        inverse_mass_matrix=None,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        dense_mass=False,
        target_accept_prob=0.8,
        trajectory_length=None,
        max_tree_depth=10,
        init_strategy=init_to_uniform,
        find_heuristic_step_size=False,
        forward_mode_differentiation=False,
        regularize_mass_matrix=True,
    ):
        super(NUTS, self).__init__(
            potential_fn=potential_fn,
            model=model,
            kinetic_fn=kinetic_fn,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
            adapt_step_size=adapt_step_size,
            adapt_mass_matrix=adapt_mass_matrix,
            dense_mass=dense_mass,
            target_accept_prob=target_accept_prob,
            trajectory_length=trajectory_length,
            init_strategy=init_strategy,
            find_heuristic_step_size=find_heuristic_step_size,
            forward_mode_differentiation=forward_mode_differentiation,
            regularize_mass_matrix=regularize_mass_matrix,
        )
        self._max_tree_depth = max_tree_depth
        self._algo = "NUTS"
