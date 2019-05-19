import math
import os

import tqdm

import jax.numpy as np
from jax import jit, partial, random
from jax.flatten_util import ravel_pytree
from jax.random import PRNGKey

import numpyro.distributions as dist
from numpyro.hmc_util import IntegratorState, build_tree, find_reasonable_step_size, velocity_verlet, warmup_adapter
from numpyro.util import cond, fori_loop, laxtuple

HMCState = laxtuple('HMCState', ['z', 'z_grad', 'potential_energy', 'num_steps', 'accept_prob',
                                 'step_size', 'inverse_mass_matrix', 'rng'])


def _get_num_steps(step_size, trajectory_length):
    num_steps = np.array(trajectory_length / step_size, dtype=np.int32)
    return np.where(num_steps < 1, np.array(1, dtype=np.int32), num_steps)


def _sample_momentum(unpack_fn, inverse_mass_matrix, rng):
    if inverse_mass_matrix.ndim == 1:
        r = dist.Normal(0., np.sqrt(np.reciprocal(inverse_mass_matrix))).sample(rng)
        return unpack_fn(r)
    elif inverse_mass_matrix.ndim == 2:
        raise NotImplementedError


def _euclidean_ke(inverse_mass_matrix, r):
    r, _ = ravel_pytree(r)

    if inverse_mass_matrix.ndim == 2:
        v = np.matmul(inverse_mass_matrix, r)
    elif inverse_mass_matrix.ndim == 1:
        v = np.multiply(inverse_mass_matrix, r)

    return 0.5 * np.dot(v, r)


def hmc(potential_fn, kinetic_fn=None, algo='NUTS'):
    r"""
    Hamiltonian Monte Carlo inference, using either fixed number of
    steps or the No U-Turn Sampler (NUTS) with adaptive path length.

    **References:**

    1. *MCMC Using Hamiltonian Dynamics*, Radford M. Neal
    2. *The No-U-turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo*,
       Matthew D. Hoffman, and Andrew Gelman.

    :param potential_fn: Python callable that computes the potential energy
        given input parameters. The input parameters to `potential_fn` can be
        any python collection type, provided that `init_samples` argument to
        `init_kernel` has the same type.
    :param kinetic_fn: Python callable that returns the kinetic energy given
        inverse mass matrix and momentum. If not provided, the default is
        euclidean kinetic energy.
    :param str algo: Whether to run ``HMC`` with fixed number of steps or ``NUTS``
        with adaptive path length. Default is ``NUTS``.
    :return init_kernel, sample_kernel: Returns a tuple of callables, the first
        one to initialize the sampler, and the second one to generate samples
        given an existing one.

    **Example**

    .. testsetup::

        import jax
        from jax import random
        import jax.numpy as np
        import numpyro.distributions as dist
        from numpyro.handlers import sample
        from numpyro.hmc_util import initialize_model
        from numpyro.mcmc import hmc
        from numpyro.util import fori_collect

    .. doctest::

        >>> true_coefs = np.array([1., 2., 3.])
        >>> data = random.normal(random.PRNGKey(2), (2000, 3))
        >>> dim = 3
        >>> labels = dist.Bernoulli(logits=(true_coefs * data).sum(-1)).sample(random.PRNGKey(3))
        >>>
        >>> def model(data, labels):
        ...     coefs_mean = np.zeros(dim)
        ...     coefs = sample('beta', dist.Normal(coefs_mean, np.ones(3)))
        ...     return sample('y', dist.Bernoulli(logits=(coefs * data).sum(-1)), obs=labels)
        >>>
        >>> init_params, potential_fn, transform_fn = initialize_model(random.PRNGKey(0), model, data, labels)
        >>> init_kernel, sample_kernel = hmc(potential_fn, algo='NUTS')
        >>> hmc_state = init_kernel(init_params,
        ...                         trajectory_length=10,
        ...                         num_warmup_steps=300)
        >>> hmc_states = fori_collect(500, sample_kernel, hmc_state,
        ...                           transform=lambda x: transform_fn(x.z))
        >>> print(np.mean(hmc_states['beta'], axis=0))
        [0.9153987 2.0754058 2.9621222]
    """
    if kinetic_fn is None:
        kinetic_fn = _euclidean_ke
    vv_init, vv_update = velocity_verlet(potential_fn, kinetic_fn)
    trajectory_len = None
    max_treedepth = None
    momentum_generator = None
    wa_update = None

    def init_kernel(init_samples,
                    num_warmup_steps,
                    step_size=1.0,
                    adapt_step_size=True,
                    adapt_mass_matrix=True,
                    diag_mass=True,
                    target_accept_prob=0.8,
                    trajectory_length=2*math.pi,
                    max_tree_depth=10,
                    run_warmup=True,
                    progbar=True,
                    heuristic_step_size=True,
                    rng=PRNGKey(0)):
        """
        Initializes the HMC sampler.

        :param init_samples: Initial parameters to begin sampling. The type can
            must be consistent with the input type to `potential_fn`.
        :param int num_warmup_steps: Number of warmup steps; samples generated
            during warmup are discarded.
        :param float step_size: Determines the size of a single step taken by the
            verlet integrator while computing the trajectory using Hamiltonian
            dynamics. If not specified, it will be set to 1.
        :param bool adapt_step_size: A flag to decide if we want to adapt step_size
            during warm-up phase using Dual Averaging scheme.
        :param bool adapt_mass_matrix: A flag to decide if we want to adapt mass
            matrix during warm-up phase using Welford scheme.
        :param bool diag_mass: A flag to decide if mass matrix is diagonal (default)
            or dense (if set to ``False``).
        :param float target_accept_prob: Target acceptance probability for step size
            adaptation using Dual Averaging. Increasing this value will lead to a smaller
            step size, hence the sampling will be slower but more robust. Default to 0.8.
        :param float trajectory_length: Length of a MCMC trajectory for HMC. Default
            value is :math:`2\\pi`.
        :param int max_tree_depth: Max depth of the binary tree created during the doubling
            scheme of NUTS sampler. Defaults to 10.
        :param bool run_warmup: Flag to decide whether warmup is run. If ``True``,
            `init_kernel` returns an initial :data:`HMCState` that can be used to
            generate samples using MCMC. Else, returns the arguments and callable
            that does the initial adaptation.
        :param bool progbar: Whether to enable progress bar updates. Defaults to
            ``True``.
        :param bool heuristic_step_size: If ``True``, a coarse grained adjustment of
            step size is done at the beginning of each adaptation window to achieve
            `target_acceptance_prob`.
        :param jax.random.PRNGKey rng: random key to be used as the source of
            randomness.
        """
        step_size = float(step_size)
        nonlocal momentum_generator, wa_update, trajectory_len, max_treedepth
        trajectory_len = float(trajectory_length)
        max_treedepth = max_tree_depth
        z = init_samples
        z_flat, unravel_fn = ravel_pytree(z)
        momentum_generator = partial(_sample_momentum, unravel_fn)

        wa_kwargs = {}
        # FIXME: compiling find_reasonable_step_size is so slow for some models
        if heuristic_step_size:
            wa_kwargs["find_reasonable_step_size"] = partial(find_reasonable_step_size,
                                                             potential_fn, kinetic_fn,
                                                             momentum_generator)

        wa_init, wa_update = warmup_adapter(num_warmup_steps,
                                            adapt_step_size=adapt_step_size,
                                            adapt_mass_matrix=adapt_mass_matrix,
                                            diag_mass=diag_mass,
                                            target_accept_prob=target_accept_prob,
                                            **wa_kwargs)

        rng_hmc, rng_wa = random.split(rng)
        wa_state = wa_init(z, rng_wa, step_size, mass_matrix_size=np.size(z_flat))
        r = momentum_generator(wa_state.inverse_mass_matrix, rng)
        vv_state = vv_init(z, r)
        hmc_state = HMCState(vv_state.z, vv_state.z_grad, vv_state.potential_energy, 0, 0.,
                             wa_state.step_size, wa_state.inverse_mass_matrix, rng_hmc)

        wa_update = jit(wa_update)
        if run_warmup:
            # JIT if progress bar updates not required
            if not progbar:
                hmc_state, _ = jit(fori_loop, static_argnums=(2,))(0, num_warmup_steps,
                                                                   warmup_update,
                                                                   (hmc_state, wa_state))
            else:
                for i in tqdm.trange(num_warmup_steps):
                    hmc_state, wa_state = warmup_update(i, (hmc_state, wa_state))
            return hmc_state
        else:
            return hmc_state, wa_state, warmup_update

    def warmup_update(t, states):
        hmc_state, wa_state = states
        hmc_state = sample_kernel(hmc_state)
        wa_state = wa_update(t, hmc_state.accept_prob, hmc_state.z, wa_state)
        hmc_state = hmc_state.update(step_size=wa_state.step_size,
                                     inverse_mass_matrix=wa_state.inverse_mass_matrix)
        return hmc_state, wa_state

    def _hmc_next(step_size, inverse_mass_matrix, vv_state, rng):
        num_steps = _get_num_steps(step_size, trajectory_len)
        vv_state_new = fori_loop(0, num_steps,
                                 lambda i, val: vv_update(step_size, inverse_mass_matrix, val),
                                 vv_state)
        energy_old = vv_state.potential_energy + kinetic_fn(inverse_mass_matrix, vv_state.r)
        energy_new = vv_state_new.potential_energy + kinetic_fn(inverse_mass_matrix, vv_state_new.r)
        delta_energy = energy_new - energy_old
        delta_energy = np.where(np.isnan(delta_energy), np.inf, delta_energy)
        accept_prob = np.clip(np.exp(-delta_energy), a_max=1.0)
        transition = random.bernoulli(rng, accept_prob)
        vv_state = cond(transition,
                        vv_state_new, lambda state: state,
                        vv_state, lambda state: state)
        return vv_state, num_steps, accept_prob

    def _nuts_next(step_size, inverse_mass_matrix, vv_state, rng):
        binary_tree = build_tree(vv_update, kinetic_fn, vv_state,
                                 inverse_mass_matrix, step_size, rng,
                                 max_tree_depth=max_treedepth)
        accept_prob = binary_tree.sum_accept_probs / binary_tree.num_proposals
        num_steps = binary_tree.num_proposals
        vv_state = vv_state.update(z=binary_tree.z_proposal,
                                   potential_energy=binary_tree.z_proposal_pe,
                                   z_grad=binary_tree.z_proposal_grad)
        return vv_state, num_steps, accept_prob

    _next = _nuts_next if algo == 'NUTS' else _hmc_next

    @jit
    def sample_kernel(hmc_state):
        """
        Given an existing :data:`HMCState`, run HMC with fixed (possibly adapted)
        step size and return a new :data:`HMCState`.

        :param hmc_state: Current sample (and associated state).
        :return: new proposed :data:`HMCState` from simulating
            Hamiltonian dynamics given existing state.
        """
        rng, rng_momentum, rng_transition = random.split(hmc_state.rng, 3)
        r = momentum_generator(hmc_state.inverse_mass_matrix, rng_momentum)
        vv_state = IntegratorState(hmc_state.z, r, hmc_state.potential_energy, hmc_state.z_grad)
        vv_state, num_steps, accept_prob = _next(hmc_state.step_size,
                                                 hmc_state.inverse_mass_matrix,
                                                 vv_state, rng_transition)
        return HMCState(vv_state.z, vv_state.z_grad, vv_state.potential_energy, num_steps,
                        accept_prob, hmc_state.step_size, hmc_state.inverse_mass_matrix, rng)

    # Make `init_kernel` and `sample_kernel` visible from the global scope once
    # `hmc` is called for sphinx doc generation.
    if 'SPHINX_BUILD' in os.environ:
        hmc.init_kernel = init_kernel
        hmc.sample_kernel = sample_kernel

    return init_kernel, sample_kernel
