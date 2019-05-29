import math
import os
from collections import namedtuple

import tqdm

import jax.numpy as np
from jax import jit, partial, random
from jax.flatten_util import ravel_pytree
from jax.random import PRNGKey
from jax.tree_util import register_pytree_node

import numpyro.distributions as dist
from numpyro.diagnostics import summary
from numpyro.hmc_util import IntegratorState, build_tree, find_reasonable_step_size, velocity_verlet, warmup_adapter
from numpyro.util import cond, fori_collect, fori_loop, identity

HMCState = namedtuple('HMCState', ['i', 'z', 'z_grad', 'potential_energy', 'num_steps', 'accept_prob',
                                   'mean_accept_prob', 'step_size', 'inverse_mass_matrix', 'rng'])
"""
A :func:`~collections.namedtuple` consisting of the following fields:

 - **i** - iteration. This is reset to 0 after warmup.
 - **z** - Python collection representing values (unconstrained samples from
   the posterior) at latent sites.
 - **z_grad** - Gradient of potential energy w.r.t. latent sample sites.
 - **potential_energy** - Potential energy computed at the given value of ``z``.
 - **num_steps** - Number of steps in the Hamiltonian trajectory (for diagnostics).
 - **accept_prob** - Acceptance probability of the proposal. Note that ``z``
   does not correspond to the proposal if it is rejected.
 - **mean_accept_prob** - Mean acceptance probability until current iteration
   during warmup adaptation or sampling (for diagnostics).
 - **step_size** - Step size to be used by the integrator in the next iteration.
   This is adapted during warmup.
 - **inverse_mass_matrix** - The inverse mass matrix to be be used for the next
   iteration. This is adapted during warmup.
 - **rng** - random number generator seed used for the iteration.
"""


register_pytree_node(
    HMCState,
    lambda xs: (tuple(xs), None),
    lambda _, xs: HMCState(*xs)
)


HMCState.update = HMCState._replace


def _get_num_steps(step_size, trajectory_length):
    num_steps = np.clip(trajectory_length / step_size, a_min=1)
    # NB: casting to np.int64 does not take effect (returns np.int32 instead)
    # if jax_enable_x64 is False
    return num_steps.astype(np.int64)


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


def get_diagnostics_str(hmc_state):
    return '{} steps of size {:.2e}. acc. prob={:.2f}'.format(hmc_state.num_steps,
                                                              hmc_state.step_size,
                                                              hmc_state.mean_accept_prob)


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
        any python collection type, provided that `init_params` argument to
        `init_kernel` has the same type.
    :param kinetic_fn: Python callable that returns the kinetic energy given
        inverse mass matrix and momentum. If not provided, the default is
        euclidean kinetic energy.
    :param str algo: Whether to run ``HMC`` with fixed number of steps or ``NUTS``
        with adaptive path length. Default is ``NUTS``.
    :returns: a tuple of callables (`init_kernel`, `sample_kernel`): the first
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
        >>> init_params, potential_fn, constrain_fn = initialize_model(random.PRNGKey(0),
        ...                                                            model, data, labels)
        >>> init_kernel, sample_kernel = hmc(potential_fn, algo='NUTS')
        >>> hmc_state = init_kernel(init_params,
        ...                         trajectory_length=10,
        ...                         num_warmup=300)
        >>> samples = fori_collect(500, sample_kernel, hmc_state,
        ...                        transform=lambda state: constrain_fn(state.z))
        >>> print(np.mean(samples['beta'], axis=0))  # doctest: +SKIP
        [0.9153987 2.0754058 2.9621222]
    """
    if kinetic_fn is None:
        kinetic_fn = _euclidean_ke
    vv_init, vv_update = velocity_verlet(potential_fn, kinetic_fn)
    trajectory_len = None
    max_treedepth = None
    momentum_generator = None
    wa_update = None

    def init_kernel(init_params,
                    num_warmup,
                    step_size=1.0,
                    adapt_step_size=True,
                    adapt_mass_matrix=True,
                    diag_mass=True,
                    target_accept_prob=0.8,
                    trajectory_length=2*math.pi,
                    max_tree_depth=10,
                    run_warmup=True,
                    progbar=True,
                    rng=PRNGKey(0)):
        """
        Initializes the HMC sampler.

        :param init_params: Initial parameters to begin sampling. The type must
            be consistent with the input type to `potential_fn`.
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
        z = init_params
        z_flat, unravel_fn = ravel_pytree(z)
        momentum_generator = partial(_sample_momentum, unravel_fn)

        find_reasonable_ss = partial(find_reasonable_step_size,
                                     potential_fn, kinetic_fn,
                                     momentum_generator)

        wa_init, wa_update = warmup_adapter(num_warmup,
                                            adapt_step_size=adapt_step_size,
                                            adapt_mass_matrix=adapt_mass_matrix,
                                            diag_mass=diag_mass,
                                            target_accept_prob=target_accept_prob,
                                            find_reasonable_step_size=find_reasonable_ss)

        rng_hmc, rng_wa = random.split(rng)
        wa_state = wa_init(z, rng_wa, step_size, mass_matrix_size=np.size(z_flat))
        r = momentum_generator(wa_state.inverse_mass_matrix, rng)
        vv_state = vv_init(z, r)
        hmc_state = HMCState(0, vv_state.z, vv_state.z_grad, vv_state.potential_energy, 0, 0., 0.,
                             wa_state.step_size, wa_state.inverse_mass_matrix, rng_hmc)

        wa_update = jit(wa_update)
        if run_warmup:
            # JIT if progress bar updates not required
            if not progbar:
                hmc_state, _ = jit(fori_loop, static_argnums=(2,))(0, num_warmup,
                                                                   warmup_update,
                                                                   (hmc_state, wa_state))
            else:
                with tqdm.trange(num_warmup, desc='warmup') as t:
                    for i in t:
                        hmc_state, wa_state = warmup_update(i, (hmc_state, wa_state))
                        # TODO: set refresh=True when its performance issue is resolved
                        t.set_postfix_str(get_diagnostics_str(hmc_state), refresh=False)
            # Reset `i` and `mean_accept_prob` for fresh diagnostics.
            hmc_state.update(i=0, mean_accept_prob=0)
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
        itr = hmc_state.i + 1
        mean_accept_prob = hmc_state.mean_accept_prob + (accept_prob - hmc_state.mean_accept_prob) / itr
        return HMCState(itr, vv_state.z, vv_state.z_grad, vv_state.potential_energy, num_steps,
                        accept_prob, mean_accept_prob, hmc_state.step_size, hmc_state.inverse_mass_matrix,
                        rng)

    # Make `init_kernel` and `sample_kernel` visible from the global scope once
    # `hmc` is called for sphinx doc generation.
    if 'SPHINX_BUILD' in os.environ:
        hmc.init_kernel = init_kernel
        hmc.sample_kernel = sample_kernel

    return init_kernel, sample_kernel


def mcmc(num_warmup, num_samples, init_params, sampler='hmc',
         constrain_fn=None, print_summary=True, **sampler_kwargs):
    """
    Convenience wrapper for MCMC samplers -- runs warmup, prints
    diagnostic summary and returns a collections of samples
    from the posterior.

    :param num_warmup: Number of warmup steps.
    :param num_samples: Number of samples to generate from the Markov chain.
    :param init_params: Initial parameters to begin sampling. The type can
        must be consistent with the input type to `potential_fn`.
    :param sampler: currently, only `hmc` is implemented (default).
    :param constrain_fn: Callable that converts a collection of unconstrained
        sample values returned from the sampler to constrained values that
        lie within the support of the sample sites.
    :param print_summary: Whether to print diagnostics summary for
        each sample site. Default is ``True``.
    :param `**sampler_kwargs`: Sampler specific keyword arguments.

         - *HMC*: Refer to :func:`~numpyro.mcmc.hmc` and
           :func:`~numpyro.mcmc.hmc.init_kernel` for accepted arguments. Note
           that all arguments must be provided as keywords.

    :return: collection of samples from the posterior.
    """
    if sampler == 'hmc':
        if constrain_fn is None:
            constrain_fn = identity
        potential_fn = sampler_kwargs.pop('potential_fn')
        kinetic_fn = sampler_kwargs.pop('kinetic_fn', None)
        algo = sampler_kwargs.pop('algo', 'NUTS')
        progbar = sampler_kwargs.pop('progbar', True)

        init_kernel, sample_kernel = hmc(potential_fn, kinetic_fn, algo)
        hmc_state = init_kernel(init_params, num_warmup, progbar=progbar, **sampler_kwargs)
        samples = fori_collect(num_samples, sample_kernel, hmc_state,
                               transform=lambda x: constrain_fn(x.z),
                               progbar=progbar,
                               diagnostics_fn=get_diagnostics_str,
                               progbar_desc='sample')
        if print_summary:
            summary(samples)
        return samples
    else:
        raise ValueError('sampler: {} not recognized'.format(sampler))
