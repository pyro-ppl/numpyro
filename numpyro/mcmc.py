from collections import namedtuple
import math
import os
import warnings

import tqdm

from jax import jit, partial, pmap, random, vmap
from jax.flatten_util import ravel_pytree
from jax.lib import xla_bridge
import jax.numpy as np
from jax.random import PRNGKey
from jax.tree_util import register_pytree_node, tree_map, tree_multimap

from numpyro.diagnostics import summary
from numpyro.hmc_util import IntegratorState, build_tree, find_reasonable_step_size, velocity_verlet, warmup_adapter
from numpyro.util import cond, fori_collect, fori_loop, identity

HMCState = namedtuple('HMCState', ['i', 'z', 'z_grad', 'potential_energy', 'num_steps', 'accept_prob',
                                   'mean_accept_prob', 'adapt_state', 'rng'])
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
 - **adapt_state** - A ``AdaptState`` namedtuple which contains adaptation information
   during warmup:

   + **step_size** - Step size to be used by the integrator in the next iteration.
   + **inverse_mass_matrix** - The inverse mass matrix to be used for the next
     iteration.
   + **mass_matrix_sqrt** - The square root of mass matrix to be used for the next
     iteration. In case of dense mass, this is the Cholesky factorization of the
     mass matrix.

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


def _sample_momentum(unpack_fn, mass_matrix_sqrt, rng):
    eps = random.normal(rng, np.shape(mass_matrix_sqrt)[:1])
    if mass_matrix_sqrt.ndim == 1:
        r = np.multiply(mass_matrix_sqrt, eps)
        return unpack_fn(r)
    elif mass_matrix_sqrt.ndim == 2:
        r = np.dot(mass_matrix_sqrt, eps)
        return unpack_fn(r)
    else:
        raise ValueError("Mass matrix has incorrect number of dims.")


def _euclidean_ke(inverse_mass_matrix, r):
    r, _ = ravel_pytree(r)

    if inverse_mass_matrix.ndim == 2:
        v = np.matmul(inverse_mass_matrix, r)
    elif inverse_mass_matrix.ndim == 1:
        v = np.multiply(inverse_mass_matrix, r)

    return 0.5 * np.dot(v, r)


def get_diagnostics_str(hmc_state):
    return '{} steps of size {:.2e}. acc. prob={:.2f}'.format(hmc_state.num_steps,
                                                              hmc_state.adapt_state.step_size,
                                                              hmc_state.mean_accept_prob)


def hmc(potential_fn, kinetic_fn=None, algo='NUTS'):
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
    :param kinetic_fn: Python callable that returns the kinetic energy given
        inverse mass matrix and momentum. If not provided, the default is
        euclidean kinetic energy.
    :param str algo: Whether to run ``HMC`` with fixed number of steps or ``NUTS``
        with adaptive path length. Default is ``NUTS``.
    :return: a tuple of callables (`init_kernel`, `sample_kernel`), the first
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
        ...     intercept = sample('intercept', dist.Normal(0., 10.))
        ...     return sample('y', dist.Bernoulli(logits=(coefs * data + intercept).sum(-1)), obs=labels)
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
        kinetic_fn = _euclidean_ke
    vv_init, vv_update = velocity_verlet(potential_fn, kinetic_fn)
    trajectory_len = None
    max_treedepth = None
    momentum_generator = None
    wa_update = None
    wa_steps = None
    if algo not in {'HMC', 'NUTS'}:
        raise ValueError('`algo` must be one of `HMC` or `NUTS`.')

    def init_kernel(init_params,
                    num_warmup,
                    step_size=1.0,
                    adapt_step_size=True,
                    adapt_mass_matrix=True,
                    dense_mass=False,
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
        :param bool dense_mass: A flag to decide if mass matrix is dense or
            diagonal (default when ``dense_mass=False``)
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
        nonlocal momentum_generator, wa_update, trajectory_len, max_treedepth, wa_steps
        wa_steps = num_warmup
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
                                            dense_mass=dense_mass,
                                            target_accept_prob=target_accept_prob,
                                            find_reasonable_step_size=find_reasonable_ss)

        rng_hmc, rng_wa = random.split(rng)
        wa_state = wa_init(z, rng_wa, step_size, mass_matrix_size=np.size(z_flat))
        r = momentum_generator(wa_state.mass_matrix_sqrt, rng)
        vv_state = vv_init(z, r)
        hmc_state = HMCState(0, vv_state.z, vv_state.z_grad, vv_state.potential_energy, 0, 0., 0.,
                             wa_state, rng_hmc)

        if run_warmup and num_warmup > 0:
            # JIT if progress bar updates not required
            if not progbar:
                hmc_state = jit(fori_loop, static_argnums=(2,))(
                    0, num_warmup, lambda *args: sample_kernel(args[1]), hmc_state)
            else:
                with tqdm.trange(num_warmup, desc='warmup') as t:
                    for i in t:
                        hmc_state = sample_kernel(hmc_state)
                        t.set_postfix_str(get_diagnostics_str(hmc_state), refresh=False)
        return hmc_state

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
        r = momentum_generator(hmc_state.adapt_state.mass_matrix_sqrt, rng_momentum)
        vv_state = IntegratorState(hmc_state.z, r, hmc_state.potential_energy, hmc_state.z_grad)
        vv_state, num_steps, accept_prob = _next(hmc_state.adapt_state.step_size,
                                                 hmc_state.adapt_state.inverse_mass_matrix,
                                                 vv_state, rng_transition)
        # not update adapt_state after warmup phase
        adapt_state = cond(hmc_state.i < wa_steps,
                           (hmc_state.i, accept_prob, vv_state.z, hmc_state.adapt_state),
                           lambda args: wa_update(*args),
                           hmc_state.adapt_state,
                           lambda x: x)

        itr = hmc_state.i + 1
        n = np.where(hmc_state.i < wa_steps, itr, itr - wa_steps)
        # Reset `mean_accept_prob` for fresh diagnostics.
        mean_accept_prob = np.where(hmc_state.i == wa_steps, 0., hmc_state.mean_accept_prob)
        mean_accept_prob = mean_accept_prob + (accept_prob - mean_accept_prob) / n

        return HMCState(itr, vv_state.z, vv_state.z_grad, vv_state.potential_energy, num_steps,
                        accept_prob, mean_accept_prob, adapt_state, rng)

    # Make `init_kernel` and `sample_kernel` visible from the global scope once
    # `hmc` is called for sphinx doc generation.
    if 'SPHINX_BUILD' in os.environ:
        hmc.init_kernel = init_kernel
        hmc.sample_kernel = sample_kernel

    return init_kernel, sample_kernel


def mcmc(num_warmup, num_samples, init_params, num_chains=1, sampler='hmc',
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
        ...     intercept = sample('intercept', dist.Normal(0., 10.))
        ...     return sample('y', dist.Bernoulli(logits=(coefs * data + intercept).sum(-1)), obs=labels)
        >>>
        >>> init_params, potential_fn, constrain_fn = initialize_model(random.PRNGKey(0), model,
        ...                                                            data, labels)
        >>> num_warmup, num_samples = 1000, 1000
        >>> samples = mcmc(num_warmup, num_samples, init_params,
        ...                potential_fn=potential_fn,
        ...                constrain_fn=constrain_fn)  # doctest: +SKIP
        warmup: 100%|██████████| 1000/1000 [00:09<00:00, 109.40it/s, 1 steps of size 5.83e-01. acc. prob=0.79]
        sample: 100%|██████████| 1000/1000 [00:00<00:00, 1252.39it/s, 1 steps of size 5.83e-01. acc. prob=0.85]


                           mean         sd       5.5%      94.5%      n_eff       Rhat
            coefs[0]       0.96       0.07       0.85       1.07     455.35       1.01
            coefs[1]       2.05       0.09       1.91       2.20     332.00       1.01
            coefs[2]       3.18       0.13       2.96       3.37     320.27       1.00
           intercept      -0.03       0.02      -0.06       0.00     402.53       1.00
    """
    sequential_chain = False
    if xla_bridge.device_count() < num_chains:
        sequential_chain = True
        warnings.warn('There are not enough devices to run parallel chains: expected {} but got {}.'
                      ' Chains will be drawn sequentially. If you are running `mcmc` in CPU,'
                      ' consider to disable XLA intra-op parallelism by setting the environment'
                      ' flag "XLA_FLAGS=--xla_force_host_platform_device_count={}".'
                      .format(num_chains, xla_bridge.device_count(), num_chains))
    progbar = sampler_kwargs.pop('progbar', True)
    if num_chains > 1:
        progbar = False

    if sampler == 'hmc':
        if constrain_fn is None:
            constrain_fn = identity
        potential_fn = sampler_kwargs.pop('potential_fn')
        kinetic_fn = sampler_kwargs.pop('kinetic_fn', None)
        algo = sampler_kwargs.pop('algo', 'NUTS')
        rngs = sampler_kwargs.pop('rng', vmap(PRNGKey)(np.arange(num_chains)))

        init_kernel, sample_kernel = hmc(potential_fn, kinetic_fn, algo)
        if progbar:
            hmc_state = init_kernel(init_params, num_warmup, progbar=progbar, rng=rngs[0],
                                    **sampler_kwargs)
            samples_flat = fori_collect(0, num_samples, sample_kernel, hmc_state,
                                        transform=lambda x: constrain_fn(x.z),
                                        progbar=progbar,
                                        diagnostics_fn=get_diagnostics_str,
                                        progbar_desc='sample')
            samples = tree_map(lambda x: x[np.newaxis, ...], samples_flat)
        else:
            def single_chain_mcmc(rng, init_params):
                hmc_state = init_kernel(init_params, num_warmup, run_warmup=False, rng=rng,
                                        **sampler_kwargs)
                samples = fori_collect(num_warmup, num_warmup + num_samples, sample_kernel, hmc_state,
                                       transform=lambda x: constrain_fn(x.z),
                                       progbar=progbar)
                return samples

            if num_chains == 1:
                samples_flat = single_chain_mcmc(rngs[0], init_params)
                samples = tree_map(lambda x: x[np.newaxis, ...], samples_flat)
            else:
                if sequential_chain:
                    samples = []
                    for i in range(num_chains):
                        init_params_i = tree_map(lambda x: x[i], init_params)
                        samples.append(jit(single_chain_mcmc)(rngs[i], init_params_i))
                    samples = tree_multimap(lambda *args: np.stack(args), *samples)
                else:
                    samples = pmap(single_chain_mcmc)(rngs, init_params)
                samples_flat = tree_map(lambda x: np.reshape(x, (-1,) + x.shape[2:]), samples)

        if print_summary:
            summary(samples)
        return samples_flat
    else:
        raise ValueError('sampler: {} not recognized'.format(sampler))
