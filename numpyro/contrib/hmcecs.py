"""Contributed code for HMC and NUTS energy conserving sampling adapted from <Hamiltonian Monte Carlo with Energy Conserving Subsampling>"""

from collections import namedtuple
import math
import os
import warnings

from jax import device_put, lax, partial, random, vmap,jacfwd, hessian,jit,ops
from jax.dtypes import canonicalize_dtype
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp

from numpyro.infer.hmc_util import (
    IntegratorState,
    build_tree,
    euclidean_kinetic_energy,
    find_reasonable_step_size,
    velocity_verlet,
    warmup_adapter
)
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import ParamInfo, init_to_uniform, initialize_model, log_density
from numpyro.util import cond, fori_loop, identity
import sys
sys.path.append('/home/lys/Dropbox/PhD/numpyro/numpyro/contrib/')     #TODO: remove
import numpyro.distributions as dist
from itertools import chain
from hmcecs_utils import potential_est, init_near_values,tuplemerge,\
                        model_args_sub,model_kwargs_sub,taylor_proxy,svi_proxy,neural_proxy,log_density_obs_hmcecs,log_density_prior_hmcecs,signed_estimator

HMCState = namedtuple('HMCState', ['i', 'z', 'z_grad', 'potential_energy', 'energy', 'num_steps', 'accept_prob',
                                   'mean_accept_prob', 'diverging', 'adapt_state','rng_key'])
#HMCECSState = namedtuple("HMCECState",["u","hmc_state","z_ref","ll_ref","jac_all","hess_all","ll_u"])

HMCECSState = namedtuple("HMCECState",['u', 'hmc_state', 'll_u'])

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
 - **u** - Subsample 
 - **blocks** - blocks in which the subsample is divided
 - **z_ref** - MAP estimation of the model parameters to initialize the subsampling.
 - **ll_map** - Log likelihood of the map estimated parameters.
 - **jac_map** - Jacobian vector from the map estimated parameters.
 - **hess_map** - Hessian matrix from the map estimated parameters
 - **Control variates** - Log likelihood correction
 - **ll_u** - Log likelihood of the subsample
"""


def _get_num_steps(step_size, trajectory_length):
    num_steps = jnp.clip(trajectory_length / step_size, a_min=1)
    # NB: casting to jnp.int64 does not take effect (returns jnp.int32 instead)
    # if jax_enable_x64 is False
    return num_steps.astype(canonicalize_dtype(jnp.int64))


def momentum_generator(prototype_r, mass_matrix_sqrt, rng_key):
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

@partial(jit, static_argnums=(2, 3, 4))
def _update_block(rng_key, u, n, m, g):
    """Returns indexes of the new subsample. The update mechanism selects blocks of indices within the subsample to be updated.
     The number of indexes to be updated depend on the block size, higher block size more correlation among elements in the subsample.
    :param rng_key
    :param u subsample indexes
    :param n total number of data
    :param m subsample size
    :param g block size: subsample subdivision"""

    if (g > m) or (g < 1):
        raise ValueError('Block size (g) = {} needs to = or > than 1 and smaller than the subsample size {}'.format(g,m))
    rng_key_block, rng_key_index = random.split(rng_key)
    # uniformly choose block to update
    chosen_block = random.randint(rng_key, shape=(), minval= 0, maxval=g + 1)
    idxs_new = random.randint(rng_key_index, shape=(m // g,), minval=0, maxval=n) #choose block within the subsample to update
    u_new = jnp.zeros(m, jnp.dtype(u)) #empty array with size m
    for i in range(m):
        #if index in the subsample // g = chosen block : pick new indexes from the subsample size
        #else not update: keep the same indexes
        u_new = ops.index_add(u_new, i,
                              lax.cond(i // g == chosen_block, i, lambda _: idxs_new[i % (m // g)], i, lambda _: u[i]))
    return u_new

def _sample_u_poisson(rng_key, m, l):
    """ Initialize subsamples u
    ***References***
    1.Hamiltonian Monte Carlo with Energy Conserving Subsampling
    2.The blockPoisson estimator for optimally tuned exact subsampling MCMC.
    :param m: subsample size
    :param l: lambda u blocks
    :param g: number of blocks
    """
    pois_key, sub_key = random.split(rng_key)
    block_lengths = dist.discrete.Poisson(1).sample(pois_key, (l,)) #lambda block lengths
    #u = random.randint(sub_key, (jnp.sum(block_lengths), ), 0, m)
    u = random.randint(sub_key, (jnp.sum(block_lengths), m), 0, m)
    return jnp.split(u, jnp.cumsum(block_lengths), axis=0)

@partial(jit, static_argnums=(2, 3, 4))
def _update_block_poisson(rng_key, u, m, l, g):
    """ Update block of u, where the length of the block of indexes to update is given by the Poisson distribution.
    ***References***
    1.Hamiltonian Monte Carlo with Energy Conserving Subsampling
    2.The blockPoisson estimator for optimally tuned exact subsampling MCMC.
    :param rng_key
    :param u: current subsample indexes
    :param m: Subsample size
    :param l: lambda
    :param g: Block size within subsample
    """
    if (g > m) or (g < 1):
        raise ValueError('Block size (g) = {} needs to = or > than 1 and smaller than the subsample size {}'.format(g,m))
    u = u.copy()
    block_key, sample_key = random.split(rng_key)
    num_updates = int(round(l / g, 0)) # choose lambda/g number of blocks to update
    chosen_blocks = random.randint(block_key, (num_updates,), 0, l)
    new_blocks = _sample_u_poisson(sample_key, m, num_updates)
    for i, block in enumerate(chosen_blocks):
        u[block] = new_blocks[i]
    return u


def hmc(potential_fn=None, potential_fn_gen=None, kinetic_fn=None, grad_potential_fn_gen=None,algo='NUTS'):
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
        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.infer.hmc import hmc
        >>> from numpyro.infer.util import initialize_model
        >>> from numpyro.util import fori_collect

        >>> true_coefs = jnp.array([1., 2., 3.])
        >>> data = random.normal(random.PRNGKey(2), (2000, 3))
        >>> dim = 3
        >>> labels = dist.Bernoulli(logits=(true_coefs * data).sum(-1)).sample(random.PRNGKey(3))
        >>>
        >>> def model(data, labels):
        ...     coefs_mean = jnp.zeros(dim)
        ...     coefs = numpyro.sample('beta', dist.Normal(coefs_mean, jnp.ones(3)))
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
        >>> print(jnp.mean(samples['beta'], axis=0))  # doctest: +SKIP
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
                    model = None,
                    ll_ref=None,
                    jac_all=None,
                    z_ref= None,
                    hess_all=None,
                    ll_u = None,
                    n = None,
                    m = None,
                    u= None,
                    l=None,
                    rng_key=random.PRNGKey(0),
                    subsample_method=None,
                    estimator=None,
                    proxy_fn=None,
                    proxy_u_fn = None):
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
        step_size = lax.convert_element_type(step_size, canonicalize_dtype(jnp.float64))
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
                if subsample_method == "perturb":
                    kwargs = {} if model_kwargs is None else model_kwargs
                    if estimator == "poisson":
                        pe_fn = potential_fn_gen(model=model, model_args=model_args, model_kwargs=kwargs, z=z, l=l,proxy_fn=proxy_fn, proxy_u_fn=proxy_u_fn)
                    else:
                        pe_fn = potential_fn_gen(model=model, model_args=model_args, model_kwargs=kwargs, z=z, n=n, m=m,proxy_fn=proxy_fn, proxy_u_fn=proxy_u_fn)

                else:
                    kwargs = {} if model_kwargs is None else model_kwargs
                    pe_fn = potential_fn_gen(*model_args, **kwargs)
        if grad_potential_fn_gen:
            kwargs = {} if model_kwargs is None else model_kwargs
            gpe_fn = grad_potential_fn_gen(*model_args, **kwargs)
        else:
            gpe_fn = None

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
        z_info = IntegratorState(z=z, potential_energy=pe, z_grad=z_grad)
        wa_state = wa_init(z_info, rng_key_wa, step_size,
                           inverse_mass_matrix=inverse_mass_matrix,
                           mass_matrix_size=jnp.size(ravel_pytree(z)[0]))

        r = momentum_generator(z, wa_state.mass_matrix_sqrt, rng_key_momentum)
        #vv_init, vv_update = velocity_verlet_hmcecs(pe_fn, kinetic_fn,grad_potential_fn=gpe_fn)
        vv_init, vv_update = velocity_verlet(pe_fn, kinetic_fn)

        vv_state = vv_init(z, r, potential_energy=pe, z_grad=z_grad)

        energy = kinetic_fn(wa_state.inverse_mass_matrix, vv_state.r)

        hmc_state = HMCState(0, vv_state.z, vv_state.z_grad, vv_state.potential_energy, energy,
                             0, 0., 0., False, wa_state,rng_key_hmc)
        hmc_sub_state = HMCECSState(u=u, hmc_state=hmc_state,ll_u=ll_u)

        hmc_state = tuplemerge(hmc_sub_state._asdict(),hmc_state._asdict())


        return device_put(hmc_state)

    def _hmc_next(step_size, inverse_mass_matrix, vv_state,
                  model_args, model_kwargs, rng_key,subsample_method,
                  estimator=None,
                  proxy_fn = None,
                  proxy_u_fn = None,
                  model = None,
                  ll_ref = None,
                  jac_all = None,
                  z = None,
                  z_ref = None,
                  hess_all = None,
                  ll_u = None,
                  u = None,
                  n = None,
                  m = None,
                  l=None):
        if potential_fn_gen:
            if grad_potential_fn_gen:
                kwargs = {} if model_kwargs is None else model_kwargs
                gpe_fn = grad_potential_fn_gen(*model_args, **kwargs, )
                pe_fn = potential_fn_gen(*model_args, **model_kwargs)

            else:
                if subsample_method == "perturb":
                    if estimator == "poisson":
                        pe_fn = potential_fn_gen(model=model,
                                                 model_args=model_args,
                                                 model_kwargs=model_kwargs,
                                                 z=vv_state.z,
                                                 l=l,
                                                 proxy_fn=proxy_fn,
                                                 proxy_u_fn=proxy_u_fn)

                    else:
                        pe_fn = potential_fn_gen(model=model,
                                                 model_args=model_args,
                                                 model_kwargs=model_kwargs,
                                                 z=vv_state.z,
                                                 n=n,
                                                 m=m,
                                                 proxy_fn=proxy_fn,
                                                 proxy_u_fn=proxy_u_fn)
                    kwargs = {} if model_kwargs is None else model_kwargs
                else:
                    pe_fn = potential_fn_gen(*model_args, **model_kwargs)
            nonlocal vv_update
            _, vv_update = velocity_verlet(pe_fn, kinetic_fn)

        num_steps = _get_num_steps(step_size, trajectory_len)

        vv_state_new = fori_loop(0, num_steps,
                                 lambda i, val: vv_update(step_size, inverse_mass_matrix, val),
                                 vv_state)
        energy_old = vv_state.potential_energy + kinetic_fn(inverse_mass_matrix, vv_state.r)
        energy_new = vv_state_new.potential_energy + kinetic_fn(inverse_mass_matrix, vv_state_new.r)
        delta_energy = energy_new - energy_old
        delta_energy = jnp.where(jnp.isnan(delta_energy), jnp.inf, delta_energy)
        accept_prob = jnp.clip(jnp.exp(-delta_energy), a_max=1.0)
        diverging = delta_energy > max_delta_energy
        transition = random.bernoulli(rng_key, accept_prob)
        vv_state, energy = cond(transition,
                                (vv_state_new, energy_new), identity,
                                (vv_state, energy_old), identity)

        return vv_state, energy, num_steps, accept_prob, diverging

    def _nuts_next(step_size, inverse_mass_matrix, vv_state,
                   model_args, model_kwargs, rng_key,subsample_method,
                   estimator=None,
                   proxy_fn=None,proxy_u_fn=None,
                   model=None,
                   ll_ref=None,jac_all=None,z = None,z_ref=None,hess_all=None,ll_u=None,u=None,
                   n=None,m=None,l=None):
        if potential_fn_gen:
            nonlocal vv_update
            if grad_potential_fn_gen:
                    kwargs = {} if model_kwargs is None else model_kwargs
                    gpe_fn = grad_potential_fn_gen(*model_args, **kwargs, )
                    pe_fn = potential_fn_gen(*model_args, **model_kwargs)
            else:
                if subsample_method == "perturb":
                    if estimator == "poisson":
                        pe_fn = potential_fn_gen(model=model,
                                                 model_args=model_args,
                                                 model_kwargs=model_kwargs,
                                                 z=vv_state.z,
                                                 l=l,
                                                 proxy_fn=proxy_fn,
                                                 proxy_u_fn=proxy_u_fn)



                    else:
                        pe_fn = potential_fn_gen(model=model,
                                                 model_args=model_args,
                                                 model_kwargs=model_kwargs,
                                                 z=vv_state.z,
                                                 n=n,
                                                 m=m,
                                                 proxy_fn=proxy_fn,
                                                 proxy_u_fn=proxy_u_fn)
                else:
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

    def sample_kernel(hmc_state,model_args=(),model_kwargs=None,
                      subsample_method=None,
                      estimator = None,
                      proxy_fn=None,
                      proxy_u_fn=None,
                      model=None,
                      ll_ref=None,
                      jac_all=None,
                      z=None,
                      z_ref=None,
                      hess_all=None,
                      ll_u=None,
                      u=None,n=None,m=None,l=None):
        """
        Given an existing :data:`~numpyro.infer.mcmc.HMCState`, run HMC with fixed (possibly adapted)
        step size and return a new :data:`~numpyro.infer.mcmc.HMCState`.

        :param hmc_state: Current sample (and associated state).
        :param tuple model_args: Model arguments if `potential_fn_gen` is specified.
        :param dict model_kwargs: Model keyword arguments if `potential_fn_gen` is specified.
        :param subsample_method: Indicates if hmc energy conserving method shall be implemented for subsampling
        :param proxy_fn
        :param proxy_u_fn
        :param model
        :param ll_ref
        :param jac_all
        :param z
        :param z_ref
        :param hess_all
        :param ll_u
        :param u
        :param n
        :param m
        :param l : lambda value for block poisson estimator method. Indicates the number of subsamples within a subsample
        :return: new proposed :data:`~numpyro.infer.mcmc.HMCState` from simulating
            Hamiltonian dynamics given existing state.

        """

        model_kwargs = {} if model_kwargs is None else model_kwargs
        if subsample_method =="perturb":
            if estimator == "poisson":
                model_args = [model_args_sub(u_i, model_args) for u_i in u] #here u = poisson_u
            else:
                model_args = model_args_sub(u,model_args)
        rng_key, rng_key_momentum, rng_key_transition = random.split(hmc_state.rng_key, 3)
        r = momentum_generator(hmc_state.z, hmc_state.adapt_state.mass_matrix_sqrt, rng_key_momentum)
        vv_state = IntegratorState(hmc_state.z, r, hmc_state.potential_energy, hmc_state.z_grad)

        vv_state, energy, num_steps, accept_prob, diverging = _next(hmc_state.adapt_state.step_size,
                                                                    hmc_state.adapt_state.inverse_mass_matrix,
                                                                    vv_state,
                                                                    model_args,
                                                                    model_kwargs,
                                                                    rng_key_transition,
                                                                    subsample_method,
                                                                    estimator,
                                                                    proxy_fn,
                                                                    proxy_u_fn,
                                                                    model,
                                                                    ll_ref,jac_all,z,z_ref,hess_all,ll_u,u,
                                                                    n,m,l)
        # not update adapt_state after warmup phase
        adapt_state = cond(hmc_state.i < wa_steps,
                           (hmc_state.i, accept_prob, vv_state, hmc_state.adapt_state),
                           lambda args: wa_update(*args),
                           hmc_state.adapt_state,
                           identity)

        itr = hmc_state.i + 1
        n = jnp.where(hmc_state.i < wa_steps, itr, itr - wa_steps)
        mean_accept_prob = hmc_state.mean_accept_prob + (accept_prob - hmc_state.mean_accept_prob) / n
        hmcstate = HMCState(itr, vv_state.z, vv_state.z_grad, vv_state.potential_energy, energy, num_steps,
                        accept_prob, mean_accept_prob, diverging, adapt_state,rng_key)
        hmc_sub_state = HMCECSState(u=u, hmc_state=hmc_state,ll_u=ll_u)
        hmcstate = tuplemerge(hmc_sub_state._asdict(),hmcstate._asdict())
        return hmcstate

    # Make `init_kernel` and `sample_kernel` visible from the global scope once
    # `hmc` is called for sphinx doc generation.
    if 'SPHINX_BUILD' in os.environ:
        hmc.init_kernel = init_kernel
        hmc.sample_kernel = sample_kernel

    return init_kernel, sample_kernel

def _log_prob(trace):
    """ Compute probability of each observation """
    node = trace['observations']
    return jnp.sum(node['fn'].log_prob(node['value']), 1)


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
    :param subsample_method If "perturb" is provided, the "potential_fn" function will be calculated
        using the equations from section 7.2.1 in https://jmlr.org/papers/volume18/15-205/15-205.pdf
    :param m subsample size
    :param g block size
    :param z_ref MAP estimate of the parameters
    :param covariate_fn Proxy function to calculate the covariates for the likelihood correction
    """
    def __init__(self,
                 model=None,
                 potential_fn=None,
                 grad_potential = None,
                 kinetic_fn=None,
                 step_size=1.0,
                 adapt_step_size=True,
                 adapt_mass_matrix=True,
                 dense_mass=False,
                 target_accept_prob=0.8,
                 trajectory_length=2 * math.pi,
                 init_strategy=init_to_uniform,
                 find_heuristic_step_size=False,
                 subsample_method = None,
                 estimator=None,  # poisson or not
                 proxy="taylor",
                 svi_fn=None,
                 m= None,
                 g = None,
                 z_ref= None,
                 algo = "HMC"
                 ):
        if not (model is None) ^ (potential_fn is None):
            raise ValueError('Only one of `model` or `potential_fn` must be specified.')

        self._model = model
        self._potential_fn = potential_fn
        self._grad_potential = grad_potential
        self._kinetic_fn = kinetic_fn if kinetic_fn is not None else euclidean_kinetic_energy
        self._step_size = step_size
        self._adapt_step_size = adapt_step_size
        self._adapt_mass_matrix = adapt_mass_matrix
        self._dense_mass = dense_mass
        self._target_accept_prob = target_accept_prob
        self._trajectory_length = trajectory_length
        self._algo = algo
        self._max_tree_depth = 10
        self._init_strategy = init_strategy
        self._find_heuristic_step_size = find_heuristic_step_size
        #HMCECS parameters
        self.subsample_method = subsample_method
        self.m = m if m is not None else 4
        self.g = g if g is not None else 2
        self.z_ref = z_ref
        self._n = None
        self._ll_ref = None
        self._jac_all = None
        self._hess_all = None
        self._ll_u = None
        self._u = None
        self._sign = None
        self._l = 100
        # Set on first call to init
        self._init_fn = None
        self._postprocess_fn = None
        self._sample_fn = None
        self._subsample_fn = None
        self._sign = []
        self.proxy = proxy
        self.svi_fn = svi_fn
        self._proxy_fn = None
        self._proxy_u_fn = None
        self._signed_estimator_fn = None
        self.estimator = estimator

    def _init_subsample_state(self,rng_key, model_args, model_kwargs, init_params,z_ref):
        "Compute the jacobian, hessian and log likelihood for all the data. Used with taylor expansion proxy"
        rng_key_subsample, rng_key_model, rng_key_hmc_init, rng_key_potential, rng_key = random.split(rng_key, 5)

        ld_fn = lambda args: jnp.sum(partial(log_density_obs_hmcecs, self._model, model_args, model_kwargs)(args)[0])
        self._jac_all, _ = ravel_pytree(jacfwd(ld_fn)(z_ref))
        hess_all, _ = ravel_pytree(hessian(ld_fn)(z_ref))
        k, = self._jac_all.shape
        self._hess_all = hess_all.reshape((k, k))
        ld_fn = lambda args: partial(log_density_obs_hmcecs,self._model,model_args,model_kwargs)(args)[0]
        self._ll_ref = ld_fn(z_ref)


    def _init_state(self, rng_key, model_args, model_kwargs, init_params):
        if self.subsample_method is not None:
            assert self.z_ref is not None, "Please provide a (i.e map) estimate for the parameters"
            self._n = model_args[0].shape[0]
            # Choose the covariate calculation method
            if self.proxy == "svi":
                self._proxy_fn,self._proxy_u_fn = svi_proxy(self.svi_fn,model_args,model_kwargs)
            elif self.proxy == "taylor":
                warnings.warn("Using default second order Taylor expansion, change by using the proxy flag to {svi}")
                self._init_subsample_state(rng_key, model_args, model_kwargs, init_params, self.z_ref)
                self._proxy_fn,self._proxy_u_fn = taylor_proxy(self.z_ref, self._model, self._ll_ref, self._jac_all, self._hess_all)
            if self.estimator =="poisson":
                self._l = 25 # lambda subsamples
                self._u = _sample_u_poisson(rng_key, self.m, self._l)

                #TODO: Confirm that the signed estimator is the new potential function---> If so the output has to be fixed
                self._potential_fn = lambda model,model_args,model_kwargs,z,l, proxy_fn,proxy_u_fn : lambda z:signed_estimator(model = model,model_args=model_args,
                                                                                                                               model_kwargs= model_kwargs,z=z,l=l,proxy_fn=proxy_fn,
                                                                                                                               proxy_u_fn=proxy_u_fn)[0]
                # Initialize the hmc sampler: sample_fn = sample_kernel
                self._init_fn, self._sample_fn = hmc(potential_fn_gen=self._potential_fn,
                                                     kinetic_fn=euclidean_kinetic_energy,
                                                     algo=self._algo)

                self._init_strategy = partial(init_near_values, values=self.z_ref)
                # Initialize the model parameters
                rng_key_init_model, rng_key = random.split(rng_key)
                model_args = [model_args_sub(u_i, model_args) for u_i in self._u] #TODO: The initialization function has to be initialized on a subsample

                #model_args = list(chain(*model_args))    #Highlight: This just chains all the elements in the sublist , len(lists_of_lists) = n , len(chain(list_of_lists)) = sum(n_elements_inside_list=*n
                self._init_strategy = partial(init_near_values, values=self.z_ref)
                init_params, potential_fn, postprocess_fn, model_trace = initialize_model(
                    rng_key_init_model,
                    self._model,
                    init_strategy=self._init_strategy,
                    dynamic_args=True,
                    model_args=tuple([arg[0] for arg in next(chain(model_args))]), #Pick the first non-empty block
                    model_kwargs=model_kwargs)

            else:
                self._u = random.randint(rng_key, (self.m,), 0, self._n)
                # Initialize the potential and gradient potential functions
                self._potential_fn = lambda model, model_args, model_kwargs, z, n, m, proxy_fn, proxy_u_fn : lambda  z:potential_est(model=model,
                                    model_args=model_args, model_kwargs=model_kwargs, z=z, n=n, m=m, proxy_fn=proxy_fn, proxy_u_fn=proxy_u_fn)

                # Initialize the hmc sampler: sample_fn = sample_kernel
                self._init_fn, self._sample_fn = hmc(potential_fn_gen=self._potential_fn,
                                                        kinetic_fn=euclidean_kinetic_energy,
                                                        algo=self._algo)


                self._init_strategy = partial(init_near_values, values=self.z_ref)
                # Initialize the model parameters
                rng_key_init_model, rng_key = random.split(rng_key)

                init_params, potential_fn, postprocess_fn, model_trace = initialize_model(
                    rng_key_init_model,
                    self._model,
                    init_strategy=self._init_strategy,
                    dynamic_args=True,
                    model_args=model_args_sub(self._u, model_args),
                    model_kwargs=model_kwargs)

            if (self.g > self.m) or (self.g < 1):
                    raise ValueError(
                        'Block size (g) = {} needs to = or > than 1 and smaller than the subsample size {}'.format(self.g,
                                                                                                                   self.m))
            elif (self.m > self._n):
                    raise ValueError(
                        'Subsample size (m) = {} needs to = or < than data size (n) {}'.format(self.m, self._n))

        else:
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

    @property
    def sample_field(self):
        return 'z'

    @property
    def default_fields(self):
        return ('z', 'diverging')

    def get_diagnostics_str(self, state):
        return '{} steps of size {:.2e}. acc. prob={:.2f}'.format(state.num_steps,
                                                                  state.adapt_state.step_size,
                                                                  state.mean_accept_prob)

    def _block_indices(self,size, num_blocks):
        a = jnp.repeat(jnp.arange(num_blocks - 1), size // num_blocks)
        b = jnp.repeat(num_blocks - 1, size - len(jnp.repeat(jnp.arange(num_blocks - 1), size // num_blocks)))
        return jnp.hstack((a, b))

    def init(self, rng_key, num_warmup, init_params=None, model_args=(), model_kwargs={}):
        """Initialize sampling algorithms"""
        # non-vectorized
        if rng_key.ndim == 1:
            rng_key, rng_key_init_model = random.split(rng_key)
        # vectorized
        else:
            rng_key, rng_key_init_model = jnp.swapaxes(vmap(random.split)(rng_key), 0, 1)


        init_params = self._init_state(rng_key_init_model, model_args, model_kwargs, init_params) #should work  for all cases

        if self._potential_fn and init_params is None:
            raise ValueError('Valid value of `init_params` must be provided with'
                             ' `potential_fn`.')
        if self.subsample_method == "perturb":
            if self.estimator == "poisson":
                init_model_args = [model_args_sub(u_i, model_args) for u_i in self._u]
            else:
                init_model_args = model_args_sub(self._u,model_args)
            hmc_init_fn = lambda init_params,rng_key: self._init_fn(init_params=init_params,
                                          num_warmup = num_warmup,
                                          step_size = self._step_size,
                                          adapt_step_size = self._adapt_step_size,
                                          adapt_mass_matrix = self._adapt_mass_matrix,
                                          dense_mass = self._dense_mass,
                                          target_accept_prob = self._target_accept_prob,
                                          trajectory_length=self._trajectory_length,
                                          max_tree_depth=self._max_tree_depth,
                                          find_heuristic_step_size=self._find_heuristic_step_size,
                                          model_args=init_model_args,
                                          model_kwargs=model_kwargs,
                                          subsample_method= self.subsample_method,
                                          estimator= self.estimator,
                                          model=self._model,
                                          ll_ref =self._ll_ref,
                                          jac_all=self._jac_all,
                                          z_ref=self.z_ref,
                                          hess_all = self._hess_all,
                                          ll_u = self._ll_u,
                                          n=self._n,
                                          m=self.m,
                                          u = self._u,
                                          l = self._l,
                                          proxy_fn = self._proxy_fn,
                                          proxy_u_fn = self._proxy_u_fn)

            if rng_key.ndim ==1:
                #rng_key_hmc_init = jnp.array([1000966916, 171341646])
                rng_key_hmc_init,_ = random.split(rng_key)

                init_state = hmc_init_fn(init_params, rng_key_hmc_init) #HMCState + HMCECSState
                if self.estimator == "poisson":
                    #signed pseudo-marginal algorithm with the block-Poisson estimator
                    #use the term signed PM for any pseudo-marginal algorithm that uses the technique in Lyne
                    # et al. (2015) where a pseudo-marginal sampler is run on the absolute value of the estimated
                    # posterior and subsequently sign-corrected by importance sampling. Similarly, we call the
                    # algorithm described in this section signed HMC-ECS
                    model_args = [model_args_sub(u_i, model_args)for u_i in self._u]
                    neg_ll, sign = signed_estimator(self._model,
                                                    model_args,
                                                    model_kwargs,
                                                    init_state.z,
                                                    self._l,
                                                    self._proxy_fn,
                                                    self._proxy_u_fn)
                    self._sign.append(sign)
                    self._ll_u = neg_ll

                else:
                    self._ll_u = potential_est(model=self._model,
                                               model_args=model_args_sub(self._u, model_args),
                                               model_kwargs=model_kwargs,
                                               z=init_state.z,
                                               n=self._n,
                                               m=self.m,
                                               proxy_fn=self._proxy_fn,
                                               proxy_u_fn=self._proxy_u_fn)
                hmc_init_sub_state =  HMCECSState(u=self._u,
                                                  hmc_state=init_state.hmc_state,
                                                  ll_u=self._ll_u)
                init_sub_state  = tuplemerge(init_state._asdict(),hmc_init_sub_state._asdict())

                return init_sub_state
            else: #TODO: What is this for? It does not go into it for num_chains>1
                raise ValueError("Not implemented for chains > 1")
                # XXX it is safe to run hmc_init_fn under vmap despite that hmc_init_fn changes some
                # nonlocal variables: momentum_generator, wa_update, trajectory_len, max_treedepth,
                # wa_steps because those variables do not depend on traced args: init_params, rng_key.
                init_state = vmap(hmc_init_fn)(init_params, rng_key)
                if self.estimator == "poisson":
                    model_args = [model_args_sub(u_i, model_args)for u_i in self._u]
                    neg_ll, sign = signed_estimator(self._model,
                                                    model_args,
                                                    model_kwargs,
                                                    init_state.z,
                                                    self._l,
                                                    self._proxy_fn,
                                                    self._proxy_u_fn)
                    self._sign.append(sign)
                    self._ll_u = neg_ll

                else:
                    self._ll_u = potential_est(model=self._model,
                                               model_args=model_args_sub(self._u, model_args),
                                               model_kwargs=model_kwargs,
                                               z=init_state.z,
                                               n=self._n,
                                               m=self.m,
                                               proxy_fn=self._proxy_fn,
                                               proxy_u_fn=self._proxy_u_fn)

                hmc_init_sub_fn = lambda init_params, rng_key: HMCECSState(u=self._u, hmc_state=init_state, ll_u=self._ll_u)

                init_subsample_state = vmap(hmc_init_sub_fn)(init_params,rng_key)

                sample_fn = vmap(self._sample_fn, in_axes=(0, None, None))
                HMCCombinedState = tuplemerge(init_state._asdict,init_subsample_state._asdict())
                self._sample_fn = sample_fn
                return HMCCombinedState

        else:
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
                return init_state
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

        if self.subsample_method == "perturb":
            rng_key_subsample, rng_key_transition, rng_key_likelihood, rng_key = random.split(
                state.rng_key, 4)
            if self.estimator == "poisson":
                #TODO: What to do here? does the negative likelihood need to be stored? how about the sign? store in the state?
                u_new = _sample_u_poisson(rng_key, self.m, self._l)
                neg_ll, sign = signed_estimator(model = self._model,
                                                model_args=[model_args_sub(u_i, model_args) for u_i in u_new],
                                                model_kwargs=model_kwargs,
                                                z=state.z,
                                                l =self._l,
                                                proxy_fn = self._proxy_fn,
                                                proxy_u_fn = self._proxy_u_fn)
                self._sign.append(sign)
                # Correct the negativeloglikelihood by substracting the density of the prior to calculate the potential
                llu_new = jnp.min(jnp.array([0, -neg_ll + state.ll_u]))

            else:
                u_new = _update_block(rng_key_subsample, state.u, self._n, self.m, self.g)
                # estimate likelihood of subsample with single block updated
                llu_new = self._potential_fn(model=self._model,
                                        model_args=model_args_sub(u_new,model_args),
                                        model_kwargs=model_kwargs,
                                        z=state.z,
                                        n=self._n,
                                        m=self.m,
                                        proxy_fn=self._proxy_fn,
                                        proxy_u_fn=self._proxy_u_fn)
            # accept new subsample with probability min(1,L^{hat}_{u_new}(z) - L^{hat}_{u}(z))
            # NOTE: latent variables (z aka theta) same, subsample indices (u) different by one block.
            accept_prob = jnp.clip(jnp.exp(-llu_new + state.ll_u), a_max=1.)
            transition = random.bernoulli(rng_key_transition, accept_prob)  #TODO: Why Bernouilli instead of Uniform?
            u, ll_u = cond(transition,
                           (u_new, llu_new), identity,
                           (state.u, state.ll_u), identity)


            ######## UPDATE PARAMETERS ##########

            hmc_subsamplestate = HMCECSState(u=u, hmc_state=state.hmc_state,ll_u=ll_u)
            hmc_subsamplestate = tuplemerge(hmc_subsamplestate._asdict(),state._asdict())

            return self._sample_fn(hmc_subsamplestate,
                                   model_args=model_args,
                                   model_kwargs=model_kwargs,
                                   subsample_method=self.subsample_method,
                                   estimator =self.estimator,
                                   proxy_fn = self._proxy_fn,
                                   proxy_u_fn = self._proxy_u_fn,
                                   model = self._model,
                                   ll_ref = self._ll_ref,
                                   jac_all =self._jac_all,
                                   z= state.z,
                                   z_ref = self.z_ref,
                                   hess_all = self._hess_all,
                                   ll_u = ll_u,
                                   u= u,
                                   n= self._n,
                                   m= self.m,
                                   l=self._l)






        else:
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
