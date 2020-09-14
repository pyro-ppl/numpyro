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
from numpyro.contrib.hmcecs_utils import grad_potential,potential_est,log_density_hmcecs
HMCState = namedtuple('HMCState', ['i', 'z', 'z_grad', 'potential_energy', 'energy', 'num_steps', 'accept_prob',
                                   'mean_accept_prob', 'diverging', 'adapt_state','rng_key'])
HMCECSState = namedtuple("HMCECState",["u","hmc_state","z_ref","ll_ref","jac_all","hess_all","ll_u"])
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
    """Returns the indexes from the subsample that will be updated, there is replacement.
     The number of indexes to be updated depend on the block size, higher block size more correlation among elements in the subsample.
    :param rng_key
    :param u subsample
    :param n total number of data
    :param m subsample size
    :param g block size: subsample subdivision"""
    rng_key_block, rng_key_index = random.split(rng_key)

    # uniformly choose block to update
    chosen_block = random.randint(rng_key, shape=(), minval= 0, maxval=g + 1) #TODO: assertions for g values? why minval=0?division by 0

    idxs_new = random.randint(rng_key_index, shape=(m // g,), minval=0, maxval=n) #chose block within the subsample to update

    u_new = jnp.zeros(m, jnp.dtype(u)) #empty array with size m
    for i in range(m):
        #if index in the subsample // g = chosen block : pick new indexes from the subsample size
        #else not update: keep the same indexes
        u_new = ops.index_add(u_new, i,
                              lax.cond(i // g == chosen_block, i, lambda _: idxs_new[i % (m // g)], i, lambda _: u[i]))
    return u_new


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
                    rng_key=random.PRNGKey(0),
                    u=None,
                    blocks=None,
                    hmc_state = None,
                    z_ref=None,
                    ll_map = None,
                    jac_map = None,
                    hess_map= None,
                    control_variates= None,
                    ll_u=None
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
        #nonlocal n,m,g #TODO: This needs to be activated
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
        z_info = IntegratorState(z=z, potential_energy=pe, z_grad=z_grad)
        wa_state = wa_init(z_info, rng_key_wa, step_size,
                           inverse_mass_matrix=inverse_mass_matrix,
                           mass_matrix_size=jnp.size(ravel_pytree(z)[0]))
        r = momentum_generator(z, wa_state.mass_matrix_sqrt, rng_key_momentum)
        vv_init, vv_update = velocity_verlet(pe_fn, kinetic_fn)
        vv_state = vv_init(z, r, potential_energy=pe, z_grad=z_grad)
        energy = kinetic_fn(wa_state.inverse_mass_matrix, vv_state.r)
        hmc_state = HMCState(0, vv_state.z, vv_state.z_grad, vv_state.potential_energy, energy,
                             0, 0., 0., False, wa_state,rng_key_hmc)
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
        delta_energy = jnp.where(jnp.isnan(delta_energy), jnp.inf, delta_energy)
        accept_prob = jnp.clip(jnp.exp(-delta_energy), a_max=1.0)
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
                           (hmc_state.i, accept_prob, vv_state, hmc_state.adapt_state),
                           lambda args: wa_update(*args),
                           hmc_state.adapt_state,
                           identity)

        itr = hmc_state.i + 1
        n = jnp.where(hmc_state.i < wa_steps, itr, itr - wa_steps)
        mean_accept_prob = hmc_state.mean_accept_prob + (accept_prob - hmc_state.mean_accept_prob) / n

        return HMCState(itr, vv_state.z, vv_state.z_grad, vv_state.potential_energy, energy, num_steps,
                        accept_prob, mean_accept_prob, diverging, adapt_state,rng_key)

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

def _hmcecs_potential(model, model_args, u, control_variates, jac_map, z, z_ref, hess_map, n, m):
    """Estimate the potential dynamic energy for the HMC ECS implementation. The calculation follows section 7.2.1 in https://jmlr.org/papers/volume18/15-205/15-205.pdf
    The computation has a complexity of O(1) and it's highly dependant on the quality of the map estimate"""
    ratio_pop_sub = n / m  # ratio of population size to subsample
    z_flat, _ = ravel_pytree(z)
    zmap_flat, _ = ravel_pytree(z_map)

    _, trace = log_density(model, model_args, {}, z)  # log likelihood for subsample
    z_diff = z_flat - zmap_flat

    control_variates += jac_map.T @ z_diff + .5 * z_diff.T @ hess_map @ (z_flat - 2 * zmap_flat)

    lq_sub = _log_prob(trace) - control_variates[u] #correction of the likelihood based on the difference between the estimation and the map estimate

    d_hat = ratio_pop_sub * jnp.sum(lq_sub)  # assume uniform distribution for subsample!
    l_hat = d_hat + jnp.sum(control_variates)

    lq_sub_mean = jnp.mean(lq_sub)
    sigma = ratio_pop_sub ** 2 * jnp.sum(lq_sub - lq_sub_mean)
    return l_hat - .5 * sigma, control_variates, lq_sub

def _grad_hmcecs_potential(model,model_args, model_kwargs,u, z, z_ref, n, m, jac_map, hess_map, lq_sub):
    ratio_pop_sub = n / m  # ratio of population size to subsample
    z_flat, treedef = ravel_pytree(z)
    zmap_flat, _ = ravel_pytree(z_ref)

    grad_cv = jac_map + hess_map @ (z_flat - zmap_flat)

    grad_lsub, _ = ravel_pytree(jacfwd(lambda args: partial(log_density, model, model_args, model_kwargs)(args)[0])(z)) #jacobian
    grad_lhat = jnp.sum(jac_map, 0) + jnp.sum(hess_map, 0) + ratio_pop_sub * jnp.sum(grad_lsub - grad_cv)

    lq_sub_mean = jnp.mean(lq_sub)
    grad_dhat = grad_lhat - grad_cv - jnp.mean(grad_lhat - grad_cv)

    # Note: factor 2 cancels with 1/2 from grad(L_hat) = grad_lhat - .5 * 2 * ratio_pop_sub**2 * ...
    grad_sigma = ratio_pop_sub ** 2 * (jnp.sum(lq_sub) * grad_dhat - lq_sub_mean * jnp.sum(
        grad_dhat))  # TODO: figure out lq_sub (20,) @ grad_dhat (z.shape)

    return treedef(grad_lhat - grad_sigma) #unflatten tree

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
                 m= None,
                 g = None,
                 z_ref= None
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
        self._algo = 'HMC'
        self._max_tree_depth = 10
        self._init_strategy = init_strategy
        self._find_heuristic_step_size = find_heuristic_step_size
        self._subsample_method = subsample_method
        self._m = m if m is not None else 4
        self._g = g if g is not None else 2
        self._z_ref = z_ref
        self._n = None
        # Set on first call to init
        self._init_fn = None
        self._postprocess_fn = None
        self._sample_fn = None
        self._subsample_fn = None

    def _init_subsample_state(self,rng_key, model_args, model_kwargs, init_params,z_ref):

        self._n = model_args[0].shape[0]

        u = random.randint(rng_key, (self._m,), 0, self._n)

        model_args = self.model_args_sub(u,model_args)
        model_kwargs = self.model_kwargs_sub(u, model_kwargs)

        rng_key_subsample, rng_key_model, rng_key_hmc_init, rng_key_potential, rng_key = random.split(rng_key, 5)

        ld_fn = lambda args: partial(log_density_hmcecs, self._model, model_args, model_kwargs, prior=False)(args)[0]
        print(z_ref["theta"].shape)
        print(z_ref.keys())
        exit()
        ll_ref = ld_fn(z_ref)
        jac_all, _ = ravel_pytree(jacfwd(ld_fn)(z_ref))
        hess_all, _ = ravel_pytree(hessian(ld_fn)(z_ref))
        print(jac_all.shape)
        exit()
        jac_all = jac_all.reshape(self._n, -1).sum(0)
        k, = jac_all.shape
        hess_all = hess_all.reshape(self._n, k, k).sum(0)

        self._potential_fn = lambda model, args, ll_ref, jac_all, z_ref, hess_all, n, m: \
            lambda z: potential_est(model=model, model_args=args, model_kwargs=model_kwargs, ll_ref=ll_ref,
                                    jac_all=jac_all,
                                    z=z, z_ref=z_ref, hess_all=hess_all, n=n, m=m)
        self._grad_potential = lambda model, args, ll_ref, jac_all, z_ref, hess_all, n, m: \
            lambda z: grad_potential(model=model, model_args=args, model_kwargs=model_kwargs, jac_all=jac_all, z=z,
                                     z_ref=z_ref,
                                     hess_all=hess_all, n=n, m=m)


    def _init_state(self, rng_key, model_args, model_kwargs, init_params):
        if self._subsample_method is not None:
            assert self._z_ref is not None, "Please provide a (i.e map) estimate for the parameters"
            self._init_subsample_state(rng_key, model_args, model_kwargs, init_params,self._z_ref)
            self._init_fn, self._subsample_fn = hmc(potential_fn_gen=self._potential_fn,
                                                    kinetic_fn=euclidean_kinetic_energy,
                                                    grad_potential_fn_gen=self._grad_potential,
                                                    algo='HMC')  # no need to be returned here to be used for sampling, because they are init sampler and subsampler are updated...

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
    def model_args_sub(self,u,model_args):
        """Subsample observations and features"""
        args = []
        for arg in model_args:
            if isinstance(arg, jnp.ndarray):
                args.append(jnp.take(arg, u, axis=0))
            else:
                args.append(arg)
        return args
    def model_kwargs_sub(self,u, kwargs):
        """Subsample observations and features"""
        for key_arg, val_arg in kwargs.items():
            if key_arg == "observations" or key_arg == "features":
                kwargs[key_arg] = jnp.take(val_arg, u, axis=0)
        return kwargs

    def _block_indices(self,size, num_blocks):
        a = jnp.repeat(jnp.arange(num_blocks - 1), size // num_blocks)
        b = jnp.repeat(num_blocks - 1, size - len(jnp.repeat(jnp.arange(num_blocks - 1), size // num_blocks)))
        return jnp.hstack((a, b))


    def init(self, rng_key, num_warmup, init_params=None, model_args=(), model_kwargs={}):
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
        elif self._subsample_method:
            init_state = vmap(hmc_init_fn)(init_params, rng_key)
            subsample_fn = vmap(self._subsample_fn, in_axes=(0, None, None))
            self._subsample_fn = subsample_fn
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
        return self._sample_fn(state, model_args, model_kwargs)
    def subsample(self,subsamplestate,model_args,model_kwargs):
        """
          Run HMC from the given :data:`~numpyro.infer.hmc.HMCECSState` and return the resulting
          :data:`~numpyro.infer.hmc.HMCECSState`.

          :param HMCECSState state: Represents the current state.
          :param model_args: Arguments provided to the model.
          :param model_kwargs: Keyword arguments provided to the model.
          :return: Next `subsample state` after running HMC.
          """

        rng_key_subsample, rng_key_transition, rng_key_likelihood, rng_key = random.split(subsamplestate.hmc_state.rng_key,4)

        u_new = _update_block(rng_key_subsample, subsamplestate.u, self._n, self._m, self._g)

        # estimate likelihood of subsample with single block updated
        llu_new = potential_est(model=self._model,
                                model_args=model_args,
                                model_kwargs=model_kwargs,
                                jac_all=subsamplestate.jac_all,
                                hess_all=subsamplestate.hess_all,
                                ll_ref=subsamplestate.ll_ref,
                                z=subsamplestate.hmc_state.z,
                                z_ref=subsamplestate.z_ref,
                                n=self._n, m=self._m)

        # accept new subsample with probability min(1,L^{hat}_{u_new}(z) - L^{hat}_{u}(z))
        # NOTE: latent variables (z aka theta) same, subsample indices (u) different by one block.
        accept_prob = jnp.clip(jnp.exp(-llu_new + subsamplestate.ll_u), a_max=1.)
        transition = random.bernoulli(rng_key_transition, accept_prob)
        u, ll_u = cond(transition,
                       (u_new, llu_new), identity,
                       (subsamplestate.u, subsamplestate.ll_u), identity)

        ######## UPDATE PARAMETERS ##########


        hmc_subsamplestate= HMCECSState(u=u, hmc_state=subsamplestate.hmc_state,
                                        z_ref=subsamplestate.z_ref,
                                        ll_u=ll_u,ll_ref=subsamplestate.ll_ref,
                                        jac_all=subsamplestate.jac_all,
                                        hess_all=subsamplestate.hess_all)

        return self._subsample_fn(hmc_subsamplestate,model_args=(self._model,
                                                               model_args,
                                                               subsamplestate.ll_ref,
                                                               subsamplestate.jac_all,
                                                               subsamplestate.z_ref,
                                                               subsamplestate.hess_all, self._n, self._m),model_kwargs=model_kwargs)

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
