from functools import partial

import jax
import jax.numpy as jnp
from jax import grad, value_and_grad
from jax.tree_util import tree_multimap
import numpyro
import numpyro.distributions as dist
from numpyro.distributions.util import is_identically_one
from numpyro.handlers import substitute, trace
from numpyro.util import ravel_pytree
from collections import namedtuple

IntegratorState = namedtuple('IntegratorState', ['z', 'r', 'potential_energy', 'z_grad'])
IntegratorState.__new__.__defaults__ = (None,) * len(IntegratorState._fields)


def model_args_sub(u, model_args):
    """Subsample observations and features according to u subsample indexes"""
    args = []
    for arg in model_args:
        if isinstance(arg, jnp.ndarray) and arg.shape[0] > len(u):
            args.append(jnp.take(arg, u, axis=0))
        else:
            args.append(arg)
    return tuple(args)


def model_kwargs_sub(u, kwargs):
    """Subsample observations and features"""
    for key_arg, val_arg in kwargs.items():
        if key_arg == "observations" or key_arg == "features":
            kwargs[key_arg] = jnp.take(val_arg, u, axis=0)
    return kwargs

def log_density_hmcecs(model, model_args, model_kwargs, params,prior=False):
    """
    (EXPERIMENTAL INTERFACE) Computes log of joint density for the model given
    latent values ``params``. If prior == False, the log probability of the prior probability
    over the parameters is not computed, solely the log probability of the observations

    :param model: Python callable containing NumPyro primitives.
    :param tuple model_args: args provided to the model.
    :param dict model_kwargs: kwargs provided to the model.
    :param dict params: dictionary of current parameter values keyed by site
        name.
    :return: log of joint density and a corresponding model trace
    """

    model = substitute(model, data=params)
    model_trace = trace(model).get_trace(*model_args, **model_kwargs)
    log_joint = jnp.array(0.)
    if not prior:
        for site in model_trace.values():
            if site['type'] == 'sample' and site['is_observed'] and not isinstance(site['fn'], dist.PRNGIdentity):
                value = site['value']
                intermediates = site['intermediates']
                scale = site['scale']
                if intermediates:
                    log_prob = site['fn'].log_prob(value, intermediates)
                else:
                    log_prob = site['fn'].log_prob(value) #TODO: The shape here is duplicated

                if (scale is not None) and (not is_identically_one(scale)):
                    log_prob = scale * log_prob

                return log_prob, model_trace
    else:
        for site in model_trace.values():
            if site['type'] == 'sample' and not isinstance(site['fn'], dist.PRNGIdentity) and not site['is_observed']: #Prior prob
                value = site['value']
                intermediates = site['intermediates']
                scale = site['scale']
                if intermediates:
                    log_prob = site['fn'].log_prob(value, intermediates)
                else:
                    log_prob = site['fn'].log_prob(value)

                if (scale is not None) and (not is_identically_one(scale)):
                    log_prob = scale * log_prob

                log_prob = jnp.sum(log_prob)
                log_joint = log_joint + log_prob
        return log_joint, model_trace

def grad_potential(model, model_args, model_kwargs,z, z_ref, jac_all, hess_all, n, m,u=None, *args, **kwargs):
    """Calculate the gradient of the potential energy function for the current subsample"""
    if any(arg.shape[0] > m for arg in model_args):
        model_args = model_args_sub(u,model_args)
    k, = jac_all.shape
    z_flat, treedef = ravel_pytree(z)
    zref_flat, _ = ravel_pytree(z_ref)
    z_diff = z_flat - zref_flat

    ld_fn = lambda args: partial(log_density_hmcecs, model, model_args, model_kwargs,prior=False)(args)[0]

    jac_ref, _ = ravel_pytree(jax.jacfwd(ld_fn)(z_ref))
    hess_ref, _ = ravel_pytree(jax.hessian(ld_fn)(z_ref))

    jac_ref = jac_ref.reshape(m, k)
    hess_ref = hess_ref.reshape(m, k, k)

    grad_sum = jac_all + hess_all.dot(z_diff)
    jac_sub, _ = ravel_pytree(jax.jacfwd(ld_fn)(z))

    ll_sub, _ = log_density_hmcecs(model, model_args, model_kwargs, z,prior=False)  # log likelihood for subsample with current theta
    ll_ref, _ = log_density_hmcecs(model, model_args, model_kwargs, z_ref,prior=False)  # log likelihood for subsample with reference theta

    diff = ll_sub - (ll_ref + jac_ref @ z_diff + .5 * z_diff @ hess_ref @ z_diff.T)

    jac_sub = jac_sub.reshape(jac_ref.shape) - jac_ref

    grad_d_k = jac_sub - z_diff.dot(hess_ref)

    gradll = -(grad_sum + n / m * (jac_sub.sum(0) - hess_ref.sum(0).dot(z_diff))) + n ** 2 / (m ** 2) * (
            diff - diff.mean(0)).T.dot(grad_d_k - grad_d_k.mean(0))

    ld_fn = lambda args: partial(log_density_hmcecs, model, model_args, model_kwargs,prior=True)(args)[0]
    jac_sub, _ = ravel_pytree(jax.jacfwd(ld_fn)(z))

    return treedef(gradll - jac_sub)

def reducer( accum, d ):
   accum.update(d)
   return accum

def  tuplemerge( *dictionaries ):
   from functools import reduce
   merged = reduce( reducer, dictionaries, {} )
   return namedtuple('HMCCombinedState', merged )(**merged) # <==== Gist of the gist

def potential_est(model, model_args, model_kwargs,ll_ref, jac_all, hess_all, z, z_ref, n, m,u=None):
    """Estimate the potential dynamic energy for the HMC ECS implementation. The calculation follows section 7.2.1 in https://jmlr.org/papers/volume18/15-205/15-205.pdf
        The computation has a complexity of O(1) and it's highly dependant on the quality of the map estimate"""

    if any(arg.shape[0] > m for arg in model_args):
        model_args = model_args_sub(u,model_args)

    # Agrees with reference upto constant factor on prior
    k, = jac_all.shape  # number of features
    z_flat, _ = ravel_pytree(z)
    zref_flat, _ = ravel_pytree(z_ref)


    z_diff = z_flat - zref_flat

    ld_fn = lambda args: partial(log_density_hmcecs, model, model_args, model_kwargs,prior=False)(args)[0]

    jac_sub, _ = ravel_pytree(jax.jacfwd(ld_fn)(z_ref))
    hess_sub, _ = ravel_pytree(jax.hessian(ld_fn)(z_ref))

    proxy = jnp.sum(ll_ref) + jac_all.T @ z_diff + .5 * z_diff.T @ hess_all @ z_diff

    ll_sub, _ = log_density_hmcecs(model, model_args, model_kwargs, z,prior=False)  # log likelihood for subsample with current theta
    ll_ref, _ = log_density_hmcecs(model, model_args, model_kwargs, z_ref,prior=False)  # log likelihood for subsample with reference theta

    diff = ll_sub - (ll_ref + jac_sub.reshape((m, k)) @ z_diff + .5 * z_diff @ hess_sub.reshape((m, k, k)) @ z_diff.T)
    l_hat = proxy + n / m * jnp.sum(diff)

    sigma = n ** 2 / m * jnp.var(diff)

    ll_prior, _ = log_density_hmcecs(model, model_args, model_kwargs, z,prior=True) #TODO: work with hierachical models

    return (-l_hat + .5 * sigma) - ll_prior

def velocity_verlet_hmcecs(potential_fn, kinetic_fn, grad_potential_fn=None):
    r"""
    Second order symplectic integrator that uses the velocity verlet algorithm
    for position `z` and momentum `r`.

    :param potential_fn: Python callable that computes the potential energy
        given input parameters. The input parameters to `potential_fn` can be
        any python collection type. If HMCECS is used the gradient of the potential
        energy funtion is calculated
    :param kinetic_fn: Python callable that returns the kinetic energy given
        inverse mass matrix and momentum.
    :return: a pair of (`init_fn`, `update_fn`).
    """

    compute_value_grad = value_and_grad(potential_fn) if grad_potential_fn is None \
        else lambda z: (potential_fn(z), grad_potential_fn(z))

    def init_fn(z, r, potential_energy=None, z_grad=None):
        """
        :param z: Position of the particle.
        :param r: Momentum of the particle.
        :param potential_energy: Potential energy at `z`.
        :param z_grad: gradient of potential energy at `z`.
        :return: initial state for the integrator.
        """
        if potential_energy is None or z_grad is None:
            potential_energy, z_grad = compute_value_grad(z)


        return IntegratorState(z, r, potential_energy, z_grad)

    def update_fn(step_size, inverse_mass_matrix, state):
        """
        :param float step_size: Size of a single step.
        :param inverse_mass_matrix: Inverse of mass matrix, which is used to
            calculate kinetic energy.
        :param state: Current state of the integrator.
        :return: new state for the integrator.
        """
        z, r, _, z_grad = state

        r = tree_multimap(lambda r, z_grad: r - 0.5 * step_size * z_grad, r, z_grad)  # r(n+1/2)
        r_grad = grad(kinetic_fn, argnums=1)(inverse_mass_matrix, r)
        z = tree_multimap(lambda z, r_grad: z + step_size * r_grad, z, r_grad)  # z(n+1)
        potential_energy, z_grad = compute_value_grad(z)
        r = tree_multimap(lambda r, z_grad: r - 0.5 * step_size * z_grad, r, z_grad)  # r(n+1)
        #return IntegratorState(z, r, potential_energy, z_grad)
        return IntegratorState(z, r, potential_energy, z_grad)

    return init_fn, update_fn

def init_near_values(site=None, values={}):
    """Initialize the sampling to a noisy map estimate of the parameters"""
    from functools import partial

    from numpyro.distributions.continuous import Normal
    from numpyro.infer.initialization import init_to_uniform

    if site is None:
        return partial(init_near_values(values=values))

    if site['type'] == 'sample' and not site['is_observed']:
        if site['name'] in values:
            try:
                rng_key = site['kwargs'].get('rng_key')
                sample_shape = site['kwargs'].get('sample_shape')
                return values[site['name']] + Normal(0., 1e-3).sample(rng_key, sample_shape)
            except:
                return init_to_uniform(site)

