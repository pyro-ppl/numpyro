from functools import partial

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions.util import is_identically_one
from numpyro.handlers import substitute, trace
from numpyro.util import ravel_pytree

def log_density_hmcecs(model, model_args, model_kwargs, params, prior=True):
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
    for site in model_trace.values():
        if site['type'] == 'sample' and not isinstance(site['fn'], dist.PRNGIdentity) and not site['is_observed']:
            value = site['value']
            intermediates = site['intermediates']
            scale = site['scale']
            if intermediates:
                log_prob = site['fn'].log_prob(value, intermediates)
            else:
                log_prob = site['fn'].log_prob(value)

            if (scale is not None) and (not is_identically_one(scale)):
                log_prob = scale * log_prob

            if prior:
                log_prob = jnp.sum(log_prob)
                log_joint = log_joint + log_prob
    return log_joint, model_trace



def grad_potential(model, model_args, model_kwargs,z, z_ref, jac_all, hess_all, n, m, *args, **kwargs):

    k, = jac_all.shape
    z_flat, treedef = ravel_pytree(z)
    zref_flat, _ = ravel_pytree(z_ref)
    z_diff = z_flat - zref_flat

    ld_fn = lambda args: partial(log_density, model, model_args, model_kwargs, prior = False)(args)[0]

    jac_ref, _ = ravel_pytree(jax.jacfwd(ld_fn)(z_ref))
    hess_ref, _ = ravel_pytree(jax.hessian(ld_fn)(z_ref))

    jac_ref = jac_ref.reshape(m, k)
    hess_ref = hess_ref.reshape(m, k, k)

    grad_sum = jac_all + hess_all.dot(z_diff)
    jac_sub, _ = ravel_pytree(jax.jacfwd(ld_fn)(z))

    ll_sub, _ = log_density(model, model_args, model_kwargs, z,prior=False)  # log likelihood for subsample with current theta
    ll_ref, _ = log_density(model, model_args, model_kwargs, z_ref,prior=False)  # log likelihood for subsample with reference theta

    diff = ll_sub - (ll_ref + jac_ref @ z_diff + .5 * z_diff @ hess_ref @ z_diff.T)

    jac_sub = jac_sub.reshape(jac_ref.shape) - jac_ref

    grad_d_k = jac_sub - z_diff.dot(hess_ref)

    gradll = -(grad_sum + n / m * (jac_sub.sum(0) - hess_ref.sum(0).dot(z_diff))) + n ** 2 / (m ** 2) * (
            diff - diff.mean(0)).T.dot(grad_d_k - grad_d_k.mean(0))

    ld_fn = lambda args: partial(log_density, model, model_args, model_kwargs,prior=True)(args)[0]
    jac_sub, _ = ravel_pytree(jax.jacfwd(ld_fn)(z))

    return treedef(gradll - jac_sub)


def potential_est(model, model_args, model_kwargs,ll_ref, jac_all, hess_all, z, z_ref, n, m):
    # Agrees with reference upto constant factor on prior
    k, = jac_all.shape  # number of features
    z_flat, _ = ravel_pytree(z)
    zref_flat, _ = ravel_pytree(z_ref)

    z_diff = z_flat - zref_flat

    ld_fn = lambda args: partial(log_density, model, model_args, model_kwargs,prior=False)(args)[0]

    jac_sub, _ = ravel_pytree(jax.jacfwd(ld_fn)(z_ref))
    hess_sub, _ = ravel_pytree(jax.hessian(ld_fn)(z_ref))

    proxy = jnp.sum(ll_ref) + jac_all.T @ z_diff + .5 * z_diff.T @ hess_all @ z_diff

    ll_sub, _ = log_density(model, model_args, model_kwargs, z,prior=False)  # log likelihood for subsample with current theta
    ll_ref, _ = log_density(model, model_args, model_kwargs, z_ref,prior=False)  # log likelihood for subsample with reference theta

    diff = ll_sub - (ll_ref + jac_sub.reshape((m, k)) @ z_diff + .5 * z_diff @ hess_sub.reshape((m, k, k)) @ z_diff.T)
    l_hat = proxy + n / m * jnp.sum(diff)

    sigma = n ** 2 / m * jnp.var(diff)

    ll_prior, _ = log_density(model, model_args, model_kwargs, z,prior=True)

    return (-l_hat + .5 * sigma) - ll_prior