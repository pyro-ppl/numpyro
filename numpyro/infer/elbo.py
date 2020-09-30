# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import warnings

from jax import random, vmap
from jax.lax import stop_gradient
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import numpyro.distributions as dist
from numpyro.distributions.kl import kl_divergence
from numpyro.distributions.util import scale_and_mask
from numpyro.handlers import replay, seed, substitute, trace
from numpyro.infer.util import log_density


class Trace_ELBO:
    """
    A trace implementation of ELBO-based SVI. The estimator is constructed
    along the lines of references [1] and [2]. There are no restrictions on the
    dependency structure of the model or the guide.

    This is the most basic implementation of the Evidence Lower Bound, which is the
    fundamental objective in Variational Inference. This implementation has various
    limitations (for example it only supports random variables with reparameterized
    samplers) but can be used as a template to build more sophisticated loss
    objectives.

    For more details, refer to http://pyro.ai/examples/svi_part_i.html.

    **References:**

    1. *Automated Variational Inference in Probabilistic Programming*,
       David Wingate, Theo Weber
    2. *Black Box Variational Inference*,
       Rajesh Ranganath, Sean Gerrish, David M. Blei

    :param num_particles: The number of particles/samples used to form the ELBO
        (gradient) estimators.
    """
    def __init__(self, num_particles=1):
        self.num_particles = num_particles

    def loss(self, rng_key, param_map, model, guide, *args, **kwargs):
        """
        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.

        :param jax.random.PRNGKey rng_key: random number generator seed.
        :param dict param_map: dictionary of current parameter values keyed by site
            name.
        :param model: Python callable with NumPyro primitives for the model.
        :param guide: Python callable with NumPyro primitives for the guide.
        :param args: arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: keyword arguments to the model / guide (these can possibly vary
            during the course of fitting).
        :return: negative of the Evidence Lower Bound (ELBO) to be minimized.
        """
        def single_particle_elbo(rng_key):
            model_seed, guide_seed = random.split(rng_key)
            seeded_model = seed(model, model_seed)
            seeded_guide = seed(guide, guide_seed)
            guide_log_density, guide_trace = log_density(seeded_guide, args, kwargs, param_map)
            seeded_model = replay(seeded_model, guide_trace)
            model_log_density, _ = log_density(seeded_model, args, kwargs, param_map)

            # log p(z) - log q(z)
            elbo = model_log_density - guide_log_density
            return elbo

        # Return (-elbo) since by convention we do gradient descent on a loss and
        # the ELBO is a lower bound that needs to be maximized.
        if self.num_particles == 1:
            return - single_particle_elbo(rng_key)
        else:
            rng_keys = random.split(rng_key, self.num_particles)
            return - jnp.mean(vmap(single_particle_elbo)(rng_keys))


class ELBO(Trace_ELBO):
    def __init__(self, num_particles=1):
        warnings.warn("Using ELBO directly in SVI is deprecated. Please use Trace_ELBO class instead.",
                      FutureWarning)
        super().__init__(num_particles=num_particles)


def _get_log_prob_sum(site):
    if site['intermediates']:
        log_prob = site['fn'].log_prob(site['value'], site['intermediates'])
    else:
        log_prob = site['fn'].log_prob(site['value'])
    log_prob = scale_and_mask(log_prob, site['scale'])
    return jnp.sum(log_prob)


def _check_mean_field_requirement(model_trace, guide_trace):
    """
    Checks that the guide and model sample sites are ordered identically.
    This is sufficient but not necessary for correctness.
    """
    model_sites = [name for name, site in model_trace.items()
                   if site["type"] == "sample" and name in guide_trace]
    guide_sites = [name for name, site in guide_trace.items()
                   if site["type"] == "sample" and name in model_trace]
    assert set(model_sites) == set(guide_sites)
    if model_sites != guide_sites:
        warnings.warn("Failed to verify mean field restriction on the guide. "
                      "To eliminate this warning, ensure model and guide sites "
                      "occur in the same order.\n" +
                      "Model sites:\n  " + "\n  ".join(model_sites) +
                      "Guide sites:\n  " + "\n  ".join(guide_sites))


class TraceMeanField_ELBO(Trace_ELBO):
    """
    A trace implementation of ELBO-based SVI. This is currently the only
    ELBO estimator in NumPyro that uses analytic KL divergences when those
    are available.

    .. warning:: This estimator may give incorrect results if the mean-field
        condition is not satisfied.
        The mean field condition is a sufficient but not necessary condition for
        this estimator to be correct. The precise condition is that for every
        latent variable `z` in the guide, its parents in the model must not include
        any latent variables that are descendants of `z` in the guide. Here
        'parents in the model' and 'descendants in the guide' is with respect
        to the corresponding (statistical) dependency structure. For example, this
        condition is always satisfied if the model and guide have identical
        dependency structures.
    """
    def loss(self, rng_key, param_map, model, guide, *args, **kwargs):
        """
        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.

        :param jax.random.PRNGKey rng_key: random number generator seed.
        :param dict param_map: dictionary of current parameter values keyed by site
            name.
        :param model: Python callable with NumPyro primitives for the model.
        :param guide: Python callable with NumPyro primitives for the guide.
        :param args: arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: keyword arguments to the model / guide (these can possibly vary
            during the course of fitting).
        :return: negative of the Evidence Lower Bound (ELBO) to be minimized.
        """
        def single_particle_elbo(rng_key):
            model_seed, guide_seed = random.split(rng_key)
            seeded_model = seed(model, model_seed)
            seeded_guide = seed(guide, guide_seed)
            subs_guide = substitute(seeded_guide, data=param_map)
            guide_trace = trace(subs_guide).get_trace(*args, **kwargs)
            subs_model = substitute(replay(seeded_model, guide_trace), data=param_map)
            model_trace = trace(subs_model).get_trace(*args, **kwargs)
            _check_mean_field_requirement(model_trace, guide_trace)

            elbo_particle = 0
            for name, model_site in model_trace.items():
                if model_site["type"] == "sample" and not isinstance(model_site["fn"], dist.PRNGIdentity):
                    if model_site["is_observed"]:
                        elbo_particle = elbo_particle + _get_log_prob_sum(model_site)
                    else:
                        guide_site = guide_trace[name]
                        try:
                            kl_qp = kl_divergence(guide_site["fn"], model_site["fn"])
                            kl_qp = scale_and_mask(kl_qp, scale=guide_site["scale"])
                            elbo_particle = elbo_particle - jnp.sum(kl_qp)
                        except NotImplementedError:
                            elbo_particle = elbo_particle + _get_log_prob_sum(model_site) \
                                - _get_log_prob_sum(guide_site)

            # handle auxiliary sites in the guide
            for name, site in guide_trace.items():
                if site["type"] == "sample" and name not in model_trace:
                    assert site["infer"].get("is_auxiliary")
                    elbo_particle = elbo_particle - _get_log_prob_sum(site)

            return elbo_particle

        if self.num_particles == 1:
            return - single_particle_elbo(rng_key)
        else:
            rng_keys = random.split(rng_key, self.num_particles)
            return - jnp.mean(vmap(single_particle_elbo)(rng_keys))


class RenyiELBO(Trace_ELBO):
    r"""
    An implementation of Renyi's :math:`\alpha`-divergence
    variational inference following reference [1].
    In order for the objective to be a strict lower bound, we require
    :math:`\alpha \ge 0`. Note, however, that according to reference [1], depending
    on the dataset :math:`\alpha < 0` might give better results. In the special case
    :math:`\alpha = 0`, the objective function is that of the important weighted
    autoencoder derived in reference [2].

    .. note:: Setting :math:`\alpha < 1` gives a better bound than the usual ELBO.

    :param float alpha: The order of :math:`\alpha`-divergence.
        Here :math:`\alpha \neq 1`. Default is 0.
    :param num_particles: The number of particles/samples
        used to form the objective (gradient) estimator. Default is 2.

    **References:**

    1. *Renyi Divergence Variational Inference*, Yingzhen Li, Richard E. Turner
    2. *Importance Weighted Autoencoders*, Yuri Burda, Roger Grosse, Ruslan Salakhutdinov
    """
    def __init__(self, alpha=0, num_particles=2):
        if alpha == 1:
            raise ValueError("The order alpha should not be equal to 1. Please use ELBO class"
                             "for the case alpha = 1.")
        self.alpha = alpha
        super(RenyiELBO, self).__init__(num_particles=num_particles)

    def loss(self, rng_key, param_map, model, guide, *args, **kwargs):
        """
        Evaluates the Renyi ELBO with an estimator that uses num_particles many samples/particles.

        :param jax.random.PRNGKey rng_key: random number generator seed.
        :param dict param_map: dictionary of current parameter values keyed by site
            name.
        :param model: Python callable with NumPyro primitives for the model.
        :param guide: Python callable with NumPyro primitives for the guide.
        :param args: arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: keyword arguments to the model / guide (these can possibly vary
            during the course of fitting).
        :returns: negative of the Renyi Evidence Lower Bound (ELBO) to be minimized.
        """
        def single_particle_elbo(rng_key):
            model_seed, guide_seed = random.split(rng_key)
            seeded_model = seed(model, model_seed)
            seeded_guide = seed(guide, guide_seed)
            guide_log_density, guide_trace = log_density(seeded_guide, args, kwargs, param_map)
            # NB: we only want to substitute params not available in guide_trace
            model_param_map = {k: v for k, v in param_map.items() if k not in guide_trace}
            seeded_model = replay(seeded_model, guide_trace)
            model_log_density, _ = log_density(seeded_model, args, kwargs, model_param_map)

            # log p(z) - log q(z)
            elbo = model_log_density - guide_log_density
            return elbo

        rng_keys = random.split(rng_key, self.num_particles)
        elbos = vmap(single_particle_elbo)(rng_keys)
        scaled_elbos = (1. - self.alpha) * elbos
        avg_log_exp = logsumexp(scaled_elbos) - jnp.log(self.num_particles)
        weights = jnp.exp(scaled_elbos - avg_log_exp)
        renyi_elbo = avg_log_exp / (1. - self.alpha)
        weighted_elbo = jnp.dot(stop_gradient(weights), elbos) / self.num_particles
        return - (stop_gradient(renyi_elbo - weighted_elbo) + weighted_elbo)
