def _elbo(rng, param_map, model, guide, model_args, guide_args, kwargs, num_particles=1):
    """
    This is the most basic implementation of the Evidence Lower Bound, which is the
    fundamental objective in Variational Inference. This implementation has various
    limitations (for example it only supports random variables with reparameterized
    samplers) but can be used as a template to build more sophisticated loss
    objectives.

    For more details, refer to http://pyro.ai/examples/svi_part_i.html.

    :param jax.random.PRNGKey rng: random number generator seed.
    :param dict param_map: dictionary of current parameter values keyed by site
        name.
    :param model: Python callable with Pyro primitives for the model.
    :param guide: Python callable with Pyro primitives for the guide
        (recognition network).
    :param tuple model_args: arguments to the model (these can possibly vary during
        the course of fitting).
    :param tuple guide_args: arguments to the guide (these can possibly vary during
        the course of fitting).
    :param dict kwargs: static keyword arguments to the model / guide.
    :return: negative of the Evidence Lower Bound (ELBo) to be minimized.
    """
    def single_particle_elbo(rng):
        seeded_model, seeded_guide = _seed(model, guide, rng)
        guide_log_density, guide_trace = log_density(seeded_guide, guide_args, kwargs, param_map)
        if isinstance(seeded_guide.__wrapped__, AutoContinuous):
            # first, we substitute `param_map` to `param` primitives of `model`
            seeded_model = substitute(seeded_model, param_map)
            # then creates a new `param_map` which holds base values of `sample` primitives
            base_param_map = {}
            # in autoguide, a site's value holds intermediate value
            for name, site in guide_trace.items():
                if site['type'] == 'sample':
                    base_param_map[name] = site['value']
            model_log_density, _ = log_density(seeded_model, model_args, kwargs, base_param_map,
                                               skip_dist_transforms=True)
        else:
            # NB: we only want to substitute params not available in guide_trace
            model_param_map = {k: v for k, v in param_map.items() if k not in guide_trace}
            seeded_model = replay(seeded_model, guide_trace)
            model_log_density, _ = log_density(seeded_model, model_args, kwargs, model_param_map)

        # log p(z) - log q(z)
        elbo = model_log_density - guide_log_density
        # Return (-elbo) since by convention we do gradient descent on a loss and
        # the ELBO is a lower bound that needs to be maximized.
        return -elbo

    if num_particles == 1:
        return single_particle_elbo(rng)
    else:
        rngs = random.split(rng, num_particles)
        return np.mean(vmap(single_particle_elbo)(rngs))


def elbo(num_particles=1):
    """
    This is the most basic implementation of the Evidence Lower Bound, which is the
    fundamental objective in Variational Inference. This implementation has various
    limitations (for example it only supports random variables with reparameterized
    samplers) but can be used as a template to build more sophisticated loss
    objectives.

    For more details, refer to http://pyro.ai/examples/svi_part_i.html.

    :param int num_particles: The number of particles/samples used to form the ELBO
        (gradient) estimators.
    """
    return partial(_elbo, num_particles=num_particles)
