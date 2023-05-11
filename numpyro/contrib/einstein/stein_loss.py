import jax.numpy as jnp
from jax import random, vmap

from numpyro.handlers import replay, seed
from numpyro.infer import ELBO
from numpyro.infer.util import (
    log_density,
)
from numpyro.util import _validate_model, check_model_guide_match


class SteinLoss(ELBO):
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

    def __init__(self, elbo_num_particles=1):
        self.num_elbo_particles = elbo_num_particles

    def loss(
            self, rng_key, param_map, model, guide, *args, **kwargs
    ):
        log_model_densities, log_guide_densities, mutable_state = self.inner_loss_with_mutable_state(rng_key, param_map,
                                                                                                     model, guide,
                                                                                                     *args, **kwargs)
        return log_model_densities, log_guide_densities

    def inner_loss_with_mutable_state(
            self, rng_key, param_map, model, guide, *args, **kwargs
    ):
        def single_particle_elbo(rng_key):
            params = param_map.copy()
            model_seed, guide_seed = random.split(rng_key)
            seeded_model = seed(model, model_seed)
            seeded_guide = seed(guide, guide_seed)
            guide_log_density, guide_trace = log_density(
                seeded_guide, args, kwargs, param_map
            )
            mutable_params = {
                name: site["value"]
                for name, site in guide_trace.items()
                if site["type"] == "mutable"
            }
            params.update(mutable_params)
            seeded_model = replay(seeded_model, guide_trace)
            model_log_density, model_trace = log_density(
                seeded_model, args, kwargs, params
            )
            check_model_guide_match(model_trace, guide_trace)
            _validate_model(model_trace, plate_warning="loose")
            mutable_params.update(
                {
                    name: site["value"]
                    for name, site in model_trace.items()
                    if site["type"] == "mutable"
                }
            )

            if mutable_params:
                if self.num_elbo_particles == 1:
                    return model_log_density, guide_log_density, mutable_params
                else:
                    raise ValueError(
                        "Currently, we only support mutable states with num_particles=1."
                    )
            else:
                return model_log_density, guide_log_density, None

        # Return (-elbo) since by convention we do gradient descent on a loss and
        # the ELBO is a lower bound that needs to be maximized.
        if self.num_elbo_particles == 1:
            model_log_density, guide_log_density, mutable_state = single_particle_elbo(rng_key)
            return model_log_density, guide_log_density, mutable_state
        else:
            rng_keys = random.split(rng_key, self.num_elbo_particles)
            model_log_densities, guide_log_densities, mutable_state = vmap(single_particle_elbo)(rng_keys)
            return model_log_densities, guide_log_densities, mutable_state
