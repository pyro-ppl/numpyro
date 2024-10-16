# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from jax import numpy as jnp, random, vmap
from jax.nn import logsumexp

from numpyro.contrib.einstein.stein_util import batch_ravel_pytree
from numpyro.handlers import replay, seed
from numpyro.infer.util import log_density
from numpyro.util import _validate_model, check_model_guide_match


class SteinLoss:
    def __init__(self, elbo_num_particles=1, stein_num_particles=1):
        self.elbo_num_particles = elbo_num_particles
        self.stein_num_particles = stein_num_particles

    def single_particle_loss(
        self,
        rng_key,
        model,
        guide,
        selected_particle,
        unravel_pytree,
        flat_particles,
        select_index,
        model_args,
        model_kwargs,
        param_map,
    ):
        guide_key, model_key = random.split(rng_key, 2)

        # 2. Draw from selected mixture component
        guide_keys = random.split(guide_key, self.stein_num_particles)

        seeded_chosen = seed(guide, guide_keys[select_index])
        log_chosen_density, chosen_trace = log_density(
            seeded_chosen, model_args, model_kwargs, {**param_map, **selected_particle}
        )

        # 3. Score mixture guide
        def log_component_density(i):
            log_cdensity, component_trace = log_density(
                replay(seed(guide, guide_key[i]), chosen_trace),
                model_args,
                model_kwargs,
                {**param_map, **unravel_pytree(flat_particles[i])},
            )
            # Validate
            check_model_guide_match(component_trace, chosen_trace)
            return log_cdensity

        log_guide_density = logsumexp(
            vmap(log_component_density)(jnp.arange(self.stein_num_particles))
        )

        # 4. Score model
        seeded_model = seed(model, model_key)
        log_model_density, model_trace = log_density(
            replay(seeded_model, chosen_trace),
            model_args,
            model_kwargs,
            {**param_map, **selected_particle},
        )

        # Validation
        check_model_guide_match(model_trace, chosen_trace)
        _validate_model(model_trace, plate_warning="loose")
        elbo = log_model_density - log_guide_density
        return elbo

    def loss(self, rng_key, param_map, model, guide, particles, *args, **kwargs):
        if not particles:
            raise ValueError("Stein mixture undefined for empty guide.")

        flat_particles, unravel_pytree, _ = batch_ravel_pytree(particles, nbatch_dims=1)

        select_key, score_key = random.split(rng_key)
        assigns = random.randint(
            select_key,
            (self.elbo_num_particles,),
            minval=0,
            maxval=self.stein_num_particles,
        )
        score_keys = random.split(score_key, self.elbo_num_particles)
        elbos = vmap(
            lambda key, assign: self.single_particle_loss(
                rng_key=key,
                model=model,
                guide=guide,
                selected_particle=unravel_pytree(flat_particles[assign]),
                unravel_pytree=unravel_pytree,
                flat_particles=flat_particles,
                select_index=assign,
                model_args=args,
                model_kwargs=kwargs,
                param_map=param_map,
            )
            - jnp.log(self.stein_num_particles)
        )(score_keys, assigns)
        return -elbos.mean()
