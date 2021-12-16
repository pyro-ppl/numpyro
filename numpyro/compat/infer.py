# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

from jax import jit

import numpyro
from numpyro.compat.pyro import get_param_store
from numpyro.infer import elbo, hmc, mcmc, svi


class HMC(hmc.HMC):
    def __init__(
        self,
        model=None,
        potential_fn=None,
        step_size=1,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        full_mass=False,
        use_multinomial_sampling=True,
        transforms=None,
        max_plate_nesting=None,
        jit_compile=False,
        jit_options=None,
        ignore_jit_warnings=False,
        trajectory_length=2 * math.pi,
        target_accept_prob=0.8,
    ):
        super(HMC, self).__init__(
            model=model,
            potential_fn=potential_fn,
            step_size=step_size,
            adapt_step_size=adapt_step_size,
            adapt_mass_matrix=adapt_mass_matrix,
            dense_mass=full_mass,
            target_accept_prob=target_accept_prob,
            trajectory_length=trajectory_length,
        )


class NUTS(hmc.NUTS):
    def __init__(
        self,
        model=None,
        potential_fn=None,
        step_size=1,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        full_mass=False,
        use_multinomial_sampling=True,
        transforms=None,
        max_plate_nesting=None,
        jit_compile=False,
        jit_options=None,
        ignore_jit_warnings=False,
        trajectory_length=2 * math.pi,
        target_accept_prob=0.8,
        max_tree_depth=10,
    ):
        if potential_fn is not None:
            raise ValueError(
                "Only `model` argument is supported in generic module;"
                " `potential_fn` is not supported."
            )
        super(NUTS, self).__init__(
            model=model,
            potential_fn=potential_fn,
            step_size=step_size,
            adapt_step_size=adapt_step_size,
            adapt_mass_matrix=adapt_mass_matrix,
            dense_mass=full_mass,
            target_accept_prob=target_accept_prob,
            trajectory_length=trajectory_length,
            max_tree_depth=max_tree_depth,
        )


class MCMC(object):
    def __init__(
        self,
        kernel,
        num_samples,
        num_warmup=None,
        initial_params=None,
        num_chains=1,
        hook_fn=None,
        mp_context=None,
        disable_progbar=False,
        disable_validation=True,
        transforms=None,
    ):
        if num_warmup is None:
            num_warmup = num_samples
        self._initial_params = initial_params
        self._mcmc = mcmc.MCMC(
            kernel,
            num_warmup,
            num_samples,
            num_chains=num_chains,
            progress_bar=(not disable_progbar),
        )

    def run(self, *args, rng_key=None, **kwargs):
        if rng_key is None:
            rng_key = numpyro.prng_key()
        self._mcmc.run(rng_key, *args, init_params=self._initial_params, **kwargs)

    def get_samples(self, num_samples=None, group_by_chain=False):
        if num_samples is not None:
            raise ValueError("`num_samples` arg unsupported in NumPyro.")
        return self._mcmc.get_samples(group_by_chain=group_by_chain)

    def summary(self, prob=0.9):
        self._mcmc.print_summary()


class SVI(svi.SVI):
    def __init__(
        self,
        model,
        guide,
        optim,
        loss,
        loss_and_grads=None,
        num_samples=10,
        num_steps=0,
        **kwargs
    ):
        super(SVI, self).__init__(model=model, guide=guide, optim=optim, loss=loss)
        self.svi_state = None

    def evaluate_loss(self, *args, **kwargs):
        return self.evaluate(self.svi_state, *args, **kwargs)

    def step(self, *args, rng_key=None, **kwargs):
        if self.svi_state is None:
            if rng_key is None:
                rng_key = numpyro.prng_key()
            self.svi_state = self.init(rng_key, *args, **kwargs)
        try:
            self.svi_state, loss = jit(self.update)(self.svi_state, *args, **kwargs)
        except TypeError as e:
            if "not a valid JAX type" in str(e):
                raise TypeError(
                    "NumPyro backend requires args, kwargs to be arrays or tuples, "
                    "dicts of arrays."
                ) from e
            else:
                raise e
        params = jit(super(SVI, self).get_params)(self.svi_state)
        get_param_store().update(params)
        return loss

    def get_params(self):
        return super(SVI, self).get_params(self.svi_state)


class Trace_ELBO(elbo.Trace_ELBO):
    def __init__(
        self,
        num_particles=1,
        max_plate_nesting=float("inf"),
        max_iarange_nesting=None,  # DEPRECATED
        vectorize_particles=False,
        strict_enumeration_warning=True,
        ignore_jit_warnings=False,
        jit_options=None,
        retain_graph=None,
        tail_adaptive_beta=-1.0,
    ):
        super(Trace_ELBO, self).__init__(num_particles=num_particles)


# JIT is enabled by default
JitTrace_ELBO = Trace_ELBO
