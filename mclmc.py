

import argparse
from collections import namedtuple
import os

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import random


import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC
from numpyro.infer.mcmc import MCMCKernel
import blackjax
from numpyro.infer.util import initialize_model
from blackjax.util import pytree_size
from blackjax.mcmc.integrators import (
    IntegratorState)


FullState = namedtuple("FullState", ["position", "momentum", "logdensity", "logdensity_grad", "rng_key"])

class MCLMC(MCMCKernel):
    """
    Microcanonical Langevin Monte Carlo (MCLMC) kernel.
    
    :param model: Python callable containing Pyro primitives.
    :param step_size: Initial step size for the Langevin dynamics.
    :param num_steps: Number of steps to take in each MCMC iteration.
    :param integrator_type: Type of integrator to use (e.g. "mclachlan").
    :param diagonal_preconditioning: Whether to use diagonal preconditioning.
    :param num_tuning_steps: Number of tuning steps to use.
    :param desired_energy_var: Desired energy variance for tuning.
    """


    
    def __init__(
        self,
        model=None,
        desired_energy_var=5e-4,
        diagonal_preconditioning=True,
    ):
        if model is None:
            raise ValueError("Model must be specified for MCLMC")
        self._model = model
        self._diagonal_preconditioning = diagonal_preconditioning
        self._desired_energy_var = desired_energy_var
        self._init_fn = None
        self._sample_fn = None
        self._postprocess_fn = None

    @property
    def model(self):
        return self._model

    @property
    def sample_field(self):
        return "position"
    
    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        """
        Initialize the MCLMC kernel.
        
        :param rng_key: Random number generator key
        :param num_warmup: Number of warmup steps
        :param init_params: Initial parameters
        :param model_args: Model arguments
        :param model_kwargs: Model keyword arguments
        :return: Initial state
        """

        init_model_key, init_state_key, run_key, rng_key_tune = jax.random.split(rng_key, 4)

        init_params, potential_fn_gen, _, _ = initialize_model(
            init_model_key,
            self._model,
            model_args=(),
            dynamic_args=True,
        )

        logdensity_fn = lambda position: -potential_fn_gen()(position)
        initial_position = init_params.z
        self.logdensity_fn = logdensity_fn

        sampler_state = blackjax.mcmc.mclmc.init(
        position=initial_position,
        logdensity_fn=self.logdensity_fn,
        rng_key=init_state_key,
        )

        kernel = lambda inverse_mass_matrix: blackjax.mcmc.mclmc.build_kernel(
            logdensity_fn=logdensity_fn,
            integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
            inverse_mass_matrix=inverse_mass_matrix,
        )

        self.dim = pytree_size(initial_position)

        # num_steps is a dummy param here
        (
            blackjax_state_after_tuning,
            blackjax_mclmc_sampler_params,
            num_tuning_integrator_steps,
        ) = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            num_steps=100,
            state=sampler_state,
            rng_key=rng_key_tune,
            diagonal_preconditioning=True,
            frac_tune3=num_warmup / (3 * 100),
            frac_tune2=num_warmup / (3 * 100),
            frac_tune1=num_warmup / (3 * 100),
            desired_energy_var=5e-4
        )

        self.adapt_state = blackjax_mclmc_sampler_params

        return FullState(blackjax_state_after_tuning.position, blackjax_state_after_tuning.momentum, blackjax_state_after_tuning.logdensity, blackjax_state_after_tuning.logdensity_grad, run_key)
    

    def sample(self, state, model_args, model_kwargs):
        """
        Run MCLMC from the given state and return the resulting state.
        
        :param state: Current state
        :param model_args: Model arguments
        :param model_kwargs: Model keyword arguments
        :return: Next state after running MCLMC
        """

        mclmc_state = IntegratorState(state.position, state.momentum, state.logdensity, state.logdensity_grad)
        rng_key, rng_key_sample = jax.random.split(state.rng_key, 2)

        kernel = blackjax.mcmc.mclmc.build_kernel(
            logdensity_fn=self.logdensity_fn,
            integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
            inverse_mass_matrix=self.adapt_state.inverse_mass_matrix,
        )
    
        new_state, info = kernel(
            rng_key=rng_key_sample,
            state=mclmc_state,
            step_size=self.adapt_state.step_size,
            L=self.adapt_state.L
        )

        return FullState(new_state.position, new_state.momentum, new_state.logdensity, new_state.logdensity_grad, rng_key)
           
if __name__ == "__main__":

    def gaussian_2d_model():
        """
        A simple 2D Gaussian model with mean [0, 0] and covariance [[1, 0.5], [0.5, 1]].
        """
        x = numpyro.sample("x", dist.Normal(0.0, 1.0))
        y = numpyro.sample("y", dist.Normal(0.0, 1.0))
        numpyro.sample("obs", dist.Normal(x + y, 0.5), obs=jnp.array([0.0]))
        return x + y


    def run_inference(model, args, rng_key):
        """
        Run MCMC inference on the given model.
        
        :param model: The model to run inference on
        :param args: Command line arguments
        :param rng_key: Random number generator key
        :return: MCMC object
        """
        kernel = MCLMC(
            model=model,
            diagonal_preconditioning=True,
            desired_energy_var=5e-4,
        )
        
        mcmc = MCMC(
            kernel,
            num_warmup=1000,
            num_samples=1000,
            num_chains=1,
            progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
        )
        
        mcmc.run(rng_key)
        mcmc.print_summary(exclude_deterministic=False)
        
        samples = mcmc.get_samples()
        plt.figure(figsize=(8, 8))
        plt.scatter(samples['x'], samples['y'], alpha=0.5)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('MCLMC samples from 2D Gaussian')
        plt.grid(True)
        plt.savefig('mclmc_samples.png')
        plt.close()
        
        return mcmc

        
    rng_key = random.PRNGKey(0)
    mcmc = run_inference(gaussian_2d_model, args=None, rng_key=rng_key)

