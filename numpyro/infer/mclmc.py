# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple

import jax

from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import initialize_model
from numpyro.util import identity

try:
    import blackjax
    from blackjax.mcmc.integrators import IntegratorState
    from blackjax.util import pytree_size

    _BLACKJAX_AVAILABLE = True
except ImportError:
    _BLACKJAX_AVAILABLE = False
    blackjax = None
    IntegratorState = None
    pytree_size = None

FullState = namedtuple(
    "FullState", ["position", "momentum", "logdensity", "logdensity_grad", "rng_key"]
)


class MCLMC(MCMCKernel):
    """
    Microcanonical Langevin Monte Carlo (MCLMC) kernel.

    MCLMC is a gradient-based MCMC algorithm that uses Hamiltonian dynamics
    on an extended state space. It requires the `blackjax` package.

    **References:**

    1. *Microcanonical Hamiltonian Monte Carlo*,
       Jakob Robnik, G. Bruno De Luca, Eva Silverstein, Uroš Seljak
       https://arxiv.org/abs/2212.08549

    .. note:: The model must have at least 2 latent dimensions for MCLMC to work
        (this is a limitation of the blackjax implementation).

    :param model: Python callable containing Pyro :mod:`~numpyro.primitives`.
    :param float desired_energy_var: Target energy variance for step size and
        trajectory length tuning. Smaller values lead to more conservative
        step sizes. Defaults to 5e-4.
    :param bool diagonal_preconditioning: Whether to use diagonal preconditioning
        for the mass matrix. Defaults to True.
    """

    def __init__(
        self,
        model=None,
        desired_energy_var=5e-4,
        diagonal_preconditioning=True,
    ):
        if not _BLACKJAX_AVAILABLE:
            raise ImportError(
                "MCLMC requires the 'blackjax' package. "
                "Please install it with: pip install blackjax"
            )
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

    @property
    def default_fields(self):
        return (self.sample_field,)

    def get_diagnostics_str(self, state):
        """
        Return a diagnostics string for the progress bar.
        """
        return "step_size={:.2e}, L={:.2e}".format(
            self.adapt_state.step_size, self.adapt_state.L
        )

    def postprocess_fn(self, args, kwargs):
        """
        Get a function that transforms unconstrained values at sample sites to values
        constrained to the site's support, in addition to returning deterministic
        sites in the model.

        :param args: Arguments to the model.
        :param kwargs: Keyword arguments to the model.
        """
        if self._postprocess_fn is None:
            return identity
        return self._postprocess_fn(*args, **kwargs)

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

        init_model_key, init_state_key, run_key, rng_key_tune = jax.random.split(
            rng_key, 4
        )

        init_params, potential_fn_gen, postprocess_fn, _ = initialize_model(
            init_model_key,
            self._model,
            model_args=model_args,
            model_kwargs=model_kwargs,
            dynamic_args=True,
        )
        self._postprocess_fn = postprocess_fn

        def logdensity_fn(position):
            return -potential_fn_gen(*model_args, **model_kwargs)(position)

        initial_position = init_params.z
        self.logdensity_fn = logdensity_fn

        sampler_state = blackjax.mcmc.mclmc.init(
            position=initial_position,
            logdensity_fn=self.logdensity_fn,
            rng_key=init_state_key,
        )

        def kernel(inverse_mass_matrix):
            return blackjax.mcmc.mclmc.build_kernel(
                logdensity_fn=logdensity_fn,
                integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
                inverse_mass_matrix=inverse_mass_matrix,
            )

        self.dim = pytree_size(initial_position)

        # num_steps is a dummy param here (used for tuning fractions)
        num_tuning_steps = 100
        (
            blackjax_state_after_tuning,
            blackjax_mclmc_sampler_params,
            _,
        ) = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            num_steps=num_tuning_steps,
            state=sampler_state,
            rng_key=rng_key_tune,
            diagonal_preconditioning=self._diagonal_preconditioning,
            frac_tune3=num_warmup / (3 * num_tuning_steps),
            frac_tune2=num_warmup / (3 * num_tuning_steps),
            frac_tune1=num_warmup / (3 * num_tuning_steps),
            desired_energy_var=self._desired_energy_var,
        )

        self.adapt_state = blackjax_mclmc_sampler_params

        return FullState(
            blackjax_state_after_tuning.position,
            blackjax_state_after_tuning.momentum,
            blackjax_state_after_tuning.logdensity,
            blackjax_state_after_tuning.logdensity_grad,
            run_key,
        )

    def sample(self, state, model_args, model_kwargs):
        """
        Run MCLMC from the given state and return the resulting state.

        :param state: Current state
        :param model_args: Model arguments
        :param model_kwargs: Model keyword arguments
        :return: Next state after running MCLMC
        """

        mclmc_state = IntegratorState(
            state.position, state.momentum, state.logdensity, state.logdensity_grad
        )
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
            L=self.adapt_state.L,
        )

        return FullState(
            new_state.position,
            new_state.momentum,
            new_state.logdensity,
            new_state.logdensity_grad,
            rng_key,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_postprocess_fn"] = None
        return state
