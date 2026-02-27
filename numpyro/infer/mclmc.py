# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple

import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp

from numpyro.diagnostics import effective_sample_size
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import initialize_model
from numpyro.util import identity

MCLMCState = namedtuple(
    "MCLMCState", ["position", "momentum", "logdensity", "logdensity_grad"]
)
MCLMCInfo = namedtuple("MCLMCInfo", ["logdensity", "kinetic_change", "energy_change"])
MCLMCAdaptationState = namedtuple(
    "MCLMCAdaptationState", ["L", "step_size", "inverse_mass_matrix"]
)
FullState = namedtuple(
    "FullState", ["position", "momentum", "logdensity", "logdensity_grad", "rng_key"]
)

# First momentum-stage coefficient in the 5-stage McLachlan splitting scheme.
_MCLACHLAN_B1 = 0.1931833275037836
# Palindromic integrator coefficients for one isokinetic McLachlan update.
_MCLACHLAN_COEFS = (_MCLACHLAN_B1, 0.5, 1 - 2 * _MCLACHLAN_B1, 0.5, _MCLACHLAN_B1)
# When NaNs are detected during adaptation, shrink step size by this factor.
_DELTA_NAN_STEP_SIZE_FACTOR = 0.8


def _pytree_size(pytree):
    return sum(jnp.size(leaf) for leaf in jax.tree.leaves(pytree))


def _generate_unit_vector(rng_key, position):
    flat_position, unravel_fn = ravel_pytree(position)
    sample = jax.random.normal(
        rng_key, shape=flat_position.shape, dtype=flat_position.dtype
    )
    return unravel_fn(sample / jnp.linalg.norm(sample))


def _incremental_value_update(
    expectation, incremental_val, weight=1.0, zero_prevention=0.0
):
    total, average = incremental_val
    average = jax.tree.map(
        lambda exp, av: jnp.where(
            (total * av + weight * exp) == 0.0,
            0.0,
            (total * av + weight * exp) / (total + weight + zero_prevention),
        ),
        expectation,
        average,
    )
    return total + weight, average


def _init_mclmc(position, logdensity_fn, rng_key):
    if _pytree_size(position) < 2:
        raise ValueError(
            "The target distribution must have more than 1 dimension for MCLMC."
        )
    logdensity, logdensity_grad = jax.value_and_grad(logdensity_fn)(position)
    return MCLMCState(
        position=position,
        momentum=_generate_unit_vector(rng_key, position),
        logdensity=logdensity,
        logdensity_grad=logdensity_grad,
    )


def _position_update(position, kinetic_grad, step_size, coef, logdensity_fn):
    new_position = jax.tree.map(
        lambda x, grad: x + step_size * coef * grad,
        position,
        kinetic_grad,
    )
    logdensity, logdensity_grad = jax.value_and_grad(logdensity_fn)(new_position)
    return new_position, logdensity, logdensity_grad


def _normalized_flatten(x, tol=1e-13):
    norm = jnp.linalg.norm(x)
    return jnp.where(norm > tol, x / norm, x), norm


def _esh_dynamics_momentum_update_one_step(
    momentum,
    logdensity_grad,
    step_size,
    coef,
    inverse_mass_matrix,
    previous_kinetic_energy_change=None,
):
    sqrt_inverse_mass_matrix = jnp.sqrt(inverse_mass_matrix)
    flatten_grads, unravel_fn = ravel_pytree(logdensity_grad)
    flatten_grads = flatten_grads * sqrt_inverse_mass_matrix
    flatten_momentum, _ = ravel_pytree(momentum)
    dims = flatten_momentum.shape[0]

    normalized_gradient, gradient_norm = _normalized_flatten(flatten_grads)
    momentum_proj = jnp.dot(flatten_momentum, normalized_gradient)
    delta = step_size * coef * gradient_norm / (dims - 1)
    zeta = jnp.exp(-delta)
    new_momentum_raw = (
        normalized_gradient * (1 - zeta) * (1 + zeta + momentum_proj * (1 - zeta))
        + 2 * zeta * flatten_momentum
    )
    new_momentum_normalized, _ = _normalized_flatten(new_momentum_raw)
    next_momentum = unravel_fn(new_momentum_normalized)
    kinetic_grad = unravel_fn(new_momentum_normalized * sqrt_inverse_mass_matrix)
    kinetic_energy_change = (
        delta
        - jnp.log(2.0)
        + jnp.log(1 + momentum_proj + (1 - momentum_proj) * zeta**2)
    ) * (dims - 1)
    if previous_kinetic_energy_change is not None:
        kinetic_energy_change = kinetic_energy_change + previous_kinetic_energy_change
    return next_momentum, kinetic_grad, kinetic_energy_change


def _isokinetic_mclachlan_step(state, step_size, logdensity_fn, inverse_mass_matrix):
    position, momentum, _, logdensity_grad = state
    kinetic_energy_change = None

    for i, coef in enumerate(_MCLACHLAN_COEFS[:-1]):
        if i % 2 == 0:
            momentum, kinetic_grad, kinetic_energy_change = (
                _esh_dynamics_momentum_update_one_step(
                    momentum=momentum,
                    logdensity_grad=logdensity_grad,
                    step_size=step_size,
                    coef=coef,
                    inverse_mass_matrix=inverse_mass_matrix,
                    previous_kinetic_energy_change=kinetic_energy_change,
                )
            )
        else:
            position, logdensity, logdensity_grad = _position_update(
                position=position,
                kinetic_grad=kinetic_grad,
                step_size=step_size,
                coef=coef,
                logdensity_fn=logdensity_fn,
            )

    momentum, _, kinetic_energy_change = _esh_dynamics_momentum_update_one_step(
        momentum=momentum,
        logdensity_grad=logdensity_grad,
        step_size=step_size,
        coef=_MCLACHLAN_COEFS[-1],
        inverse_mass_matrix=inverse_mass_matrix,
        previous_kinetic_energy_change=kinetic_energy_change,
    )
    return MCLMCState(
        position, momentum, logdensity, logdensity_grad
    ), kinetic_energy_change


def _partially_refresh_momentum(momentum, rng_key, step_size, L):
    flat_momentum, unravel_fn = ravel_pytree(momentum)
    dim = flat_momentum.shape[0]
    nu = jnp.sqrt((jnp.exp(2 * step_size / L) - 1.0) / dim)
    z = nu * jax.random.normal(
        rng_key, shape=flat_momentum.shape, dtype=flat_momentum.dtype
    )
    new_momentum = unravel_fn((flat_momentum + z) / jnp.linalg.norm(flat_momentum + z))
    return jax.lax.cond(
        jnp.isinf(L), lambda _: momentum, lambda _: new_momentum, operand=None
    )


def _maruyama_step(
    init_state, step_size, L, rng_key, logdensity_fn, inverse_mass_matrix
):
    key1, key2 = jax.random.split(rng_key)
    state = init_state._replace(
        momentum=_partially_refresh_momentum(
            momentum=init_state.momentum,
            rng_key=key1,
            L=L,
            step_size=step_size * 0.5,
        )
    )
    state, kinetic_change = _isokinetic_mclachlan_step(
        state=state,
        step_size=step_size,
        logdensity_fn=logdensity_fn,
        inverse_mass_matrix=inverse_mass_matrix,
    )
    state = state._replace(
        momentum=_partially_refresh_momentum(
            momentum=state.momentum,
            rng_key=key2,
            L=L,
            step_size=step_size * 0.5,
        )
    )
    return state, kinetic_change


def _handle_nans(previous_state, next_state, info, key):
    new_momentum = _generate_unit_vector(key, previous_state.position)
    flat_position, _ = ravel_pytree(next_state.position)
    flat_momentum, _ = ravel_pytree(next_state.momentum)
    nonans = jnp.logical_and(
        jnp.all(jnp.isfinite(flat_position)), jnp.all(jnp.isfinite(flat_momentum))
    )
    state, info = jax.lax.cond(
        nonans,
        lambda: (next_state, info),
        lambda: (
            previous_state._replace(momentum=new_momentum),
            MCLMCInfo(
                logdensity=previous_state.logdensity,
                energy_change=jnp.zeros_like(info.energy_change),
                kinetic_change=jnp.zeros_like(info.kinetic_change),
            ),
        ),
    )
    return state, info


def _build_kernel(logdensity_fn, inverse_mass_matrix):
    def kernel(rng_key, state, L, step_size):
        kernel_key, nan_key = jax.random.split(rng_key)
        next_state, kinetic_change = _maruyama_step(
            init_state=state,
            step_size=step_size,
            L=L,
            rng_key=kernel_key,
            logdensity_fn=logdensity_fn,
            inverse_mass_matrix=inverse_mass_matrix,
        )
        energy_change = kinetic_change - next_state.logdensity + state.logdensity
        next_state, info = _handle_nans(
            previous_state=state,
            next_state=next_state,
            info=MCLMCInfo(
                logdensity=next_state.logdensity,
                energy_change=energy_change,
                kinetic_change=kinetic_change,
            ),
            key=nan_key,
        )
        return next_state, info

    return kernel


def _adaptation_handle_nans(
    previous_state, next_state, step_size, step_size_max, kinetic_change, key
):
    flat_position, _ = ravel_pytree(next_state.position)
    flat_momentum, _ = ravel_pytree(next_state.momentum)
    nonans = jnp.logical_and(
        jnp.all(jnp.isfinite(flat_position)), jnp.all(jnp.isfinite(flat_momentum))
    )
    state, step_size, kinetic_change = jax.tree.map(
        lambda new, old: jax.lax.select(nonans, jnp.nan_to_num(new), old),
        (next_state, step_size_max, kinetic_change),
        (previous_state, step_size * _DELTA_NAN_STEP_SIZE_FACTOR, 0.0),
    )
    state = jax.lax.cond(
        jnp.isnan(next_state.logdensity),
        lambda: state._replace(
            momentum=_generate_unit_vector(key, previous_state.position)
        ),
        lambda: state,
    )
    return nonans, state, step_size, kinetic_change


def _make_l_step_size_adaptation(
    kernel_fn,
    dim,
    frac_tune1,
    frac_tune2,
    diagonal_preconditioning,
    desired_energy_var=1e-3,
    trust_in_estimate=1.5,
    num_effective_samples=150,
):
    decay_rate = (num_effective_samples - 1.0) / (num_effective_samples + 1.0)

    def predictor(previous_state, params, adaptive_state, rng_key):
        time, x_average, step_size_max = adaptive_state
        rng_key, nan_key = jax.random.split(rng_key)
        next_state, info = kernel_fn(params.inverse_mass_matrix)(
            rng_key=rng_key,
            state=previous_state,
            L=params.L,
            step_size=params.step_size,
        )
        success, state, step_size_max, energy_change = _adaptation_handle_nans(
            previous_state=previous_state,
            next_state=next_state,
            step_size=params.step_size,
            step_size_max=step_size_max,
            kinetic_change=info.energy_change,
            key=nan_key,
        )
        xi = jnp.square(energy_change) / (dim * desired_energy_var) + 1e-8
        weight = jnp.exp(-0.5 * jnp.square(jnp.log(xi) / (6.0 * trust_in_estimate)))
        x_average = decay_rate * x_average + weight * (
            xi / jnp.power(params.step_size, 6.0)
        )
        time = decay_rate * time + weight
        step_size = jnp.power(x_average / time, -1.0 / 6.0)
        step_size = jnp.where(step_size < step_size_max, step_size, step_size_max)
        params_new = params._replace(step_size=step_size)
        return state, params_new, (time, x_average, step_size_max), success

    def step(iteration_state, weight_and_key):
        mask, rng_key = weight_and_key
        state, params, adaptive_state, streaming_avg = iteration_state
        state, params, adaptive_state, success = predictor(
            state, params, adaptive_state, rng_key
        )
        x = ravel_pytree(state.position)[0]
        streaming_avg = _incremental_value_update(
            expectation=jnp.array([x, jnp.square(x)]),
            incremental_val=streaming_avg,
            weight=mask * success * params.step_size,
        )
        return (state, params, adaptive_state, streaming_avg), None

    def run_steps(xs, state, params):
        return jax.lax.scan(
            step,
            init=(
                state,
                params,
                (0.0, 0.0, jnp.inf),
                (0.0, jnp.array([jnp.zeros(dim), jnp.zeros(dim)])),
            ),
            xs=xs,
        )[0]

    def adaptation(state, params, num_steps, rng_key):
        num_steps1 = round(num_steps * frac_tune1)
        num_steps2 = round(num_steps * frac_tune2)
        keys = jax.random.split(rng_key, num_steps1 + num_steps2 + 1)
        tune_keys, final_key = keys[:-1], keys[-1]
        mask = jnp.concatenate((jnp.zeros(num_steps1), jnp.ones(num_steps2)))

        state, params, _, (_, average) = run_steps((mask, tune_keys), state, params)
        L = params.L
        inverse_mass_matrix = params.inverse_mass_matrix
        if num_steps2 > 1:
            x_average, x_squared_average = average[0], average[1]
            variances = x_squared_average - jnp.square(x_average)
            L = jnp.sqrt(jnp.sum(variances))
            if diagonal_preconditioning:
                inverse_mass_matrix = variances
                params = params._replace(inverse_mass_matrix=inverse_mass_matrix)
                L = jnp.sqrt(dim)
            steps = round(num_steps2 / 3)
            keys = jax.random.split(final_key, steps)
            state, params, _, (_, _) = run_steps((jnp.ones(steps), keys), state, params)
        return state, MCLMCAdaptationState(L, params.step_size, inverse_mass_matrix)

    return adaptation


def _make_adaptation_l(kernel, frac, lfactor):
    def adaptation_l(state, params, num_steps, rng_key):
        num_steps3 = round(num_steps * frac)
        keys = jax.random.split(rng_key, num_steps3)

        def step(curr_state, key):
            next_state, _ = kernel(
                rng_key=key,
                state=curr_state,
                L=params.L,
                step_size=params.step_size,
            )
            return next_state, next_state.position

        state, samples = jax.lax.scan(step, init=state, xs=keys)
        flat_samples = jax.vmap(lambda x: ravel_pytree(x)[0])(samples)
        ess = effective_sample_size(flat_samples[None, ...])
        return state, params._replace(
            L=lfactor * params.step_size * jnp.mean(num_steps3 / ess)
        )

    return adaptation_l


def _mclmc_find_l_and_step_size(
    mclmc_kernel,
    num_steps,
    state,
    rng_key,
    frac_tune1=0.1,
    frac_tune2=0.1,
    frac_tune3=0.1,
    desired_energy_var=5e-4,
    trust_in_estimate=1.5,
    num_effective_samples=150,
    diagonal_preconditioning=True,
    params=None,
    lfactor=0.4,
):
    dim = _pytree_size(state.position)
    if params is None:
        params = MCLMCAdaptationState(
            L=jnp.sqrt(dim),
            step_size=jnp.sqrt(dim) * 0.25,
            inverse_mass_matrix=jnp.ones((dim,)),
        )

    part1_key, part2_key = jax.random.split(rng_key, 2)
    num_steps1 = round(num_steps * frac_tune1)
    num_steps2 = round(num_steps * frac_tune2)
    num_steps2 += diagonal_preconditioning * (num_steps2 // 3)
    num_steps3 = round(num_steps * frac_tune3)

    state, params = _make_l_step_size_adaptation(
        kernel_fn=mclmc_kernel,
        dim=dim,
        frac_tune1=frac_tune1,
        frac_tune2=frac_tune2,
        desired_energy_var=desired_energy_var,
        trust_in_estimate=trust_in_estimate,
        num_effective_samples=num_effective_samples,
        diagonal_preconditioning=diagonal_preconditioning,
    )(state, params, num_steps, part1_key)

    total_num_tuning_integrator_steps = num_steps1 + num_steps2
    if num_steps3 >= 2:
        state, params = _make_adaptation_l(
            kernel=mclmc_kernel(params.inverse_mass_matrix),
            frac=frac_tune3,
            lfactor=lfactor,
        )(state, params, num_steps, part2_key)
        total_num_tuning_integrator_steps += num_steps3
    return state, params, total_num_tuning_integrator_steps


class MCLMC(MCMCKernel):
    """
    Microcanonical Langevin Monte Carlo (MCLMC) kernel.

    This kernel implements an isokinetic integrator with stochastic momentum
    refreshment. During warmup, it automatically tunes step size, momentum
    decoherence length ``L``, and optionally a diagonal preconditioner.
    The resulting state can be used with :class:`~numpyro.infer.mcmc.MCMC`.

    Example
    -------

    A minimal 2D model:

    .. code-block:: python

        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC
        from numpyro.infer.mclmc import MCLMC

        def model():
            numpyro.sample("x", dist.Normal(jnp.array([0.0, 0.0]), 1.0).to_event(1))

        kernel = MCLMC(model=model)
        mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, progress_bar=False)
        mcmc.run(jax.random.key(0))
        samples = mcmc.get_samples()

    Model with observed data and tuned energy variance:

    .. code-block:: python

        def model(X, y=None):
            w = numpyro.sample("w", dist.Normal(jnp.zeros(X.shape[-1]), 1.0))
            logits = X @ w
            numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)

        kernel = MCLMC(
            model=model,
            desired_energy_var=5e-4,
            diagonal_preconditioning=True,
        )
        mcmc = MCMC(kernel, num_warmup=1500, num_samples=1000, progress_bar=False)
        mcmc.run(jax.random.key(1), X, y)

    **References:**

    1. *Microcanonical Hamiltonian Monte Carlo*,
       Jakob Robnik, G. Bruno De Luca, Eva Silverstein, Uroš Seljak
       https://arxiv.org/abs/2212.08549

    .. note:: The model must have at least 2 unconstrained latent dimensions.
        This limitation comes from the isokinetic MCLMC dynamics.

    :param model: Python callable containing NumPyro primitives.
    :param float desired_energy_var: Target energy variance used in warmup to tune
        step size. Smaller values generally lead to more conservative integration
        steps. Defaults to ``5e-4``.
    :param bool diagonal_preconditioning: Whether warmup should estimate a diagonal
        inverse mass matrix. If ``False``, adaptation uses isotropic scaling.
        Defaults to ``True``.
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
        return "step_size={:.2e}, L={:.2e}".format(
            self.adapt_state.step_size, self.adapt_state.L
        )

    def postprocess_fn(self, args, kwargs):
        if self._postprocess_fn is None:
            return identity
        return self._postprocess_fn(*args, **kwargs)

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        init_model_key, init_state_key, run_key, tune_key = jax.random.split(rng_key, 4)
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

        self.logdensity_fn = logdensity_fn
        sampler_state = _init_mclmc(
            position=init_params.z,
            logdensity_fn=self.logdensity_fn,
            rng_key=init_state_key,
        )

        def kernel_fn(inverse_mass_matrix):
            return _build_kernel(
                logdensity_fn=self.logdensity_fn,
                inverse_mass_matrix=inverse_mass_matrix,
            )

        num_tuning_steps = 100
        tuned_state, self.adapt_state, _ = _mclmc_find_l_and_step_size(
            mclmc_kernel=kernel_fn,
            num_steps=num_tuning_steps,
            state=sampler_state,
            rng_key=tune_key,
            diagonal_preconditioning=self._diagonal_preconditioning,
            frac_tune1=num_warmup / (3 * num_tuning_steps),
            frac_tune2=num_warmup / (3 * num_tuning_steps),
            frac_tune3=num_warmup / (3 * num_tuning_steps),
            desired_energy_var=self._desired_energy_var,
        )
        self._kernel = _build_kernel(
            logdensity_fn=self.logdensity_fn,
            inverse_mass_matrix=self.adapt_state.inverse_mass_matrix,
        )
        return FullState(
            tuned_state.position,
            tuned_state.momentum,
            tuned_state.logdensity,
            tuned_state.logdensity_grad,
            run_key,
        )

    def sample(self, state, model_args, model_kwargs):
        mclmc_state = MCLMCState(
            state.position, state.momentum, state.logdensity, state.logdensity_grad
        )
        rng_key, sample_key = jax.random.split(state.rng_key, 2)
        new_state, _ = self._kernel(
            rng_key=sample_key,
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
