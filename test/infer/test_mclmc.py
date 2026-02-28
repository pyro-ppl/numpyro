# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0


from numpy.testing import assert_allclose
import pytest

import jax
from jax import random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC
import numpyro.infer.mclmc as mclmc_module
from numpyro.infer.mclmc import MCLMC


def _two_dim_model():
    numpyro.sample("x", dist.Normal(jnp.array([0.0, 0.0]), 1.0).to_event(1))


def _model_with_args(loc, scale=1.0):
    numpyro.sample("x", dist.Normal(loc, scale).to_event(1))


def _gaussian_logdensity(x):
    return -0.5 * jnp.sum(jnp.square(x))


def _make_test_state(key=None):
    if key is None:
        key = random.PRNGKey(0)
    return mclmc_module._init_mclmc(
        position=jnp.array([0.3, -0.7]),
        logdensity_fn=_gaussian_logdensity,
        rng_key=key,
    )


def test_pytree_size_counts_all_leaves():
    pytree = {"a": jnp.zeros((2, 3)), "b": [jnp.ones((4,)), jnp.ones(())]}
    assert mclmc_module._pytree_size(pytree) == 11


def test_generate_unit_vector_has_unit_norm():
    position = jnp.array([1.0, 2.0, 3.0])
    vec = mclmc_module._generate_unit_vector(random.PRNGKey(0), position)
    flat_vec, _ = jax.flatten_util.ravel_pytree(vec)
    assert_allclose(jnp.linalg.norm(flat_vec), 1.0, atol=1e-6)
    assert flat_vec.shape == position.shape


def test_incremental_value_update_weighted_average():
    avg = mclmc_module._StreamingAverage(
        total=jnp.array(0.0), average=jnp.array([0.0, 0.0])
    )
    avg = mclmc_module._incremental_value_update(
        expectation=jnp.array([2.0, 4.0]), incremental_val=avg, weight=2.0
    )
    avg = mclmc_module._incremental_value_update(
        expectation=jnp.array([4.0, 8.0]), incremental_val=avg, weight=2.0
    )
    assert_allclose(avg.average, jnp.array([3.0, 6.0]), atol=1e-6)
    assert_allclose(avg.total, 4.0, atol=1e-6)


def test_incremental_value_update_zero_numerator_safe():
    avg = mclmc_module._StreamingAverage(total=jnp.array(0.0), average=jnp.array([0.0]))
    updated = mclmc_module._incremental_value_update(
        expectation=jnp.array([0.0]), incremental_val=avg, weight=0.0
    )
    assert_allclose(updated.average, jnp.array([0.0]))


def test_init_mclmc_rejects_low_dimension():
    with pytest.raises(
        ValueError, match="target distribution must have more than 1 dimension"
    ):
        mclmc_module._init_mclmc(
            position=jnp.array([0.0]),
            logdensity_fn=lambda x: -0.5 * jnp.sum(x**2),
            rng_key=random.PRNGKey(0),
        )


def test_init_mclmc_returns_valid_state():
    state = _make_test_state(random.PRNGKey(1))
    flat_momentum, _ = jax.flatten_util.ravel_pytree(state.momentum)
    assert jnp.isfinite(state.logdensity)
    assert jnp.all(jnp.isfinite(state.logdensity_grad))
    assert_allclose(jnp.linalg.norm(flat_momentum), 1.0, atol=1e-6)


def test_position_update_matches_expected_gaussian_update():
    position = jnp.array([1.0, 2.0])
    kinetic_grad = jnp.array([0.5, -1.0])
    new_position, logdensity, grad = mclmc_module._position_update(
        position=position,
        kinetic_grad=kinetic_grad,
        step_size=0.1,
        coef=0.5,
        logdensity_fn=_gaussian_logdensity,
    )
    expected_position = jnp.array([1.025, 1.95])
    assert_allclose(new_position, expected_position, atol=1e-7)
    assert_allclose(logdensity, _gaussian_logdensity(expected_position), atol=1e-7)
    assert_allclose(grad, -expected_position, atol=1e-7)


def test_normalized_flatten_for_nonzero_and_zero_vectors():
    normalized, norm = mclmc_module._normalized_flatten(jnp.array([3.0, 4.0]))
    assert_allclose(normalized, jnp.array([0.6, 0.8]), atol=1e-7)
    assert_allclose(norm, 5.0, atol=1e-7)

    normalized_zero, norm_zero = mclmc_module._normalized_flatten(jnp.zeros((3,)))
    assert_allclose(normalized_zero, jnp.zeros((3,)), atol=1e-7)
    assert_allclose(norm_zero, 0.0, atol=1e-7)


def test_esh_dynamics_momentum_update_matches_naive_formula():
    step_size = 1e-3
    key0, key1 = random.split(random.PRNGKey(62))
    gradient = random.uniform(key0, shape=(3,))
    momentum = random.uniform(key1, shape=(3,))
    momentum = momentum / jnp.linalg.norm(momentum)

    gradient_norm = jnp.linalg.norm(gradient)
    gradient_normalized = gradient / gradient_norm
    delta = step_size * gradient_norm / (momentum.shape[0] - 1)
    naive_next = (
        momentum
        + gradient_normalized
        * (
            jnp.sinh(delta)
            + jnp.dot(gradient_normalized, momentum * (jnp.cosh(delta) - 1))
        )
    ) / (jnp.cosh(delta) + jnp.dot(gradient_normalized, momentum * jnp.sinh(delta)))
    naive_next = naive_next / jnp.linalg.norm(naive_next)

    next_momentum, _, _ = mclmc_module._esh_dynamics_momentum_update_one_step(
        momentum=momentum,
        logdensity_grad=gradient,
        step_size=step_size,
        coef=1.0,
        inverse_mass_matrix=jnp.ones((3,)),
    )
    assert_allclose(next_momentum, naive_next, atol=1e-6)


def test_isokinetic_mclachlan_step_returns_finite_state_and_unit_momentum():
    state = _make_test_state(random.PRNGKey(0))
    next_state, kinetic_change = mclmc_module._isokinetic_mclachlan_step(
        state=state,
        step_size=1e-3,
        logdensity_fn=_gaussian_logdensity,
        inverse_mass_matrix=jnp.ones((2,)),
    )
    flat_momentum, _ = jax.flatten_util.ravel_pytree(next_state.momentum)
    assert jnp.isfinite(kinetic_change)
    assert jnp.isfinite(next_state.logdensity)
    assert jnp.all(jnp.isfinite(next_state.logdensity_grad))
    assert_allclose(jnp.linalg.norm(flat_momentum), 1.0, atol=1e-5)


def test_partially_refresh_momentum_respects_infinite_l():
    momentum = jnp.array([1.0, 0.0])
    refreshed_inf = mclmc_module._partially_refresh_momentum(
        momentum=momentum,
        rng_key=random.PRNGKey(0),
        step_size=0.1,
        L=jnp.inf,
    )
    assert_allclose(refreshed_inf, momentum)

    refreshed = mclmc_module._partially_refresh_momentum(
        momentum=momentum,
        rng_key=random.PRNGKey(0),
        step_size=0.1,
        L=1.0,
    )
    assert_allclose(jnp.linalg.norm(refreshed), 1.0, atol=1e-6)


def test_maruyama_step_returns_finite_values():
    state = _make_test_state(random.PRNGKey(0))
    next_state, kinetic_change = mclmc_module._maruyama_step(
        init_state=state,
        step_size=1e-2,
        L=1.0,
        rng_key=random.PRNGKey(1),
        logdensity_fn=_gaussian_logdensity,
        inverse_mass_matrix=jnp.ones((2,)),
    )
    assert jnp.isfinite(kinetic_change)
    assert jnp.isfinite(next_state.logdensity)
    assert jnp.all(jnp.isfinite(next_state.logdensity_grad))
    assert mclmc_module._state_is_finite(next_state)


def test_state_is_finite_detects_nan_and_inf():
    state = _make_test_state(random.PRNGKey(0))
    assert mclmc_module._state_is_finite(state)

    nan_state = state._replace(position=jnp.array([jnp.nan, 0.0]))
    inf_state = state._replace(momentum=jnp.array([jnp.inf, 0.0]))
    assert not mclmc_module._state_is_finite(nan_state)
    assert not mclmc_module._state_is_finite(inf_state)


def test_fallback_state_with_fresh_momentum_preserves_position_and_logdensity():
    state = _make_test_state(random.PRNGKey(0))
    new_state = mclmc_module._fallback_state_with_fresh_momentum(
        previous_state=state, key=random.PRNGKey(2)
    )
    assert_allclose(new_state.position, state.position)
    assert_allclose(new_state.logdensity, state.logdensity)
    assert_allclose(jnp.linalg.norm(new_state.momentum), 1.0, atol=1e-6)


def test_handle_nans_keeps_valid_state_and_falls_back_for_invalid_state():
    previous = _make_test_state(random.PRNGKey(0))
    valid_next = previous._replace(position=previous.position + 0.1)
    info = mclmc_module.MCLMCInfo(
        logdensity=valid_next.logdensity,
        kinetic_change=jnp.array(0.3),
        energy_change=jnp.array(0.2),
    )
    state_ok, info_ok = mclmc_module._handle_nans(
        previous_state=previous, next_state=valid_next, info=info, key=random.PRNGKey(1)
    )
    assert_allclose(state_ok.position, valid_next.position)
    assert_allclose(info_ok.energy_change, info.energy_change)

    invalid_next = valid_next._replace(position=jnp.array([jnp.nan, 0.0]))
    state_bad, info_bad = mclmc_module._handle_nans(
        previous_state=previous,
        next_state=invalid_next,
        info=info,
        key=random.PRNGKey(3),
    )
    assert_allclose(state_bad.position, previous.position)
    assert_allclose(info_bad.logdensity, previous.logdensity)
    assert_allclose(info_bad.energy_change, 0.0)
    assert_allclose(info_bad.kinetic_change, 0.0)


def test_build_kernel_single_step_outputs_finite_state_and_info():
    kernel = mclmc_module._build_kernel(
        logdensity_fn=_gaussian_logdensity, inverse_mass_matrix=jnp.ones((2,))
    )
    state = _make_test_state(random.PRNGKey(0))
    next_state, info = kernel(
        rng_key=random.PRNGKey(1), state=state, L=1.0, step_size=1e-2
    )
    assert mclmc_module._state_is_finite(next_state)
    assert jnp.isfinite(info.logdensity)
    assert jnp.isfinite(info.energy_change)
    assert jnp.isfinite(info.kinetic_change)


def test_adaptation_handle_nans_behavior():
    previous = _make_test_state(random.PRNGKey(0))
    next_state = previous._replace(position=previous.position + 0.1)
    success, state, new_step_size_max, new_kinetic = (
        mclmc_module._adaptation_handle_nans(
            previous_state=previous,
            next_state=next_state,
            step_size=jnp.array(0.2),
            step_size_max=jnp.array(0.5),
            kinetic_change=jnp.array(0.1),
            key=random.PRNGKey(2),
        )
    )
    assert success
    assert_allclose(state.position, next_state.position)
    assert_allclose(new_step_size_max, 0.5)
    assert_allclose(new_kinetic, 0.1)

    invalid = next_state._replace(
        position=jnp.array([jnp.nan, 0.0]), logdensity=jnp.nan
    )
    success, state, new_step_size_max, new_kinetic = (
        mclmc_module._adaptation_handle_nans(
            previous_state=previous,
            next_state=invalid,
            step_size=jnp.array(0.2),
            step_size_max=jnp.array(0.5),
            kinetic_change=jnp.array(0.1),
            key=random.PRNGKey(3),
        )
    )
    assert not success
    assert_allclose(new_step_size_max, 0.2 * mclmc_module._DELTA_NAN_STEP_SIZE_FACTOR)
    assert_allclose(new_kinetic, 0.0)
    assert_allclose(state.position, previous.position)


def test_make_l_step_size_adaptation_returns_finite_positive_params():
    dim = 2
    initial_state = _make_test_state(random.PRNGKey(0))
    params = mclmc_module.MCLMCAdaptationState(
        L=jnp.sqrt(dim),
        step_size=0.2,
        inverse_mass_matrix=jnp.ones((dim,)),
    )
    adaptation = mclmc_module._make_l_step_size_adaptation(
        kernel_fn=lambda imm: mclmc_module._build_kernel(_gaussian_logdensity, imm),
        dim=dim,
        frac_tune1=0.2,
        frac_tune2=0.2,
        diagonal_preconditioning=True,
    )
    state, new_params = adaptation(
        initial_state,
        params,
        num_steps=30,
        rng_key=random.PRNGKey(1),
    )
    assert mclmc_module._state_is_finite(state)
    assert jnp.isfinite(new_params.L) and (new_params.L > 0)
    assert jnp.isfinite(new_params.step_size) and (new_params.step_size > 0)
    assert new_params.inverse_mass_matrix.shape == (dim,)


def test_make_adaptation_l_nominal_case_updates_l():
    state = _make_test_state(random.PRNGKey(0))
    params = mclmc_module.MCLMCAdaptationState(
        L=jnp.array(1.0),
        step_size=jnp.array(0.1),
        inverse_mass_matrix=jnp.ones((2,)),
    )
    kernel = mclmc_module._build_kernel(_gaussian_logdensity, jnp.ones((2,)))
    adaptation_l = mclmc_module._make_adaptation_l(kernel=kernel, frac=0.5, lfactor=0.4)
    _, new_params = adaptation_l(
        state=state,
        params=params,
        num_steps=12,
        rng_key=random.PRNGKey(2),
    )
    assert jnp.isfinite(new_params.L)
    assert new_params.L > 0


def test_mclmc_find_l_and_step_size_returns_expected_phase_accounting():
    state = _make_test_state(random.PRNGKey(0))
    state, params, total_steps = mclmc_module._mclmc_find_l_and_step_size(
        mclmc_kernel=lambda imm: mclmc_module._build_kernel(_gaussian_logdensity, imm),
        num_steps=20,
        state=state,
        rng_key=random.PRNGKey(1),
        frac_tune1=0.2,
        frac_tune2=0.2,
        frac_tune3=0.2,
        diagonal_preconditioning=True,
    )
    expected_num_steps1 = round(20 * 0.2)
    expected_num_steps2 = round(20 * 0.2) + (round(20 * 0.2) // 3)
    expected_num_steps3 = round(20 * 0.2)
    assert (
        total_steps == expected_num_steps1 + expected_num_steps2 + expected_num_steps3
    )
    assert mclmc_module._state_is_finite(state)
    assert jnp.isfinite(params.L) and (params.L > 0)
    assert jnp.isfinite(params.step_size) and (params.step_size > 0)
    assert params.inverse_mass_matrix.shape == (2,)


def test_mclmc_model_required():
    """Test that ValueError is raised when model is None."""
    with pytest.raises(ValueError, match="Model must be specified"):
        MCLMC(model=None)


def test_mclmc_normal():
    """Test MCLMC with a 2D normal distribution."""
    true_mean = jnp.array([1.0, 2.0])
    true_std = jnp.array([0.5, 1.0])
    num_warmup, num_samples = 1000, 2000

    def model():
        numpyro.sample("x", dist.Normal(true_mean, true_std).to_event(1))

    kernel = MCLMC(model=model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(0))
    samples = mcmc.get_samples()

    assert "x" in samples
    assert samples["x"].shape == (num_samples, 2)
    assert_allclose(jnp.mean(samples["x"], axis=0), true_mean, atol=0.1)
    assert_allclose(jnp.std(samples["x"], axis=0), true_std, atol=0.2)


def test_mclmc_gaussian_2d():
    """Test MCLMC with a 2D Gaussian model with observation."""
    num_warmup, num_samples = 1000, 1000

    def model():
        x = numpyro.sample("x", dist.Normal(0.0, 1.0))
        y = numpyro.sample("y", dist.Normal(0.0, 1.0))
        numpyro.sample("obs", dist.Normal(x + y, 0.5), obs=jnp.array(0.0))

    kernel = MCLMC(
        model=model,
        diagonal_preconditioning=True,
        desired_energy_var=5e-4,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(0))
    samples = mcmc.get_samples()

    assert "x" in samples
    assert "y" in samples
    assert samples["x"].shape == (num_samples,)
    assert samples["y"].shape == (num_samples,)
    # With obs=0, x+y should be close to 0, so means should be near 0
    assert_allclose(jnp.mean(samples["x"]) + jnp.mean(samples["y"]), 0.0, atol=0.2)


def test_mclmc_logistic_regression():
    """Test MCLMC with a logistic regression model."""
    N, dim = 1000, 3
    num_warmup, num_samples = 1000, 2000

    key1, key2, key3 = random.split(random.PRNGKey(0), 3)
    data = random.normal(key1, (N, dim))
    true_coefs = jnp.arange(1.0, dim + 1.0)
    logits = jnp.sum(true_coefs * data, axis=-1)
    labels = dist.Bernoulli(logits=logits).sample(key2)

    # Closure pattern is used here for compactness.
    def model():
        coefs = numpyro.sample("coefs", dist.Normal(jnp.zeros(dim), jnp.ones(dim)))
        logits = jnp.sum(coefs * data, axis=-1)
        numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=labels)

    kernel = MCLMC(model=model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(key3)
    samples = mcmc.get_samples()

    assert "coefs" in samples
    assert samples["coefs"].shape == (num_samples, dim)
    assert_allclose(jnp.mean(samples["coefs"], 0), true_coefs, atol=0.5)


def test_mclmc_sample_shape():
    """Test that MCLMC produces samples with expected shapes."""
    num_warmup, num_samples = 500, 500

    def model():
        numpyro.sample("a", dist.Normal(0, 1))
        numpyro.sample("b", dist.Normal(0, 1).expand([3]))
        numpyro.sample("c", dist.Normal(0, 1).expand([2, 4]))

    kernel = MCLMC(model=model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(0))
    samples = mcmc.get_samples()

    assert samples["a"].shape == (num_samples,)
    assert samples["b"].shape == (num_samples, 3)
    assert samples["c"].shape == (num_samples, 2, 4)


def test_mclmc_model_args_and_kwargs():
    """Test that model_args/model_kwargs are respected during inference."""
    true_mean = jnp.array([1.5, -0.5])
    true_scale = 0.8
    num_warmup, num_samples = 500, 1000

    kernel = MCLMC(model=_model_with_args)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(1), true_mean, scale=true_scale)
    samples = mcmc.get_samples()["x"]

    assert samples.shape == (num_samples, 2)
    assert_allclose(jnp.mean(samples, axis=0), true_mean, atol=0.2)
    assert_allclose(jnp.std(samples, axis=0), true_scale, atol=0.2)


def test_mclmc_rejects_one_dimensional_latent_space():
    """Test that MCLMC rejects models with fewer than 2 latent dimensions."""

    def one_dim_model():
        numpyro.sample("x", dist.Normal(0.0, 1.0))

    kernel = MCLMC(model=one_dim_model)
    mcmc = MCMC(
        kernel,
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        progress_bar=False,
    )
    with pytest.raises(
        ValueError,
        match="target distribution must have more than 1 dimension",
    ):
        mcmc.run(random.PRNGKey(0))


def test_mclmc_small_warmup_runs():
    """Test small warmup edge case where adaptation phases are tiny."""
    kernel = MCLMC(model=_two_dim_model)
    mcmc = MCMC(
        kernel,
        num_warmup=3,
        num_samples=20,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(2))
    samples = mcmc.get_samples()["x"]
    assert samples.shape == (20, 2)


def test_mclmc_public_properties_and_diagnostics():
    kernel = MCLMC(model=_two_dim_model)
    assert kernel.model is _two_dim_model
    assert kernel.sample_field == "position"
    assert kernel.default_fields == ("position",)
    assert kernel.get_diagnostics_str(None) == ""
    kernel.adapt_state = mclmc_module.MCLMCAdaptationState(
        L=jnp.array(1.2), step_size=jnp.array(0.05), inverse_mass_matrix=jnp.ones((2,))
    )
    assert "step_size=" in kernel.get_diagnostics_str(None)
    assert "L=" in kernel.get_diagnostics_str(None)


def test_mclmc_postprocess_fn_identity_when_uninitialized():
    kernel = MCLMC(model=_two_dim_model)
    fn = kernel.postprocess_fn((), {})
    x = {"z": jnp.array([1.0, 2.0])}
    out = fn(x)
    assert out is x


def test_mclmc_sample_raises_if_not_initialized():
    kernel = MCLMC(model=_two_dim_model)
    state = mclmc_module.FullState(
        position=jnp.array([0.0, 0.0]),
        momentum=jnp.array([1.0, 0.0]),
        logdensity=jnp.array(0.0),
        logdensity_grad=jnp.array([0.0, 0.0]),
        rng_key=random.PRNGKey(0),
    )
    with pytest.raises(RuntimeError, match="must be initialized"):
        kernel.sample(state, (), {})


def test_mclmc_init_and_sample_direct_api():
    kernel = MCLMC(model=_two_dim_model)
    state = kernel.init(
        rng_key=random.PRNGKey(0),
        num_warmup=30,
        init_params=None,
        model_args=(),
        model_kwargs={},
    )
    assert isinstance(state, mclmc_module.FullState)
    next_state = kernel.sample(state, (), {})
    assert isinstance(next_state, mclmc_module.FullState)
    assert next_state.position["x"].shape == (2,)


def test_mclmc_postprocess_fn_after_init_returns_callable():
    kernel = MCLMC(model=_two_dim_model)
    kernel.init(
        rng_key=random.PRNGKey(1),
        num_warmup=10,
        init_params=None,
        model_args=(),
        model_kwargs={},
    )
    fn = kernel.postprocess_fn((), {})
    assert callable(fn)


def test_mclmc_getstate_clears_postprocess_fn():
    kernel = MCLMC(model=_two_dim_model)
    kernel._postprocess_fn = lambda *args, **kwargs: lambda x: x
    state = kernel.__getstate__()
    assert state["_postprocess_fn"] is None
    assert state["_model"] is _two_dim_model


def test_mclmc_adaptation_l_handles_bad_ess(monkeypatch):
    """Test ESS guard keeps L finite for degenerate ESS estimates."""
    state = mclmc_module.MCLMCState(
        position=jnp.array([0.0, 0.0]),
        momentum=jnp.array([1.0, 0.0]),
        logdensity=jnp.array(0.0),
        logdensity_grad=jnp.array([0.0, 0.0]),
    )
    params = mclmc_module.MCLMCAdaptationState(
        L=jnp.array(1.0),
        step_size=jnp.array(0.1),
        inverse_mass_matrix=jnp.ones((2,)),
    )

    def dummy_kernel(rng_key, state, L, step_size):
        del rng_key, L, step_size
        return state, mclmc_module.MCLMCInfo(
            logdensity=state.logdensity,
            kinetic_change=jnp.array(0.0),
            energy_change=jnp.array(0.0),
        )

    monkeypatch.setattr(
        mclmc_module,
        "effective_sample_size",
        lambda _: jnp.array([0.0, jnp.nan, jnp.inf]),
    )

    adaptation_l = mclmc_module._make_adaptation_l(
        kernel=dummy_kernel,
        frac=0.5,
        lfactor=0.4,
    )
    _, new_params = adaptation_l(
        state=state,
        params=params,
        num_steps=10,
        rng_key=random.PRNGKey(0),
    )
    assert jnp.isfinite(new_params.L)
    assert new_params.L > 0
