# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from numpy.testing import assert_allclose
import pytest

from jax import random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import (
    MCMC,
    NUTS,
    ExternalKernel,
    Predictive,
    constrain_samples,
    get_log_density_fn,
    init_to_value,
)
from numpyro.infer.external import _resolve_init_position
from numpyro.infer.util import ParamInfo, potential_energy


@pytest.fixture(scope="module")
def blackjax():
    """Import blackjax lazily so its module-level constants don't trip the
    conftest's first-test ``live_arrays() == 0`` invariant."""
    return pytest.importorskip("blackjax", minversion="1.5")


def _linreg_model(x, y=None):
    """Simple linear regression model used as the integration target."""
    a = numpyro.sample("a", dist.Normal(0.0, 2.0))
    b = numpyro.sample("b", dist.HalfNormal(2.0))
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    mu = numpyro.deterministic("mu", a + b * x)
    with numpyro.plate("data", len(x)):
        numpyro.sample("likelihood", dist.Normal(mu, sigma), obs=y)


def _linreg_data(seed=0, n=80):
    key = random.key(seed)
    k1, k2 = random.split(key)
    x = random.normal(k1, (n,))
    y = 1.0 + 2.0 * x + 0.3 * random.normal(k2, (n,))
    return x, y


def _make_build_nuts(blackjax_mod, num_warmup=500):
    """Factory: closes blackjax over the build_kernel callback for NUTS."""

    def build(rng_key, logdensity_fn, init_position):
        adapt = blackjax_mod.window_adaptation(blackjax_mod.nuts, logdensity_fn)
        (state, params), _ = adapt.run(rng_key, init_position, num_steps=num_warmup)
        final = blackjax_mod.nuts(logdensity_fn, **params)
        return state, final.step, lambda s: s.position

    return build


def _make_build_mclmc(blackjax_mod, num_warmup=1000):
    """Factory: closes blackjax over the build_kernel callback for MCLMC."""
    from blackjax.mcmc.integrators import isokinetic_mclachlan

    def build(rng_key, logdensity_fn, init_position):
        key_init, key_tune = random.split(rng_key)
        init_state = blackjax_mod.mcmc.mclmc.init(
            position=init_position,
            logdensity_fn=logdensity_fn,
            rng_key=key_init,
        )

        def kernel_factory(inverse_mass_matrix):
            return blackjax_mod.mcmc.mclmc.build_kernel(
                logdensity_fn=logdensity_fn,
                inverse_mass_matrix=inverse_mass_matrix,
                integrator=isokinetic_mclachlan,
            )

        state, params, _ = blackjax_mod.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel_factory,
            num_steps=num_warmup,
            state=init_state,
            rng_key=key_tune,
        )
        final_kernel = kernel_factory(params.inverse_mass_matrix)

        def step_fn(rng_key, state):
            return final_kernel(rng_key, state, params.L, params.step_size)

        return state, step_fn, lambda s: s.position

    return build


def test_get_log_density_fn_negates_potential():
    """logdensity_fn returns exactly the negation of potential_energy at the init point."""
    x, y = _linreg_data()
    info = get_log_density_fn(random.key(0), _linreg_model, model_args=(x, y))
    pe = potential_energy(_linreg_model, (x, y), {}, info.init_position)
    assert_allclose(info.logdensity_fn(info.init_position), -pe, rtol=1e-5)


def test_get_log_density_fn_postprocess_includes_deterministics():
    """postprocess maps unconstrained to constrained and includes deterministic sites."""
    x, y = _linreg_data()
    info = get_log_density_fn(random.key(0), _linreg_model, model_args=(x, y))
    out = info.postprocess_fn(info.init_position)
    assert set(out.keys()) >= {"a", "b", "sigma", "mu"}
    # sigma has Exponential prior -> constrained to be positive
    assert float(out["sigma"]) > 0.0
    # mu shape follows x
    assert out["mu"].shape == x.shape


def test_constrain_samples_batched_chain():
    """constrain_samples with batch_ndims=1 vmaps over a chain of unconstrained draws."""
    x, y = _linreg_data()
    info = get_log_density_fn(random.key(0), _linreg_model, model_args=(x, y))
    n = 5
    raw = {
        k: jnp.broadcast_to(v, (n,) + jnp.shape(v))
        for k, v in info.init_position.items()
    }
    out = constrain_samples(raw, _linreg_model, model_args=(x, y), batch_ndims=1)
    assert out["sigma"].shape == (n,)
    assert out["mu"].shape == (n, x.shape[0])  # deterministic site


def test_constrain_samples_two_batch_dims():
    """constrain_samples with batch_ndims=2 vmaps twice (chains x samples)."""
    x, y = _linreg_data()
    info = get_log_density_fn(random.key(0), _linreg_model, model_args=(x, y))
    chains, draws = 4, 3
    raw = {
        k: jnp.broadcast_to(v, (chains, draws) + jnp.shape(v))
        for k, v in info.init_position.items()
    }
    out = constrain_samples(raw, _linreg_model, model_args=(x, y), batch_ndims=2)
    assert out["sigma"].shape == (chains, draws)
    assert out["mu"].shape == (chains, draws, x.shape[0])


def test_constrain_samples_rejects_negative_ndims():
    """constrain_samples raises for negative batch_ndims."""
    x, y = _linreg_data()
    info = get_log_density_fn(random.key(0), _linreg_model, model_args=(x, y))
    with pytest.raises(ValueError):
        constrain_samples(
            info.init_position, _linreg_model, model_args=(x, y), batch_ndims=-1
        )


def test_external_kernel_rejects_nonzero_warmup(blackjax):
    """ExternalKernel.init raises if MCMC's num_warmup is non-zero (warmup belongs in build_kernel)."""
    kernel = ExternalKernel(_linreg_model, build_kernel=_make_build_nuts(blackjax))
    mcmc = MCMC(kernel, num_warmup=10, num_samples=10, progress_bar=False)
    x, y = _linreg_data()
    with pytest.raises(ValueError, match="num_warmup=0"):
        mcmc.run(random.key(0), x, y)


def test_external_kernel_requires_model_xor_potential_fn(blackjax):
    """Constructor demands exactly one of `model` / `potential_fn`."""
    build = _make_build_nuts(blackjax)
    with pytest.raises(ValueError):
        ExternalKernel(build_kernel=build)
    with pytest.raises(ValueError):
        ExternalKernel(
            model=_linreg_model,
            potential_fn=lambda p: jnp.zeros(()),
            build_kernel=build,
        )


def test_external_kernel_vectorized_chain_method_not_supported(blackjax):
    """A clear NotImplementedError is raised under chain_method='vectorized'."""
    kernel = ExternalKernel(_linreg_model, build_kernel=_make_build_nuts(blackjax))
    mcmc = MCMC(
        kernel,
        num_warmup=0,
        num_samples=10,
        num_chains=2,
        chain_method="vectorized",
        progress_bar=False,
    )
    x, y = _linreg_data()
    with pytest.raises(NotImplementedError, match="vectorized"):
        mcmc.run(random.key(0), x, y)


def test_external_kernel_nuts_recovers_posterior(blackjax):
    """ExternalKernel + blackjax NUTS recovers ground-truth params within MCSE."""
    x, y = _linreg_data()
    kernel = ExternalKernel(_linreg_model, build_kernel=_make_build_nuts(blackjax))
    mcmc = MCMC(kernel, num_warmup=0, num_samples=1000, progress_bar=False)
    mcmc.run(random.key(1), x, y)
    samples = mcmc.get_samples()
    assert samples["a"].mean() == pytest.approx(1.0, abs=0.15)
    assert samples["b"].mean() == pytest.approx(2.0, abs=0.15)
    assert samples["sigma"].mean() == pytest.approx(0.3, abs=0.15)
    assert samples["mu"].shape == (1000, x.shape[0])
    assert (samples["sigma"] > 0).all()


def test_external_kernel_nuts_matches_builtin_nuts(blackjax):
    """Posterior means agree across implementations within a wide MCSE band."""
    x, y = _linreg_data()
    ref = MCMC(
        NUTS(_linreg_model), num_warmup=500, num_samples=1000, progress_bar=False
    )
    ref.run(random.key(2), x, y)
    ref_samples = ref.get_samples()

    ext = MCMC(
        ExternalKernel(_linreg_model, build_kernel=_make_build_nuts(blackjax)),
        num_warmup=0,
        num_samples=1000,
        progress_bar=False,
    )
    ext.run(random.key(2), x, y)
    ext_samples = ext.get_samples()

    for k in ("a", "b", "sigma"):
        assert ext_samples[k].mean() == pytest.approx(ref_samples[k].mean(), abs=0.1), k


def test_external_kernel_mclmc_recovers_posterior(blackjax):
    """ExternalKernel + blackjax MCLMC produces sensible posterior means."""
    x, y = _linreg_data()
    kernel = ExternalKernel(_linreg_model, build_kernel=_make_build_mclmc(blackjax))
    mcmc = MCMC(kernel, num_warmup=0, num_samples=4000, progress_bar=False)
    mcmc.run(random.key(3), x, y)
    samples = mcmc.get_samples()
    assert samples["a"].mean() == pytest.approx(1.0, abs=0.25)
    assert samples["b"].mean() == pytest.approx(2.0, abs=0.25)
    assert samples["sigma"].mean() == pytest.approx(0.3, abs=0.25)


def test_external_kernel_extra_fields_nuts_diagnostics(blackjax):
    """Diagnostics in the inner blackjax NUTS info are reachable via dotted extra_fields."""
    x, y = _linreg_data()
    kernel = ExternalKernel(_linreg_model, build_kernel=_make_build_nuts(blackjax))
    mcmc = MCMC(kernel, num_warmup=0, num_samples=200, progress_bar=False)
    mcmc.run(random.key(4), x, y, extra_fields=("info.is_divergent",))
    extras = mcmc.get_extra_fields()
    assert "info.is_divergent" in extras
    assert extras["info.is_divergent"].shape == (200,)


def test_external_kernel_predictive_integration(blackjax):
    """get_samples() output is directly consumable by Predictive."""
    x, y = _linreg_data()
    kernel = ExternalKernel(_linreg_model, build_kernel=_make_build_nuts(blackjax))
    mcmc = MCMC(kernel, num_warmup=0, num_samples=200, progress_bar=False)
    mcmc.run(random.key(5), x, y)
    samples = mcmc.get_samples()
    preds = Predictive(_linreg_model, posterior_samples=samples)(random.key(6), x)
    assert preds["likelihood"].shape == (200, x.shape[0])


def test_external_kernel_multi_chain_sequential(blackjax):
    """num_chains=4 under chain_method='sequential' yields per-chain samples."""
    x, y = _linreg_data()
    kernel = ExternalKernel(_linreg_model, build_kernel=_make_build_nuts(blackjax))
    mcmc = MCMC(
        kernel,
        num_warmup=0,
        num_samples=300,
        num_chains=4,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(random.key(7), x, y)
    samples_by_chain = mcmc.get_samples(group_by_chain=True)
    assert samples_by_chain["a"].shape == (4, 300)
    assert samples_by_chain["mu"].shape == (4, 300, x.shape[0])


def test_external_kernel_potential_fn_path(blackjax):
    """ExternalKernel(potential_fn=...) skips initialize_model and uses init_params verbatim.

    Targets a 3-D isotropic Gaussian via a pre-built potential function;
    posterior mean recovers the origin and postprocess is identity (no
    transforms because there is no NumPyro model).
    """
    dim = 3

    def potential_fn(position):
        return 0.5 * (position["x"] ** 2).sum()

    init_params = {"x": jnp.zeros(dim)}
    kernel = ExternalKernel(
        potential_fn=potential_fn, build_kernel=_make_build_nuts(blackjax)
    )
    mcmc = MCMC(kernel, num_warmup=0, num_samples=1000, progress_bar=False)
    mcmc.run(random.key(8), init_params=init_params)
    samples = mcmc.get_samples()
    assert samples["x"].shape == (1000, dim)
    # Identity postprocess: keys are the position keys, no extras.
    assert set(samples.keys()) == {"x"}
    assert jnp.linalg.norm(samples["x"].mean(axis=0)) < 0.2


def test_external_kernel_extra_fields_inner_state(blackjax):
    """Dotted extra_fields can drill into the inner sampler state (Blackjax HMCState)."""
    x, y = _linreg_data()
    kernel = ExternalKernel(_linreg_model, build_kernel=_make_build_nuts(blackjax))
    mcmc = MCMC(kernel, num_warmup=0, num_samples=200, progress_bar=False)
    mcmc.run(random.key(9), x, y, extra_fields=("inner.logdensity",))
    extras = mcmc.get_extra_fields()
    assert "inner.logdensity" in extras
    assert extras["inner.logdensity"].shape == (200,)
    # logdensity is a real-valued scalar per step; finite and varying
    assert jnp.isfinite(extras["inner.logdensity"]).all()


def test_external_kernel_init_params_unwraps_param_info(blackjax):
    """Passing a ParamInfo namedtuple as init_params is unwrapped like HMC does."""
    from numpyro.infer.util import initialize_model

    x, y = _linreg_data()
    model_info = initialize_model(
        random.key(10), _linreg_model, model_args=(x, y), dynamic_args=True
    )
    # Pass the full ParamInfo (not just .z) — this is what HMC accepts.
    kernel = ExternalKernel(_linreg_model, build_kernel=_make_build_nuts(blackjax))
    mcmc = MCMC(kernel, num_warmup=0, num_samples=100, progress_bar=False)
    mcmc.run(random.key(11), x, y, init_params=model_info.param_info)
    samples = mcmc.get_samples()
    assert samples["a"].shape == (100,)
    assert isinstance(model_info.param_info, ParamInfo)  # sanity


def test_external_kernel_bad_build_kernel_return(blackjax):
    """A `build_kernel` returning the wrong shape surfaces a focused TypeError."""

    def bad_build(rng_key, logdensity_fn, init_position):
        # Forgot get_position
        return None, lambda k, s: (s, None)

    x, y = _linreg_data()
    kernel = ExternalKernel(_linreg_model, build_kernel=bad_build)
    mcmc = MCMC(kernel, num_warmup=0, num_samples=1, progress_bar=False)
    with pytest.raises(TypeError, match="3-tuple"):
        mcmc.run(random.key(12), x, y)


def test_external_kernel_bad_build_kernel_non_iterable(blackjax):
    """A `build_kernel` returning a non-iterable also surfaces the focused error."""

    def bad_build(rng_key, logdensity_fn, init_position):
        return object()  # not a tuple at all

    x, y = _linreg_data()
    kernel = ExternalKernel(_linreg_model, build_kernel=bad_build)
    mcmc = MCMC(kernel, num_warmup=0, num_samples=1, progress_bar=False)
    with pytest.raises(TypeError, match="3-tuple"):
        mcmc.run(random.key(13), x, y)


def test_constrain_samples_single_position():
    """batch_ndims=0 applies constrain_fn to a single position without vmap."""
    x, y = _linreg_data()
    info = get_log_density_fn(random.key(0), _linreg_model, model_args=(x, y))
    out = constrain_samples(
        info.init_position, _linreg_model, model_args=(x, y), batch_ndims=0
    )
    assert set(out.keys()) >= {"a", "b", "sigma", "mu"}
    assert out["sigma"].shape == ()  # scalar, no leading batch dim
    assert out["mu"].shape == x.shape
    assert float(out["sigma"]) > 0.0


def test_external_kernel_potential_fn_target_variance(blackjax):
    """potential_fn path mixes properly — variance ~ 1.0, not a stuck chain."""
    dim = 3

    def potential_fn(position):
        return 0.5 * (position["x"] ** 2).sum()

    init_params = {"x": jnp.zeros(dim)}
    kernel = ExternalKernel(
        potential_fn=potential_fn, build_kernel=_make_build_nuts(blackjax)
    )
    mcmc = MCMC(kernel, num_warmup=0, num_samples=1500, progress_bar=False)
    mcmc.run(random.key(14), init_params=init_params)
    samples = mcmc.get_samples()
    # A degenerate / stuck sampler would have ~zero variance even though the
    # mean is also ~0; the variance check rules that out.
    sample_var = jnp.var(samples["x"], axis=0)
    assert (sample_var > 0.5).all(), f"variance too low: {sample_var}"
    assert (sample_var < 2.0).all(), f"variance too high: {sample_var}"


def test_external_kernel_inner_logdensity_varies(blackjax):
    """The collected `inner.logdensity` should vary across the chain."""
    x, y = _linreg_data()
    kernel = ExternalKernel(_linreg_model, build_kernel=_make_build_nuts(blackjax))
    mcmc = MCMC(kernel, num_warmup=0, num_samples=200, progress_bar=False)
    mcmc.run(random.key(15), x, y, extra_fields=("inner.logdensity",))
    series = mcmc.get_extra_fields()["inner.logdensity"]
    assert jnp.isfinite(series).all()
    # A constant placeholder would have std == 0; a healthy chain visits
    # different states and the log-density wanders.
    assert float(series.std()) > 1e-3


def test_external_kernel_diagnostics_fn(blackjax):
    """diagnostics_fn is invoked via get_diagnostics_str(state)."""
    calls: list[str] = []

    def diag(state):
        msg = "tag"
        calls.append(msg)
        return msg

    x, y = _linreg_data()
    kernel = ExternalKernel(
        _linreg_model,
        build_kernel=_make_build_nuts(blackjax),
        diagnostics_fn=diag,
    )
    state = kernel.init(random.key(16), 0, None, model_args=(x, y), model_kwargs={})
    assert kernel.get_diagnostics_str(state) == "tag"
    assert calls == ["tag"]


def test_external_kernel_run_twice_same_kernel(blackjax):
    """The same MCMC object can be run twice with different rng_keys."""
    x, y = _linreg_data()
    mcmc = MCMC(
        ExternalKernel(_linreg_model, build_kernel=_make_build_nuts(blackjax)),
        num_warmup=0,
        num_samples=400,
        progress_bar=False,
    )
    mcmc.run(random.key(17), x, y)
    means1 = {k: float(v.mean()) for k, v in mcmc.get_samples().items() if k != "mu"}
    mcmc.run(random.key(18), x, y)
    means2 = {k: float(v.mean()) for k, v in mcmc.get_samples().items() if k != "mu"}
    for k in ("a", "b", "sigma"):
        assert means1[k] == pytest.approx(means2[k], abs=0.2), k


def _vector_param_model(x, y=None):
    """Linear regression with a vector parameter `w` (3-D)."""
    w = numpyro.sample("w", dist.Normal(jnp.zeros(3), 2.0).to_event(1))
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    mu = numpyro.deterministic("mu", x @ w)
    with numpyro.plate("data", x.shape[0]):
        numpyro.sample("likelihood", dist.Normal(mu, sigma), obs=y)


def test_external_kernel_vector_parameter(blackjax):
    """Models with vector-shaped sites recover the true weights."""
    key_x, key_eps = random.split(random.key(19))
    n, d = 80, 3
    X = random.normal(key_x, (n, d))
    w_true = jnp.array([1.0, -0.5, 2.0])
    y = X @ w_true + 0.2 * random.normal(key_eps, (n,))

    kernel = ExternalKernel(
        _vector_param_model, build_kernel=_make_build_nuts(blackjax)
    )
    mcmc = MCMC(kernel, num_warmup=0, num_samples=800, progress_bar=False)
    mcmc.run(random.key(20), X, y)
    samples = mcmc.get_samples()
    assert samples["w"].shape == (800, d)
    assert jnp.allclose(samples["w"].mean(axis=0), w_true, atol=0.15)


def test_external_kernel_multi_chain_parallel(blackjax):
    """num_chains=2 under chain_method='parallel' yields per-chain samples."""
    from jax import local_device_count

    numpyro.set_host_device_count(2)
    if local_device_count() < 2:
        pytest.skip("requires at least 2 host devices")

    x, y = _linreg_data()
    kernel = ExternalKernel(_linreg_model, build_kernel=_make_build_nuts(blackjax))
    mcmc = MCMC(
        kernel,
        num_warmup=0,
        num_samples=200,
        num_chains=2,
        chain_method="parallel",
        progress_bar=False,
    )
    mcmc.run(random.key(21), x, y)
    samples_by_chain = mcmc.get_samples(group_by_chain=True)
    assert samples_by_chain["a"].shape == (2, 200)
    assert samples_by_chain["mu"].shape == (2, 200, x.shape[0])
    # Different chains should explore differently — at least their means
    # shouldn't be identical to machine precision.
    chain_means = samples_by_chain["a"].mean(axis=1)
    assert float(jnp.abs(chain_means[0] - chain_means[1])) > 1e-6


def test_get_log_density_fn_forward_mode(blackjax):
    """forward_mode_differentiation=True still recovers the posterior."""
    x, y = _linreg_data()
    info = get_log_density_fn(
        random.key(22),
        _linreg_model,
        model_args=(x, y),
        forward_mode_differentiation=True,
    )
    # The logdensity_fn is structurally identical regardless of grad mode;
    # the flag affects internal validation in initialize_model. Confirm it
    # still produces a finite log-density and a valid init.
    assert jnp.isfinite(info.logdensity_fn(info.init_position))
    # Drive a short NUTS chain via ExternalKernel using the same flag.
    kernel = ExternalKernel(
        _linreg_model,
        build_kernel=_make_build_nuts(blackjax),
        forward_mode_differentiation=True,
    )
    mcmc = MCMC(kernel, num_warmup=0, num_samples=400, progress_bar=False)
    mcmc.run(random.key(23), x, y)
    samples = mcmc.get_samples()
    assert samples["a"].mean() == pytest.approx(1.0, abs=0.2)
    assert samples["b"].mean() == pytest.approx(2.0, abs=0.2)


def test_get_log_density_fn_with_init_to_value():
    """A non-default init_strategy is respected by `get_log_density_fn`."""
    x, y = _linreg_data()
    # Use init_to_value with values in *constrained* space; numpyro inverts.
    pinned = {"a": jnp.array(0.5), "b": jnp.array(1.2), "sigma": jnp.array(0.7)}
    info = get_log_density_fn(
        random.key(24),
        _linreg_model,
        model_args=(x, y),
        init_strategy=init_to_value(values=pinned),
    )
    # After postprocess (constrained + transforms), values should be ~pinned.
    constrained = info.postprocess_fn(info.init_position)
    for k, v in pinned.items():
        assert constrained[k] == pytest.approx(float(v), abs=1e-5), k


def test_resolve_init_position_branches():
    """Direct unit tests for the three branches of `_resolve_init_position`."""
    pinned = {"x": jnp.zeros(3)}
    default = {"x": jnp.ones(3)}

    # Branch 1: raw dict passes through unchanged.
    assert _resolve_init_position(pinned, default=None) is pinned

    # Branch 2: ParamInfo unwraps to `.z`.
    param_info = ParamInfo(z=pinned, potential_energy=jnp.zeros(()), z_grad=pinned)
    assert _resolve_init_position(param_info, default=None) is pinned

    # Branch 3: None falls back to default when supplied.
    assert _resolve_init_position(None, default=default) is default

    # Branch 4: None + no default raises.
    with pytest.raises(ValueError, match="cannot be None"):
        _resolve_init_position(None, default=None)
