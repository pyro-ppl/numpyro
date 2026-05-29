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
)
from numpyro.infer.util import potential_energy


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

    def build(rng_key, logdensity_fn, init_position):
        from blackjax.mcmc.integrators import isokinetic_mclachlan

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
    out = info.postprocess(info.init_position)
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
