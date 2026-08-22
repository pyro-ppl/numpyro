# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the composable Gibbs kernels in `numpyro.infer.gibbs`."""

from functools import partial
import pickle

import numpy as np
from numpy.testing import assert_allclose
import pytest

from jax import jit, random, vmap
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular

import numpyro
from numpyro.contrib.control_flow import scan
import numpyro.distributions as dist
from numpyro.infer import (
    HMC,
    MCMC,
    NUTS,
    BarkerMH,
    CustomGibbs,
    DiscreteGibbs,
    DiscreteHMCGibbs,
    Gibbs,
    HMCGibbs,
    MixedHMC,
)
from numpyro.infer.gibbs import GibbsState
from numpyro.infer.gibbs_util import (
    GIBBS_SITES_KWARG,
    any_changed,
    conditioned,
    discrete_latent_sites,
    with_conditioning,
)
from numpyro.infer.hmc_gibbs import HMCGibbsState
from numpyro.util import cond, identity


def _linear_regression_gibbs_fn(X, XX, XY, Y, rng_key, gibbs_sites, hmc_sites):
    N, P = X.shape
    sigma = (
        jnp.exp(hmc_sites["log_sigma"])
        if "log_sigma" in hmc_sites
        else hmc_sites["sigma"]
    )
    sigma_sq = jnp.square(sigma)
    covar_inv = XX / sigma_sq + jnp.eye(P)
    L = cho_factor(covar_inv, lower=True)[0]
    L_inv = solve_triangular(L, jnp.eye(P), lower=True)
    loc = cho_solve((L, True), XY) / sigma_sq
    beta_proposal = dist.MultivariateNormal(loc=loc, scale_tril=L_inv).sample(rng_key)
    return {"beta": beta_proposal}


def _linear_data(seed, N, P, sigma):
    np.random.seed(seed)
    X = np.random.randn(N * P).reshape((N, P))
    XX = np.matmul(np.transpose(X), X)
    Y = X[:, 0] + sigma * np.random.randn(N)
    XY = np.sum(X * Y[:, None], axis=0)
    return X, XX, XY, Y


def xy_model():
    x = numpyro.sample("x", dist.Normal(0.0, 2.0))
    y = numpyro.sample("y", dist.Normal(0.0, 2.0))
    numpyro.sample("obs", dist.Normal(x + y, 1.0), obs=jnp.array([1.0]))


def xy_gibbs_fn(rng_key, gibbs_sites, hmc_sites):
    y = hmc_sites["y"]
    return {"x": dist.Normal(0.8 * (1 - y), jnp.sqrt(0.8)).sample(rng_key)}


def test_gibbs_util():
    model = conditioned(xy_model)
    assert conditioned(model) is model
    kwargs = with_conditioning({"a": 1, GIBBS_SITES_KWARG: {"x": 0.0}}, {"y": 1.0})
    assert kwargs == {"a": 1, GIBBS_SITES_KWARG: {"x": 0.0, "y": 1.0}}
    with pytest.raises(ValueError, match="non-latent"):
        with_conditioning({}, {"z": 1.0}, allowed=frozenset({"x"}))
    assert not any_changed({"x": jnp.ones(2)}, {"x": jnp.ones(2)})
    assert any_changed({"x": jnp.ones(2), "c": 1}, {"x": jnp.ones(2), "c": 2})
    assert not any_changed({}, {})


@pytest.mark.parametrize("kernel_cls", [HMC, NUTS])
def test_linear_model_log_sigma(
    kernel_cls, N=100, P=50, sigma=0.11, num_warmup=500, num_samples=500
):
    X, XX, XY, Y = _linear_data(0, N, P, sigma)

    def model(X, Y):
        N, P = X.shape
        log_sigma = numpyro.sample("log_sigma", dist.Normal(1.0))
        sigma = jnp.exp(log_sigma)
        beta = numpyro.sample("beta", dist.Normal(jnp.zeros(P), jnp.ones(P)))
        mean = jnp.sum(beta * X, axis=-1)
        numpyro.deterministic("mean", mean)
        numpyro.sample("obs", dist.Normal(mean, sigma), obs=Y)

    gibbs_fn = partial(_linear_regression_gibbs_fn, X, XX, XY, Y)
    kernel = Gibbs([(CustomGibbs(gibbs_fn), ["beta"]), (kernel_cls(model), None)])
    mcmc = MCMC(
        kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=False
    )
    mcmc.run(random.key(0), X, Y)
    samples = mcmc.get_samples()
    assert set(samples) == {"beta", "log_sigma", "mean"}
    assert samples["mean"].shape == (num_samples, N)
    beta_mean = np.mean(samples["beta"], axis=0)
    assert_allclose(beta_mean, np.array([1.0] + [0.0] * (P - 1)), atol=0.05)
    sigma_mean = np.exp(np.mean(samples["log_sigma"], axis=0))
    assert_allclose(sigma_mean, sigma, atol=0.25)


@pytest.mark.parametrize("kernel_cls", [HMC, NUTS])
def test_linear_model_sigma(
    kernel_cls, N=90, P=40, sigma=0.07, num_warmup=500, num_samples=500
):
    X, XX, XY, Y = _linear_data(1, N, P, sigma)

    def model(X, Y):
        N, P = X.shape
        sigma = numpyro.sample("sigma", dist.HalfCauchy(1.0))
        beta = numpyro.sample("beta", dist.Normal(jnp.zeros(P), jnp.ones(P)))
        mean = jnp.sum(beta * X, axis=-1)
        numpyro.sample("obs", dist.Normal(mean, sigma), obs=Y)

    gibbs_fn = partial(_linear_regression_gibbs_fn, X, XX, XY, Y)
    kernel = Gibbs([(CustomGibbs(gibbs_fn), ["beta"]), (kernel_cls(model), None)])
    mcmc = MCMC(
        kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=False
    )
    mcmc.run(random.key(0), X, Y)
    beta_mean = np.mean(mcmc.get_samples()["beta"], axis=0)
    assert_allclose(beta_mean, np.array([1.0] + [0.0] * (P - 1)), atol=0.05)
    sigma_mean = np.mean(mcmc.get_samples()["sigma"], axis=0)
    assert_allclose(sigma_mean, sigma, atol=0.25)


@pytest.mark.parametrize("kernel_cls", [HMC, NUTS])
def test_gaussian_model(kernel_cls, D=2, num_warmup=5000, num_samples=5000):
    np.random.seed(0)
    cov = np.random.randn(4 * D * D).reshape((2 * D, 2 * D))
    cov = jnp.matmul(jnp.transpose(cov), cov) + 0.25 * jnp.eye(2 * D)
    cov00 = cov[:D, :D]
    cov01 = cov[:D, D:]
    cov10 = cov[D:, :D]
    cov11 = cov[D:, D:]
    cov_01_cov11_inv = jnp.matmul(cov01, jnp.linalg.inv(cov11))
    cov_10_cov00_inv = jnp.matmul(cov10, jnp.linalg.inv(cov00))
    posterior_cov0 = cov00 - jnp.matmul(cov_01_cov11_inv, cov10)
    posterior_cov1 = cov11 - jnp.matmul(cov_10_cov00_inv, cov01)

    def model():
        numpyro.sample(
            "x", dist.MultivariateNormal(jnp.zeros(2 * D), covariance_matrix=cov)
        )

    def gaussian_gibbs_fn(rng_key, hmc_sites, gibbs_sites):
        x1 = hmc_sites["x1"]
        posterior_loc0 = jnp.matmul(cov_01_cov11_inv, x1)
        x0_proposal = dist.MultivariateNormal(
            loc=posterior_loc0, covariance_matrix=posterior_cov0
        ).sample(rng_key)
        return {"x0": x0_proposal}

    def split_model():
        x0 = numpyro.sample(
            "x0", dist.MultivariateNormal(jnp.zeros(D), covariance_matrix=cov00)
        )
        numpyro.sample(
            "x1",
            dist.MultivariateNormal(
                jnp.matmul(cov_10_cov00_inv, x0), covariance_matrix=posterior_cov1
            ),
        )

    kernel = Gibbs(
        [
            (CustomGibbs(gaussian_gibbs_fn), ["x0"]),
            (kernel_cls(split_model, dense_mass=True), None),
        ]
    )
    mcmc = MCMC(
        kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=False
    )
    mcmc.run(random.key(0))
    x0_mean = np.mean(mcmc.get_samples()["x0"], axis=0)
    x1_mean = np.mean(mcmc.get_samples()["x1"], axis=0)
    x0_std = np.std(mcmc.get_samples()["x0"], axis=0)
    x1_std = np.std(mcmc.get_samples()["x1"], axis=0)
    assert_allclose(x0_mean, np.zeros(D), atol=0.25)
    assert_allclose(x1_mean, np.zeros(D), atol=0.25)
    assert_allclose(x0_std, np.sqrt(np.diagonal(cov00)), rtol=0.05)
    assert_allclose(x1_std, np.sqrt(np.diagonal(cov11)), rtol=0.1)


def test_matches_hmc_gibbs_exactly():
    kernel = Gibbs([(CustomGibbs(xy_gibbs_fn), ["x"]), (NUTS(xy_model), None)])
    mcmc = MCMC(kernel, num_warmup=50, num_samples=50, progress_bar=False)
    mcmc.run(random.key(0))
    ref = MCMC(
        HMCGibbs(NUTS(xy_model), xy_gibbs_fn, ["x"]),
        num_warmup=50,
        num_samples=50,
        progress_bar=False,
    )
    ref.run(random.key(0))
    for name in ("x", "y"):
        assert_allclose(mcmc.get_samples()[name], ref.get_samples()[name])
    assert isinstance(ref.last_state, HMCGibbsState)
    assert ref.last_state.hmc_state is ref.last_state.block_states[-1]
    assert ref.sampler.inner_kernel is ref.sampler.blocks[1][0]
    assert ref.sampler.model is xy_model


def test_facade_extra_fields():
    def model():
        c = numpyro.sample("c", dist.Bernoulli(0.8))
        numpyro.sample("x", dist.Normal(c, 1.0))

    mcmc = MCMC(
        DiscreteHMCGibbs(NUTS(model)), num_warmup=20, num_samples=20, progress_bar=False
    )
    mcmc.run(
        random.key(0),
        extra_fields=("hmc_state.potential_energy", "block_states.1.diverging"),
    )
    extra = mcmc.get_extra_fields()
    assert extra["hmc_state.potential_energy"].shape == (20,)
    assert extra["block_states.1.diverging"].shape == (20,)
    assert "acc. prob" in mcmc.sampler.get_diagnostics_str(mcmc.last_state)


def test_mixed_hmc_as_block():
    def model():
        c = numpyro.sample("c", dist.Bernoulli(0.8))
        x = numpyro.sample("x", dist.Normal(c, 1.0))
        y = numpyro.sample("y", dist.Normal(0.0, 2.0))
        numpyro.sample("obs", dist.Normal(x + y, 1.0), obs=jnp.array([1.0]))

    def y_gibbs_fn(rng_key, gibbs_sites, hmc_sites):
        x = hmc_sites["x"]
        return {"y": dist.Normal(0.8 * (1 - x), jnp.sqrt(0.8)).sample(rng_key)}

    kernel = Gibbs(
        [
            (CustomGibbs(y_gibbs_fn), ["y"]),
            (MixedHMC(HMC(model, trajectory_length=1.2), num_discrete_updates=2), None),
        ]
    )
    mcmc = MCMC(kernel, num_warmup=500, num_samples=5000, progress_bar=False)
    mcmc.run(random.key(0))
    samples = mcmc.get_samples()
    assert set(samples) == {"c", "x", "y"}
    ref = MCMC(
        DiscreteHMCGibbs(NUTS(model)),
        num_warmup=500,
        num_samples=5000,
        progress_bar=False,
    )
    ref.run(random.key(0))
    for name in ("c", "x", "y"):
        assert_allclose(samples[name].mean(), ref.get_samples()[name].mean(), atol=0.15)


def test_custom_gibbs_returned_keys_validated():
    def bad_gibbs_fn(rng_key, gibbs_sites, hmc_sites):
        return {"z": jnp.zeros(())}

    kernel = Gibbs([(CustomGibbs(bad_gibbs_fn), ["x"]), (NUTS(xy_model), None)])
    mcmc = MCMC(kernel, num_warmup=2, num_samples=2, progress_bar=False)
    with pytest.raises(ValueError, match="exactly the sites"):
        mcmc.run(random.key(0))
    with pytest.raises(ValueError, match="initial values"):
        CustomGibbs(bad_gibbs_fn).init(random.key(0), 1, None, (), {})
    with pytest.raises(ValueError, match="callable"):
        CustomGibbs(None)


def test_nested_composite_with_deterministic_sites():
    def model():
        x = numpyro.sample("x", dist.Normal(0.0, 2.0))
        y = numpyro.sample("y", dist.Normal(0.0, 2.0))
        z = numpyro.sample("z", dist.Normal(0.0, 2.0))
        numpyro.deterministic("s", x + y + z)
        numpyro.sample("obs", dist.Normal(x + y + z, 1.0), obs=jnp.array([1.0]))

    def gibbs_fn(rng_key, gibbs_sites, hmc_sites):
        # conditional of x given the other two sites: prior N(0, 4), likelihood N(1 - y - z, 1)
        rest = hmc_sites["y"] + hmc_sites["z"]
        assert set(hmc_sites) == {"y", "z"}
        return {"x": dist.Normal(0.8 * (1 - rest), jnp.sqrt(0.8)).sample(rng_key)}

    inner = Gibbs([(CustomGibbs(gibbs_fn), ["x"]), (NUTS(model), ["y"])])
    kernel = Gibbs([(inner, ["x", "y"]), (NUTS(model), None)])
    mcmc = MCMC(kernel, num_warmup=500, num_samples=2000, progress_bar=False)
    mcmc.run(random.key(0))
    samples = mcmc.get_samples()
    assert set(samples) == {"x", "y", "z", "s"}
    assert_allclose(samples["s"], samples["x"] + samples["y"] + samples["z"], rtol=1e-5)
    # posterior of the sum: prior N(0, 12), obs 1 with unit variance
    assert_allclose(samples["s"].mean(), 12 / 13, atol=0.15)
    assert_allclose(samples["s"].std(), np.sqrt(12 / 13), rtol=0.15)


def test_three_hmc_blocks():
    def model():
        x = numpyro.sample("x", dist.Normal(0.0, 1.0))
        y = numpyro.sample("y", dist.HalfNormal(1.0))
        z = numpyro.sample("z", dist.Normal(0.0, 1.0))
        numpyro.sample("obs", dist.Normal(x + z, y), obs=jnp.array([0.5, -0.5, 1.0]))

    kernel = Gibbs([(NUTS(model), ["x"]), (NUTS(model), ["y"]), (NUTS(model), None)])
    mcmc = MCMC(kernel, num_warmup=500, num_samples=2000, progress_bar=False)
    mcmc.run(random.key(0))
    ref = MCMC(NUTS(model), num_warmup=500, num_samples=2000, progress_bar=False)
    ref.run(random.key(0))
    for name in ("x", "y", "z"):
        assert_allclose(
            mcmc.get_samples()[name].mean(), ref.get_samples()[name].mean(), atol=0.15
        )
        assert mcmc.get_samples()[name].std() > 0.2


@pytest.mark.parametrize(
    "blocks, match",
    [
        ([], "at least one block"),
        (
            [(NUTS(xy_model), ["x"]), (NUTS(xy_model), ["x", "y"])],
            "more than one block",
        ),
        ([(NUTS(xy_model), ["x"])], "not owned by any block"),
        ([(NUTS(xy_model), ["w"]), (NUTS(xy_model), None)], "not latent sample sites"),
        ([(NUTS(xy_model), None), (NUTS(xy_model), None)], "At most one block"),
        ([(BarkerMH(xy_model), None)], "does not implement `refresh`"),
        ([(NUTS(xy_model), ["x"]), (NUTS(lambda: None), None)], "same model"),
        ([(NUTS(potential_fn=lambda z: 0.0), None)], "potential function"),
        ([(CustomGibbs(xy_gibbs_fn), ["x"])], "built on a model"),
    ],
)
def test_invalid_blocks(blocks, match):
    with pytest.raises(ValueError, match=match):
        kernel = Gibbs(blocks)
        kernel.init(random.key(0), 10, None, (), {})


def test_init_errors():
    def discrete_model():
        c = numpyro.sample("c", dist.Bernoulli(0.3))
        numpyro.sample("x", dist.Normal(c, 1.0))

    with pytest.raises(ValueError, match="Discrete latent sites"):
        Gibbs([(NUTS(discrete_model), None)]).init(random.key(0), 10, None, (), {})

    def subsample_model(data):
        mean = numpyro.sample("mean", dist.Normal())
        with numpyro.plate("batch", data.shape[0], subsample_size=2):
            numpyro.sample("obs", dist.Normal(mean, 1), obs=numpyro.subsample(data, 0))

    with pytest.raises(ValueError, match="subsample plates"):
        Gibbs([(NUTS(subsample_model), None)]).init(
            random.key(0), 10, None, (jnp.ones(5),), {}
        )

    def dynamic_support_model():
        lb = numpyro.sample("lb", dist.Normal(0.0, 1.0))
        numpyro.sample("y", dist.Uniform(lb, lb + 1.0))

    with pytest.raises(ValueError, match="value-dependent supports"):
        Gibbs(
            [(NUTS(dynamic_support_model), ["y"]), (NUTS(dynamic_support_model), None)]
        ).init(random.key(0), 10, None, (), {})

    kernel = Gibbs([(CustomGibbs(xy_gibbs_fn), ["x"]), (NUTS(xy_model), None)])
    with pytest.raises(ValueError, match="unknown sites"):
        kernel.init(random.key(0), 10, {"w": jnp.zeros(())}, (), {})
    with pytest.raises(ValueError, match="single random key"):
        kernel.init(random.split(random.key(0), 2), 10, None, (), {})


def test_init_params_and_gated_refresh():
    kernel = Gibbs([(CustomGibbs(xy_gibbs_fn), ["x"]), (NUTS(xy_model), None)])
    init_params = {"x": jnp.array(0.25), "y": jnp.array(-0.75)}
    state = kernel.init(random.key(0), 10, init_params, (), {})
    assert isinstance(state, GibbsState)
    assert_allclose(state.z["x"], 0.25)
    assert_allclose(state.z["y"], -0.75)
    assert set(init_params) == {"x", "y"}
    hmc_state = state.block_states[1]
    # the HMC block's cached potential is consistent with the initial value of x
    hmc = kernel.blocks[1][0]
    refreshed = hmc.refresh(hmc_state, (), with_conditioning({}, {"x": state.z["x"]}))
    assert_allclose(refreshed.potential_energy, hmc_state.potential_energy)
    # a gated refresh under jit keeps the state structure
    gated = jit(
        lambda pred, s: cond(
            pred,
            s,
            lambda s: hmc.refresh(s, (), {GIBBS_SITES_KWARG: {"x": 1.0}}),
            s,
            identity,
        )
    )
    assert not jnp.allclose(
        gated(True, hmc_state).potential_energy,
        gated(False, hmc_state).potential_energy,
    )


def test_lifecycle_and_extra_fields():
    kernel = Gibbs([(CustomGibbs(xy_gibbs_fn), ["x"]), (NUTS(xy_model), None)])
    mcmc = MCMC(kernel, num_warmup=20, num_samples=20, progress_bar=False)
    mcmc.warmup(random.key(0))
    mcmc.run(
        random.key(1),
        extra_fields=("block_states.1.diverging", "block_states.1.num_steps"),
    )
    extra = mcmc.get_extra_fields()
    assert extra["block_states.1.diverging"].shape == (20,)
    assert extra["block_states.1.num_steps"].shape == (20,)
    # a second run reuses the initialized kernel
    mcmc.run(random.key(2))
    assert set(mcmc.get_samples()) == {"x", "y"}
    # pickle then continue from the post warmup state
    mcmc2 = pickle.loads(pickle.dumps(mcmc))
    mcmc2.post_warmup_state = mcmc2.last_state
    mcmc2.run(random.key(3))
    assert set(mcmc2.get_samples()) == {"x", "y"}
    assert mcmc2.sampler.get_diagnostics_str(mcmc2.last_state)


@pytest.mark.filterwarnings("ignore:There are not enough devices")
@pytest.mark.parametrize("chain_method", ["sequential", "parallel", vmap])
def test_chain_methods(chain_method):
    kernel = Gibbs([(CustomGibbs(xy_gibbs_fn), ["x"]), (NUTS(xy_model), None)])
    mcmc = MCMC(
        kernel,
        num_warmup=20,
        num_samples=20,
        num_chains=2,
        chain_method=chain_method,
        progress_bar=False,
    )
    mcmc.run(random.key(0))
    mcmc.run(random.key(1))
    assert mcmc.get_samples(group_by_chain=True)["x"].shape == (2, 20)


def test_jit_model_args():
    def model(scale):
        x = numpyro.sample("x", dist.Normal(0.0, scale))
        y = numpyro.sample("y", dist.Normal(0.0, scale))
        numpyro.sample("obs", dist.Normal(x + y, 1.0), obs=jnp.array([1.0]))

    kernel = Gibbs([(CustomGibbs(xy_gibbs_fn), ["x"]), (NUTS(model), None)])
    mcmc = MCMC(
        kernel, num_warmup=20, num_samples=20, progress_bar=False, jit_model_args=True
    )
    mcmc.run(random.key(0), 2.0)
    mcmc.run(random.key(1), 3.0)
    assert set(mcmc.get_samples()) == {"x", "y"}


def test_scan_model():
    def model(T=5):
        x0 = numpyro.sample("x0", dist.Normal(0.0, 1.0))
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))

        def transition(x, t):
            x_new = numpyro.sample("x", dist.Normal(x, sigma))
            numpyro.sample("obs", dist.Normal(x_new, 0.5), obs=jnp.float32(t))
            return x_new, x_new

        scan(transition, x0, jnp.arange(T))

    def sigma_gibbs_fn(rng_key, gibbs_sites, hmc_sites):
        return {"sigma": dist.HalfNormal(1.0).sample(rng_key)}

    kernel = Gibbs([(CustomGibbs(sigma_gibbs_fn), ["sigma"]), (NUTS(model), None)])
    mcmc = MCMC(kernel, num_warmup=20, num_samples=20, progress_bar=False)
    mcmc.run(random.key(0))
    assert mcmc.get_samples()["x"].shape == (20, 5)


def _discrete_blocks(model, inner_kernel=NUTS, **kwargs):
    return Gibbs(
        [
            (DiscreteGibbs(model, **kwargs), discrete_latent_sites),
            (inner_kernel(model), None),
        ]
    )


def _discrete_model():
    numpyro.sample("x", dist.Bernoulli(0.7).expand([3]))
    numpyro.sample("y", dist.Binomial(10, 0.3))


def test_discrete_gibbs_standalone():
    kernel = DiscreteGibbs(_discrete_model)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=5000, progress_bar=False)
    mcmc.run(random.key(0))
    samples = mcmc.get_samples()
    assert_allclose(jnp.mean(samples["x"], 0), 0.7 * jnp.ones(3), atol=0.05)
    assert_allclose(jnp.mean(samples["y"], 0), 0.3 * 10, atol=0.1)
    assert kernel._sites == ("x", "y")
    assert_allclose(kernel._support_sizes_flat, np.array([2, 2, 2, 11]))
    # refresh recomputes the potential energy at the current values
    state = mcmc.last_state
    pe = kernel.get_potential_fn((), {})(state.z)
    assert_allclose(kernel.refresh(state, (), {}).potential_energy, pe, rtol=1e-5)
    assert_allclose(state.potential_energy, pe, rtol=1e-5)
    # a gated refresh under jit keeps the state structure
    gated = jit(
        lambda p, s: cond(p, s, lambda s: kernel.refresh(s, (), {}), s, identity)
    )
    assert gated(True, state).potential_energy.dtype == state.potential_energy.dtype
    # a second run re-initializes the kernel
    mcmc.run(random.key(1))
    # pickle drops the prepared model only
    kernel2 = pickle.loads(pickle.dumps(kernel))
    assert kernel2._prepared_model is None and kernel2._sites == ("x", "y")


def test_discrete_gibbs_errors():
    def mixed_model():
        c = numpyro.sample("c", dist.Bernoulli(0.8))
        numpyro.sample("x", dist.Normal(c, 1.0))

    with pytest.raises(ValueError, match="cannot sample the latent sites"):
        DiscreteGibbs(mixed_model).init(random.key(0), 10, None, (), {})
    with pytest.raises(ValueError, match="Cannot detect any discrete"):
        DiscreteGibbs(xy_model).init(random.key(0), 10, None, (), {})
    with pytest.raises(RuntimeError, match="init"):
        DiscreteGibbs(mixed_model).get_potential_fn()


@pytest.mark.parametrize("num_chains", [1, 2])
@pytest.mark.filterwarnings("ignore:There are not enough devices:UserWarning")
def test_discrete_gibbs_multiple_sites_chain(num_chains):
    def model():
        numpyro.sample("x", dist.Bernoulli(0.7).expand([3]))
        numpyro.sample("y", dist.Binomial(10, 0.3))

    mcmc = MCMC(
        _discrete_blocks(model),
        num_warmup=1000,
        num_samples=10000,
        num_chains=num_chains,
        progress_bar=False,
    )
    mcmc.run(random.key(0))
    samples = mcmc.get_samples()
    assert_allclose(jnp.mean(samples["x"], 0), 0.7 * jnp.ones(3), atol=0.05)
    assert_allclose(jnp.mean(samples["y"], 0), 0.3 * 10, atol=0.1)


def test_discrete_gibbs_enum():
    def model():
        numpyro.sample("x", dist.Bernoulli(0.7), infer={"enumerate": "parallel"})
        y = numpyro.sample("y", dist.Binomial(10, 0.3))
        numpyro.deterministic("y2", y**2)
        z = numpyro.sample("z", dist.Normal(0.0, 1.0))
        numpyro.sample("obs", dist.Normal(z + y, 1.0), obs=jnp.array(3.0))

    kernel = _discrete_blocks(model)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=10000, progress_bar=False)
    mcmc.run(random.key(0))
    samples = mcmc.get_samples()
    assert set(samples) == {"y", "y2", "z"}
    assert kernel.blocks[0][0]._enum
    # y | z has prior Binomial(10, 0.3) and likelihood N(3 - z, 1); the posterior mean of y
    # is pulled towards 3
    assert 2.5 < jnp.mean(samples["y"]) < 3.5
    assert_allclose(samples["y2"], samples["y"] ** 2)


def test_discrete_gibbs_enum_potential_marginalizes():
    def model():
        x = numpyro.sample("x", dist.Bernoulli(0.7), infer={"enumerate": "parallel"})
        numpyro.sample("y", dist.Bernoulli(0.3))
        numpyro.sample("obs", dist.Normal(x, 1.0), obs=jnp.array(0.5))

    kernel = DiscreteGibbs(model)
    state = kernel.init(random.key(0), 10, None, (), {})
    pe = kernel.get_potential_fn((), {})({"y": jnp.array(1)})
    # exact marginal over x
    likelihood = 0.7 * jnp.exp(dist.Normal(1.0, 1.0).log_prob(0.5)) + 0.3 * jnp.exp(
        dist.Normal(0.0, 1.0).log_prob(0.5)
    )
    expected = -(jnp.log(0.3) + jnp.log(likelihood))
    assert_allclose(pe, expected, rtol=1e-5)
    assert state.z.keys() == {"y"}


@pytest.mark.parametrize("random_walk", [False, True])
@pytest.mark.parametrize("modified", [False, True])
def test_discrete_gibbs_bernoulli(random_walk, modified):
    def model():
        numpyro.sample("c", dist.Bernoulli(0.8))

    kernel = _discrete_blocks(model, random_walk=random_walk, modified=modified)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=10000, progress_bar=False)
    mcmc.run(random.key(0))
    samples = mcmc.get_samples()["c"]
    assert_allclose(jnp.mean(samples), 0.8, atol=0.05)


def test_discrete_gibbs_improper_uniform():
    def model():
        numpyro.sample("c", dist.Bernoulli(0.8))
        numpyro.sample(
            "u", dist.ImproperUniform(dist.constraints.unit_interval, (), ())
        )

    mcmc = MCMC(
        _discrete_blocks(model), num_warmup=10, num_samples=10, progress_bar=False
    )
    mcmc.run(random.key(0))


@pytest.mark.parametrize("modified", [False, True])
def test_discrete_gibbs_gmm_1d(modified):
    def model(probs, locs):
        c = numpyro.sample("c", dist.Categorical(probs))
        numpyro.sample("x", dist.Normal(locs[c], 0.5))

    probs = jnp.array([0.15, 0.3, 0.3, 0.25])
    locs = jnp.array([-2, 0, 2, 4])
    kernel = Gibbs(
        [
            (DiscreteGibbs(model, modified=modified), discrete_latent_sites),
            (NUTS(model, trajectory_length=1.2), None),
        ]
    )
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=200000, progress_bar=False)
    mcmc.run(random.key(0), probs, locs)
    samples = mcmc.get_samples()
    assert_allclose(jnp.mean(samples["x"]), 1.3, atol=0.1)
    assert_allclose(jnp.var(samples["x"]), 4.36, atol=0.4)
    assert_allclose(jnp.mean(samples["c"]), 1.65, atol=0.1)
    assert_allclose(jnp.var(samples["c"]), 1.03, atol=0.1)


def test_three_blocks_doctest_model():
    def model(probs, locs):
        c = numpyro.sample("c", dist.Categorical(probs))
        x = numpyro.sample("x", dist.Normal(locs[c], 0.5))
        y = numpyro.sample("y", dist.Normal(0.0, 2.0))
        numpyro.sample("obs", dist.Normal(x + y, 1.0), obs=jnp.array([1.0]))

    def gibbs_fn(rng_key, gibbs_sites, hmc_sites):
        x = hmc_sites["x"]
        assert set(hmc_sites) == {"c", "x"}
        return {"y": dist.Normal(0.8 * (1 - x), jnp.sqrt(0.8)).sample(rng_key)}

    kernel = Gibbs(
        [
            (DiscreteGibbs(model), discrete_latent_sites),
            (CustomGibbs(gibbs_fn), ["y"]),
            (NUTS(model), None),
        ]
    )
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000, progress_bar=False)
    mcmc.run(
        random.key(0),
        jnp.array([0.15, 0.3, 0.3, 0.25]),
        jnp.array([-2.0, 0.0, 2.0, 4.0]),
    )
    samples = mcmc.get_samples()
    assert set(samples) == {"c", "x", "y"}

    # reference: NUTS on the marginalized model
    def ref_model(probs, locs):
        x = numpyro.sample(
            "x", dist.MixtureSameFamily(dist.Categorical(probs), dist.Normal(locs, 0.5))
        )
        y = numpyro.sample("y", dist.Normal(0.0, 2.0))
        numpyro.sample("obs", dist.Normal(x + y, 1.0), obs=jnp.array([1.0]))

    ref = MCMC(NUTS(ref_model), num_warmup=1000, num_samples=20000, progress_bar=False)
    ref.run(
        random.key(0),
        jnp.array([0.15, 0.3, 0.3, 0.25]),
        jnp.array([-2.0, 0.0, 2.0, 4.0]),
    )
    for name in ("x", "y"):
        assert_allclose(samples[name].mean(), ref.get_samples()[name].mean(), atol=0.1)
        assert_allclose(samples[name].std(), ref.get_samples()[name].std(), rtol=0.1)
