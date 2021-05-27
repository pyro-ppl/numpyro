# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import numpy as np
from numpy.testing import assert_allclose
import pytest

from jax import hessian, jacrev, random, vmap
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve, inv, solve_triangular

import numpyro
import numpyro.distributions as dist
from numpyro.infer import HMC, HMCECS, MCMC, NUTS, DiscreteHMCGibbs, HMCGibbs, MixedHMC
from numpyro.infer.util import log_density


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


@pytest.mark.parametrize("kernel_cls", [HMC, NUTS])
def test_linear_model_log_sigma(
    kernel_cls, N=100, P=50, sigma=0.11, warmup_steps=500, num_samples=500
):
    np.random.seed(0)
    X = np.random.randn(N * P).reshape((N, P))
    XX = np.matmul(np.transpose(X), X)
    Y = X[:, 0] + sigma * np.random.randn(N)
    XY = np.sum(X * Y[:, None], axis=0)

    def model(X, Y):
        N, P = X.shape

        log_sigma = numpyro.sample("log_sigma", dist.Normal(1.0))
        sigma = jnp.exp(log_sigma)
        beta = numpyro.sample("beta", dist.Normal(jnp.zeros(P), jnp.ones(P)))
        mean = jnp.sum(beta * X, axis=-1)
        numpyro.deterministic("mean", mean)

        numpyro.sample("obs", dist.Normal(mean, sigma), obs=Y)

    gibbs_fn = partial(_linear_regression_gibbs_fn, X, XX, XY, Y)

    hmc_kernel = kernel_cls(model)
    kernel = HMCGibbs(hmc_kernel, gibbs_fn=gibbs_fn, gibbs_sites=["beta"])
    mcmc = MCMC(kernel, warmup_steps, num_samples, progress_bar=False)

    mcmc.run(random.PRNGKey(0), X, Y)

    beta_mean = np.mean(mcmc.get_samples()["beta"], axis=0)
    assert_allclose(beta_mean, np.array([1.0] + [0.0] * (P - 1)), atol=0.05)

    sigma_mean = np.exp(np.mean(mcmc.get_samples()["log_sigma"], axis=0))
    assert_allclose(sigma_mean, sigma, atol=0.25)


@pytest.mark.parametrize("kernel_cls", [HMC, NUTS])
def test_linear_model_sigma(
    kernel_cls, N=90, P=40, sigma=0.07, warmup_steps=500, num_samples=500
):
    np.random.seed(1)
    X = np.random.randn(N * P).reshape((N, P))
    XX = np.matmul(np.transpose(X), X)
    Y = X[:, 0] + sigma * np.random.randn(N)
    XY = np.sum(X * Y[:, None], axis=0)

    def model(X, Y):
        N, P = X.shape

        sigma = numpyro.sample("sigma", dist.HalfCauchy(1.0))
        beta = numpyro.sample("beta", dist.Normal(jnp.zeros(P), jnp.ones(P)))
        mean = jnp.sum(beta * X, axis=-1)

        numpyro.sample("obs", dist.Normal(mean, sigma), obs=Y)

    gibbs_fn = partial(_linear_regression_gibbs_fn, X, XX, XY, Y)

    hmc_kernel = kernel_cls(model)
    kernel = HMCGibbs(hmc_kernel, gibbs_fn=gibbs_fn, gibbs_sites=["beta"])
    mcmc = MCMC(kernel, warmup_steps, num_samples, progress_bar=False)

    mcmc.run(random.PRNGKey(0), X, Y)

    beta_mean = np.mean(mcmc.get_samples()["beta"], axis=0)
    assert_allclose(beta_mean, np.array([1.0] + [0.0] * (P - 1)), atol=0.05)

    sigma_mean = np.mean(mcmc.get_samples()["sigma"], axis=0)
    assert_allclose(sigma_mean, sigma, atol=0.25)


@pytest.mark.parametrize("kernel_cls", [HMC, NUTS])
def test_gaussian_model(kernel_cls, D=2, warmup_steps=5000, num_samples=5000):
    np.random.seed(0)
    cov = np.random.randn(4 * D * D).reshape((2 * D, 2 * D))
    cov = jnp.matmul(jnp.transpose(cov), cov) + 0.25 * jnp.eye(2 * D)

    cov00 = cov[:D, :D]
    cov01 = cov[:D, D:]
    cov10 = cov[D:, :D]
    cov11 = cov[D:, D:]

    cov_01_cov11_inv = jnp.matmul(cov01, inv(cov11))
    cov_10_cov00_inv = jnp.matmul(cov10, inv(cov00))

    posterior_cov0 = cov00 - jnp.matmul(cov_01_cov11_inv, cov10)
    posterior_cov1 = cov11 - jnp.matmul(cov_10_cov00_inv, cov01)

    # we consider a model in which (x0, x1) ~ MVN(0, cov)

    def gaussian_gibbs_fn(rng_key, hmc_sites, gibbs_sites):
        x1 = hmc_sites["x1"]
        posterior_loc0 = jnp.matmul(cov_01_cov11_inv, x1)
        x0_proposal = dist.MultivariateNormal(
            loc=posterior_loc0, covariance_matrix=posterior_cov0
        ).sample(rng_key)
        return {"x0": x0_proposal}

    def model():
        x0 = numpyro.sample(
            "x0", dist.MultivariateNormal(loc=jnp.zeros(D), covariance_matrix=cov00)
        )
        posterior_loc1 = jnp.matmul(cov_10_cov00_inv, x0)
        numpyro.sample(
            "x1",
            dist.MultivariateNormal(
                loc=posterior_loc1, covariance_matrix=posterior_cov1
            ),
        )

    hmc_kernel = kernel_cls(model, dense_mass=True)
    kernel = HMCGibbs(hmc_kernel, gibbs_fn=gaussian_gibbs_fn, gibbs_sites=["x0"])
    mcmc = MCMC(kernel, warmup_steps, num_samples, progress_bar=False)

    mcmc.run(random.PRNGKey(0))

    x0_mean = np.mean(mcmc.get_samples()["x0"], axis=0)
    x1_mean = np.mean(mcmc.get_samples()["x1"], axis=0)

    x0_std = np.std(mcmc.get_samples()["x0"], axis=0)
    x1_std = np.std(mcmc.get_samples()["x1"], axis=0)

    assert_allclose(x0_mean, np.zeros(D), atol=0.2)
    assert_allclose(x1_mean, np.zeros(D), atol=0.2)

    assert_allclose(x0_std, np.sqrt(np.diagonal(cov00)), rtol=0.05)
    assert_allclose(x1_std, np.sqrt(np.diagonal(cov11)), rtol=0.1)


@pytest.mark.parametrize(
    "kernel, inner_kernel, kwargs",
    [(MixedHMC, HMC, {"num_discrete_updates": 5}), (DiscreteHMCGibbs, NUTS, {})],
)
@pytest.mark.parametrize("num_chains", [1, 2])
@pytest.mark.filterwarnings("ignore:There are not enough devices:UserWarning")
def test_discrete_gibbs_multiple_sites_chain(kernel, inner_kernel, kwargs, num_chains):
    def model():
        numpyro.sample("x", dist.Bernoulli(0.7).expand([3]))
        numpyro.sample("y", dist.Binomial(10, 0.3))

    sampler = kernel(inner_kernel(model), **kwargs)
    mcmc = MCMC(sampler, 1000, 10000, num_chains=num_chains, progress_bar=False)
    mcmc.run(random.PRNGKey(0))
    mcmc.print_summary()
    samples = mcmc.get_samples()
    assert_allclose(jnp.mean(samples["x"], 0), 0.7 * jnp.ones(3), atol=0.01)
    assert_allclose(jnp.mean(samples["y"], 0), 0.3 * 10, atol=0.1)


@pytest.mark.parametrize(
    "kernel, inner_kernel, kwargs",
    [(MixedHMC, HMC, {"num_discrete_updates": 6}), (DiscreteHMCGibbs, NUTS, {})],
)
def test_discrete_gibbs_enum(kernel, inner_kernel, kwargs):
    def model():
        numpyro.sample("x", dist.Bernoulli(0.7), infer={"enumerate": "parallel"})
        y = numpyro.sample("y", dist.Binomial(10, 0.3))
        numpyro.deterministic("y2", y ** 2)

    sampler = kernel(inner_kernel(model), **kwargs)
    mcmc = MCMC(sampler, 1000, 10000, progress_bar=False)
    mcmc.run(random.PRNGKey(0))
    samples = mcmc.get_samples()
    assert_allclose(jnp.mean(samples["y"], 0), 0.3 * 10, atol=0.1)


@pytest.mark.parametrize("random_walk", [False, True])
@pytest.mark.parametrize(
    "kernel, inner_kernel, kwargs",
    [
        (MixedHMC, HMC, {"num_discrete_updates": 6}),
        (DiscreteHMCGibbs, NUTS, {"modified": True}),
        (DiscreteHMCGibbs, NUTS, {"modified": False}),
    ],
)
def test_discrete_gibbs_bernoulli(random_walk, kernel, inner_kernel, kwargs):
    def model():
        numpyro.sample("c", dist.Bernoulli(0.8))

    sampler = kernel(inner_kernel(model), random_walk=random_walk, **kwargs)
    mcmc = MCMC(sampler, 1000, 10000, progress_bar=False)
    mcmc.run(random.PRNGKey(0))
    samples = mcmc.get_samples()["c"]
    assert_allclose(jnp.mean(samples), 0.8, atol=0.05)


@pytest.mark.parametrize("modified", [False, True])
@pytest.mark.parametrize(
    "kernel, inner_kernel, kwargs",
    [(MixedHMC, HMC, {"num_discrete_updates": 20}), (DiscreteHMCGibbs, NUTS, {})],
)
def test_discrete_gibbs_gmm_1d(modified, kernel, inner_kernel, kwargs):
    def model(probs, locs):
        c = numpyro.sample("c", dist.Categorical(probs))
        numpyro.sample("x", dist.Normal(locs[c], 0.5))

    probs = jnp.array([0.15, 0.3, 0.3, 0.25])
    locs = jnp.array([-2, 0, 2, 4])
    sampler = kernel(
        inner_kernel(model, trajectory_length=1.2), modified=modified, **kwargs
    )
    mcmc = MCMC(sampler, 1000, 200000, progress_bar=False)
    mcmc.run(random.PRNGKey(0), probs, locs)
    samples = mcmc.get_samples()
    assert_allclose(jnp.mean(samples["x"]), 1.3, atol=0.1)
    assert_allclose(jnp.var(samples["x"]), 4.36, atol=0.4)
    assert_allclose(jnp.mean(samples["c"]), 1.65, atol=0.1)
    assert_allclose(jnp.var(samples["c"]), 1.03, atol=0.1)


@pytest.mark.parametrize("num_blocks", [1, 2, 50, 100])
def test_block_update_partitioning(num_blocks):
    plate_size = 10000, 100

    plate_sizes = {"N": plate_size}
    gibbs_sites = {"N": jnp.arange(plate_size[1])}
    gibbs_state = {}

    new_gibbs_sites, new_gibbs_state = numpyro.infer.hmc_gibbs._block_update(
        plate_sizes, num_blocks, random.PRNGKey(2), gibbs_sites, gibbs_state
    )
    block_size = 100 // num_blocks
    for name in gibbs_sites:
        assert (
            block_size == jnp.not_equal(gibbs_sites[name], new_gibbs_sites[name]).sum()
        )

    assert gibbs_state == new_gibbs_state


def test_enum_subsample_smoke():
    def model(data):
        x = numpyro.sample("x", dist.Bernoulli(0.5))
        with numpyro.plate("N", data.shape[0], subsample_size=100, dim=-1):
            batch = numpyro.subsample(data, event_dim=0)
            numpyro.sample("obs", dist.Normal(x, 1), obs=batch)

    data = random.normal(random.PRNGKey(0), (10000,)) + 1
    kernel = HMCECS(NUTS(model), num_blocks=10)
    mcmc = MCMC(kernel, 10, 10)
    mcmc.run(random.PRNGKey(0), data)


@pytest.mark.parametrize("kernel_cls", [HMC, NUTS])
@pytest.mark.parametrize("num_block", [1, 2, 50])
@pytest.mark.parametrize("subsample_size", [50, 150])
def test_hmcecs_normal_normal(kernel_cls, num_block, subsample_size):
    true_loc = jnp.array([0.3, 0.1, 0.9])
    num_warmup, num_samples = 200, 200
    data = true_loc + dist.Normal(jnp.zeros(3), jnp.ones(3)).sample(
        random.PRNGKey(1), (10000,)
    )

    def model(data, subsample_size):
        mean = numpyro.sample("mean", dist.Normal().expand((3,)).to_event(1))
        with numpyro.plate(
            "batch", data.shape[0], dim=-2, subsample_size=subsample_size
        ):
            sub_data = numpyro.subsample(data, 0)
            numpyro.sample("obs", dist.Normal(mean, 1), obs=sub_data)

    ref_params = {
        "mean": true_loc + dist.Normal(true_loc, 5e-2).sample(random.PRNGKey(0))
    }
    proxy_fn = HMCECS.taylor_proxy(ref_params)

    kernel = HMCECS(kernel_cls(model), proxy=proxy_fn)
    mcmc = MCMC(kernel, num_warmup, num_samples)
    mcmc.run(random.PRNGKey(0), data, subsample_size)

    samples = mcmc.get_samples()
    assert_allclose(np.mean(mcmc.get_samples()["mean"], axis=0), true_loc, atol=0.1)
    assert len(samples["mean"]) == num_samples


@pytest.mark.parametrize("subsample_size", [5, 10, 15])
def test_taylor_proxy_norm(subsample_size):
    data_key, tr_key, rng_key = random.split(random.PRNGKey(0), 3)
    ref_params = jnp.array([0.1, 0.5, -0.2])
    sigma = 0.1

    data = ref_params + dist.Normal(jnp.zeros(3), jnp.ones(3)).sample(data_key, (100,))
    n, _ = data.shape

    def model(data, subsample_size):
        mean = numpyro.sample(
            "mean", dist.Normal(ref_params, jnp.ones_like(ref_params))
        )
        with numpyro.plate(
            "data", data.shape[0], subsample_size=subsample_size, dim=-2
        ) as idx:
            numpyro.sample("obs", dist.Normal(mean, sigma), obs=data[idx])

    def log_prob_fn(params):
        return vmap(dist.Normal(params, sigma).log_prob)(data).sum(-1)

    log_prob = log_prob_fn(ref_params)
    log_norm_jac = jacrev(log_prob_fn)(ref_params)
    log_norm_hessian = hessian(log_prob_fn)(ref_params)

    tr = numpyro.handlers.trace(numpyro.handlers.seed(model, tr_key)).get_trace(
        data, subsample_size
    )
    plate_sizes = {"data": (n, subsample_size)}

    proxy_constructor = HMCECS.taylor_proxy({"mean": ref_params})
    proxy_fn, gibbs_init, gibbs_update = proxy_constructor(
        tr, plate_sizes, model, (data, subsample_size), {}
    )

    def taylor_expand_2nd_order(idx, pos):
        return (
            log_prob[idx]
            + (log_norm_jac[idx] @ pos)
            + 0.5 * (pos @ log_norm_hessian[idx]) @ pos
        )

    def taylor_expand_2nd_order_sum(pos):
        return (
            log_prob.sum()
            + log_norm_jac.sum(0) @ pos
            + 0.5 * pos @ log_norm_hessian.sum(0) @ pos
        )

    for _ in range(5):
        split_key, perturbe_key, rng_key = random.split(rng_key, 3)
        perturbe_params = ref_params + dist.Normal(0.1, 0.1).sample(
            perturbe_key, ref_params.shape
        )
        subsample_idx = random.randint(rng_key, (subsample_size,), 0, n)
        gibbs_site = {"data": subsample_idx}
        proxy_state = gibbs_init(None, gibbs_site)
        actual_proxy_sum, actual_proxy_sub = proxy_fn(
            {"data": perturbe_params}, ["data"], proxy_state
        )
        assert_allclose(
            actual_proxy_sub["data"],
            taylor_expand_2nd_order(subsample_idx, perturbe_params - ref_params),
            rtol=1e-5,
        )
        assert_allclose(
            actual_proxy_sum["data"],
            taylor_expand_2nd_order_sum(perturbe_params - ref_params),
            rtol=1e-5,
        )


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("kernel_cls", [HMC, NUTS])
def test_estimate_likelihood(kernel_cls):
    data_key, tr_key, sub_key, rng_key = random.split(random.PRNGKey(0), 4)
    ref_params = jnp.array([0.1, 0.5, -0.2])
    sigma = 0.1
    data = ref_params + dist.Normal(jnp.zeros(3), jnp.ones(3)).sample(
        data_key, (10_000,)
    )
    n, _ = data.shape
    num_warmup = 200
    num_samples = 200
    num_blocks = 20

    def model(data):
        mean = numpyro.sample(
            "mean", dist.Normal(ref_params, jnp.ones_like(ref_params))
        )
        with numpyro.plate("N", data.shape[0], subsample_size=100, dim=-2) as idx:
            numpyro.sample("obs", dist.Normal(mean, sigma), obs=data[idx])

    proxy_fn = HMCECS.taylor_proxy({"mean": ref_params})
    kernel = HMCECS(kernel_cls(model), proxy=proxy_fn, num_blocks=num_blocks)
    mcmc = MCMC(kernel, num_warmup, num_samples)

    mcmc.run(random.PRNGKey(0), data, extra_fields=["hmc_state.potential_energy"])

    pes = mcmc.get_extra_fields()["hmc_state.potential_energy"]
    samples = mcmc.get_samples()
    pes_full = vmap(
        lambda sample: log_density(
            model, (data,), {}, {**sample, **{"N": jnp.arange(n)}}
        )[0]
    )(samples)

    assert jnp.var(jnp.exp(-pes - pes_full)) < 1.0
