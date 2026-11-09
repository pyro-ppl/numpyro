# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import numpy as np
from numpy.testing import assert_allclose
import pytest

import jax
from jax import hessian, jacrev, random, vmap
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve, inv, solve_triangular

import numpyro
import numpyro.distributions as dist
from numpyro.infer import (
    HMC,
    HMCECS,
    MCMC,
    NUTS,
    SA,
    DiscreteGibbs,
    DiscreteHMCGibbs,
    HMCGibbs,
    MixedHMC,
)
from numpyro.infer.gibbs import plate_gibbs_update, plate_log_factors, with_conditioning
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
    kernel_cls, N=100, P=50, sigma=0.11, num_warmup=500, num_samples=500
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
    mcmc = MCMC(
        kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=False
    )

    mcmc.run(random.key(0), X, Y)

    beta_mean = np.mean(mcmc.get_samples()["beta"], axis=0)
    assert_allclose(beta_mean, np.array([1.0] + [0.0] * (P - 1)), atol=0.05)

    sigma_mean = np.exp(np.mean(mcmc.get_samples()["log_sigma"], axis=0))
    assert_allclose(sigma_mean, sigma, atol=0.25)


@pytest.mark.parametrize("kernel_cls", [HMC, NUTS])
def test_linear_model_sigma(
    kernel_cls, N=90, P=40, sigma=0.07, num_warmup=500, num_samples=500
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
    mcmc = MCMC(
        sampler,
        num_warmup=1000,
        num_samples=10000,
        num_chains=num_chains,
        progress_bar=False,
    )
    mcmc.run(random.key(0))
    mcmc.print_summary()
    samples = mcmc.get_samples()
    assert_allclose(jnp.mean(samples["x"], 0), 0.7 * jnp.ones(3), atol=0.05)
    assert_allclose(jnp.mean(samples["y"], 0), 0.3 * 10, atol=0.1)


@pytest.mark.parametrize(
    "kernel, inner_kernel, kwargs",
    [(MixedHMC, HMC, {"num_discrete_updates": 6}), (DiscreteHMCGibbs, NUTS, {})],
)
def test_discrete_gibbs_enum(kernel, inner_kernel, kwargs):
    def model():
        numpyro.sample("x", dist.Bernoulli(0.7), infer={"enumerate": "parallel"})
        y = numpyro.sample("y", dist.Binomial(10, 0.3))
        numpyro.deterministic("y2", y**2)

    sampler = kernel(inner_kernel(model), **kwargs)
    mcmc = MCMC(sampler, num_warmup=1000, num_samples=10000, progress_bar=False)
    mcmc.run(random.key(0))
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
    mcmc = MCMC(sampler, num_warmup=1000, num_samples=10000, progress_bar=False)
    mcmc.run(random.key(0))
    samples = mcmc.get_samples()["c"]
    assert_allclose(jnp.mean(samples), 0.8, atol=0.05)


@pytest.mark.parametrize("random_walk", [False, True])
@pytest.mark.parametrize(
    "kernel, inner_kernel, kwargs",
    [
        (MixedHMC, HMC, {"num_discrete_updates": 6}),
        (DiscreteHMCGibbs, NUTS, {"modified": True}),
        (DiscreteHMCGibbs, NUTS, {"modified": False}),
    ],
)
def test_discrete_gibbs_discrete_uniform(random_walk, kernel, inner_kernel, kwargs):
    # the support {3, ..., 7} does not start at 0
    obs = np.array([4.0, 6.5])

    def model():
        x = numpyro.sample("x", dist.DiscreteUniform(3, 7).expand([2]))
        # a continuous latent, so that MixedHMC also refracts with a nonzero offset
        numpyro.sample("y", dist.Normal())
        numpyro.sample("obs", dist.Normal(x, 1.0), obs=obs)

    sampler = kernel(inner_kernel(model), random_walk=random_walk, **kwargs)
    mcmc = MCMC(sampler, num_warmup=1000, num_samples=10000, progress_bar=False)
    mcmc.run(random.key(0))
    samples = mcmc.get_samples()["x"]
    support = np.arange(3, 8)
    log_weights = -0.5 * (obs[:, None] - support) ** 2
    expected = np.exp(log_weights - np.logaddexp.reduce(log_weights, -1, keepdims=True))
    actual = np.mean(np.asarray(samples)[:, :, None] == support, 0)
    assert_allclose(actual, expected, atol=0.03)


def test_improper_uniform():
    def model():
        numpyro.sample("c", dist.Bernoulli(0.8))
        numpyro.sample(
            "u", dist.ImproperUniform(dist.constraints.unit_interval, (), ())
        )

    sampler = DiscreteHMCGibbs(NUTS(model))
    mcmc = MCMC(sampler, num_warmup=10, num_samples=10, progress_bar=False)
    mcmc.run(random.key(0))


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
    mcmc = MCMC(sampler, num_warmup=1000, num_samples=200000, progress_bar=False)
    mcmc.run(random.key(0), probs, locs)
    samples = mcmc.get_samples()
    assert_allclose(jnp.mean(samples["x"]), 1.3, atol=0.1)
    assert_allclose(jnp.var(samples["x"]), 4.36, atol=0.4)
    assert_allclose(jnp.mean(samples["c"]), 1.65, atol=0.1)
    assert_allclose(jnp.var(samples["c"]), 1.03, atol=0.1)


def _plate_mixture_data():
    probs = jnp.array([0.2, 0.5, 0.3])
    offsets = jnp.array([-2.0, 0.0, 2.0])
    y = jnp.array([-2.5, -1.0, 0.1, 0.3, 1.9, 2.4, -0.4, 0.9, -1.6, 1.2, 0.0, 2.9])
    return y, probs, offsets


def _plate_mixture(y, probs, offsets):
    # z is plated and every factor that reads it is in the same plate
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    with numpyro.plate("N", y.shape[0]):
        z = numpyro.sample("z", dist.Categorical(probs))
        numpyro.sample("y", dist.Normal(mu + offsets[z], 1.0), obs=y)


def _nested_plate_mixture(y, probs, offsets):
    # the likelihood is in an extra plate; DiscreteUniform has a nonzero support low
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    with numpyro.plate("N", y.shape[0]):
        z = numpyro.sample("z", dist.DiscreteUniform(1, 3))
        with numpyro.plate("M", 2, dim=-2):
            numpyro.sample(
                "y", dist.Normal(mu + offsets[z - 1], 1.0), obs=jnp.stack([y, y + 0.5])
            )


def _mixed_sites_mixture(y, probs, offsets):
    # w is not plated and keeps the coordinate-wise sweep; z is updated plate-wise
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    w = numpyro.sample("w", dist.Bernoulli(0.5))
    with numpyro.plate("N", y.shape[0]):
        z = numpyro.sample("z", dist.Categorical(probs))
        numpyro.sample("y", dist.Normal(mu + offsets[z] + w, 1.0), obs=y)


def _unplated_mixture(y, probs, offsets):
    c = numpyro.sample("c", dist.Categorical(probs))
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    numpyro.sample("y", dist.Normal(mu + offsets[c], 1.0).expand([y.shape[0]]), obs=y)


def _factor_outside_plate(y, probs, offsets):
    # a factor outside the plate couples the elements of z: no plate-wise update
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    with numpyro.plate("N", y.shape[0]):
        z = numpyro.sample("z", dist.Categorical(probs))
        numpyro.sample("y", dist.Normal(mu + offsets[z], 1.0), obs=y)
    numpyro.factor("coupling", -0.5 * jnp.square(jnp.sum(z) - 8.0))


def _site_outside_reads_plate(y, probs, offsets):
    # a latent site outside the plate depends on z: no plate-wise update
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    with numpyro.plate("N", y.shape[0]):
        z = numpyro.sample("z", dist.Categorical(probs))
    w = numpyro.sample("w", dist.Bernoulli(jax.nn.sigmoid(jnp.sum(z) - 8.0)))
    with numpyro.plate("N", y.shape[0]):
        numpyro.sample("y", dist.Normal(mu + offsets[z] + w, 1.0), obs=y)


def _sum_inside_plate(y, probs, offsets):
    # a reduction over the plate inside the plate: the elements of z are not independent
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    with numpyro.plate("N", y.shape[0]):
        z = numpyro.sample("z", dist.Categorical(probs))
        numpyro.sample("y", dist.Normal(mu + offsets[z] + 0.3 * z.sum(), 1.0), obs=y)


def _ising():
    # a neighbour coupling inside the plates: the elements of s are not independent
    with numpyro.plate("rows", 6, dim=-2), numpyro.plate("cols", 6, dim=-1):
        s = numpyro.sample("s", dist.Bernoulli(0.5))
        spin = 2.0 * s - 1.0
        bonds = jnp.roll(spin, -1, axis=-1) + jnp.roll(spin, -1, axis=-2)
        numpyro.factor("coupling", 0.4 * spin * bonds)


_PLATE_MODELS = {
    "plate": _plate_mixture,
    "nested": _nested_plate_mixture,
    "mixed": _mixed_sites_mixture,
}


@pytest.mark.parametrize(
    "model, plate_sites, sweep_sites",
    [
        (_plate_mixture, ("z",), ()),
        (_nested_plate_mixture, ("z",), ()),
        (_mixed_sites_mixture, ("z",), ("w",)),
        (_unplated_mixture, (), ("c",)),
        (_factor_outside_plate, (), ("z",)),
        (_site_outside_reads_plate, (), ("z", "w")),
    ],
)
def test_discrete_gibbs_plate_blocks(model, plate_sites, sweep_sites):
    model_args = _plate_mixture_data()
    kernel = DiscreteHMCGibbs(NUTS(model), plate_aware=True)
    kernel.init(random.key(0), 10, None, model_args, {})
    discrete_kernel = kernel.blocks[0][0]
    assert tuple(block.name for block in discrete_kernel.plate_blocks) == plate_sites
    assert discrete_kernel._sweep_sites == sweep_sites
    for block in discrete_kernel.plate_blocks:
        assert block.support_size == 3
        assert block.shape == (model_args[0].shape[0],)
        assert block.plate_dims == (-1,)
        assert block.factors == ("z", "y")


def test_discrete_gibbs_plate_aware_off():
    # the default and plate_aware=False: no plate block, every site is swept coordinate-wise
    model_args = _plate_mixture_data()
    for kwargs in ({}, {"plate_aware": False}):
        kernel = DiscreteHMCGibbs(NUTS(_mixed_sites_mixture), **kwargs)
        kernel.init(random.key(0), 10, None, model_args, {})
        discrete_kernel = kernel.blocks[0][0]
        assert discrete_kernel.plate_blocks == ()
        assert set(discrete_kernel._sweep_sites) == {"w", "z"}


@pytest.mark.parametrize("model", [_sum_inside_plate, _ising])
def test_discrete_gibbs_plate_check(model):
    # the factors inside the plates read other elements of the site; the check at init
    # finds it and the site keeps the coordinate-wise sweep
    if model is _ising:
        kernel = DiscreteGibbs(model, plate_aware=True)
        model_args = ()
    else:
        kernel = DiscreteHMCGibbs(NUTS(model), plate_aware=True)
        model_args = _plate_mixture_data()
    with pytest.warns(UserWarning, match="keeps the coordinate-wise sweep"):
        kernel.init(random.key(0), 10, None, model_args, {})
    discrete_kernel = kernel if model is _ising else kernel.blocks[0][0]
    assert discrete_kernel.plate_blocks == ()
    assert len(discrete_kernel._sweep_sites) == 1


def test_discrete_gibbs_plate_numpy_args():
    # the model computes with its NumPy argument in Python
    y, probs, offsets = _plate_mixture_data()

    def model(y):
        scale = float(np.std(y))
        mu = numpyro.sample("mu", dist.Normal(0.0, scale))
        with numpyro.plate("N", y.shape[0]):
            z = numpyro.sample("z", dist.Categorical(probs))
            numpyro.sample("y", dist.Normal(mu + offsets[z], 1.0), obs=y)

    kernel = DiscreteHMCGibbs(NUTS(model), plate_aware=True)
    kernel.init(random.key(0), 10, None, (np.asarray(y),), {})
    assert [block.name for block in kernel.blocks[0][0].plate_blocks] == ["z"]


@pytest.mark.filterwarnings("ignore:Out-of-support values")
def test_discrete_gibbs_plate_infinite_energy():
    # The initial z has zero likelihood. After the plate update the potential energy is
    # computed afresh instead of carried as inf - inf, and the swept site keeps moving.
    lo = jnp.array([0.0, 2.0, 4.0])
    y = jnp.array([0.5, 2.5, 4.5, 0.2, 2.2, 4.8])

    def model(y):
        numpyro.sample("w", dist.Bernoulli(0.5))
        with numpyro.plate("N", y.shape[0]):
            z = numpyro.sample("z", dist.Categorical(jnp.ones(3) / 3))
            numpyro.sample(
                "y", dist.Uniform(lo[z], lo[z] + 1.0, validate_args=True), obs=y
            )

    kernel = DiscreteGibbs(model, plate_aware=True)
    init_params = {"w": jnp.array(0), "z": jnp.zeros(6, dtype=jnp.int32)}
    state = kernel.init(random.key(0), 0, init_params, (y,), {})
    assert [block.name for block in kernel.plate_blocks] == ["z"]
    assert jnp.isinf(state.potential_energy)
    potential_fn = kernel.get_potential_fn((y,), {})
    sample = jax.jit(lambda state: kernel.sample(state, (y,), {}))
    w = []
    for _ in range(20):
        state = sample(state)
        w.append(state.z["w"])
    assert_allclose(state.z["z"], jnp.array([0, 1, 2, 0, 1, 2]))
    assert_allclose(state.potential_energy, potential_fn(state.z), rtol=1e-6)
    assert len(set(np.asarray(w).tolist())) == 2


@pytest.mark.parametrize("model_name", ["plate", "nested"])
def test_discrete_gibbs_plate_log_factors(model_name):
    # The per-element log factors of the plate-wise update are the full conditionals that
    # the coordinate-wise sweep evaluates, one potential energy per element and value.
    y, probs, offsets = model_args = _plate_mixture_data()
    model = _PLATE_MODELS[model_name]
    kernel = DiscreteHMCGibbs(NUTS(model), plate_aware=True)
    kernel.init(random.key(0), 10, None, model_args, {})
    discrete_kernel = kernel.blocks[0][0]
    (block,) = discrete_kernel.plate_blocks
    low = block.support_low
    z = {"z": jnp.array([0, 1, 2, 1, 0, 2, 1, 1, 2, 0, 1, 2]) + low}
    kwargs = with_conditioning({}, {"mu": jnp.array(0.3)})

    log_factors = plate_log_factors(
        discrete_kernel._prepared_model, model_args, kwargs, block, z
    )
    assert log_factors.shape == (3, y.shape[0])

    potential_fn = discrete_kernel.get_potential_fn(model_args, kwargs)
    expected = jnp.stack(
        [
            jnp.stack(
                [
                    -potential_fn({"z": z["z"].at[i].set(v + low)})
                    for v in range(block.support_size)
                ]
            )
            for i in range(y.shape[0])
        ],
        axis=-1,
    )
    # the two sum the same terms in a different order; the potential also sums the
    # factors of the other elements, which cancel in the normalization
    x64 = expected.dtype == jnp.float64
    assert_allclose(
        jax.nn.log_softmax(log_factors, axis=0),
        jax.nn.log_softmax(expected, axis=0),
        atol=1e-9 if x64 else 1e-4,
    )

    # the potential energy after the update is carried from the log-factor delta
    pe = potential_fn(z)
    z_new, pe_new = plate_gibbs_update(random.key(1), z, pe, block, log_factors)
    assert jnp.any(z_new["z"] != z["z"])
    assert_allclose(pe_new, potential_fn(z_new), rtol=1e-12 if x64 else 1e-5)


@pytest.mark.parametrize("random_walk", [False, True])
@pytest.mark.parametrize("modified", [False, True])
def test_discrete_gibbs_plate_update_stationarity(random_walk, modified):
    # At a fixed state every proposal leaves the full conditionals invariant: from
    # z0 ~ softmax(log_factors), one plate-wise step gives z1 with the same marginals.
    model_args = _plate_mixture_data()
    kernel = DiscreteHMCGibbs(NUTS(_plate_mixture), plate_aware=True)
    kernel.init(random.key(0), 10, None, model_args, {})
    discrete_kernel = kernel.blocks[0][0]
    (block,) = discrete_kernel.plate_blocks
    z = {"z": jnp.array([0, 1, 2, 1, 0, 2, 1, 1, 2, 0, 1, 2])}
    kwargs = with_conditioning({}, {"mu": jnp.array(-0.2)})
    # the factors of an element do not read the other elements: the same at every z
    log_factors = plate_log_factors(
        discrete_kernel._prepared_model, model_args, kwargs, block, z
    )
    expected = jax.nn.softmax(log_factors, axis=0)

    def step(key):
        key_init, key_update = random.split(key)
        z0 = {"z": random.categorical(key_init, log_factors, axis=0)}
        z1, _ = plate_gibbs_update(
            key_update, z0, jnp.zeros(()), block, log_factors, random_walk, modified
        )
        return z1["z"]

    draws = vmap(step)(random.split(random.key(1), 8000))
    assert draws.shape == (8000, 12)
    actual = jnp.mean(draws[:, None, :] == jnp.arange(3)[None, :, None], axis=0)
    assert_allclose(actual, expected, atol=0.03)


@pytest.mark.parametrize(
    "model_name, random_walk, modified",
    [("mixed", False, False), ("plate", True, True)],
)
def test_discrete_gibbs_plate_aware_posterior(model_name, random_walk, modified):
    # Plate-wise and coordinate-wise updates target the same posterior.
    model_args = _plate_mixture_data()
    model = _PLATE_MODELS[model_name]
    means = {}
    for plate_aware in (False, True):
        kernel = DiscreteHMCGibbs(
            NUTS(model),
            random_walk=random_walk,
            modified=modified,
            plate_aware=plate_aware,
        )
        mcmc = MCMC(kernel, num_warmup=500, num_samples=20000, progress_bar=False)
        mcmc.run(random.key(0), *model_args)
        assert len(kernel.blocks[0][0].plate_blocks) == int(plate_aware)
        samples = mcmc.get_samples()
        assert samples["z"].shape == (20000, 12)
        means[plate_aware] = {k: jnp.mean(v, axis=0) for k, v in samples.items()}
    assert_allclose(means[True]["z"], means[False]["z"], atol=0.05)
    for name in means[False]:
        if name != "z":
            assert_allclose(means[True][name], means[False][name], atol=0.1)


def test_enum_subsample_smoke():
    def model(data):
        x = numpyro.sample("x", dist.Bernoulli(0.5), infer={"enumerate": "parallel"})
        with numpyro.plate("N", data.shape[0], subsample_size=100, dim=-1):
            batch = numpyro.subsample(data, event_dim=0)
            numpyro.sample("obs", dist.Normal(x, 1), obs=batch)

    data = random.normal(random.key(0), (10000,)) + 1
    kernel = HMCECS(NUTS(model), num_blocks=10)
    mcmc = MCMC(kernel, num_warmup=10, num_samples=10)
    mcmc.run(random.key(0), data)


def test_enum_subsample_error():
    def model(data):
        x = numpyro.sample("x", dist.Bernoulli(0.5), infer={"enumerate": "parallel"})
        with numpyro.plate("N", data.shape[0], subsample_size=100, dim=-1):
            batch = numpyro.subsample(data, event_dim=0)
            numpyro.sample("obs", dist.Normal(x, 1), obs=batch)

    data = random.normal(random.key(0), (10000,)) + 1
    kernel = HMCECS(NUTS(model), num_blocks=10, proxy=HMCECS.taylor_proxy({}))
    mcmc = MCMC(kernel, num_warmup=10, num_samples=10)
    with pytest.raises(RuntimeError, match=".*discrete latent.*"):
        mcmc.run(random.key(0), data)


@pytest.mark.parametrize("kernel_cls", [HMC, NUTS])
@pytest.mark.parametrize("num_block", [1, 2, 50])
@pytest.mark.parametrize("subsample_size", [50, 150])
@pytest.mark.parametrize("degree", [1, 2])
def test_hmcecs_normal_normal(kernel_cls, num_block, subsample_size, degree):
    true_loc = jnp.array([0.3, 0.1, 0.9])
    num_warmup, num_samples = 200, 200
    data = true_loc + dist.Normal(jnp.zeros(3), jnp.ones(3)).sample(
        random.key(1), (10000,)
    )

    def model(data, subsample_size):
        mean = numpyro.sample("mean", dist.Normal().expand((3,)).to_event(1))
        with numpyro.plate(
            "batch", data.shape[0], dim=-1, subsample_size=subsample_size
        ):
            sub_data = numpyro.subsample(data, 1)
            numpyro.sample("obs", dist.Normal(mean, 1).to_event(), obs=sub_data)

    ref_params = {"mean": true_loc + dist.Normal(true_loc, 5e-2).sample(random.key(0))}
    proxy_fn = HMCECS.taylor_proxy(ref_params, degree=degree)

    kernel = HMCECS(kernel_cls(model), proxy=proxy_fn)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(random.key(0), data, subsample_size)

    samples = mcmc.get_samples()
    assert_allclose(np.mean(mcmc.get_samples()["mean"], axis=0), true_loc, atol=0.1)
    assert len(samples["mean"]) == num_samples


@pytest.mark.parametrize("subsample_size", [5, 10, 15])
def test_taylor_proxy_norm(subsample_size):
    data_key, tr_key, rng_key = random.split(random.key(0), 3)
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
    ref_params = jnp.array([0.1, 0.5, -0.2])
    sigma = 0.1
    data = ref_params + dist.Normal(jnp.zeros(3), jnp.ones(3)).sample(
        random.key(0), (2000,)
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
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)

    mcmc.run(random.key(0), data, extra_fields=["hmc_state.potential_energy"])

    pes = mcmc.get_extra_fields()["hmc_state.potential_energy"]
    samples = mcmc.get_samples()
    pes_full = vmap(
        lambda sample: log_density(
            model, (data,), {}, {**sample, **{"N": jnp.arange(n)}}
        )[0]
    )(samples)

    assert jnp.var(jnp.exp(-pes - pes_full)) < 1.0


def test_hmcecs_multiple_plates():
    true_loc = jnp.array([0.3, 0.1, 0.9])
    num_warmup, num_samples = 2, 2
    data = true_loc + dist.Normal(jnp.zeros(3), jnp.ones(3)).sample(
        random.key(1), (1000,)
    )

    def model(data):
        mean = numpyro.sample("mean", dist.Normal().expand((3,)).to_event(1))
        with numpyro.plate("batch", data.shape[0], dim=-2, subsample_size=10):
            sub_data = numpyro.subsample(data, 0)
            with numpyro.plate("dim", 3):
                numpyro.sample("obs", dist.Normal(mean, 1), obs=sub_data)

    ref_params = {"mean": true_loc + dist.Normal(true_loc, 5e-2).sample(random.key(0))}
    proxy_fn = HMCECS.taylor_proxy(ref_params)

    kernel = HMCECS(NUTS(model), proxy=proxy_fn)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(random.key(0), data)


def test_callable_chain_method():
    def model():
        x = numpyro.sample("x", dist.Normal(0.0, 2.0))
        y = numpyro.sample("y", dist.Normal(0.0, 2.0))
        numpyro.sample("obs", dist.Normal(x + y, 1.0), obs=jnp.array([1.0]))

    def gibbs_fn(rng_key, gibbs_sites, hmc_sites):
        y = hmc_sites["y"]
        new_x = dist.Normal(0.8 * (1 - y), jnp.sqrt(0.8)).sample(rng_key)
        return {"x": new_x}

    hmc_kernel = NUTS(model)
    kernel = HMCGibbs(hmc_kernel, gibbs_fn=gibbs_fn, gibbs_sites=["x"])
    mcmc = MCMC(
        kernel,
        num_warmup=100,
        num_chains=2,
        num_samples=100,
        chain_method=vmap,
        progress_bar=False,
    )
    mcmc.run(random.key(0))
    samples = mcmc.get_samples()
    assert set(samples.keys()) == {"x", "y"}


def _subsample_model(data, subsample_size):
    mean = numpyro.sample("mean", dist.Normal().expand((3,)).to_event(1))
    with numpyro.plate("batch", data.shape[0], dim=-1, subsample_size=subsample_size):
        sub_data = numpyro.subsample(data, 1)
        numpyro.sample("obs", dist.Normal(mean, 1).to_event(), obs=sub_data)


@pytest.mark.parametrize("use_proxy", [False, True])
def test_hmcecs_lifecycle(use_proxy):
    # regression test: the model used to be re-wrapped with `estimate_likelihood` at every
    # `init`, so `warmup()` followed by `run()` raised on duplicated site names
    true_loc = jnp.array([0.3, 0.1, 0.9])
    data = true_loc + dist.Normal(jnp.zeros(3), jnp.ones(3)).sample(
        random.key(1), (1000,)
    )
    proxy = HMCECS.taylor_proxy({"mean": true_loc}, degree=2) if use_proxy else None
    kernel = HMCECS(NUTS(_subsample_model), proxy=proxy)
    mcmc = MCMC(kernel, num_warmup=20, num_samples=20, progress_bar=False)
    mcmc.warmup(random.key(0), data, 50)
    mcmc.run(random.key(1), data, 50)
    mcmc.run(random.key(2), data, 50)
    assert mcmc.get_samples()["mean"].shape == (20, 3)


def test_hmc_gibbs_reuses_initialized_inner_kernel():
    def model():
        x = numpyro.sample("x", dist.Normal(0.0, 2.0))
        y = numpyro.sample("y", dist.Normal(0.0, 2.0))
        numpyro.sample("obs", dist.Normal(x + y, 1.0), obs=jnp.array([1.0]))

    def gibbs_fn(rng_key, gibbs_sites, hmc_sites):
        y = hmc_sites["y"]
        return {"x": dist.Normal(0.8 * (1 - y), jnp.sqrt(0.8)).sample(rng_key)}

    inner = NUTS(model)
    MCMC(inner, num_warmup=10, num_samples=10, progress_bar=False).run(random.key(0))
    kernel = HMCGibbs(inner, gibbs_fn=gibbs_fn, gibbs_sites=["x"])
    init_params = {"x": jnp.array(0.5), "y": jnp.array(-0.5)}
    mcmc = MCMC(kernel, num_warmup=10, num_samples=10, progress_bar=False)
    mcmc.run(random.key(0), init_params=init_params)
    assert set(mcmc.get_samples()) == {"x", "y"}
    # `init_params` is not mutated by the kernel
    assert set(init_params) == {"x", "y"}
    # the wrapped copy did not leak into the user's kernel
    assert inner.model is model


def test_hmcecs_conditioned_site_in_subsample_plate():
    def model(data):
        mean = numpyro.sample("mean", dist.Normal())
        with numpyro.plate("batch", data.shape[0], subsample_size=2):
            local = numpyro.sample("local", dist.Normal(mean, 1.0))
            numpyro.sample("obs", dist.Normal(local, 1), obs=numpyro.subsample(data, 0))

    kernel = HMCECS(NUTS(model))
    with pytest.raises(ValueError, match="inside a subsample plate"):
        kernel.init(
            random.key(0),
            10,
            None,
            (jnp.ones(5),),
            {"_gibbs_sites": {"local": jnp.ones(2)}},
        )


def test_hmc_gibbs_public_names():
    from numpyro.infer.hmc_gibbs import (  # noqa: F401
        HMCECSState,
        HMCGibbsState,
        estimate_likelihood,
        taylor_proxy,
    )

    with pytest.raises(ValueError, match="HMC or NUTS"):
        HMCECS(SA(lambda: None))
    with pytest.raises(ValueError, match="potential function"):
        HMCECS(NUTS(potential_fn=lambda z: 0.0))
