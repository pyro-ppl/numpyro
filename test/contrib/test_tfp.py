# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
from numpy.testing import assert_allclose
import pytest

from jax import jit, random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import AffineTransform
from numpyro.infer import MCMC, NUTS
from numpyro.infer.reparam import TransformReparam


def test_dist_pytree():
    from tensorflow_probability.substrates.jax import distributions as tfd

    from numpyro.contrib.tfp.distributions import TFPDistribution

    @jit
    def f(x):
        with numpyro.handlers.seed(rng_seed=0), numpyro.handlers.trace() as tr:
            numpyro.sample("x", tfd.Normal(x, 1))
        return tr["x"]["fn"]

    res = f(0.0)

    assert isinstance(res, TFPDistribution)
    assert res.loc == 0
    assert res.scale == 1


@pytest.mark.filterwarnings("ignore:can't resolve package")
def test_transformed_distributions():
    from tensorflow_probability.substrates.jax import (
        bijectors as tfb,
        distributions as tfd,
    )

    d = dist.TransformedDistribution(dist.Normal(0, 1), dist.transforms.ExpTransform())
    d1 = tfd.TransformedDistribution(tfd.Normal(0, 1), tfb.Exp())
    x = random.normal(random.PRNGKey(0), (1000,))
    d_x = d.log_prob(x).sum()
    d1_x = d1.log_prob(x).sum()
    assert_allclose(d_x, d1_x)


@pytest.mark.filterwarnings("ignore:can't resolve package")
def test_logistic_regression():
    from tensorflow_probability.substrates.jax import distributions as tfd

    N, dim = 3000, 3
    num_warmup, num_samples = (1000, 1000)
    data = random.normal(random.PRNGKey(0), (N, dim))
    true_coefs = jnp.arange(1.0, dim + 1.0)
    logits = jnp.sum(true_coefs * data, axis=-1)
    labels = tfd.Bernoulli(logits=logits).sample(seed=random.PRNGKey(1))

    def model(labels):
        coefs = numpyro.sample("coefs", tfd.Normal(jnp.zeros(dim), jnp.ones(dim)))
        logits = numpyro.deterministic("logits", jnp.sum(coefs * data, axis=-1))
        return numpyro.sample("obs", tfd.Bernoulli(logits=logits), obs=labels)

    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(random.PRNGKey(2), labels)
    mcmc.print_summary()
    samples = mcmc.get_samples()
    assert samples["logits"].shape == (num_samples, N)
    expected_coefs = jnp.array([0.97, 2.05, 3.18])
    assert_allclose(jnp.mean(samples["coefs"], 0), expected_coefs, atol=0.22)


@pytest.mark.filterwarnings("ignore:can't resolve package")
# TODO: remove after https://github.com/tensorflow/probability/issues/1072 is resolved
@pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
def test_beta_bernoulli():
    from tensorflow_probability.substrates.jax import distributions as tfd

    num_warmup, num_samples = (500, 2000)

    def model(data):
        alpha = jnp.array([1.1, 1.1])
        beta = jnp.array([1.1, 1.1])
        p_latent = numpyro.sample("p_latent", tfd.Beta(alpha, beta))
        numpyro.sample("obs", tfd.Bernoulli(p_latent), obs=data)
        return p_latent

    true_probs = jnp.array([0.9, 0.1])
    data = tfd.Bernoulli(true_probs).sample(
        seed=random.PRNGKey(1), sample_shape=(1000, 2)
    )
    kernel = NUTS(model=model, trajectory_length=0.1)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(random.PRNGKey(2), data)
    mcmc.print_summary()
    samples = mcmc.get_samples()
    assert_allclose(jnp.mean(samples["p_latent"], 0), true_probs, atol=0.05)


def make_kernel_fn(target_log_prob_fn):
    import tensorflow_probability.substrates.jax as tfp

    return tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=0.5 / jnp.sqrt(0.5 ** jnp.arange(4)[..., None]),
        num_leapfrog_steps=5,
    )


@pytest.mark.parametrize(
    "kernel, kwargs",
    [
        ("HamiltonianMonteCarlo", dict(step_size=0.05, num_leapfrog_steps=10)),
        ("NoUTurnSampler", dict(step_size=0.05)),
        ("RandomWalkMetropolis", dict()),
        ("SliceSampler", dict(step_size=1.0, max_doublings=5)),
        (
            "UncalibratedHamiltonianMonteCarlo",
            dict(step_size=0.05, num_leapfrog_steps=10),
        ),
        ("UncalibratedRandomWalk", dict()),
    ],
)
@pytest.mark.filterwarnings("ignore:can't resolve package")
# TODO: remove after https://github.com/tensorflow/probability/issues/1072 is resolved
@pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
def test_mcmc_kernels(kernel, kwargs):
    from numpyro.contrib.tfp import mcmc

    if ("CI" in os.environ) and kernel == "SliceSampler":
        # TODO: Look into this issue if some users are using SliceSampler
        # with NumPyro model.
        pytest.skip("SliceSampler freezes CI for unknown reason.")

    kernel_class = getattr(mcmc, kernel)

    true_coef = 0.9
    num_warmup, num_samples = 1000, 1000

    def model(data):
        alpha = numpyro.sample("alpha", dist.Uniform(0, 1))
        with numpyro.handlers.reparam(config={"loc": TransformReparam()}):
            loc = numpyro.sample(
                "loc",
                dist.TransformedDistribution(
                    dist.Uniform(0, 1), AffineTransform(0, alpha)
                ),
            )
        numpyro.sample("obs", dist.Normal(loc, 0.1), obs=data)

    data = true_coef + random.normal(random.PRNGKey(0), (1000,))
    tfp_kernel = kernel_class(model=model, **kwargs)
    mcmc = MCMC(tfp_kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.warmup(random.PRNGKey(2), data, collect_warmup=True)
    warmup_samples = mcmc.get_samples()
    mcmc.run(random.PRNGKey(3), data)
    samples = mcmc.get_samples()
    assert len(warmup_samples["loc"]) == num_warmup
    assert len(samples["loc"]) == num_samples
    assert_allclose(jnp.mean(samples["loc"], 0), true_coef, atol=0.05)


@pytest.mark.parametrize(
    "kernel, kwargs",
    [
        ("MetropolisAdjustedLangevinAlgorithm", dict(step_size=1.0)),
        ("RandomWalkMetropolis", dict()),
        ("SliceSampler", dict(step_size=1.0, max_doublings=5)),
        ("UncalibratedLangevin", dict(step_size=0.1)),
        (
            "ReplicaExchangeMC",
            dict(
                inverse_temperatures=0.5 ** jnp.arange(4), make_kernel_fn=make_kernel_fn
            ),
        ),
    ],
)
@pytest.mark.parametrize("num_chains", [1, 2])
@pytest.mark.skipif(
    "XLA_FLAGS" not in os.environ,
    reason="without this mark, we have duplicated tests in Travis",
)
@pytest.mark.filterwarnings("ignore:There are not enough devices:UserWarning")
@pytest.mark.filterwarnings("ignore:can't resolve package")
# TODO: remove after https://github.com/tensorflow/probability/issues/1072 is resolved
@pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
def test_unnormalized_normal_chain(kernel, kwargs, num_chains):
    from numpyro.contrib.tfp import mcmc

    # TODO: remove when this issue is fixed upstream
    # https://github.com/tensorflow/probability/pull/1087
    if num_chains == 2 and kernel == "ReplicaExchangeMC":
        pytest.xfail("ReplicaExchangeMC is not fully compatible with omnistaging yet.")

    kernel_class = getattr(mcmc, kernel)

    true_mean, true_std = 1.0, 0.5
    num_warmup, num_samples = (1000, 8000)

    def potential_fn(z):
        return 0.5 * ((z - true_mean) / true_std) ** 2

    init_params = jnp.array(0.0) if num_chains == 1 else jnp.array([0.0, 2.0])
    tfp_kernel = kernel_class(potential_fn=potential_fn, **kwargs)
    mcmc = MCMC(
        tfp_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(0), init_params=init_params)
    mcmc.print_summary()
    hmc_states = mcmc.get_samples()
    assert_allclose(jnp.mean(hmc_states), true_mean, rtol=0.07)
    assert_allclose(jnp.std(hmc_states), true_std, rtol=0.07)


# test if sampling from tfp distributions works as expected using
# numpyro sample function: numpyro.sample("name", dist) (bug)
@pytest.mark.filterwarnings("ignore:can't resolve package")
@pytest.mark.filterwarnings("ignore:Importing distributions")
def test_sample_tfp_distributions():
    from tensorflow_probability.substrates.jax import distributions as tfd

    from numpyro.contrib.tfp.distributions import TFPDistribution

    # test no error raised
    d = TFPDistribution[tfd.Normal](0, 1)
    with numpyro.handlers.seed(rng_seed=random.PRNGKey(0)):
        numpyro.sample("normal", d)

    # test intermediates are []
    value, intermediates = d(sample_intermediates=True, rng_key=random.PRNGKey(0))
    assert intermediates == []


# test that sampling from unwrapped tensorflow_probability distributions works as
# expected using numpyro.sample primitive
@pytest.mark.parametrize(
    "dist,args",
    [
        ["Bernoulli", (0,)],
        ["Beta", (1, 1)],
        ["Binomial", (10, 0)],
        ["Categorical", ([0, 1, -1],)],
        ["Cauchy", (0, 1)],
        ["Dirichlet", ([1, 2, 0.5],)],
        ["Exponential", (1,)],
        ["InverseGamma", (1, 1)],
        ["Normal", (0, 1)],
        ["OrderedLogistic", ([0, 1], 0.5)],
        ["Pareto", (1,)],
    ],
)
def test_sample_unwrapped_tfp_distributions(dist, args):
    from tensorflow_probability.substrates.jax import distributions as tfd

    # test no error is raised
    with numpyro.handlers.seed(rng_seed=random.PRNGKey(0)):
        # since we import tfd inside the test, distributions have to be parametrized as
        # strings, which is why we use getattr here
        numpyro.sample("sample", getattr(tfd, dist)(*args))


# test mixture distributions
def test_sample_unwrapped_mixture_same_family():
    from tensorflow_probability.substrates.jax import distributions as tfd

    # test no error is raised
    with numpyro.handlers.seed(rng_seed=random.PRNGKey(0)):
        numpyro.sample(
            "sample",
            tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=[0.3, 0.7]),
                components_distribution=tfd.Normal(
                    loc=[-1.0, 1], scale=[0.1, 0.5]  # One for each component.
                ),
            ),
        )


# test that MCMC works with unwrapped tensorflow_probability distributions
def test_mcmc_unwrapped_tfp_distributions():
    from tensorflow_probability.substrates.jax import distributions as tfd

    def model(y):
        theta = numpyro.sample("p", tfd.Beta(1, 1))

        with numpyro.plate("plate", y.size):
            numpyro.sample("y", tfd.Bernoulli(probs=theta), obs=y)

    mcmc = MCMC(NUTS(model), num_warmup=1000, num_samples=1000)
    mcmc.run(random.PRNGKey(0), jnp.array([0, 0, 1, 1, 1]))
    samples = mcmc.get_samples()

    assert_allclose(jnp.mean(samples["p"]), 4 / 7, atol=0.05)


@pytest.mark.parametrize("shape", [(), (4,), (2, 3)], ids=str)
@pytest.mark.filterwarnings("ignore:Importing distributions from numpyro.contrib")
@pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
def test_kl_normal_normal(shape):
    from tensorflow_probability.substrates.jax import distributions as tfd

    from numpyro.contrib.tfp.distributions import TFPDistribution

    p = TFPDistribution[tfd.Normal](
        np.random.normal(size=shape), np.exp(np.random.normal(size=shape))
    )
    q = TFPDistribution[tfd.Normal](
        np.random.normal(size=shape), np.exp(np.random.normal(size=shape))
    )
    actual = dist.kl_divergence(p, q)
    x = p.sample(random.PRNGKey(0), (10000,)).copy()
    expected = jnp.mean((p.log_prob(x) - q.log_prob(x)), 0)
    assert_allclose(actual, expected, rtol=0.05)
