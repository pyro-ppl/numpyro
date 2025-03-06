# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pickle

import numpy as np
from numpy.testing import assert_allclose
import pytest

import jax
from jax import random
import jax.numpy as jnp

import numpyro
from numpyro.contrib.funsor import config_kl
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.constraints import (
    boolean,
    circular,
    corr_cholesky,
    corr_matrix,
    greater_than,
    interval,
    l1_ball,
    lower_cholesky,
    nonnegative_integer,
    ordered_vector,
    positive,
    positive_definite,
    positive_integer,
    positive_ordered_vector,
    real,
    real_matrix,
    real_vector,
    scaled_unit_lower_cholesky,
    simplex,
    softplus_lower_cholesky,
    softplus_positive,
    sphere,
    unit_interval,
)
from numpyro.infer import (
    HMC,
    HMCECS,
    MCMC,
    NUTS,
    SA,
    SVI,
    BarkerMH,
    DiscreteHMCGibbs,
    MixedHMC,
    Predictive,
    TraceEnum_ELBO,
)
from numpyro.infer.autoguide import AutoDelta, AutoDiagonalNormal, AutoNormal


def normal_model():
    numpyro.sample("x", dist.Normal(0, 1))


def bernoulli_model():
    numpyro.sample("x", dist.Bernoulli(0.5))


def logistic_regression():
    data = jnp.arange(10)
    x = numpyro.sample("x", dist.Normal(0, 1))
    with numpyro.plate("N", 10, subsample_size=2):
        batch = numpyro.subsample(data, 0)
        numpyro.sample("obs", dist.Bernoulli(logits=x), obs=batch)


def gmm(data, K):
    mix_proportions = numpyro.sample("phi", dist.Dirichlet(jnp.ones(K)))
    with numpyro.plate("num_clusters", K, dim=-1):
        cluster_means = numpyro.sample("cluster_means", dist.Normal(jnp.arange(K), 1.0))
    with numpyro.plate("data", data.shape[0], dim=-1):
        assignments = numpyro.sample(
            "assignments",
            dist.Categorical(mix_proportions),
            infer={"enumerate": "parallel"},
        )
        numpyro.sample("obs", dist.Normal(cluster_means[assignments], 1.0), obs=data)


@pytest.mark.parametrize("kernel", [BarkerMH, HMC, NUTS, SA])
def test_pickle_hmc(kernel):
    mcmc = MCMC(kernel(normal_model), num_warmup=10, num_samples=10)
    mcmc.run(random.PRNGKey(0))
    pickled_mcmc = pickle.loads(pickle.dumps(mcmc))
    jax.tree.all(
        jax.tree.map(assert_allclose, mcmc.get_samples(), pickled_mcmc.get_samples())
    )


@pytest.mark.parametrize("kernel", [BarkerMH, HMC, NUTS, SA])
def test_pickle_hmc_enumeration(kernel):
    K, N = 3, 1000

    true_cluster_means = jnp.array([1.0, 5.0, 10.0])
    true_mix_proportions = jnp.array([0.1, 0.3, 0.6])
    cluster_assignments = dist.Categorical(true_mix_proportions).sample(
        random.PRNGKey(0), (N,)
    )
    data = dist.Normal(true_cluster_means[cluster_assignments], 1.0).sample(
        random.PRNGKey(1)
    )
    mcmc = MCMC(kernel(gmm), num_warmup=10, num_samples=10)
    mcmc.run(random.PRNGKey(0), data, K)
    pickled_mcmc = pickle.loads(pickle.dumps(mcmc))
    jax.tree.all(
        jax.tree.map(assert_allclose, mcmc.get_samples(), pickled_mcmc.get_samples())
    )


@pytest.mark.parametrize("kernel", [DiscreteHMCGibbs, MixedHMC])
def test_pickle_discrete_hmc(kernel):
    mcmc = MCMC(kernel(HMC(bernoulli_model)), num_warmup=10, num_samples=10)
    mcmc.run(random.PRNGKey(0))
    pickled_mcmc = pickle.loads(pickle.dumps(mcmc))
    jax.tree.all(
        jax.tree.map(assert_allclose, mcmc.get_samples(), pickled_mcmc.get_samples())
    )


def test_pickle_hmcecs():
    mcmc = MCMC(HMCECS(NUTS(logistic_regression)), num_warmup=10, num_samples=10)
    mcmc.run(random.PRNGKey(0))
    pickled_mcmc = pickle.loads(pickle.dumps(mcmc))
    jax.tree.all(
        jax.tree.map(assert_allclose, mcmc.get_samples(), pickled_mcmc.get_samples())
    )


def poisson_regression(x, N):
    rate = numpyro.sample("param", dist.Gamma(1.0, 1.0))
    batch_size = len(x) if x is not None else None
    with numpyro.plate("batch", N, batch_size):
        numpyro.sample("x", dist.Poisson(rate), obs=x)


@pytest.mark.parametrize("guide_class", [AutoDelta, AutoDiagonalNormal, AutoNormal])
def test_pickle_autoguide(guide_class):
    x = np.random.poisson(1.0, size=(100,))

    guide = guide_class(poisson_regression)
    optim = numpyro.optim.Adam(1e-2)
    svi = SVI(poisson_regression, guide, optim, numpyro.infer.Trace_ELBO())
    svi_result = svi.run(random.PRNGKey(1), 3, x, len(x))
    pickled_guide = pickle.loads(pickle.dumps(guide))

    predictive = Predictive(
        poisson_regression,
        guide=pickled_guide,
        params=svi_result.params,
        num_samples=1,
        return_sites=["param", "x"],
    )
    samples = predictive(random.PRNGKey(1), None, 1)
    assert set(samples.keys()) == {"param", "x"}


def test_pickle_singleton_constraint():
    # some numpyro constraint classes such as constraints._Real, are only accessible
    # through their public singleton instance, (such as constraint.real). This test
    # ensures that pickling and unpickling singleton instances does not re-create
    # additional instances, which is the default behavior of pickle, and which would
    # break singleton semantics.
    singleton_constraints = (
        boolean,
        circular,
        corr_cholesky,
        corr_matrix,
        l1_ball,
        lower_cholesky,
        nonnegative_integer,
        ordered_vector,
        positive,
        positive_definite,
        positive_integer,
        positive_ordered_vector,
        real,
        real_matrix,
        real_vector,
        scaled_unit_lower_cholesky,
        simplex,
        softplus_lower_cholesky,
        softplus_positive,
        sphere,
        unit_interval,
    )
    for cnstr in singleton_constraints:
        roundtripped_cnstr = pickle.loads(pickle.dumps(cnstr))
        # make sure that the unpickled constraint is the original singleton constraint
        assert roundtripped_cnstr is cnstr

    # Test that it remains possible to pickle newly-created, non-singleton constraints.
    # because these constraints are neither singleton nor exposed as top-level variables
    # of the numpyro.distributions.constraints module, these objects are not pickled by
    # reference, but by value.
    int_cstr = interval(1.0, 2.0)
    roundtripped_int_cstr = pickle.loads(pickle.dumps(int_cstr))
    assert type(roundtripped_int_cstr) is type(int_cstr)
    assert int_cstr.lower_bound == roundtripped_int_cstr.lower_bound
    assert int_cstr.upper_bound == roundtripped_int_cstr.upper_bound

    gt_cstr = greater_than(1.0)
    roundtripped_gt_cstr = pickle.loads(pickle.dumps(gt_cstr))
    assert type(roundtripped_gt_cstr) is type(gt_cstr)
    assert gt_cstr.lower_bound == roundtripped_gt_cstr.lower_bound


def test_mcmc_pickle_post_warmup():
    mcmc = MCMC(NUTS(normal_model), num_warmup=10, num_samples=10)
    mcmc.run(random.PRNGKey(0))
    pickled_mcmc = pickle.loads(pickle.dumps(mcmc))
    pickled_mcmc.post_warmup_state = pickled_mcmc.last_state
    pickled_mcmc.run(random.PRNGKey(1))


def bernoulli_regression(data):
    f = numpyro.sample("beta", dist.Beta(1.0, 1.0))
    with numpyro.plate("N", len(data)):
        numpyro.sample("obs", dist.Bernoulli(f), obs=data)


def test_beta_bernoulli():
    data = jnp.array([1.0] * 8 + [0.0] * 2)

    def guide(data):
        alpha_q = numpyro.param("alpha_q", 1.0, constraint=constraints.positive)
        beta_q = numpyro.param("beta_q", 1.0, constraint=constraints.positive)
        numpyro.sample("beta", dist.Beta(alpha_q, beta_q))

    pickled_model = pickle.loads(pickle.dumps(config_kl(bernoulli_regression)))
    optim = numpyro.optim.Adam(1e-2)
    svi = SVI(config_kl(bernoulli_regression), guide, optim, TraceEnum_ELBO())
    svi_result = svi.run(random.PRNGKey(0), 3, data)
    params = svi_result.params

    svi = SVI(pickled_model, guide, optim, TraceEnum_ELBO())
    svi_result = svi.run(random.PRNGKey(0), 3, data)
    pickled_params = svi_result.params

    jax.tree.all(jax.tree.map(assert_allclose, params, pickled_params))
