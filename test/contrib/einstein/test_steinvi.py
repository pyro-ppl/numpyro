# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from functools import partial
import string

import numpy as np
import numpy.random as nrandom
from numpy.testing import assert_allclose
import pytest

from jax import numpy as jnp, random

import numpyro
from numpyro import handlers
from numpyro.contrib.einstein import (
    ASVGD,
    SVGD,
    GraphicalKernel,
    IMQKernel,
    LinearKernel,
    RandomFeatureKernel,
    RBFKernel,
    SteinVI,
)
from numpyro.contrib.einstein.stein_kernels import MixtureKernel
import numpyro.distributions as dist
from numpyro.distributions import Bernoulli, Normal, Poisson
from numpyro.distributions.transforms import AffineTransform
from numpyro.infer import Trace_ELBO, init_to_mean, init_to_value
from numpyro.infer.autoguide import (
    AutoBNAFNormal,
    AutoDAIS,
    AutoDelta,
    AutoDiagonalNormal,
    AutoIAFNormal,
    AutoLaplaceApproximation,
    AutoLowRankMultivariateNormal,
    AutoMultivariateNormal,
    AutoNormal,
    AutoSemiDAIS,
    AutoSurrogateLikelihoodDAIS,
)
from numpyro.infer.initialization import (
    init_to_feasible,
    init_to_median,
    init_to_sample,
    init_to_uniform,
)
from numpyro.infer.reparam import TransformReparam
from numpyro.optim import Adam

KERNELS = [
    RBFKernel(),
    LinearKernel(),
    IMQKernel(),
    GraphicalKernel(),
    RandomFeatureKernel(),
]

np.set_printoptions(precision=100)
TKERNEL = namedtuple("TestSteinKernel", ["kernel", "particle_info", "loss_fn", "kval"])


class WrappedGraphicalKernel(GraphicalKernel):
    def __init__(self, mode):
        super().__init__(mode=mode, local_kernel_fns={"p1": RBFKernel("norm")})


class WrappedMixtureKernel(MixtureKernel):
    def __init__(self, mode):
        super().__init__(
            mode=mode,
            ws=np.array([0.2, 0.8]),
            kernel_fns=[RBFKernel(mode), RBFKernel(mode)],
        )


def uniform_normal():
    true_coef = 0.9

    def model(data):
        alpha = numpyro.sample("alpha", dist.Uniform(0, 1))
        with numpyro.handlers.reparam(config={"loc": TransformReparam()}):
            loc = numpyro.sample(
                "loc",
                dist.TransformedDistribution(
                    dist.Uniform(0, 1).mask(False), AffineTransform(0, alpha)
                ),
            )
        with numpyro.plate("data", data.shape[0]):
            numpyro.sample("obs", dist.Normal(loc, 0.1), obs=data)

    data = true_coef + random.normal(random.PRNGKey(0), (10,))
    return true_coef, (data,), model


def regression():
    N, dim = 10, 3
    data = random.normal(random.PRNGKey(0), (N, dim))
    true_coefs = np.arange(1.0, dim + 1.0)
    logits = np.sum(true_coefs * data, axis=-1)
    labels = dist.Bernoulli(logits=logits).sample(random.PRNGKey(1))

    def model(features, labels):
        coefs = numpyro.sample(
            "coefs", dist.Normal(np.zeros(dim), np.ones(dim)).to_event(1)
        )

        logits = numpyro.deterministic("logits", np.sum(coefs * features, axis=-1))

        with numpyro.plate("data", labels.shape[0]):
            return numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=labels)

    return true_coefs, (data, labels), model


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("problem", (uniform_normal, regression))
@pytest.mark.parametrize("method", ("ASVGD", "SVGD", "SteinVI"))
def test_run_smoke(kernel, problem, method):
    true_coefs, data, model = problem()
    if method == "ASVGD":
        stein = ASVGD(model, Adam(1e-1), kernel, num_stein_particles=1)
    if method == "SVGD":
        stein = SVGD(model, Adam(1e-1), kernel, num_stein_particles=1)
    if method == "SteinVI":
        stein = SteinVI(
            model, AutoNormal(model), Adam(1e-1), kernel, num_stein_particles=1
        )

    stein.run(random.PRNGKey(0), 1, *data)


########################################
# Stein Interior
########################################


@pytest.mark.parametrize(
    "auto_guide",
    [
        AutoIAFNormal,
        AutoBNAFNormal,
        AutoSemiDAIS,
        AutoSurrogateLikelihoodDAIS,
        AutoDAIS,
    ],
)
def test_incompatible_autoguide(auto_guide):
    def model():
        return

    if auto_guide.__name__ == "AutoSurrogateLikelihoodDAIS":
        guide = auto_guide(model, model)
    elif auto_guide.__name__ == "AutoSemiDAIS":
        guide = auto_guide(model, model, model)
    else:
        guide = auto_guide(model)

    try:
        SteinVI(
            model,
            guide,
            Adam(1.0),
            RBFKernel(),
            num_stein_particles=1,
        )
        pytest.fail()
    except AssertionError:
        return


@pytest.mark.parametrize(
    "init_loc",
    [
        init_to_value,
        init_to_feasible,
        partial(init_to_value),
        partial(init_to_feasible),
        partial(init_to_uniform, radius=0),
    ],
)
def test_incompatible_init_locs(init_loc):
    def model():
        return

    try:
        SteinVI(
            model,
            AutoDelta(model, init_loc_fn=init_loc),
            Adam(1.0),
            RBFKernel(),
            num_stein_particles=1,
        )
        pytest.fail()
    except AssertionError:
        return


def test_stein_reinit():
    num_particles = 4

    def model():
        numpyro.sample("x", Normal(0, 1.0))

    stein = SteinVI(
        model,
        AutoDelta(model),
        Adam(1.0),
        RBFKernel(),
        num_stein_particles=num_particles,
    )

    for i in range(2):
        with handlers.seed(rng_seed=i):
            params = stein.get_params(stein.init(numpyro.prng_key()))
            xs = params["x_auto_loc"]
            assert jnp.unique(xs).shape == xs.shape


@pytest.mark.parametrize(
    "auto_class",
    [
        AutoMultivariateNormal,
        AutoNormal,
        AutoLowRankMultivariateNormal,
        AutoLaplaceApproximation,
        AutoDelta,
        AutoDiagonalNormal,
    ],
)
@pytest.mark.parametrize(
    "init_loc_fn",
    [
        init_to_uniform,
        init_to_median,
        init_to_mean,
        init_to_sample,
    ],
)
@pytest.mark.parametrize("num_particles", [1, 2, 10])
def test_init_auto_guide(auto_class, init_loc_fn, num_particles):
    latent_dim = 3

    def model(obs):
        a = numpyro.sample("a", Normal(0, 1).expand((latent_dim,)).to_event(1))
        return numpyro.sample("obs", Bernoulli(logits=a), obs=obs)

    obs = Bernoulli(0.5).sample(random.PRNGKey(0), (10, latent_dim))

    rng_key = random.PRNGKey(0)
    guide_key, stein_key = random.split(rng_key)

    guide = auto_class(model, init_loc_fn=init_loc_fn())

    steinvi = SteinVI(
        model,
        guide,
        Adam(1.0),
        RBFKernel(),
        num_stein_particles=num_particles,
    )
    state = steinvi.init(stein_key, obs)

    init_params = steinvi.get_params(state)

    inner_guide = auto_class(model, init_loc_fn=init_loc_fn())

    with handlers.seed(rng_seed=guide_key), handlers.trace() as inner_guide_tr:
        inner_guide(obs)

    for name, site in inner_guide_tr.items():
        if site.get("type") == "param":
            assert name in init_params
            inner_param = site
            init_value = init_params[name]
            expected_shape = (num_particles, *np.shape(inner_param["value"]))
            assert init_value.shape == expected_shape
            if "auto_loc" in name or name == "b":
                assert np.all(init_value != np.zeros(expected_shape))
                assert np.unique(init_value).shape == init_value.reshape(-1).shape
            elif "scale" in name:
                assert_allclose(init_value[init_value != 0.0], 0.1, rtol=1e-6)


@pytest.mark.parametrize("num_particles", [1, 2, 10])
def test_init_custom_guide(num_particles):
    latent_dim = 3

    def guide(obs):
        aloc = numpyro.param(
            "aloc", lambda rng_key: Normal().sample(rng_key, (latent_dim,))
        )
        numpyro.sample("a", Normal(aloc, 1).to_event(1))

    def model(obs):
        a = numpyro.sample("a", Normal(0, 1).expand((latent_dim,)).to_event(1))
        return numpyro.sample("obs", Bernoulli(logits=a), obs=obs)

    obs = Bernoulli(0.5).sample(random.PRNGKey(0), (10, latent_dim))

    rng_key = random.PRNGKey(0)
    guide_key, stein_key = random.split(rng_key)

    steinvi = SteinVI(
        model,
        guide,
        Adam(1.0),
        RBFKernel(),
        num_stein_particles=num_particles,
    )
    init_params = steinvi.get_params(steinvi.init(stein_key, obs))

    init_value = init_params["aloc"]
    expected_shape = (num_particles, latent_dim)

    assert expected_shape == init_value.shape
    assert np.all(init_value != np.zeros(expected_shape))
    assert np.unique(init_value).shape == init_value.reshape(-1).shape


@pytest.mark.parametrize("length", [1, 2, 3, 6])
@pytest.mark.parametrize("depth", [1, 3, 5])
@pytest.mark.parametrize("t", [list, tuple])  # add dict, set
def test_param_size(length, depth, t):
    def nest(v, d):
        if d == 0:
            return v
        return nest(t([v]), d - 1)

    seed = random.PRNGKey(nrandom.randint(0, 10_000))
    sizes = Poisson(5).sample(seed, (length, nrandom.randint(0, 10))) + 1
    total_size = sum(map(lambda size: size.prod(), sizes))
    uparam = t(nest(np.empty(tuple(size)), nrandom.randint(0, depth)) for size in sizes)
    stein = SteinVI(id, id, Adam(1.0), RBFKernel())
    assert stein._param_size(uparam) == total_size, f"Failed for seed {seed}"


@pytest.mark.parametrize("num_particles", [1, 2, 7, 10])
@pytest.mark.parametrize("num_params", [0, 1, 2, 10, 20])
def test_calc_particle_info(num_params, num_particles):
    seed = random.PRNGKey(nrandom.randint(0, 10_000))
    sizes = Poisson(5).sample(seed, (100, nrandom.randint(0, 10))) + 1

    uparam = tuple(np.empty(tuple(size)) for size in sizes)
    uparams = {string.ascii_lowercase[i]: uparam for i in range(num_params)}

    par_param_size = sum(map(lambda size: size.prod(), sizes)) // num_particles
    expected_start_end = zip(
        par_param_size * np.arange(num_params),
        par_param_size * np.arange(1, num_params + 1),
    )
    expected_pinfo = dict(zip(string.ascii_lowercase[:num_params], expected_start_end))

    stein = SteinVI(id, id, Adam(1.0), Trace_ELBO(), RBFKernel())
    pinfo, _ = stein._calc_particle_info(uparams, num_particles)

    for k in pinfo.keys():
        assert pinfo[k] == expected_pinfo[k], f"Failed for seed {seed}"


def test_calc_particle_info_nested():
    num_params = 3
    num_particles = 10
    seed = random.PRNGKey(42)
    sizes = Poisson(5).sample(seed, (100, nrandom.randint(1, 10))) + 1
    uparam = tuple(np.empty(tuple(size)) for size in sizes)
    uparams = {
        string.ascii_lowercase[i]: {
            string.ascii_lowercase[j]: uparam for j in range(num_params)
        }
        for i in range(num_params)
    }

    stein = SteinVI(id, id, Adam(1.0), RBFKernel())
    pinfo, _ = stein._calc_particle_info(uparams, num_particles)
    start = 0
    tot_size = sum(map(lambda size: size.prod(), sizes)) // num_particles
    for val in pinfo.values():
        for v in val.values():
            assert v == (start, start + tot_size)
            start += tot_size
