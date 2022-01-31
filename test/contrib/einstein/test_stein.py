# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from copy import copy
import string

import numpy as np
from numpy.ma.testutils import assert_array_approx_equal
import numpy.random as nrandom
from numpy.testing import assert_allclose
import pytest

from jax import random

import numpyro
from numpyro import handlers
from numpyro.contrib.einstein import (
    GraphicalKernel,
    IMQKernel,
    LinearKernel,
    RBFKernel,
    SteinVI,
    kernels,
)
from numpyro.contrib.einstein.kernels import (
    HessianPrecondMatrix,
    MixtureKernel,
    PrecondMatrixKernel,
    RandomFeatureKernel,
)
from numpyro.contrib.einstein.util import posdef, sqrth, sqrth_and_inv_sqrth
import numpyro.distributions as dist
from numpyro.distributions import Bernoulli, Normal, Poisson
from numpyro.distributions.transforms import AffineTransform
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import (
    AutoDelta,
    AutoDiagonalNormal,
    AutoLaplaceApproximation,
    AutoLowRankMultivariateNormal,
    AutoMultivariateNormal,
    AutoNormal,
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
    kernels.RBFKernel(),
    kernels.LinearKernel(),
    kernels.IMQKernel(),
    kernels.GraphicalKernel(),
    kernels.RandomFeatureKernel(),
]

np.set_printoptions(precision=100)
TKERNEL = namedtuple("TestSteinKernel", ["kernel", "particle_info", "loss_fn", "kval"])
PARTICLES_2D = np.array([[1.0, 2.0], [-10.0, 10.0], [7.0, 3.0], [2.0, -1]])

TPARTICLES_2D = (np.array([1.0, 2.0]), np.array([10.0, 5.0]))  # transformed particles


class WrappedGraphicalKernel(GraphicalKernel):
    def __init__(self, mode):
        super().__init__(mode=mode, local_kernel_fns={"p1": RBFKernel("norm")})


class WrappedPrecondMatrixKernel(PrecondMatrixKernel):
    def __init__(self, mode):
        super().__init__(
            HessianPrecondMatrix(), RBFKernel(mode=mode), precond_mode="const"
        )


class WrappedMixtureKernel(MixtureKernel):
    def __init__(self, mode):
        super().__init__(
            mode=mode,
            ws=np.array([0.2, 0.8]),
            kernel_fns=[RBFKernel(mode), RBFKernel(mode)],
        )


KERNEL_TEST_CASES = [
    TKERNEL(
        RBFKernel,
        lambda d: {},
        lambda x: x,
        {
            "norm": 0.040711474,
            "vector": np.array([0.056071877, 0.7260586]),
            "matrix": np.array([[0.040711474, 0.0], [0.0, 0.040711474]]),
        },
    ),
    TKERNEL(RandomFeatureKernel, lambda d: {}, lambda x: x, {"norm": 15.251404}),
    TKERNEL(
        IMQKernel,
        lambda d: {},
        lambda x: x,
        {"norm": 0.104828484, "vector": np.array([0.11043153, 0.31622776])},
    ),
    TKERNEL(LinearKernel, lambda d: {}, lambda x: x, {"norm": 21.0}),
    TKERNEL(
        WrappedMixtureKernel,
        lambda d: {},
        lambda x: x,
        {"matrix": np.array([[0.040711474, 0.0], [0.0, 0.040711474]])},
    ),
    TKERNEL(
        WrappedGraphicalKernel,
        lambda d: {"p1": (0, d)},
        lambda x: x,
        {"matrix": np.array([[0.040711474, 0.0], [0.0, 0.040711474]])},
    ),
    TKERNEL(
        WrappedPrecondMatrixKernel,
        lambda d: {},
        lambda x: -0.02 / 12 * x[0] ** 4 - 0.5 / 12 * x[1] ** 4 - x[0] * x[1],
        {
            "matrix": np.array(
                [[2.3780507e-04, -1.6688075e-05], [-1.6688075e-05, 1.2849815e-05]]
            )
        },
    ),
]

TEST_IDS = [t[0].__name__ for t in KERNEL_TEST_CASES]

PARTICLES = [(PARTICLES_2D, TPARTICLES_2D)]


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


########################################
#  Stein Exterior (Smoke tests)
########################################


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize(
    "init_loc_fn",
    (init_to_uniform(), init_to_sample(), init_to_median(), init_to_feasible()),
)
@pytest.mark.parametrize("auto_guide", (AutoDelta, AutoNormal))
@pytest.mark.parametrize("problem", (uniform_normal, regression))
def test_steinvi_smoke(kernel, auto_guide, init_loc_fn, problem):
    true_coefs, data, model = problem()
    stein = SteinVI(
        model,
        auto_guide(model, init_loc_fn=init_loc_fn),
        Adam(1e-1),
        Trace_ELBO(),
        kernel,
    )
    stein.run(random.PRNGKey(0), 1, *data)


########################################
# Stein Interior
########################################


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize(
    "init_loc_fn",
    (init_to_uniform(), init_to_sample(), init_to_median(), init_to_feasible()),
)
@pytest.mark.parametrize("auto_guide", (AutoDelta, AutoNormal))  # add transforms
@pytest.mark.parametrize("problem", (regression, uniform_normal))
def test_get_params(kernel, auto_guide, init_loc_fn, problem):
    _, data, model = problem()
    guide, optim, elbo = (
        auto_guide(model, init_loc_fn=init_loc_fn),
        Adam(1e-1),
        Trace_ELBO(),
    )

    stein = SteinVI(model, guide, optim, elbo, kernel)
    stein_params = stein.get_params(stein.init(random.PRNGKey(0), *data))

    svi = SVI(model, guide, optim, elbo)
    svi_params = svi.get_params(svi.init(random.PRNGKey(0), *data))
    assert svi_params.keys() == stein_params.keys()

    for name, svi_param in svi_params.items():
        assert (
            stein_params[name].shape
            == np.repeat(svi_param[None, ...], stein.num_particles, axis=0).shape
        )


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
        init_to_feasible,
        init_to_median,
        init_to_sample,
    ],
)
@pytest.mark.parametrize("num_particles", [1, 2, 10])
def test_auto_guide(auto_class, init_loc_fn, num_particles):
    latent_dim = 3

    def model(obs):
        a = numpyro.sample("a", Normal(0, 1))
        return numpyro.sample("obs", Bernoulli(logits=a), obs=obs)

    obs = Bernoulli(0.5).sample(random.PRNGKey(0), (10, latent_dim))

    rng_key = random.PRNGKey(0)
    guide_key, stein_key = random.split(rng_key)
    inner_guide = auto_class(model, init_loc_fn=init_loc_fn())

    with handlers.seed(rng_seed=guide_key), handlers.trace() as inner_guide_tr:
        inner_guide(obs)

    steinvi = SteinVI(
        model,
        auto_class(model, init_loc_fn=init_loc_fn()),
        Adam(1.0),
        Trace_ELBO(),
        RBFKernel(),
        num_particles=num_particles,
    )
    state = steinvi.init(stein_key, obs)
    init_params = steinvi.get_params(state)

    for name, site in inner_guide_tr.items():
        if site.get("type") == "param":
            assert name in init_params
            inner_param = site
            init_value = init_params[name]
            expected_shape = (num_particles, *np.shape(inner_param["value"]))
            assert init_value.shape == expected_shape
            if "auto_loc" in name or name == "b":
                assert np.alltrue(init_value != np.zeros(expected_shape))
                assert np.unique(init_value).shape == init_value.reshape(-1).shape
            elif "scale" in name:
                assert_array_approx_equal(init_value, np.full(expected_shape, 0.1))
            else:
                assert_array_approx_equal(init_value, np.full(expected_shape, 0.0))


def test_svgd_loss_and_grads():
    true_coefs, data, model = uniform_normal()
    guide = AutoDelta(model)
    loss = Trace_ELBO()
    stein_uparams = {
        "alpha_auto_loc": np.array(
            [
                -1.2,
            ]
        ),
        "loc_base_auto_loc": np.array(
            [
                1.53,
            ]
        ),
    }
    stein = SteinVI(model, guide, Adam(0.1), loss, RBFKernel())
    stein.init(random.PRNGKey(0), *data)
    svi = SVI(model, guide, Adam(0.1), loss)
    svi.init(random.PRNGKey(0), *data)
    expected_loss = loss.loss(
        random.PRNGKey(1), svi.constrain_fn(stein_uparams), model, guide, *data
    )
    stein_loss, stein_grad = stein._svgd_loss_and_grads(
        random.PRNGKey(1), stein_uparams, *data
    )
    assert expected_loss == stein_loss


@pytest.mark.parametrize(
    "kernel, particle_info, loss_fn, kval", KERNEL_TEST_CASES, ids=TEST_IDS
)
@pytest.mark.parametrize("mode", ["norm", "vector", "matrix"])
@pytest.mark.parametrize("particles, tparticles", PARTICLES)
def test_apply_kernel(
    kernel, particles, particle_info, loss_fn, tparticles, mode, kval
):
    if mode not in kval:
        pytest.skip()
    (d,) = tparticles[0].shape
    kernel_fn = kernel(mode=mode)
    kernel_fn.init(random.PRNGKey(0), particles.shape)
    kernel_fn = kernel_fn.compute(particles, particle_info(d), loss_fn)
    v = np.ones_like(kval[mode])
    stein = SteinVI(id, id, Adam(1.0), Trace_ELBO(), kernel(mode))
    value = stein._apply_kernel(kernel_fn, *tparticles, v)
    kval_ = copy(kval)
    if mode == "matrix":
        kval_[mode] = np.dot(kval_[mode], v)
    assert_allclose(value, kval_[mode], atol=1e-9)


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
    stein = SteinVI(id, id, Adam(1.0), Trace_ELBO(), RBFKernel())
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
    pinfo = stein._calc_particle_info(uparams, num_particles)

    for k in pinfo.keys():
        assert pinfo[k] == expected_pinfo[k], f"Failed for seed {seed}"


########################################
# Stein Kernels
########################################


@pytest.mark.parametrize(
    "kernel, particle_info, loss_fn, kval", KERNEL_TEST_CASES, ids=TEST_IDS
)
@pytest.mark.parametrize("particles, tparticles", PARTICLES)
@pytest.mark.parametrize("mode", ["norm", "vector", "matrix"])
def test_kernel_forward(
    kernel, particles, particle_info, loss_fn, tparticles, mode, kval
):
    if mode not in kval:
        return
    (d,) = tparticles[0].shape
    kernel_fn = kernel(mode=mode)
    kernel_fn.init(random.PRNGKey(0), particles.shape)
    kernel_fn = kernel_fn.compute(particles, particle_info(d), loss_fn)
    value = kernel_fn(*tparticles)

    assert_allclose(value, kval[mode], atol=1e-9)


@pytest.mark.parametrize("batch_shape", [(), (2,), (3, 1)])
def test_posdef(batch_shape):
    dim = 4
    x = np.random.normal(size=batch_shape + (dim, dim + 1))
    m = x @ np.swapaxes(x, -2, -1)
    assert_allclose(posdef(m), m, rtol=1e-5)


@pytest.mark.parametrize("batch_shape", [(), (2,), (3, 1)])
def test_sqrth(batch_shape):
    dim = 4
    x = np.random.normal(size=batch_shape + (dim, dim + 1))
    m = x @ np.swapaxes(x, -2, -1)
    s = sqrth(m)
    assert_allclose(s @ np.swapaxes(s, -2, -1), m, rtol=1e-5)


@pytest.mark.parametrize("batch_shape", [(), (2,), (3, 1)])
def test_sqrth_and_inv_sqrth(batch_shape):
    dim = 4
    x = np.random.normal(size=batch_shape + (dim, dim + 1))
    m = x @ np.swapaxes(x, -2, -1)
    s, i, si = sqrth_and_inv_sqrth(m)
    assert_allclose(s @ np.swapaxes(s, -2, -1), m, rtol=1e-5)
    assert_allclose(i, np.linalg.inv(m), rtol=1e-5)
    assert_allclose(si @ np.swapaxes(si, -2, -1), i, rtol=1e-5)
