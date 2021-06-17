import string
from collections import namedtuple

import jax.numpy as jnp
import numpy as np
import numpy.random as nrandom
import pytest
from jax import random
from numpy.testing import assert_allclose

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.einstein import (
    Stein,
    RBFKernel,
    IMQKernel,
    LinearKernel,
    GraphicalKernel,
)
from numpyro.contrib.einstein import kernels
from numpyro.contrib.einstein.kernels import (
    HessianPrecondMatrix,
    MixtureKernel,
    PrecondMatrixKernel,
    RandomFeatureKernel,
)
from numpyro.contrib.einstein.utils import posdef, sqrth, sqrth_and_inv_sqrth
from numpyro.distributions import Poisson
from numpyro.distributions.transforms import AffineTransform
from numpyro.infer import Trace_ELBO, HMC, NUTS, SVI
from numpyro.infer.autoguide import AutoDelta, AutoNormal
from numpyro.infer.initialization import (
    init_to_uniform,
    init_to_sample,
    init_to_median,
    init_to_feasible,
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

jnp.set_printoptions(precision=100)
TKERNEL = namedtuple("TestSteinKernel", ["kernel", "particle_info", "loss_fn", "kval"])
PARTICLES_2D = jnp.array([[1.0, 2.0], [-10.0, 10.0], [7.0, 3.0], [2.0, -1]])

TPARTICLES_2D = (jnp.array([1.0, 2.0]), jnp.array([10.0, 5.0]))  # transformed particles


class WrappedGraphicalKernel(GraphicalKernel):
    def __init__(self, mode):
        super().__init__(mode=mode, local_kernel_fns={"p1": RBFKernel("norm")})


class WrappedPrecondMatrixKernel(PrecondMatrixKernel):
    def __init__(self, mode):
        super().__init__(HessianPrecondMatrix(), RBFKernel(mode=mode), precond_mode="const")


class WrappedMixtureKernel(MixtureKernel):
    def __init__(self, mode):
        super().__init__(mode=mode, ws=jnp.array([0.2, 0.8]), kernel_fns=[RBFKernel(mode), RBFKernel(mode)], )


KERNEL_TEST_CASES = [
    TKERNEL(
        RBFKernel,
        lambda d: {},
        lambda x: x,
        {
            "norm": 0.040711474,
            "vector": jnp.array([0.056071877, 0.7260586]),
            "matrix": jnp.array([[0.040711474, 0.0], [0.0, 0.040711474]]),
        },
    ),
    TKERNEL(RandomFeatureKernel, lambda d: {}, lambda x: x, {"norm": 15.251404}),
    TKERNEL(
        IMQKernel,
        lambda d: {},
        lambda x: x,
        {"norm": 0.104828484, "vector": jnp.array([0.11043153, 0.31622776])},
    ),
    TKERNEL(LinearKernel, lambda d: {}, lambda x: x, {"norm": 21.0}),
    TKERNEL(WrappedMixtureKernel, lambda d: {}, lambda x: x,
            {"matrix": jnp.array([[0.040711474, 0.0], [0.0, 0.040711474]])},
            ),
    TKERNEL(
        WrappedGraphicalKernel,
        lambda d: {"p1": (0, d)},
        lambda x: x,
        {"matrix": jnp.array([[0.040711474, 0.0], [0.0, 0.040711474]])},
    ),
    TKERNEL(
        WrappedPrecondMatrixKernel,
        lambda d: {},
        lambda x: -0.02 / 12 * x[0] ** 4 - 0.5 / 12 * x[1] ** 4 - x[0] * x[1],
        {
            "matrix": jnp.array(
                [[2.3780507e-04, -1.6688075e-05], [-1.6688075e-05, 1.2849815e-05]]
            )
        },
    ),
]

TEST_IDS = [t[0].__name__ for t in KERNEL_TEST_CASES]

PARTICLES = [(PARTICLES_2D, TPARTICLES_2D)]


@pytest.mark.parametrize("particles, tparticles", PARTICLES)
########################################
#  Stein Exterior
########################################

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
        numpyro.sample("obs", dist.Normal(loc, 0.1), obs=data)

    data = true_coef + random.normal(random.PRNGKey(0), (1000,))
    return true_coef, (data,), model


def regression():
    N, dim = 1000, 3
    data = random.normal(random.PRNGKey(0), (N, dim))
    true_coefs = jnp.arange(1.0, dim + 1.0)
    logits = jnp.sum(true_coefs * data, axis=-1)
    labels = dist.Bernoulli(logits=logits).sample(random.PRNGKey(1))

    def model(features, labels):
        coefs = numpyro.sample("coefs", dist.Normal(jnp.zeros(dim), jnp.ones(dim)))
        logits = numpyro.deterministic("logits", jnp.sum(coefs * features, axis=-1))
        return numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=labels)

    return true_coefs, (data, labels), model


# init_with_noise(),  # consider
# init_to_value()))
@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize(
    "init_strategy",
    (init_to_uniform(), init_to_sample(), init_to_median(), init_to_feasible()),
)
@pytest.mark.parametrize("auto_guide", (AutoDelta, AutoNormal))  # add transforms
@pytest.mark.parametrize("problem", (uniform_normal, regression))
def test_init_strategy(kernel, auto_guide, init_strategy, problem):
    true_coefs, data, model = problem()
    stein = Stein(
        model,
        auto_guide(model),
        Adam(1e-1),
        Trace_ELBO(),
        kernel,
        init_strategy=init_strategy,
    )
    state, loss = stein.run(random.PRNGKey(0), 100, *data)
    stein.get_params(state)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("sp_criterion", ("infl", "rand"))
@pytest.mark.parametrize("sp_mode", ("local", "global"))
@pytest.mark.parametrize("num_mcmc_particles", (1, 2, 10))
@pytest.mark.parametrize("mcmc_warmup", (1, 2, 100))
@pytest.mark.parametrize("mcmc_samples", (1, 2, 100))
@pytest.mark.parametrize("mcmc_kernel", (HMC, NUTS))
@pytest.mark.parametrize("auto_guide", (AutoDelta, AutoNormal))  # add transforms
@pytest.mark.parametrize("problem", (uniform_normal, regression))
def test_stein_point(
        kernel,
        sp_criterion,
        auto_guide,
        sp_mode,
        num_mcmc_particles,
        mcmc_warmup,
        mcmc_samples,
        mcmc_kernel,
        problem,
):
    true_coefs, data, model = problem()
    stein = Stein(
        model,
        auto_guide(model),
        Adam(1e-1),
        Trace_ELBO(),
        kernel,
        sp_mode=sp_mode,
        sp_mcmc_crit=sp_criterion,
        mcmc_kernel=mcmc_kernel,
    )
    state, loss = stein.run(random.PRNGKey(0), 100, *data)
    stein.get_params(state)


########################################
# Stein Interior
########################################


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize(
    "init_strategy",
    (init_to_uniform(), init_to_sample(), init_to_median(), init_to_feasible()),
)
@pytest.mark.parametrize("auto_guide", (AutoDelta, AutoNormal))  # add transforms
@pytest.mark.parametrize("problem", (regression, uniform_normal))
def test_get_params(kernel, auto_guide, init_strategy, problem):
    _, data, model = problem()
    guide, optim, elbo = auto_guide(model), Adam(1e-1), Trace_ELBO()

    stein = Stein(model, guide, optim, elbo, kernel, init_strategy=init_strategy)
    stein_params = stein.get_params(stein.init(random.PRNGKey(0), *data))

    svi = SVI(model, guide, optim, elbo)
    svi_params = svi.get_params(svi.init(random.PRNGKey(0), *data))
    assert svi_params.keys() == stein_params.keys()

    for name, svi_param in svi_params.items():
        assert (
                stein_params[name].shape
                == jnp.repeat(svi_param[None, ...], stein.num_particles, axis=0).shape
        )


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize(
    "init_strategy",
    (init_to_uniform(), init_to_sample(), init_to_median(), init_to_feasible()),
)
@pytest.mark.parametrize("auto_guide", (AutoDelta, AutoNormal))  # add transforms
@pytest.mark.parametrize("problem", (regression, uniform_normal))
def test_update_evaluate(kernel, auto_guide, init_strategy, problem):
    _, data, model = problem()
    guide, optim, elbo = auto_guide(model), Adam(1e-1), Trace_ELBO()

    stein = Stein(model, guide, optim, elbo, kernel, init_strategy=init_strategy)
    state = stein.init(random.PRNGKey(0), *data)
    _, update_loss = stein.update(state, *data)
    state = stein.init(random.PRNGKey(0), *data)
    eval_loss = stein.evaluate(state, *data)
    assert eval_loss == update_loss


@pytest.mark.parametrize("sp_mode", ["local", "global"])
@pytest.mark.parametrize("subset_idxs", [[], [1], [0, 2, 1, 3]])
def test_score_sp_mcmc(sp_mode, subset_idxs):
    true_coef, data, model = uniform_normal()
    if sp_mode == "local" and not subset_idxs == []:
        pytest.skip()
    stein_uparams = {}
    sp_mcmc_subset_uparams = {}
    classic_uparams = {}
    stein = Stein(
        model, AutoDelta(model), Adam(1e-1), Trace_ELBO(), RBFKernel(), sp_mode=sp_mode
    )
    stein._score_sp_mcmc(
        random.PRNGKey(0),
        subset_idxs,
        stein_uparams,
        sp_mcmc_subset_uparams,
        classic_uparams,
        *data,
        **{},
    )


def test_svgd_loss_and_grads():
    pass


@pytest.mark.parametrize("num_mcmc_particles", (1, 2, 10))
@pytest.mark.parametrize("mcmc_warmup", (1, 2, 100))
@pytest.mark.parametrize("mcmc_samples", (1, 2, 100))
@pytest.mark.parametrize("mcmc_kernel", (HMC, NUTS))
def test_sp_mcmc(num_mcmc_particles, mcmc_warmup, mcmc_samples, mcmc_kernel):
    uconstr_params = {}
    stein = Stein()
    stein._sp_mcmc(random.PRNGKey(0), uconstr_params)


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
    v = jnp.ones_like(kval[mode])
    stein = Stein(id, id, Adam(1.0), Trace_ELBO(), kernel(mode))
    value = stein._apply_kernel(kernel_fn, *tparticles, v)
    if mode == "matrix":
        kval[mode] = jnp.dot(kval[mode], v)
    assert_allclose(value, kval[mode], atol=1e-9)


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
    uparam = t(
        nest(jnp.empty(tuple(size)), nrandom.randint(0, depth)) for size in sizes
    )
    stein = Stein(id, id, Adam(1.0), Trace_ELBO(), RBFKernel())
    assert stein._param_size(uparam) == total_size, f"Failed for seed {seed}"


@pytest.mark.parametrize("num_particles", [1, 2, 7, 10])
@pytest.mark.parametrize("num_params", [0, 1, 2, 10, 20])
def test_calc_particle_info(num_params, num_particles):
    seed = random.PRNGKey(nrandom.randint(0, 10_000))
    sizes = Poisson(5).sample(seed, (100, nrandom.randint(0, 10))) + 1

    uparam = tuple(jnp.empty(tuple(size)) for size in sizes)
    uparams = {string.ascii_lowercase[i]: uparam for i in range(num_params)}

    par_param_size = sum(map(lambda size: size.prod(), sizes)) // num_particles
    expected_start_end = zip(
        par_param_size * np.arange(num_params),
        par_param_size * np.arange(1, num_params + 1),
    )
    expected_pinfo = dict(zip(string.ascii_lowercase[:num_params], expected_start_end))

    stein = Stein(id, id, Adam(1.0), Trace_ELBO(), RBFKernel())
    pinfo = stein._calc_particle_info(uparams, num_particles)

    for k in pinfo.keys():
        assert pinfo[k] == expected_pinfo[k], f"Failed for seed {seed}"


########################################
# Variational Interface
########################################
@pytest.mark.parametrize("callback", [])
def test_callsback(callback):
    pass


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
