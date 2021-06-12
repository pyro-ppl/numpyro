import pytest
from jax import random
from numpyro.contrib.einstein.vi import VI

import numpyro.distributions as dist
from numpyro.infer.reparam import TransformReparam
from numpyro.distributions.transforms import AffineTransform

from numpyro.contrib.einstein import Stein
from numpyro.contrib.einstein import kernels
from numpyro.infer import Trace_ELBO, HMC, NUTS, SVI, log_likelihood, Predictive
from numpyro.infer.autoguide import AutoDelta, AutoNormal
from numpyro.infer.initialization import (
    init_to_uniform,
    init_to_sample,
    init_with_noise,
    init_to_median,
    init_to_feasible,
    init_to_value,
)
from numpyro.infer.util import log_density
from numpyro.optim import Adam
import numpyro
import jax.numpy as jnp

KERNELS = [
    kernels.RBFKernel(),
    kernels.LinearKernel(),
    kernels.IMQKernel(),
    kernels.GraphicalKernel(),
    kernels.RandomFeatureKernel(),
]


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
@pytest.mark.parametrize("sp_criterion", ("local", "rand"))
@pytest.mark.parametrize("sp_mode", ("local", "global"))
@pytest.mark.parametrize("num_mcmc_particles", (1, 10, 20))
@pytest.mark.parametrize("mcmc_warmup", (1, 2, 100, 500))
@pytest.mark.parametrize("mcmc_samples", (1, 2, 100, 500))
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


def test_score_sp_mcmc():
    pass


def test_svgd_loss_and_grads():
    pass


def test_update():
    pass


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize(
    "init_strategy",
    (init_to_uniform(), init_to_sample(), init_to_median(), init_to_feasible()),
)
@pytest.mark.parametrize("auto_guide", (AutoDelta, AutoNormal))  # add transforms
@pytest.mark.parametrize("problem", (uniform_normal, regression))
def test_init(kernel, auto_guide, init_strategy, problem):
    true_coefs, data, model = problem()
    stein = Stein(
        model,
        auto_guide(model),
        Adam(1e-1),
        Trace_ELBO(),
        kernel,
        init_strategy=init_strategy,
    )
    state = stein.init(random.PRNGKey(0), *data)
    stein.get_params(state)


########################################
# Variational Interface
########################################
@pytest.mark.parametrize("callback", [])
def test_callsback(callback):
    pass
