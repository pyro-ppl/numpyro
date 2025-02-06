# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

from numpy.random import RandomState
from numpy.testing import assert_allclose
import pytest

import jax
from jax import jacobian, jit, lax, random, vmap
from jax.example_libraries.stax import Dense
import jax.numpy as jnp
import optax
from optax import piecewise_constant_schedule

import numpyro
from numpyro import handlers, optim
from numpyro.contrib.control_flow import scan
import numpyro.distributions as dist
from numpyro.distributions import constraints, transforms
from numpyro.distributions.flows import InverseAutoregressiveTransform
from numpyro.handlers import substitute
from numpyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from numpyro.infer.autoguide import (
    AutoBatchedLowRankMultivariateNormal,
    AutoBatchedMultivariateNormal,
    AutoBNAFNormal,
    AutoDAIS,
    AutoDelta,
    AutoDiagonalNormal,
    AutoGuideList,
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
    init_to_value,
)
from numpyro.infer.reparam import TransformReparam
from numpyro.infer.util import Predictive
from numpyro.nn.auto_reg_nn import AutoregressiveNN
from numpyro.util import fori_loop

init_strategy = init_to_median(num_samples=2)


@pytest.mark.parametrize(
    "auto_class",
    [
        AutoDiagonalNormal,
        AutoDAIS,
        AutoIAFNormal,
        AutoBNAFNormal,
        AutoMultivariateNormal,
        AutoLaplaceApproximation,
        AutoLowRankMultivariateNormal,
        AutoNormal,
        AutoDelta,
        AutoGuideList,
    ],
)
def test_beta_bernoulli(auto_class):
    data = jnp.array([[1.0] * 8 + [0.0] * 2, [1.0] * 4 + [0.0] * 6]).T
    N = len(data)

    def model(data):
        f = numpyro.sample("beta", dist.Beta(jnp.ones(2), jnp.ones(2)).to_event())
        with numpyro.plate("N", N):
            numpyro.sample("obs", dist.Bernoulli(f).to_event(1), obs=data)

    adam = optim.Adam(0.01)
    if auto_class == AutoDAIS:
        guide = auto_class(model, init_loc_fn=init_strategy, base_dist="cholesky")
    elif auto_class == AutoGuideList:
        guide = AutoGuideList(model)
        guide.append(AutoNormal(handlers.block(model, hide=[])))
    else:
        guide = auto_class(model, init_loc_fn=init_strategy)
    svi = SVI(model, guide, adam, Trace_ELBO())
    svi_state = svi.init(random.PRNGKey(1), data)

    def body_fn(i, val):
        svi_state, loss = svi.update(val, data)
        return svi_state

    svi_state = fori_loop(0, 3000, body_fn, svi_state)
    params = svi.get_params(svi_state)

    true_coefs = (jnp.sum(data, axis=0) + 1) / (data.shape[0] + 2)
    # test .sample_posterior method
    posterior_samples = guide.sample_posterior(
        random.PRNGKey(1), params, sample_shape=(1000,)
    )
    posterior_mean = jnp.mean(posterior_samples["beta"], 0)
    assert_allclose(posterior_mean, true_coefs, atol=0.05)

    # assume AutoGuideList does not have AutoDAIS, AutoDelta, AutoIAFNormal, or AutoBNAFNormal
    if auto_class not in (AutoDAIS, AutoIAFNormal, AutoBNAFNormal):
        median = guide.median(params)
        assert median["beta"].shape == (2,)
        # test .quantile method
        if auto_class is not AutoDelta:
            quantiles = guide.quantiles(params, [0.2, 0.5, 0.8])
            assert quantiles["beta"].shape == (3, 2)

    # Predictive can be instantiated from posterior samples...
    predictive = Predictive(model, posterior_samples=posterior_samples)
    predictive_samples = predictive(random.PRNGKey(1), None)
    assert predictive_samples["obs"].shape == (1000, N, 2)

    # ... or from the guide + params
    predictive = Predictive(model, guide=guide, params=params, num_samples=1000)
    predictive_samples = predictive(random.PRNGKey(1), None)
    assert predictive_samples["obs"].shape == (1000, N, 2)


@pytest.mark.parametrize(
    "auto_class",
    [
        AutoDiagonalNormal,
        AutoIAFNormal,
        AutoDAIS,
        AutoBNAFNormal,
        AutoMultivariateNormal,
        AutoLaplaceApproximation,
        AutoLowRankMultivariateNormal,
        AutoNormal,
        AutoDelta,
        AutoGuideList,
    ],
)
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceMeanField_ELBO])
def test_logistic_regression(auto_class, Elbo):
    N, dim = 3000, 3
    data = random.normal(random.PRNGKey(0), (N, dim))
    true_coefs = jnp.arange(1.0, dim + 1.0)
    logits = jnp.sum(true_coefs * data, axis=-1)
    labels = dist.Bernoulli(logits=logits).sample(random.PRNGKey(1))

    def model(data=None, labels=None):
        coefs = numpyro.sample("coefs", dist.Normal(0, 1).expand([dim]).to_event())
        if data is not None:
            logits = numpyro.deterministic("logits", jnp.sum(coefs * data, axis=-1))
            with numpyro.plate("N", len(data)):
                return numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=labels)

    adam = optim.Adam(0.01)
    rng_key_init = random.PRNGKey(1)
    if auto_class == AutoGuideList:
        guide = AutoGuideList(model)
        guide.append(AutoNormal(handlers.block(model, hide=["logits"])))
    else:
        guide = auto_class(model, init_loc_fn=init_strategy)
    svi = SVI(model, guide, adam, Elbo())
    svi_state = svi.init(rng_key_init, data, labels)

    # smoke test if analytic KL is used
    # assume that AutoGuideList has AutoNormal
    if (
        auto_class is AutoNormal or auto_class is AutoGuideList
    ) and Elbo is TraceMeanField_ELBO:
        _, mean_field_loss = svi.update(svi_state, data, labels)
        svi.loss = Trace_ELBO()
        _, elbo_loss = svi.update(svi_state, data, labels)
        svi.loss = TraceMeanField_ELBO()
        assert abs(mean_field_loss - elbo_loss) > 0.5

    def body_fn(i, val):
        svi_state, loss = svi.update(val, data, labels)
        return svi_state

    svi_state = fori_loop(0, 2000, body_fn, svi_state)
    params = svi.get_params(svi_state)
    # assume that AutoGuideList does not have AutoDAIS, AutoIAFNormal, or AutoBNAFNormal
    if auto_class not in (AutoDAIS, AutoIAFNormal, AutoBNAFNormal):
        median = guide.median(params)
        assert_allclose(median["coefs"], true_coefs, rtol=0.1)
        # test .quantile method
        if auto_class is not AutoDelta:
            quantiles = guide.quantiles(params, [0.2, 0.5])
            assert_allclose(quantiles["coefs"][1], true_coefs, rtol=0.1)
    # test .sample_posterior method
    posterior_samples = guide.sample_posterior(
        random.PRNGKey(1), params, sample_shape=(1000,)
    )
    expected_coefs = jnp.array([0.97, 2.05, 3.18])
    assert_allclose(jnp.mean(posterior_samples["coefs"], 0), expected_coefs, rtol=0.1)


def test_mvn_quantile():
    def model():
        numpyro.sample("x", dist.Normal(0, 1).expand([2]).to_event(1))

    guide = AutoMultivariateNormal(model)
    with handlers.seed(rng_seed=random.PRNGKey(0)):
        guide()
    params = {
        "auto_loc": jnp.zeros(2),
        "auto_scale_tril": jnp.array([[1.0, 0.0], [0.5, 0.5]]),
    }
    actual = guide.quantiles(params, quantiles=0.3)["x"]
    mvn = dist.MultivariateNormal(
        params["auto_loc"], scale_tril=params["auto_scale_tril"]
    )
    expected = dist.Normal(params["auto_loc"], jnp.sqrt(mvn.variance)).icdf(
        jnp.array(0.3)
    )
    assert_allclose(actual, expected, rtol=1e-5)


def test_iaf():
    # test for substitute logic for exposed methods `sample_posterior` and `get_transforms`
    N, dim = 3000, 3
    data = random.normal(random.PRNGKey(0), (N, dim))
    true_coefs = jnp.arange(1.0, dim + 1.0)
    logits = jnp.sum(true_coefs * data, axis=-1)
    labels = dist.Bernoulli(logits=logits).sample(random.PRNGKey(1))

    def model(data, labels):
        coefs = numpyro.sample("coefs", dist.Normal(jnp.zeros(dim), jnp.ones(dim)))
        offset = numpyro.sample("offset", dist.Uniform(-1, 1))
        logits = offset + jnp.sum(coefs * data, axis=-1)
        return numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=labels)

    adam = optim.Adam(0.01)
    rng_key_init = random.PRNGKey(1)
    guide = AutoIAFNormal(model)
    svi = SVI(model, guide, adam, Trace_ELBO())
    svi_state = svi.init(rng_key_init, data, labels)
    params = svi.get_params(svi_state)

    x = random.normal(random.PRNGKey(0), (dim + 1,))
    rng_key = random.PRNGKey(1)
    actual_sample = guide.sample_posterior(rng_key, params)
    actual_output = guide._unpack_latent(guide.get_transform(params)(x))

    flows = []
    for i in range(guide.num_flows):
        if i > 0:
            flows.append(transforms.PermuteTransform(jnp.arange(dim + 1)[::-1]))
        arn_init, arn_apply = AutoregressiveNN(
            dim + 1,
            [dim + 1, dim + 1],
            permutation=jnp.arange(dim + 1),
            skip_connections=guide._skip_connections,
            nonlinearity=guide._nonlinearity,
        )
        arn = partial(arn_apply, params["auto_arn__{}$params".format(i)])
        flows.append(InverseAutoregressiveTransform(arn))
    flows.append(guide._unpack_latent)

    transform = transforms.ComposeTransform(flows)
    _, rng_key_sample = random.split(rng_key)
    expected_sample = transform(
        dist.Normal(jnp.zeros(dim + 1), 1).sample(rng_key_sample)
    )
    expected_output = transform(x)
    assert_allclose(actual_sample["coefs"], expected_sample["coefs"])
    assert_allclose(
        actual_sample["offset"],
        transforms.biject_to(constraints.interval(-1, 1))(expected_sample["offset"]),
    )

    jax.tree.all(jax.tree.map(assert_allclose, actual_output, expected_output))


def test_uniform_normal():
    true_coef = 0.9
    data = true_coef + random.normal(random.PRNGKey(0), (1000,))

    def model(data):
        alpha = numpyro.sample("alpha", dist.Uniform(0, 1))
        with numpyro.handlers.reparam(config={"loc": TransformReparam()}):
            loc = numpyro.sample(
                "loc",
                dist.TransformedDistribution(
                    dist.Uniform(0, 1), transforms.AffineTransform(0, alpha)
                ),
            )
        with numpyro.plate("N", len(data)):
            numpyro.sample("obs", dist.Normal(loc, 0.1), obs=data)

    adam = optim.Adam(0.01)
    rng_key_init = random.PRNGKey(1)
    guide = AutoDiagonalNormal(model)
    svi = SVI(model, guide, adam, Trace_ELBO())
    svi_state = svi.init(rng_key_init, data)

    def body_fn(i, val):
        svi_state, loss = svi.update(val, data)
        return svi_state

    svi_state = fori_loop(0, 1000, body_fn, svi_state)
    params = svi.get_params(svi_state)
    median = guide.median(params)
    assert_allclose(median["loc"], true_coef, rtol=0.05)
    # test .quantile method
    median = guide.quantiles(params, [0.2, 0.5])
    assert_allclose(median["loc"][1], true_coef, rtol=0.1)


def test_param():
    # this test the validity of model having
    # param sites contain composed transformed constraints
    rng_keys = random.split(random.PRNGKey(0), 3)
    a_minval = 1
    a_init = jnp.exp(random.normal(rng_keys[0])) + a_minval
    b_init = jnp.exp(random.normal(rng_keys[1]))
    x_init = random.normal(rng_keys[2])

    def model():
        a = numpyro.param("a", a_init, constraint=constraints.greater_than(a_minval))
        b = numpyro.param("b", b_init, constraint=constraints.positive)
        numpyro.sample("x", dist.Normal(a, b))

    # this class is used to force init value of `x` to x_init
    class _AutoGuide(AutoDiagonalNormal):
        def __call__(self, *args, **kwargs):
            return substitute(
                super(_AutoGuide, self).__call__, {"_auto_latent": x_init[None]}
            )(*args, **kwargs)

    adam = optim.Adam(0.01)
    rng_key_init = random.PRNGKey(1)
    guide = _AutoGuide(model)
    svi = SVI(model, guide, adam, Trace_ELBO())
    svi_state = svi.init(rng_key_init)

    params = svi.get_params(svi_state)
    assert_allclose(params["a"], a_init, rtol=1e-6)
    assert_allclose(params["b"], b_init, rtol=1e-6)
    assert_allclose(params["auto_loc"], guide._init_latent, rtol=1e-6)
    assert_allclose(params["auto_scale"], jnp.ones(1) * guide._init_scale, rtol=1e-6)

    actual_loss = svi.evaluate(svi_state)
    assert jnp.isfinite(actual_loss)
    expected_loss = dist.Normal(guide._init_latent, guide._init_scale).log_prob(
        x_init
    ) - dist.Normal(a_init, b_init).log_prob(x_init)
    assert_allclose(actual_loss, expected_loss, rtol=1e-6)


def test_dynamic_supports():
    true_coef = 0.9
    data = true_coef + random.normal(random.PRNGKey(0), (1000,))

    def actual_model(data):
        alpha = numpyro.sample("alpha", dist.Uniform(0, 1))
        with numpyro.handlers.reparam(config={"loc": TransformReparam()}):
            loc = numpyro.sample(
                "loc",
                dist.TransformedDistribution(
                    dist.Uniform(0, 1), transforms.AffineTransform(0, alpha)
                ),
            )
        with numpyro.plate("N", len(data)):
            numpyro.sample("obs", dist.Normal(loc, 0.1), obs=data)

    def expected_model(data):
        alpha = numpyro.sample("alpha", dist.Uniform(0, 1))
        loc = numpyro.sample("loc", dist.Uniform(0, 1)) * alpha
        with numpyro.plate("N", len(data)):
            numpyro.sample("obs", dist.Normal(loc, 0.1), obs=data)

    adam = optim.Adam(0.01)
    rng_key_init = random.PRNGKey(1)

    guide = AutoDiagonalNormal(actual_model)
    svi = SVI(actual_model, guide, adam, Trace_ELBO())
    svi_state = svi.init(rng_key_init, data)
    actual_opt_params = adam.get_params(svi_state.optim_state)
    actual_params = svi.get_params(svi_state)
    actual_values = guide.median(actual_params)
    actual_loss = svi.evaluate(svi_state, data)

    guide = AutoDiagonalNormal(expected_model)
    svi = SVI(expected_model, guide, adam, Trace_ELBO())
    svi_state = svi.init(rng_key_init, data)
    expected_opt_params = adam.get_params(svi_state.optim_state)
    expected_params = svi.get_params(svi_state)
    expected_values = guide.median(expected_params)
    expected_loss = svi.evaluate(svi_state, data)

    # test auto_loc, auto_scale
    jax.tree.all(jax.tree.map(assert_allclose, actual_opt_params, expected_opt_params))
    jax.tree.all(jax.tree.map(assert_allclose, actual_params, expected_params))
    # test latent values
    assert_allclose(actual_values["alpha"], expected_values["alpha"])
    assert_allclose(actual_values["loc_base"], expected_values["loc"])
    assert_allclose(actual_loss, expected_loss)


def test_laplace_approximation_warning():
    def model(x, y):
        a = numpyro.sample("a", dist.Normal(0, 10))
        b = numpyro.sample("b", dist.Normal(0, 10).expand([3]).to_event())
        mu = a + b[0] * x + b[1] * x**2 + b[2] * x**3
        with numpyro.plate("N", len(x)):
            numpyro.sample("y", dist.Normal(mu, 0.001), obs=y)

    x = random.normal(random.PRNGKey(0), (3,))
    y = 1 + 2 * x + 3 * x**2 + 4 * x**3
    guide = AutoLaplaceApproximation(model)
    svi = SVI(model, guide, optim.Adam(0.1), Trace_ELBO(), x=x, y=y)
    init_state = svi.init(random.PRNGKey(0))
    svi_state = fori_loop(0, 10000, lambda i, val: svi.update(val)[0], init_state)
    params = svi.get_params(svi_state)
    with pytest.warns(UserWarning, match="Hessian of log posterior"):
        guide.sample_posterior(random.PRNGKey(1), params)


def test_laplace_approximation_custom_hessian():
    def model(x, y):
        a = numpyro.sample("a", dist.Normal(0, 10))
        b = numpyro.sample("b", dist.Normal(0, 10))
        mu = a + b * x
        with numpyro.plate("N", len(x)):
            numpyro.sample("y", dist.Normal(mu, 1), obs=y)

    x = random.normal(random.PRNGKey(0), (100,))
    y = 1 + 2 * x
    guide = AutoLaplaceApproximation(
        model, hessian_fn=lambda f, x: jacobian(jacobian(f))(x)
    )
    svi = SVI(model, guide, optim.Adam(0.1), Trace_ELBO(), x=x, y=y)
    svi_result = svi.run(random.PRNGKey(0), 10000, progress_bar=False)
    guide.get_transform(svi_result.params)


def test_improper():
    y = random.normal(random.PRNGKey(0), (100,))

    def model(y):
        lambda1 = numpyro.sample(
            "lambda1", dist.ImproperUniform(dist.constraints.real, (), ())
        )
        lambda2 = numpyro.sample(
            "lambda2", dist.ImproperUniform(dist.constraints.real, (), ())
        )
        sigma = numpyro.sample(
            "sigma", dist.ImproperUniform(dist.constraints.positive, (), ())
        )
        mu = numpyro.deterministic("mu", lambda1 + lambda2)
        with numpyro.plate("N", len(y)):
            numpyro.sample("y", dist.Normal(mu, sigma), obs=y)

    guide = AutoDiagonalNormal(model)
    svi = SVI(model, guide, optim.Adam(0.003), Trace_ELBO(), y=y)
    svi_state = svi.init(random.PRNGKey(2))
    lax.scan(lambda state, i: svi.update(state), svi_state, jnp.zeros(10000))


def test_module():
    x = random.normal(random.PRNGKey(0), (100, 10))
    y = random.normal(random.PRNGKey(1), (100,))

    def model(x, y):
        nn = numpyro.module("nn", Dense(1), (10,))
        mu = nn(x).squeeze(-1)
        sigma = numpyro.sample("sigma", dist.HalfNormal(1))
        with numpyro.plate("N", len(y)):
            numpyro.sample("y", dist.Normal(mu, sigma), obs=y)

    guide = AutoDiagonalNormal(model)
    svi = SVI(model, guide, optim.Adam(0.003), Trace_ELBO(), x=x, y=y)
    svi_state = svi.init(random.PRNGKey(2))
    lax.scan(lambda state, i: svi.update(state), svi_state, jnp.zeros(1000))


@pytest.mark.parametrize("auto_class", [AutoNormal])
def test_subsample_guide(auto_class):
    # The model adapted from tutorial/source/easyguide.ipynb
    def model(batch, subsample, full_size):
        drift = numpyro.sample("drift", dist.LogNormal(-1, 0.5))
        with handlers.substitute(data={"data": subsample}):
            plate = numpyro.plate("data", full_size, subsample_size=len(subsample))
        assert plate.size == 50

        def transition_fn(z_prev, y_curr):
            with plate:
                z_curr = numpyro.sample("state", dist.Normal(z_prev, drift))
                y_curr = numpyro.sample(
                    "obs", dist.Bernoulli(logits=z_curr), obs=y_curr
                )
            return z_curr, y_curr

        _, result = scan(
            transition_fn, jnp.zeros(len(subsample)), batch, length=num_time_steps
        )
        return result

    def create_plates(batch, subsample, full_size):
        with handlers.substitute(data={"data": subsample}):
            return numpyro.plate("data", full_size, subsample_size=subsample.shape[0])

    guide = auto_class(model, create_plates=create_plates)

    full_size = 50
    batch_size = 20
    num_time_steps = 8
    with handlers.seed(rng_seed=0):
        data = model(None, jnp.arange(full_size), full_size)
    assert data.shape == (num_time_steps, full_size)

    svi = SVI(model, guide, optim.Adam(0.02), Trace_ELBO())
    svi_state = svi.init(
        random.PRNGKey(0),
        data[:, :batch_size],
        jnp.arange(batch_size),
        full_size=full_size,
    )
    update_fn = jit(svi.update, static_argnums=(3,))
    for epoch in range(2):
        beg = 0
        while beg < full_size:
            end = min(full_size, beg + batch_size)
            subsample = jnp.arange(beg, end)
            batch = data[:, beg:end]
            beg = end
            svi_state, loss = update_fn(svi_state, batch, subsample, full_size)


@pytest.mark.parametrize(
    "auto_class",
    [
        AutoDiagonalNormal,
        AutoMultivariateNormal,
        AutoLaplaceApproximation,
        AutoLowRankMultivariateNormal,
        AutoNormal,
        AutoDelta,
    ],
)
def test_autoguide_deterministic(auto_class):
    def model(y=None):
        n, len_y = (y.size, len(y)) if y is not None else (1, 1)

        mu = numpyro.sample("mu", dist.Normal(0, 5))
        sigma = numpyro.param("sigma", 1, constraint=constraints.positive)

        with numpyro.plate("N", len_y):
            y = numpyro.sample("y", dist.Normal(mu, sigma).expand((n,)), obs=y)
        numpyro.deterministic("z", (y - mu) / sigma)

    mu, sigma = 2, 3
    y = mu + sigma * random.normal(random.PRNGKey(0), shape=(300,))
    y_train = y[:200]
    y_test = y[200:]

    guide = auto_class(model)
    optimiser = numpyro.optim.Adam(step_size=0.01)
    svi = SVI(model, guide, optimiser, Trace_ELBO())

    svi_result = svi.run(random.PRNGKey(0), num_steps=500, y=y_train)
    params = svi_result.params
    posterior_samples = guide.sample_posterior(
        random.PRNGKey(0), params, sample_shape=(1000,)
    )

    posterior_samples.pop("z")
    predictive = Predictive(model, posterior_samples, params=params)
    predictive_samples = predictive(random.PRNGKey(0), y_test)

    assert predictive_samples["y"].shape == (1000, 100)
    assert predictive_samples["z"].shape == (1000, 100)
    assert_allclose(
        (predictive_samples["y"] - posterior_samples["mu"][..., None])
        / params["sigma"],
        predictive_samples["z"],
        atol=0.05,
    )


@pytest.mark.parametrize("size,dim", [(10, -2), (5, -1)])
def test_plate_inconsistent(size, dim):
    def model():
        with numpyro.plate("a", 10, dim=-1):
            numpyro.sample("x", dist.Normal(0, 1))
        with numpyro.plate("a", size, dim=dim):
            numpyro.sample("y", dist.Normal(0, 1))

    guide = AutoDelta(model)
    svi = SVI(model, guide, numpyro.optim.Adam(step_size=0.1), Trace_ELBO())
    with pytest.raises(AssertionError, match="has inconsistent dim or size"):
        svi.run(random.PRNGKey(0), 10)


@pytest.mark.parametrize(
    "auto_class",
    [
        AutoDelta,
        AutoDiagonalNormal,
        AutoMultivariateNormal,
        AutoNormal,
        AutoLowRankMultivariateNormal,
        AutoLaplaceApproximation,
    ],
)
@pytest.mark.parametrize(
    "init_loc_fn",
    [
        init_to_feasible,
        init_to_median,
        init_to_sample,
        init_to_uniform,
    ],
)
@pytest.mark.filterwarnings("ignore:.*enumerate.*:FutureWarning")
def test_discrete_helpful_error(auto_class, init_loc_fn):
    def model():
        p = numpyro.sample("p", dist.Beta(2.0, 2.0))
        x = numpyro.sample("x", dist.Bernoulli(p))
        with numpyro.plate("N", 2):
            numpyro.sample(
                "obs",
                dist.Bernoulli(p * x + (1 - p) * (1 - x)),
                obs=jnp.array([1.0, 0.0]),
            )

    guide = auto_class(model, init_loc_fn=init_loc_fn)
    with pytest.raises(ValueError, match=".*handle discrete.*"):
        handlers.seed(guide, 0)()


@pytest.mark.parametrize(
    "auto_class",
    [
        AutoDelta,
        AutoDiagonalNormal,
        AutoMultivariateNormal,
        AutoNormal,
        AutoLowRankMultivariateNormal,
        AutoLaplaceApproximation,
    ],
)
@pytest.mark.parametrize(
    "init_loc_fn",
    [
        init_to_feasible,
        init_to_median,
        init_to_sample,
        init_to_uniform,
    ],
)
def test_sphere_helpful_error(auto_class, init_loc_fn):
    def model():
        x = numpyro.sample("x", dist.Normal(0.0, 1.0).expand([2]).to_event(1))
        y = numpyro.sample("y", dist.ProjectedNormal(x))
        numpyro.sample("obs", dist.Normal(y, 1), obs=jnp.array([1.0, 0.0]))

    guide = auto_class(model, init_loc_fn=init_loc_fn)
    with pytest.raises(ValueError, match=".*ProjectedNormalReparam.*"):
        handlers.seed(guide, 0)()


def test_autodais_subsampling_error():
    data = jnp.array([1.0] * 8 + [0.0] * 2)

    def model(data):
        f = numpyro.sample("beta", dist.Beta(1, 1))
        with numpyro.plate("plate", 20, 10, dim=-1):
            numpyro.sample("obs", dist.Bernoulli(f), obs=data)

    adam = optim.Adam(0.01)
    guide = AutoDAIS(model)
    svi = SVI(model, guide, adam, Trace_ELBO())

    with pytest.raises(NotImplementedError, match=".*data subsampling.*"):
        svi.init(random.PRNGKey(1), data)


def test_subsample_model_with_deterministic():
    def model():
        x = numpyro.sample("x", dist.Normal(0, 1))
        numpyro.deterministic("x2", x * 2)
        with numpyro.plate("N", 10, subsample_size=5):
            numpyro.sample("obs", dist.Normal(x, 1), obs=jnp.ones(5))

    guide = AutoNormal(model)
    svi = SVI(model, guide, optim.Adam(1.0), Trace_ELBO())
    svi_result = svi.run(random.PRNGKey(0), 10)
    samples = guide.sample_posterior(random.PRNGKey(1), svi_result.params)
    assert "x2" in samples


def test_autocontinuous_local_error():
    def model():
        with numpyro.plate("N", 10, subsample_size=4):
            numpyro.sample("x", dist.Normal(0, 1))

    guide = AutoDiagonalNormal(model)
    svi = SVI(model, guide, optim.Adam(1.0), Trace_ELBO())
    with pytest.raises(ValueError, match="local latent variables"):
        svi.init(random.PRNGKey(0))


def test_init_to_scalar_value():
    def model():
        numpyro.sample("x", dist.Normal(0, 1))

    guide = AutoDiagonalNormal(model, init_loc_fn=init_to_value(values={"x": 1.0}))
    svi = SVI(model, guide, optim.Adam(1.0), Trace_ELBO())
    svi.init(random.PRNGKey(0))


def test_autosemidais(N=18, P=3, sigma_obs=0.1, num_steps=45 * 1000, num_samples=5000):
    X = RandomState(0).randn(N, P)
    Y = X[:, 0] - 0.5 * X[:, 1] + sigma_obs * RandomState(1).randn(N)

    def global_model():
        return numpyro.sample("theta", dist.Normal(jnp.zeros(P), 1).to_event(1))

    def local_model(subsample_size, theta):
        with numpyro.plate("N", N, subsample_size=subsample_size):
            X_batch = numpyro.subsample(X, event_dim=1)
            Y_batch = numpyro.subsample(Y, event_dim=0)
            tau = numpyro.sample("tau", dist.Gamma(5.0, 5.0))
            numpyro.sample(
                "obs",
                dist.Normal(X_batch @ theta, sigma_obs / jnp.sqrt(tau)),
                obs=Y_batch,
            )

    def model12():
        return local_model(12, global_model())

    def model16():
        return local_model(16, global_model())

    def _get_optim():
        scheduler = piecewise_constant_schedule(
            1.0e-3, {15 * 1000: 1.0e-4, 30 * 1000: 1.0e-5}
        )
        return optax.chain(
            optax.scale_by_adam(), optax.scale_by_schedule(scheduler), optax.scale(-1.0)
        )

    base_guide = AutoNormal(global_model)
    guide16 = AutoSemiDAIS(
        model16, partial(local_model, 16), base_guide, K=4, eta_max=0.25, eta_init=0.005
    )
    svi_result16 = SVI(model16, guide16, _get_optim(), Trace_ELBO()).run(
        random.PRNGKey(0), num_steps
    )

    assert svi_result16.params["auto_eta_coeff"].mean() > 0.2

    samples16 = guide16.sample_posterior(random.PRNGKey(1), svi_result16.params)
    assert samples16["theta"].shape == (P,) and samples16["tau"].shape == (16,)

    dais_elbo16 = Trace_ELBO(num_particles=num_samples).loss(
        random.PRNGKey(0), svi_result16.params, model16, guide16
    )
    dais_elbo16 = -dais_elbo16.item()

    guide12 = AutoSemiDAIS(
        model12, partial(local_model, 12), base_guide, K=4, eta_max=0.25, eta_init=0.005
    )
    with handlers.seed(rng_seed=0):
        guide12()  # initialize guide since we are not training this guide
    samples12 = guide12.sample_posterior(random.PRNGKey(1), svi_result16.params)
    assert samples12["theta"].shape == (P,) and samples12["tau"].shape == (12,)

    dais_elbo12 = Trace_ELBO(num_particles=num_samples).loss(
        random.PRNGKey(0), svi_result16.params, model12, guide12
    )
    dais_elbo12 = -dais_elbo12.item()
    assert_allclose(dais_elbo12, dais_elbo16, atol=0.05)

    def create_plates():
        return numpyro.plate("N", N, subsample_size=16)

    mf_guide = AutoNormal(model16, create_plates=create_plates)
    mf_svi_result = SVI(model16, mf_guide, _get_optim(), Trace_ELBO()).run(
        random.PRNGKey(0), num_steps
    )

    mf_elbo = Trace_ELBO(num_particles=num_samples).loss(
        random.PRNGKey(0), mf_svi_result.params, model16, mf_guide
    )
    mf_elbo = -mf_elbo.item()
    assert dais_elbo16 > mf_elbo + 0.1

    with handlers.substitute(
        data={"N": jnp.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])}
    ):
        samples_one = guide12.sample_posterior(random.PRNGKey(1), svi_result16.params)

    with handlers.substitute(
        data={"N": jnp.array([0, 2, 9, 10, 11, 12, 13, 14, 3, 4, 5, 6])}
    ):
        samples_two = guide12.sample_posterior(random.PRNGKey(1), svi_result16.params)
    assert_allclose(samples_one["theta"], samples_two["theta"])
    assert_allclose(samples_one["tau"][:2], samples_two["tau"][:2])
    assert jnp.min(jnp.abs(samples_one["tau"][2:] - samples_two["tau"][2:])) > 1e-5


def test_autosemidais_admissible_smoke():
    def global_model():
        theta = numpyro.sample("theta", dist.Normal(0, 1))
        tau = numpyro.sample("tau", dist.LogNormal(jnp.zeros(2), 1).to_event(1))
        return {"tau": tau.mean(), "theta": theta}

    def local_model1(global_latents):
        tau = global_latents["tau"]
        theta = global_latents["theta"]
        with numpyro.plate("inner", 10, subsample_size=5, dim=-1):
            with numpyro.plate("outer", 20, dim=-2):
                sigma1 = numpyro.sample("sigma", dist.LogNormal(0.0, 1.0))
                sigma2 = numpyro.sample(
                    "log_sigma", dist.Normal(jnp.zeros(2), 1.0).to_event(1)
                )
                sigma2 = jnp.exp(sigma2.mean(-1))
                assert sigma1.shape == (20, 5) and sigma2.shape == (20, 5)
                numpyro.sample(
                    "obs",
                    dist.Normal(theta, tau * sigma1 * sigma2),
                    obs=jnp.ones((20, 5)),
                )

    def model():
        return local_model1(global_model())

    base_guide = AutoNormal(global_model)
    guide = AutoSemiDAIS(model, local_model1, base_guide)
    svi = SVI(model, guide, optim.Adam(0.01), Trace_ELBO())
    svi_result = svi.run(random.PRNGKey(0), 10)
    samples = guide.sample_posterior(random.PRNGKey(1), svi_result.params)
    assert samples["theta"].shape == () and samples["tau"].shape == (2,)
    assert samples["sigma"].shape == (20, 5)
    assert samples["log_sigma"].shape == (20, 5, 2)

    def local_model2(global_latents):
        tau = global_latents["tau"]
        theta = global_latents["theta"]
        with numpyro.plate("inner", 10, subsample_size=5, dim=-1):
            sigma1 = numpyro.sample("sigma", dist.LogNormal(0.0, 1.0))
            sigma2 = numpyro.sample(
                "log_sigma", dist.Normal(jnp.zeros(2), 1.0).to_event(1)
            )
            sigma2 = jnp.exp(sigma2.mean(-1))
            numpyro.sample(
                "obs", dist.Normal(theta, tau * sigma1 * sigma2), obs=jnp.ones(5)
            )

    def model():
        return local_model2(global_model())

    base_guide = AutoNormal(global_model)
    guide = AutoSemiDAIS(model, local_model2, base_guide)
    svi = SVI(model, guide, optim.Adam(0.01), Trace_ELBO())
    svi_result = svi.run(random.PRNGKey(0), 10)
    samples = guide.sample_posterior(random.PRNGKey(1), svi_result.params)
    assert samples["theta"].shape == () and samples["tau"].shape == (2,)
    assert samples["sigma"].shape == (5,) and samples["log_sigma"].shape == (5, 2)


def test_autosemidais_local_only():
    data = jnp.linspace(0, 1, 10)

    def model():
        with numpyro.plate("N", 10, subsample_size=5, dim=-1):
            batch = numpyro.subsample(data, event_dim=0)
            numpyro.sample("x", dist.Normal(batch, 1))

    def create_plates():
        return numpyro.plate("N", 10, subsample_size=5, dim=-1)

    local_guide = AutoNormal(model, create_plates=create_plates)
    guide = AutoSemiDAIS(model, model, None, local_guide=local_guide)
    svi = SVI(model, guide, optim.Adam(0.01), Trace_ELBO())
    svi_result = svi.run(random.PRNGKey(0), 10)
    samples = guide.sample_posterior(
        random.PRNGKey(1), svi_result.params, sample_shape=(100,)
    )
    assert samples["x"].shape == (100, 5)


def test_autosemidais_inadmissible_smoke():
    def global_model():
        return numpyro.sample("theta", dist.Normal(0, 1))

    def local_model1(theta):
        with numpyro.plate("a", 10, 5):
            numpyro.sample("x", dist.Normal(0, 1))
            with numpyro.plate("b", 20, 10):
                numpyro.sample("y", dist.Normal(0, 1))

    def local_model2(theta):
        with numpyro.plate("a", 10, 5):
            numpyro.sample("y", dist.Normal(0, 1), obs=jnp.ones(5))

    def model():
        return local_model1(global_model())

    base_guide = AutoNormal(global_model)
    guide = AutoSemiDAIS(model, local_model1, base_guide)
    svi = SVI(model, guide, optim.Adam(0.01), Trace_ELBO())

    with pytest.raises(AssertionError, match="contains exactly 1 plate"):
        svi.run(random.PRNGKey(0), 10)

    def model2():
        return local_model2(global_model())

    guide = AutoSemiDAIS(model2, local_model2, base_guide)
    svi = SVI(model2, guide, optim.Adam(0.01), Trace_ELBO())
    with pytest.raises(RuntimeError, match="are no local variables"):
        svi.run(random.PRNGKey(0), 10)


def test_autosldais(
    N=64, subsample_size=48, num_surrogate=32, D=3, num_steps=40000, num_samples=2000
):
    def _model(X, Y):
        theta = numpyro.sample(
            "theta", dist.Normal(jnp.zeros(D), jnp.ones(D)).to_event(1)
        )
        with numpyro.plate("N", N, subsample_size=subsample_size):
            X_batch = numpyro.subsample(X, event_dim=1)
            Y_batch = numpyro.subsample(Y, event_dim=0)
            numpyro.sample("obs", dist.Bernoulli(logits=theta @ X_batch.T), obs=Y_batch)

    def _surrogate_model(X_surr, Y_surr):
        theta = numpyro.sample(
            "theta", dist.Normal(jnp.zeros(D), jnp.ones(D)).to_event(1)
        )
        omegas = numpyro.param(
            "omegas",
            2.0 * jnp.ones(num_surrogate),
            constraint=dist.constraints.positive,
        )

        with numpyro.plate("N", num_surrogate), numpyro.handlers.scale(scale=omegas):
            numpyro.sample("obs", dist.Bernoulli(logits=theta @ X_surr.T), obs=Y_surr)

    X = RandomState(0).randn(N, D)
    X[:, 2] = X[:, 0] + X[:, 1]
    logits = X[:, 0] - 0.5 * X[:, 1]
    Y = dist.Bernoulli(logits=logits).sample(random.PRNGKey(0))

    model = partial(_model, X, Y)
    surrogate_model = partial(_surrogate_model, X[:num_surrogate], Y[:num_surrogate])

    def _get_optim():
        scheduler = piecewise_constant_schedule(
            1.0e-3, {15 * 1000: 1.0e-4, 30 * 1000: 1.0e-5}
        )
        return optax.chain(
            optax.scale_by_adam(), optax.scale_by_schedule(scheduler), optax.scale(-1.0)
        )

    guide = AutoSurrogateLikelihoodDAIS(
        model, surrogate_model, K=3, eta_max=0.25, eta_init=0.005
    )
    svi_result = SVI(model, guide, _get_optim(), Trace_ELBO()).run(
        random.PRNGKey(1), num_steps
    )

    samples = guide.sample_posterior(random.PRNGKey(2), svi_result.params)
    assert samples["theta"].shape == (D,)

    dais_elbo = Trace_ELBO(num_particles=num_samples).loss(
        random.PRNGKey(0), svi_result.params, model, guide
    )
    dais_elbo = -dais_elbo.item()

    def create_plates():
        return numpyro.plate("N", N, subsample_size=subsample_size)

    mf_guide = AutoNormal(model, create_plates=create_plates)
    mf_svi_result = SVI(model, mf_guide, _get_optim(), Trace_ELBO()).run(
        random.PRNGKey(0), num_steps
    )

    mf_elbo = Trace_ELBO(num_particles=num_samples).loss(
        random.PRNGKey(1), mf_svi_result.params, model, mf_guide
    )
    mf_elbo = -mf_elbo.item()

    assert dais_elbo > mf_elbo + 0.1


def test_autodelta_capture_deterministic_variables():
    def model():
        x = numpyro.sample("x", dist.Normal())
        numpyro.deterministic("x2", x**2)

    guide = AutoDelta(model)
    svi = SVI(model, guide, optim.Adam(0.01), Trace_ELBO())
    svi_result = svi.run(random.PRNGKey(0), num_steps=1_000)
    guide_samples = guide.sample_posterior(
        rng_key=random.PRNGKey(1), params=svi_result.params
    )
    assert "x2" in guide_samples


@pytest.mark.parametrize("shape", [(), (1,), (2, 3)])
@pytest.mark.parametrize("sample_shape", [(), (1,), (2, 3)])
def test_autodelta_sample_posterior_with_sample_shape(shape, sample_shape):
    def model():
        x = numpyro.sample("x", dist.Normal().expand(shape))
        numpyro.deterministic("x2", x**2)

    guide = AutoDelta(model)
    svi = SVI(model, guide, optim.Adam(0.01), Trace_ELBO())
    svi_result = svi.run(random.PRNGKey(0), num_steps=1_000)
    guide_samples = guide.sample_posterior(
        rng_key=random.PRNGKey(1), params=svi_result.params, sample_shape=sample_shape
    )
    assert guide_samples["x"].shape == sample_shape + shape
    assert guide_samples["x2"].shape == sample_shape + shape


@pytest.mark.parametrize(
    "auto_classes",
    [
        (AutoNormal, AutoDiagonalNormal),
        (AutoNormal, AutoDAIS),
        (AutoNormal, AutoSemiDAIS),
        (AutoNormal, AutoSurrogateLikelihoodDAIS),
        (AutoNormal, AutoIAFNormal),
        (AutoNormal, AutoBNAFNormal),
        (AutoNormal, AutoMultivariateNormal),
        (AutoNormal, AutoLaplaceApproximation),
        (AutoNormal, AutoLowRankMultivariateNormal),
        (AutoNormal, AutoNormal),
        (AutoNormal, AutoDelta),
    ],
)
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceMeanField_ELBO])
def test_autoguidelist(auto_classes, Elbo):
    sigma = 0.05

    def model(x, y=None):
        a = numpyro.sample("a", dist.Normal(0, 1))
        b = numpyro.sample("b", dist.Normal(jnp.ones((2, 1)), 1))
        mu = a + x @ b
        with numpyro.plate("N", len(x), dim=-2):
            numpyro.sample("y", dist.Normal(mu, sigma), obs=y)

    N = 500
    a = 1
    b = jnp.asarray([[-0.5], [-1]])
    x = random.normal(random.PRNGKey(0), (N, 2))
    y = a + x @ b + sigma * random.normal(random.PRNGKey(1), (N, 1))

    guide = AutoGuideList(model)
    guide.append(auto_classes[0](numpyro.handlers.block(model, expose=["a"])))
    # AutoGuideList does not support AutoDAIS, AutoSemiDAIS, or AutoSurrogateLikelihoodDAIS
    if auto_classes[1] == AutoDAIS:
        with pytest.raises(
            ValueError,
            match="AutoDAIS, AutoSemiDAIS, and AutoSurrogateLikelihoodDAIS are not supported.",
        ):
            guide.append(auto_classes[1](numpyro.handlers.block(model, hide=["a"])))
        return
    if auto_classes[1] == AutoSemiDAIS:
        with pytest.raises(
            ValueError,
            match="AutoDAIS, AutoSemiDAIS, and AutoSurrogateLikelihoodDAIS are not supported.",
        ):
            guide.append(
                auto_classes[1](
                    numpyro.handlers.block(model, hide=["a"]),
                    local_model=None,
                    global_guide=None,
                )
            )
        return
    if auto_classes[1] == AutoSurrogateLikelihoodDAIS:
        with pytest.raises(
            ValueError,
            match="AutoDAIS, AutoSemiDAIS, and AutoSurrogateLikelihoodDAIS are not supported.",
        ):
            guide.append(
                auto_classes[1](
                    numpyro.handlers.block(model, hide=["a"]),
                    surrogate_model=None,
                )
            )
        return

    guide.append(auto_classes[1](numpyro.handlers.block(model, hide=["a"])))

    optimiser = numpyro.optim.Adam(step_size=0.1)
    svi = SVI(model, guide, optimiser, Elbo())

    svi_result = svi.run(random.PRNGKey(0), num_steps=500, x=x, y=y)
    params = svi_result.params
    posterior_samples = guide.sample_posterior(
        random.PRNGKey(0), params, x=x, sample_shape=(1_000,)
    )

    assert posterior_samples["a"].shape == (1_000,)
    assert posterior_samples["b"].shape == (1_000, 2, 1)

    assert_allclose(jnp.mean(posterior_samples["a"], 0), a, atol=0.05)
    assert_allclose(jnp.mean(posterior_samples["b"], 0), b, atol=0.05)

    # Predictive can be instantiated from posterior samples...
    predictive = Predictive(model, posterior_samples=posterior_samples)
    predictive_samples = predictive(random.PRNGKey(1), x)
    assert predictive_samples["y"].shape == (1_000, N, 1)

    # ... or from the guide + params
    predictive = Predictive(model, guide=guide, params=params, num_samples=1_000)
    predictive_samples = predictive(random.PRNGKey(1), x)
    assert predictive_samples["y"].shape == (1_000, N, 1)

    # median and quantiles from guide
    if any(
        auto_class in [AutoDelta, AutoIAFNormal, AutoBNAFNormal]
        for auto_class in auto_classes
    ):
        with pytest.raises(NotImplementedError):
            quantiles = guide.quantiles(params=params, quantiles=[0.2, 0.5, 0.8])
    else:
        quantiles = guide.quantiles(params=params, quantiles=[0.2, 0.5, 0.8])
        assert quantiles["a"].shape == (3,)
        assert quantiles["b"].shape == (3, 2, 1)

    # median and quantiles from partial guides
    for auto_class, part in zip(auto_classes, guide):
        if auto_class not in (AutoIAFNormal, AutoBNAFNormal):
            part.median(params=params)
        if auto_class not in [AutoDelta, AutoIAFNormal, AutoBNAFNormal]:
            part.quantiles(params=params, quantiles=[0.2, 0.5, 0.8])


def test_autoguidelist_deterministic():
    def model():
        x = numpyro.sample("x", dist.Normal())
        numpyro.deterministic("x2", x**2)

    guide = AutoGuideList(model)
    guide.append(AutoDiagonalNormal(model))
    seeded_guide = handlers.seed(guide, 8)
    with pytest.raises(ValueError, match="should not be exposed"):
        seeded_guide()


@pytest.mark.parametrize(
    "auto_class",
    [
        AutoDiagonalNormal,
        AutoDAIS,
        AutoSemiDAIS,
        AutoSurrogateLikelihoodDAIS,
        AutoIAFNormal,
        AutoBNAFNormal,
        AutoMultivariateNormal,
        AutoLaplaceApproximation,
        AutoLowRankMultivariateNormal,
        AutoNormal,
        AutoDelta,
    ],
)
@pytest.mark.parametrize("shape", [(), (1,), (2, 3)])
@pytest.mark.parametrize("sample_shape", [(), (1,), (2, 3)])
def test_autoguidelist_sample_posterior_with_sample_shape(
    auto_class, shape, sample_shape
):
    def model():
        x = numpyro.sample("x", dist.Normal().expand(shape))
        numpyro.deterministic("x2", x**2)

    guide = AutoGuideList(model)
    blocked_model = handlers.block(model, hide=["x2"])

    # AutoGuideList does not support AutoDAIS, AutoSemiDAIS, or AutoSurrogateLikelihoodDAIS
    if auto_class == AutoDAIS:
        with pytest.raises(
            ValueError,
            match="AutoDAIS, AutoSemiDAIS, and AutoSurrogateLikelihoodDAIS are not supported.",
        ):
            guide.append(auto_class(blocked_model))
        return
    if auto_class == AutoSemiDAIS:
        with pytest.raises(
            ValueError,
            match="AutoDAIS, AutoSemiDAIS, and AutoSurrogateLikelihoodDAIS are not supported.",
        ):
            guide.append(auto_class(blocked_model, local_model=None, global_guide=None))
        return
    if auto_class == AutoSurrogateLikelihoodDAIS:
        with pytest.raises(
            ValueError,
            match="AutoDAIS, AutoSemiDAIS, and AutoSurrogateLikelihoodDAIS are not supported.",
        ):
            guide.append(auto_class(blocked_model, surrogate_model=None))
        return

    guide.append(auto_class(blocked_model))
    svi = SVI(model, guide, optim.Adam(0.01), Trace_ELBO())
    if auto_class in (AutoIAFNormal, AutoBNAFNormal) and max(shape, default=0) <= 1:
        with pytest.raises(
            ValueError,
            match="latent dim = 1. Consider using AutoDiagonalNormal instead",
        ):
            svi_result = svi.run(random.PRNGKey(0), num_steps=1_000)
            guide_samples = guide.sample_posterior(
                rng_key=random.PRNGKey(1),
                params=svi_result.params,
                sample_shape=sample_shape,
            )
    else:
        svi_result = svi.run(random.PRNGKey(0), num_steps=1_000)
        guide_samples = guide.sample_posterior(
            rng_key=random.PRNGKey(1),
            params=svi_result.params,
            sample_shape=sample_shape,
        )
        assert guide_samples["x"].shape == sample_shape + shape
        # Substitute and trace to get the deterministic sites.
        trace = handlers.trace(handlers.substitute(model, guide_samples)).get_trace()
        assert trace["x2"]["value"].shape == sample_shape + shape


@pytest.mark.parametrize("use_global_dais_params", [True, False])
def test_dais_vae(use_global_dais_params):
    def model():
        with numpyro.plate("N", 10):
            numpyro.sample("x", dist.Normal(jnp.arange(-5, 5), 2))

    guide = AutoSemiDAIS(
        model, model, subsample_plate="N", use_global_dais_params=use_global_dais_params
    )
    svi = SVI(model, guide, optax.adam(0.02), Trace_ELBO())
    svi_results = svi.run(random.PRNGKey(0), 5000)
    samples = guide.sample_posterior(
        random.PRNGKey(1), svi_results.params, sample_shape=(1000,)
    )
    if use_global_dais_params:
        assert_allclose(
            samples["x"].mean(), jnp.arange(-5, 5).mean(), atol=0.1, rtol=0.1
        )
    else:
        assert_allclose(
            samples["x"].mean(axis=0), jnp.arange(-5, 5), atol=0.2, rtol=0.1
        )


@pytest.mark.parametrize(
    "auto_class",
    [
        AutoBatchedMultivariateNormal,
        AutoBatchedLowRankMultivariateNormal,
    ],
)
def test_auto_batched(auto_class) -> None:
    # Model for batched multivariate normal.
    off_diag = jnp.asarray([-0.2, 0, 0.5])
    covs = off_diag[:, None, None] + jnp.eye(4)

    def model():
        with numpyro.plate("N", off_diag.shape[0]):
            numpyro.sample("x", dist.MultivariateNormal(0, covs))

    # Run inference.
    guide = auto_class(model)
    svi = SVI(model, guide, optax.adam(0.001), Trace_ELBO())
    result = svi.run(random.PRNGKey(0), 10000)
    samples = guide.sample_posterior(
        random.PRNGKey(1), result.params, sample_shape=(1000,)
    )

    # Verify off-diagonal entries are correlated.
    empirical_covs = vmap(jnp.cov)(jnp.moveaxis(samples["x"], 0, 2))
    i, j = jnp.triu_indices(3, 1)
    empirical_off_diag = empirical_covs[:, i, j].mean(axis=1)
    corrcoef = jnp.corrcoef(off_diag, empirical_off_diag)[0, 1]
    assert corrcoef > 0.99


@pytest.mark.parametrize(
    "auto_class",
    [
        AutoBatchedMultivariateNormal,
        AutoBatchedLowRankMultivariateNormal,
    ],
)
def test_auto_batched_shapes(auto_class) -> None:
    def model(n, m):
        distribution = dist.Normal().expand([7]).to_event(1)
        with numpyro.plate("n", n):
            x = numpyro.sample("x", distribution)
        with numpyro.plate("m", m):
            y = numpyro.sample("y", distribution)
        return x, y

    with numpyro.handlers.seed(rng_seed=0):
        auto_class(model)(3, 3)

        with pytest.raises(ValueError, match="inconsistent batch shapes"):
            auto_class(model)(3, 4)

        with pytest.raises(ValueError, match="Expected 2 batch dimensions"):
            auto_class(model, batch_ndim=2)(3, 3)


@pytest.mark.parametrize(
    "auto_class", [AutoBatchedMultivariateNormal, AutoBatchedLowRankMultivariateNormal]
)
def test_auto_batched_median(auto_class) -> None:
    def model():
        distribution = dist.Normal().expand([7]).to_event(1)
        with numpyro.plate("n", 3):
            x = numpyro.sample("x", distribution)
        with numpyro.plate("m", 3):
            y = numpyro.sample("y", distribution)
        return x, y

    guide = auto_class(model)
    with numpyro.handlers.seed(rng_seed=0):
        trace = numpyro.handlers.trace(guide).get_trace()
    params = {
        key: value["value"] for key, value in trace.items() if value["type"] == "param"
    }
    median = guide.median(params)
    assert jnp.allclose(
        jnp.stack(list(median.values())).ravel(), params["auto_loc"].ravel()
    )


def test_autoguide_with_delta_site() -> None:
    def model(x):
        numpyro.sample("x", dist.Delta(3.0), obs=x)
        # Need to sample a latent variable so the guide is not empty.
        numpyro.sample("y", dist.Normal())

    guide = AutoDiagonalNormal(lambda: model(None))
    with pytest.raises(ValueError, match="has a delta distribution"):
        numpyro.handlers.seed(guide, 9)()

    # Check delta distributions are fine if observed.
    guide = AutoDiagonalNormal(lambda: model(3.0))
    numpyro.handlers.seed(guide, 9)()


@pytest.mark.parametrize(
    "guide_cls",
    [
        AutoBNAFNormal,
        AutoDAIS,
        AutoDelta,
        AutoDiagonalNormal,
        AutoLaplaceApproximation,
        AutoLowRankMultivariateNormal,
        AutoMultivariateNormal,
        AutoNormal,
    ],
)
def test_subsample(guide_cls) -> None:
    def model(n: int, x: jnp.ndarray):
        mu = numpyro.sample("mu", dist.Normal(0, 1))
        sigma = numpyro.sample("sigma", dist.HalfNormal(1))
        with numpyro.plate("n", n, subsample_size=x.size):
            numpyro.sample("x", dist.Normal(mu, sigma), obs=x)

    n = 20
    x = 5 + jax.random.normal(jax.random.key(1), (20,))
    subset = x[: n // 2]

    svi = numpyro.infer.SVI(
        model,
        guide_cls(model),
        numpyro.optim.Adam(0.1),
        numpyro.infer.Trace_ELBO(),
        n=n,
    )
    state = svi.init(jax.random.key(2), x=x)
    svi.update(state, x=subset)
