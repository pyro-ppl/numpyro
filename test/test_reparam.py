# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numpy.testing import assert_allclose
import pytest

from jax import jacobian, lax, random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import AffineTransform, ExpTransform
import numpyro.handlers as handlers
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoIAFNormal
from numpyro.infer.reparam import LocScaleReparam, NeuTraReparam, TransformReparam
from numpyro.infer.util import initialize_model
from numpyro.optim import Adam


# Test helper to extract a few central moments from samples.
def get_moments(x):
    m1 = jnp.mean(x, axis=0)
    x = x - m1
    xx = x * x
    xxx = x * xx
    xxxx = xx * xx
    m2 = jnp.mean(xx, axis=0)
    m3 = jnp.mean(xxx, axis=0) / m2 ** 1.5
    m4 = jnp.mean(xxxx, axis=0) / m2 ** 2
    return jnp.stack([m1, m2, m3, m4])


@pytest.mark.parametrize("shape", [(), (4,), (2, 3)], ids=str)
def test_log_normal(shape):
    loc = np.random.rand(*shape) * 2 - 1
    scale = np.random.rand(*shape) + 0.5

    def model():
        with numpyro.plate_stack("plates", shape):
            with numpyro.plate("particles", 100000):
                return numpyro.sample("x",
                                      dist.TransformedDistribution(
                                          dist.Normal(jnp.zeros_like(loc),
                                                      jnp.ones_like(scale)),
                                          [AffineTransform(loc, scale),
                                           ExpTransform()]).expand_by([100000]))

    with handlers.trace() as tr:
        value = handlers.seed(model, 0)()
    expected_moments = get_moments(jnp.log(value))

    with numpyro.handlers.reparam(config={"x": TransformReparam()}):
        with handlers.trace() as tr:
            value = handlers.seed(model, 0)()
    assert tr["x"]["type"] == "deterministic"
    actual_moments = get_moments(jnp.log(value))
    assert_allclose(actual_moments, expected_moments, atol=0.05)


def neals_funnel(dim):
    y = numpyro.sample('y', dist.Normal(0, 3))
    with numpyro.plate('D', dim):
        numpyro.sample('x', dist.Normal(0, jnp.exp(y / 2)))


def dirichlet_categorical(data):
    concentration = jnp.array([1.0, 1.0, 1.0])
    p_latent = numpyro.sample('p', dist.Dirichlet(concentration))
    with numpyro.plate('N', data.shape[0]):
        numpyro.sample('obs', dist.Categorical(p_latent), obs=data)
    return p_latent


def test_neals_funnel_smoke():
    dim = 10

    guide = AutoIAFNormal(neals_funnel)
    svi = SVI(neals_funnel, guide, Adam(1e-10), Trace_ELBO())
    svi_state = svi.init(random.PRNGKey(0), dim)

    def body_fn(i, val):
        svi_state, loss = svi.update(val, dim)
        return svi_state

    svi_state = lax.fori_loop(0, 1000, body_fn, svi_state)
    params = svi.get_params(svi_state)

    neutra = NeuTraReparam(guide, params)
    model = neutra.reparam(neals_funnel)
    nuts = NUTS(model)
    mcmc = MCMC(nuts, num_warmup=50, num_samples=50)
    mcmc.run(random.PRNGKey(1), dim)
    samples = mcmc.get_samples()
    transformed_samples = neutra.transform_sample(samples['auto_shared_latent'])
    assert 'x' in transformed_samples
    assert 'y' in transformed_samples


@pytest.mark.parametrize('model, kwargs', [
    (neals_funnel, {'dim': 10}),
    (dirichlet_categorical, {'data': jnp.ones(10, dtype=jnp.int32)})
])
def test_reparam_log_joint(model, kwargs):
    guide = AutoIAFNormal(model)
    svi = SVI(model, guide, Adam(1e-10), Trace_ELBO(), **kwargs)
    svi_state = svi.init(random.PRNGKey(0))
    params = svi.get_params(svi_state)
    neutra = NeuTraReparam(guide, params)
    reparam_model = neutra.reparam(model)
    _, pe_fn, _, _ = initialize_model(random.PRNGKey(1), model, model_kwargs=kwargs)
    init_params, pe_fn_neutra, _, _ = initialize_model(random.PRNGKey(2), reparam_model, model_kwargs=kwargs)
    latent_x = list(init_params[0].values())[0]
    pe_transformed = pe_fn_neutra(init_params[0])
    latent_y = neutra.transform(latent_x)
    log_det_jacobian = neutra.transform.log_abs_det_jacobian(latent_x, latent_y)
    pe = pe_fn(guide._unpack_latent(latent_y))
    assert_allclose(pe_transformed, pe - log_det_jacobian)


@pytest.mark.parametrize("shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("centered", [0., 0.6, 1., None])
@pytest.mark.parametrize("dist_type", ["Normal", "StudentT"])
@pytest.mark.parametrize("event_dim", [0, 1])
def test_loc_scale(dist_type, centered, shape, event_dim):
    loc = np.random.uniform(-1., 1., shape)
    scale = np.random.uniform(0.5, 1.5, shape)
    event_dim = min(event_dim, len(shape))

    def model(loc, scale):
        with numpyro.plate_stack("plates", shape[:len(shape) - event_dim]):
            with numpyro.plate("particles", 10000):
                if "dist_type" == "Normal":
                    numpyro.sample("x", dist.Normal(loc, scale).to_event(event_dim))
                else:
                    numpyro.sample("x", dist.StudentT(10.0, loc, scale).to_event(event_dim))

    def get_expected_probe(loc, scale):
        with numpyro.handlers.trace() as trace:
            with numpyro.handlers.seed(rng_seed=0):
                model(loc, scale)
        return get_moments(trace["x"]["value"])

    if "dist_type" == "Normal":
        reparam = LocScaleReparam()
    else:
        reparam = LocScaleReparam(shape_params=["df"])

    def get_actual_probe(loc, scale):
        with numpyro.handlers.trace() as trace:
            with numpyro.handlers.seed(rng_seed=0):
                with numpyro.handlers.reparam(config={"x": reparam}):
                    model(loc, scale)
        return get_moments(trace["x"]["value"])

    expected_probe = get_expected_probe(loc, scale)
    actual_probe = get_actual_probe(loc, scale)
    assert_allclose(actual_probe, expected_probe, atol=0.1)

    expected_grad = jacobian(get_expected_probe, argnums=(0, 1))(loc, scale)
    actual_grad = jacobian(get_actual_probe, argnums=(0, 1))(loc, scale)
    assert_allclose(actual_grad[0], expected_grad[0], atol=0.05)  # loc grad
    assert_allclose(actual_grad[1], expected_grad[1], atol=0.05)  # scale grad
