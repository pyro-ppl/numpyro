# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from jax import lax, numpy as np, random
import numpy as onp
from numpy.testing import assert_allclose
import pytest

import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import AffineTransform, ExpTransform
import numpyro.handlers as handlers
from numpyro.contrib.autoguide import AutoIAFNormal
from numpyro.contrib.reparam import NeuTraReparam, TransformReparam, reparam
from numpyro.infer.util import initialize_model
from numpyro.infer import MCMC, NUTS, SVI, ELBO
from numpyro.optim import Adam


# Test helper to extract a few log central moments from samples.
def get_moments(x):
    assert (x > 0).all()
    x = np.log(x)
    m1 = np.mean(x, axis=0)
    x = x - m1
    xx = x * x
    xxx = x * xx
    xxxx = xx * xx
    m2 = np.mean(xx, axis=0)
    m3 = np.mean(xxx, axis=0) / m2 ** 1.5
    m4 = np.mean(xxxx, axis=0) / m2 ** 2
    return np.stack([m1, m2, m3, m4])


@pytest.mark.parametrize("shape", [(), (4,), (2, 3)], ids=str)
def test_log_normal(shape):
    loc = onp.random.rand(*shape) * 2 - 1
    scale = onp.random.rand(*shape) + 0.5

    def model():
        with numpyro.plate_stack("plates", shape):
            with numpyro.plate("particles", 100000):
                return numpyro.sample("x",
                                      dist.TransformedDistribution(
                                          dist.Normal(np.zeros_like(loc),
                                                      np.ones_like(scale)),
                                          [AffineTransform(loc, scale),
                                           ExpTransform()]))

    with handlers.trace() as tr:
        value = handlers.seed(model, 0)()
    assert isinstance(tr["x"]["fn"], dist.TransformedDistribution)
    expected_moments = get_moments(value)

    with reparam(config={"x": TransformReparam()}):
        with handlers.trace() as tr:
            value = handlers.seed(model, 0)()
    assert tr["x"]["type"] == "deterministic"
    actual_moments = get_moments(value)
    assert_allclose(actual_moments, expected_moments, atol=0.05)


def neals_funnel(dim):
    y = numpyro.sample('y', dist.Normal(0, 3))
    with numpyro.plate('D', dim):
        numpyro.sample('x', dist.Normal(0, np.exp(y / 2)))


def dirichlet_categorical(data):
    concentration = np.array([1.0, 1.0, 1.0])
    p_latent = numpyro.sample('p', dist.Dirichlet(concentration))
    with numpyro.plate('N', data.shape[0]):
        numpyro.sample('obs', dist.Categorical(p_latent), obs=data)
    return p_latent


def test_neals_funnel_smoke():
    dim = 10

    guide = AutoIAFNormal(neals_funnel)
    svi = SVI(neals_funnel, guide, Adam(1e-10), ELBO())
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
    (dirichlet_categorical, {'data': np.ones(10, dtype=np.int32)})
])
def test_reparam_log_joint(model, kwargs):
    guide = AutoIAFNormal(model)
    svi = SVI(model, guide, Adam(1e-10), ELBO(), **kwargs)
    svi_state = svi.init(random.PRNGKey(0))
    params = svi.get_params(svi_state)
    neutra = NeuTraReparam(guide, params)
    reparam_model = neutra.reparam(model)
    _, pe_fn, _ = initialize_model(random.PRNGKey(1), model, model_kwargs=kwargs)
    init_params, pe_fn_neutra, _ = initialize_model(random.PRNGKey(2), reparam_model, model_kwargs=kwargs)
    latent_x = list(init_params.values())[0]
    pe_transformed = pe_fn_neutra(init_params)
    latent_y = neutra.transform(latent_x)
    log_det_jacobian = neutra.transform.log_abs_det_jacobian(latent_x, latent_y)
    pe = pe_fn(guide._unpack_latent(latent_y))
    assert_allclose(pe_transformed, pe - log_det_jacobian)
