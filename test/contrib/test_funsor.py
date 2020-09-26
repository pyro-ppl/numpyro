# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from functools import partial

import numpy as np
from numpy.testing import assert_allclose
import pytest

from jax import random
import jax.numpy as jnp

from funsor import Tensor, bint, reals
import numpyro
from numpyro.contrib.control_flow import scan
from numpyro.contrib.funsor import config_enumerate, enum, markov, to_data, to_funsor
from numpyro.contrib.funsor.enum_messenger import NamedMessenger
from numpyro.contrib.funsor.infer_util import log_density
from numpyro.contrib.indexing import Vindex
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


def test_gaussian_mixture_model():
    K, N = 3, 1000

    def gmm(data):
        mix_proportions = numpyro.sample("phi", dist.Dirichlet(jnp.ones(K)))
        with numpyro.plate("num_clusters", K, dim=-1):
            cluster_means = numpyro.sample("cluster_means", dist.Normal(jnp.arange(K), 1.))
        with numpyro.plate("data", data.shape[0], dim=-1):
            assignments = numpyro.sample("assignments", dist.Categorical(mix_proportions))
            numpyro.sample("obs", dist.Normal(cluster_means[assignments], 1.), obs=data)

    true_cluster_means = jnp.array([1., 5., 10.])
    true_mix_proportions = jnp.array([0.1, 0.3, 0.6])
    cluster_assignments = dist.Categorical(true_mix_proportions).sample(random.PRNGKey(0), (N,))
    data = dist.Normal(true_cluster_means[cluster_assignments], 1.0).sample(random.PRNGKey(1))

    nuts_kernel = NUTS(gmm)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=500)
    mcmc.run(random.PRNGKey(2), data)
    samples = mcmc.get_samples()
    assert_allclose(samples["phi"].mean(0).sort(), true_mix_proportions, atol=0.05)
    assert_allclose(samples["cluster_means"].mean(0).sort(), true_cluster_means, atol=0.2)


def test_bernoulli_latent_model():
    def model(data):
        y_prob = numpyro.sample("y_prob", dist.Beta(1., 1.))
        with numpyro.plate("data", data.shape[0]):
            y = numpyro.sample("y", dist.Bernoulli(y_prob))
            z = numpyro.sample("z", dist.Bernoulli(0.65 * y + 0.1))
            numpyro.sample("obs", dist.Normal(2. * z, 1.), obs=data)

    N = 2000
    y_prob = 0.3
    y = dist.Bernoulli(y_prob).sample(random.PRNGKey(0), (N,))
    z = dist.Bernoulli(0.65 * y + 0.1).sample(random.PRNGKey(1))
    data = dist.Normal(2. * z, 1.0).sample(random.PRNGKey(2))

    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=500)
    mcmc.run(random.PRNGKey(3), data)
    samples = mcmc.get_samples()
    assert_allclose(samples["y_prob"].mean(0), y_prob, atol=0.05)


def test_change_point():
    def model(count_data):
        n_count_data = count_data.shape[0]
        alpha = 1 / jnp.mean(count_data)
        lambda_1 = numpyro.sample('lambda_1', dist.Exponential(alpha))
        lambda_2 = numpyro.sample('lambda_2', dist.Exponential(alpha))
        # this is the same as DiscreteUniform(0, 69)
        tau = numpyro.sample('tau', dist.Categorical(logits=jnp.zeros(70)))
        idx = jnp.arange(n_count_data)
        lambda_ = jnp.where(tau > idx, lambda_1, lambda_2)
        with numpyro.plate("data", n_count_data):
            numpyro.sample('obs', dist.Poisson(lambda_), obs=count_data)

    count_data = jnp.array([
        13, 24, 8, 24,  7, 35, 14, 11, 15, 11, 22, 22, 11, 57, 11,
        19, 29, 6, 19, 12, 22, 12, 18, 72, 32,  9,  7, 13, 19, 23,
        27, 20, 6, 17, 13, 10, 14,  6, 16, 15,  7,  2, 15, 15, 19,
        70, 49, 7, 53, 22, 21, 31, 19, 11,  1, 20, 12, 35, 17, 23,
        17,  4, 2, 31, 30, 13, 27,  0, 39, 37,  5, 14, 13, 22,
    ])

    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=500)
    mcmc.run(random.PRNGKey(0), count_data)
    samples = mcmc.get_samples()
    assert_allclose(samples["lambda_1"].mean(0), 18., atol=1.)
    assert_allclose(samples["lambda_2"].mean(0), 22.5, atol=1.5)


def test_gaussian_hmm():
    dim = 4
    num_steps = 10

    def model(data):
        with numpyro.plate("states", dim):
            transition = numpyro.sample("transition", dist.Dirichlet(jnp.ones(dim)))
            emission_loc = numpyro.sample("emission_loc", dist.Normal(0, 1))
            emission_scale = numpyro.sample("emission_scale", dist.LogNormal(0, 1))

        trans_prob = numpyro.sample("initialize", dist.Dirichlet(jnp.ones(dim)))
        for t, y in markov(enumerate(data)):
            x = numpyro.sample("x_{}".format(t), dist.Categorical(trans_prob))
            numpyro.sample("y_{}".format(t), dist.Normal(emission_loc[x], emission_scale[x]), obs=y)
            trans_prob = transition[x]

    def _generate_data():
        transition_probs = np.random.rand(dim, dim)
        transition_probs = transition_probs / transition_probs.sum(-1, keepdims=True)
        emissions_loc = np.arange(dim)
        emissions_scale = 1.
        state = np.random.choice(3)
        obs = [np.random.normal(emissions_loc[state], emissions_scale)]
        for _ in range(num_steps - 1):
            state = np.random.choice(dim, p=transition_probs[state])
            obs.append(np.random.normal(emissions_loc[state], emissions_scale))
        return np.stack(obs)

    data = _generate_data()
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=500)
    mcmc.run(random.PRNGKey(0), data)


def test_iteration():

    def testing():
        for i in markov(range(5)):
            v1 = to_data(Tensor(jnp.ones(2), OrderedDict([(str(i), bint(2))]), 'real'))
            v2 = to_data(Tensor(jnp.zeros(2), OrderedDict([('a', bint(2))]), 'real'))
            fv1 = to_funsor(v1, reals())
            fv2 = to_funsor(v2, reals())
            print(i, v1.shape)  # shapes should alternate
            if i % 2 == 0:
                assert v1.shape == (2,)
            else:
                assert v1.shape == (2, 1, 1)
            assert v2.shape == (2, 1)
            print(i, fv1.inputs)
            print('a', v2.shape)  # shapes should stay the same
            print('a', fv2.inputs)

    with NamedMessenger():
        testing()


def test_nesting():

    def testing():

        with markov():
            v1 = to_data(Tensor(jnp.ones(2), OrderedDict([("1", bint(2))]), 'real'))
            print(1, v1.shape)  # shapes should alternate
            assert v1.shape == (2,)

            with markov():
                v2 = to_data(Tensor(jnp.ones(2), OrderedDict([("2", bint(2))]), 'real'))
                print(2, v2.shape)  # shapes should alternate
                assert v2.shape == (2, 1)

                with markov():
                    v3 = to_data(Tensor(jnp.ones(2), OrderedDict([("3", bint(2))]), 'real'))
                    print(3, v3.shape)  # shapes should alternate
                    assert v3.shape == (2,)

                    with markov():
                        v4 = to_data(Tensor(jnp.ones(2), OrderedDict([("4", bint(2))]), 'real'))
                        print(4, v4.shape)  # shapes should alternate

                        assert v4.shape == (2, 1)

    with NamedMessenger():
        testing()


def test_staggered():

    def testing():
        for i in markov(range(12)):
            if i % 4 == 0:
                v2 = to_data(Tensor(jnp.zeros(2), OrderedDict([('a', bint(2))]), 'real'))
                fv2 = to_funsor(v2, reals())
                assert v2.shape == (2,)
                print('a', v2.shape)
                print('a', fv2.inputs)

    with NamedMessenger():
        testing()


@pytest.mark.parametrize('num_steps', [1, 10, 11])
def test_scan_enum_one_latent(num_steps):
    data = random.normal(random.PRNGKey(0), (num_steps,))
    init_probs = jnp.array([0.6, 0.4])
    transition_probs = jnp.array([[0.8, 0.2], [0.1, 0.9]])
    locs = jnp.array([-1.0, 1.0])

    def model(data):
        x = None
        for i, y in markov(enumerate(data)):
            probs = init_probs if x is None else transition_probs[x]
            x = numpyro.sample(f"x_{i}", dist.Categorical(probs))
            numpyro.sample(f"y_{i}", dist.Normal(locs[x], 1), obs=y)
        return x

    def fun_model(data):
        def transition_fn(x, y):
            probs = init_probs if x is None else transition_probs[x]
            x = numpyro.sample("x", dist.Categorical(probs))
            numpyro.sample("y", dist.Normal(locs[x], 1), obs=y)
            return x, None

        x, collections = scan(transition_fn, None, data)
        assert collections is None
        return x

    actual_log_joint = log_density(enum(config_enumerate(fun_model)), (data,), {}, {})[0]
    expected_log_joint = log_density(enum(config_enumerate(model)), (data,), {}, {})[0]
    assert_allclose(actual_log_joint, expected_log_joint)

    actual_last_x = enum(config_enumerate(fun_model))(data)
    expected_last_x = enum(config_enumerate(model))(data)
    assert_allclose(actual_last_x, expected_last_x)


def test_scan_enum_plate():
    N, D = 10, 3
    data = random.normal(random.PRNGKey(0), (N, D))
    init_probs = jnp.array([0.6, 0.4])
    transition_probs = jnp.array([[0.8, 0.2], [0.1, 0.9]])
    locs = jnp.array([-1.0, 1.0])

    def model(data):
        x = None
        D_plate = numpyro.plate("D", D, dim=-1)
        for i, y in markov(enumerate(data)):
            with D_plate:
                probs = init_probs if x is None else transition_probs[x]
                x = numpyro.sample(f"x_{i}", dist.Categorical(probs))
                numpyro.sample(f"y_{i}", dist.Normal(locs[x], 1), obs=y)

    def fun_model(data):
        def transition_fn(x, y):
            probs = init_probs if x is None else transition_probs[x]
            with numpyro.plate("D", D, dim=-1):
                x = numpyro.sample("x", dist.Categorical(probs))
                numpyro.sample("y", dist.Normal(locs[x], 1), obs=y)
            return x, None

        scan(transition_fn, None, data)

    actual_log_joint = log_density(enum(config_enumerate(fun_model), -2), (data,), {}, {})[0]
    expected_log_joint = log_density(enum(config_enumerate(model), -2), (data,), {}, {})[0]
    assert_allclose(actual_log_joint, expected_log_joint)


def test_scan_enum_separated_plates_same_dim():
    N, D1, D2 = 10, 3, 4
    data = random.normal(random.PRNGKey(0), (N, D1 + D2))
    data1, data2 = data[:, :D1], data[:, D1:]
    init_probs = jnp.array([0.6, 0.4])
    transition_probs = jnp.array([[0.8, 0.2], [0.1, 0.9]])
    locs = jnp.array([-1.0, 1.0])

    def model(data1, data2):
        x = None
        D1_plate = numpyro.plate("D1", D1, dim=-1)
        D2_plate = numpyro.plate("D2", D2, dim=-1)
        for i, (y1, y2) in markov(enumerate(zip(data1, data2))):
            probs = init_probs if x is None else transition_probs[x]
            x = numpyro.sample(f"x_{i}", dist.Categorical(probs))
            with D1_plate:
                numpyro.sample(f"y1_{i}", dist.Normal(locs[x], 1), obs=y1)
            with D2_plate:
                numpyro.sample(f"y2_{i}", dist.Normal(locs[x], 1), obs=y2)

    def fun_model(data1, data2):
        def transition_fn(x, y):
            y1, y2 = y
            probs = init_probs if x is None else transition_probs[x]
            x = numpyro.sample("x", dist.Categorical(probs))
            with numpyro.plate("D1", D1, dim=-1):
                numpyro.sample("y1", dist.Normal(locs[x], 1), obs=y1)
            with numpyro.plate("D2", D2, dim=-1):
                numpyro.sample("y2", dist.Normal(locs[x], 1), obs=y2)
            return x, None

        scan(transition_fn, None, (data1, data2))

    actual_log_joint = log_density(enum(config_enumerate(fun_model), -2), (data1, data2), {}, {})[0]
    expected_log_joint = log_density(enum(config_enumerate(model), -2), (data1, data2), {}, {})[0]
    assert_allclose(actual_log_joint, expected_log_joint)


def test_scan_enum_separated_plate_discrete():
    N, D = 10, 3
    data = random.normal(random.PRNGKey(0), (N, D))
    transition_probs = jnp.array([[0.8, 0.2], [0.1, 0.9]])
    locs = jnp.array([[-1.0, 1.0], [2.0, 3.0]])

    def model(data):
        x = 0
        D_plate = numpyro.plate("D", D, dim=-1)
        for i, y in markov(enumerate(data)):
            probs = transition_probs[x]
            x = numpyro.sample(f"x_{i}", dist.Categorical(probs))
            with D_plate:
                w = numpyro.sample(f"w_{i}", dist.Bernoulli(0.6))
                numpyro.sample(f"y_{i}", dist.Normal(Vindex(locs)[x, w], 1), obs=y)

    def fun_model(data):
        def transition_fn(x, y):
            probs = transition_probs[x]
            x = numpyro.sample("x", dist.Categorical(probs))
            with numpyro.plate("D", D, dim=-1):
                w = numpyro.sample("w", dist.Bernoulli(0.6))
                numpyro.sample("y", dist.Normal(Vindex(locs)[x, w], 1), obs=y)
            return x, None

        scan(transition_fn, 0, data)

    actual_log_joint = log_density(enum(config_enumerate(fun_model), -2), (data,), {}, {})[0]
    expected_log_joint = log_density(enum(config_enumerate(model), -2), (data,), {}, {})[0]
    assert_allclose(actual_log_joint, expected_log_joint)


def test_scan_enum_discrete_outside():
    data = random.normal(random.PRNGKey(0), (10,))
    probs = jnp.array([[[0.8, 0.2], [0.1, 0.9]],
                       [[0.7, 0.3], [0.6, 0.4]]])
    locs = jnp.array([-1.0, 1.0])

    def model(data):
        w = numpyro.sample("w", dist.Bernoulli(0.6))
        x = 0
        for i, y in markov(enumerate(data)):
            x = numpyro.sample(f"x_{i}", dist.Categorical(probs[w, x]))
            numpyro.sample(f"y_{i}", dist.Normal(locs[x], 1), obs=y)

    def fun_model(data):
        w = numpyro.sample("w", dist.Bernoulli(0.6))

        def transition_fn(x, y):
            x = numpyro.sample("x", dist.Categorical(probs[w, x]))
            numpyro.sample("y", dist.Normal(locs[x], 1), obs=y)
            return x, None

        scan(transition_fn, 0, data)

    actual_log_joint = log_density(enum(config_enumerate(fun_model)), (data,), {}, {})[0]
    expected_log_joint = log_density(enum(config_enumerate(model)), (data,), {}, {})[0]
    assert_allclose(actual_log_joint, expected_log_joint)


def test_scan_enum_two_latents():
    num_steps = 11
    data = random.normal(random.PRNGKey(0), (num_steps,))
    probs_x = jnp.array([[0.8, 0.2], [0.1, 0.9]])
    probs_w = jnp.array([[0.7, 0.3], [0.6, 0.4]])
    locs = jnp.array([[-1.0, 1.0], [2.0, 3.0]])

    def model(data):
        x = w = 0
        for i, y in markov(enumerate(data)):
            x = numpyro.sample(f"x_{i}", dist.Categorical(probs_x[x]))
            w = numpyro.sample(f"w_{i}", dist.Categorical(probs_w[w]))
            numpyro.sample(f"y_{i}", dist.Normal(locs[w, x], 1), obs=y)

    def fun_model(data):
        def transition_fn(carry, y):
            x, w = carry
            x = numpyro.sample("x", dist.Categorical(probs_x[x]))
            w = numpyro.sample("w", dist.Categorical(probs_w[w]))
            numpyro.sample("y", dist.Normal(locs[w, x], 1), obs=y)
            # also test if scan's `ys` are recorded corrected
            return (x, w), x

        scan(transition_fn, (0, 0), data)

    actual_log_joint = log_density(enum(config_enumerate(fun_model)), (data,), {}, {})[0]
    expected_log_joint = log_density(enum(config_enumerate(model)), (data,), {}, {})[0]
    assert_allclose(actual_log_joint, expected_log_joint)


def test_scan_enum_scan_enum():
    num_steps = 11
    data_x = random.normal(random.PRNGKey(0), (num_steps,))
    data_w = data_x[:-1] + 1
    probs_x = jnp.array([[0.8, 0.2], [0.1, 0.9]])
    probs_w = jnp.array([[0.7, 0.3], [0.6, 0.4]])
    locs_x = jnp.array([-1.0, 1.0])
    locs_w = jnp.array([2.0, 3.0])

    def model(data_x, data_w):
        x = w = 0
        for i, y in markov(enumerate(data_x)):
            x = numpyro.sample(f"x_{i}", dist.Categorical(probs_x[x]))
            numpyro.sample(f"y_x_{i}", dist.Normal(locs_x[x], 1), obs=y)

        for i, y in markov(enumerate(data_w)):
            w = numpyro.sample(f"w{i}", dist.Categorical(probs_w[w]))
            numpyro.sample(f"y_w_{i}", dist.Normal(locs_w[w], 1), obs=y)

    def fun_model(data_x, data_w):
        def transition_fn(name, probs, locs, x, y):
            x = numpyro.sample(name, dist.Categorical(probs[x]))
            numpyro.sample("y_" + name, dist.Normal(locs[x], 1), obs=y)
            return x, None

        scan(partial(transition_fn, "x", probs_x, locs_x), 0, data_x)
        scan(partial(transition_fn, "w", probs_w, locs_w), 0, data_w)

    actual_log_joint = log_density(enum(config_enumerate(fun_model)), (data_x, data_w), {}, {})[0]
    expected_log_joint = log_density(enum(config_enumerate(model)), (data_x, data_w), {}, {})[0]
    assert_allclose(actual_log_joint, expected_log_joint)
