# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

r"""
Example: Toy Mixture Model with Discrete Enumeration
====================================================

A toy mixture model to provide a simple example for implementing discrete enumeration::

    (A) -> [B] -> (C)

``A`` is an observed Bernoulli variable with Beta prior. ``B`` is a hidden variable which
is a mixture of two Bernoulli distributions (with Beta priors), chosen by ``A`` being true or false.
``C`` is observed, and like ``B``, is a mixture of two Bernoulli distributions (with Beta priors),
chosen by ``B`` being true or false. There is a plate over the three variables for ``num_obs``
independent observations of data.

Because ``B`` is hidden and discrete we wish to marginalize it out of the model. This is done by:

1. marking the model with ``@config_enumerate``
2. marking the ``B`` sample site in the model with ``infer={"enumerate": "parallel"}``
3. passing ``SVI`` the ``TraceEnum_ELBO`` loss function
"""

import argparse

import matplotlib.pyplot as plt

from jax import random
import jax.numpy as jnp
import optax

import numpyro
from numpyro import handlers
from numpyro.contrib.funsor import config_enumerate
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import SVI, TraceEnum_ELBO
from numpyro.ops.indexing import Vindex


def main(args):
    num_obs = args.num_obs
    num_steps = args.num_steps
    prior, CPDs, data = handlers.seed(generate_data, random.PRNGKey(0))(num_obs)
    posterior_params = train(prior, data, num_steps, num_obs)
    evaluate(CPDs, posterior_params)


def generate_data(num_obs):
    # domain = [False, True]
    prior = {
        "A": jnp.array([1.0, 10.0]),
        "B": jnp.array([[10.0, 1.0], [1.0, 10.0]]),
        "C": jnp.array([[10.0, 1.0], [1.0, 10.0]]),
    }
    CPDs = {
        "p_A": numpyro.sample("p_A", dist.Beta(prior["A"][0], prior["A"][1])),
        "p_B": numpyro.sample("p_B", dist.Beta(prior["B"][:, 0], prior["B"][:, 1])),
        "p_C": numpyro.sample("p_C", dist.Beta(prior["C"][:, 0], prior["C"][:, 1])),
    }
    data = {"A": numpyro.sample("A", dist.Bernoulli(jnp.ones(num_obs) * CPDs["p_A"]))}
    data["B"] = numpyro.sample("B", dist.Bernoulli(CPDs["p_B"][data["A"]]))
    data["C"] = numpyro.sample("C", dist.Bernoulli(CPDs["p_C"][data["B"]]))
    return prior, CPDs, data


@config_enumerate
def model(prior, obs, num_obs):
    p_A = numpyro.sample("p_A", dist.Beta(1, 1))
    p_B = numpyro.sample("p_B", dist.Beta(jnp.ones(2), jnp.ones(2)).to_event(1))
    p_C = numpyro.sample("p_C", dist.Beta(jnp.ones(2), jnp.ones(2)).to_event(1))
    with numpyro.plate("data_plate", num_obs):
        A = numpyro.sample("A", dist.Bernoulli(p_A), obs=obs["A"])
        # Vindex used to ensure proper indexing into the enumerated sample sites
        B = numpyro.sample(
            "B",
            dist.Bernoulli(Vindex(p_B)[A]),
            infer={"enumerate": "parallel"},
        )
        numpyro.sample("C", dist.Bernoulli(Vindex(p_C)[B]), obs=obs["C"])


def guide(prior, obs, num_obs):
    a = numpyro.param("a", prior["A"], constraint=constraints.positive)
    numpyro.sample("p_A", dist.Beta(a[0], a[1]))
    b = numpyro.param("b", prior["B"], constraint=constraints.positive)
    numpyro.sample("p_B", dist.Beta(b[:, 0], b[:, 1]).to_event(1))
    c = numpyro.param("c", prior["C"], constraint=constraints.positive)
    numpyro.sample("p_C", dist.Beta(c[:, 0], c[:, 1]).to_event(1))


def train(prior, data, num_steps, num_obs):
    elbo = TraceEnum_ELBO()
    svi = SVI(model, guide, optax.adam(learning_rate=0.01), loss=elbo)
    svi_result = svi.run(random.PRNGKey(0), num_steps, prior, data, num_obs)
    plt.figure()
    plt.plot(svi_result.losses)
    plt.show()
    posterior_params = svi_result.params.copy()
    posterior_params["a"] = posterior_params["a"][
        None, :
    ]  # reshape to same as other variables
    return posterior_params


def evaluate(CPDs, posterior_params):
    true_p_A, pred_p_A = get_true_pred_CPDs(CPDs["p_A"], posterior_params["a"])
    true_p_B, pred_p_B = get_true_pred_CPDs(CPDs["p_B"], posterior_params["b"])
    true_p_C, pred_p_C = get_true_pred_CPDs(CPDs["p_C"], posterior_params["c"])
    print("\np_A = True")
    print("actual:   ", true_p_A)
    print("predicted:", pred_p_A)
    print("\np_B = True | A = False/True")
    print("actual:   ", true_p_B)
    print("predicted:", pred_p_B)
    print("\np_C = True | B = False/True")
    print("actual:   ", true_p_C)
    print("predicted:", pred_p_C)


def get_true_pred_CPDs(CPD, posterior_param):
    true_p = CPD
    pred_p = posterior_param[:, 0] / jnp.sum(posterior_param, axis=1)
    return true_p, pred_p


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.17.0")
    parser = argparse.ArgumentParser(description="Toy mixture model")
    parser.add_argument("-n", "--num-steps", default=4000, type=int)
    parser.add_argument("-o", "--num-obs", default=10000, type=int)
    args = parser.parse_args()
    main(args)
