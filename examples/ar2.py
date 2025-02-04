# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

r"""
Example: AR2 process
====================

In this example we show how to use ``jax.lax.scan``
to avoid writing a (slow) Python for-loop. In this toy
example, with ``--num-data=1000``, the improvement is
of almost almost 3x.

To demonstrate, we will be implementing an AR2 process.
The idea is that we have some times series

.. math::

    y_0, y_1, ..., y_T

and we seek parameters :math:`c`, :math:`\alpha_1`, and :math:`\alpha_2`
such that for each :math:`t` between :math:`2` and :math:`T`, we have

.. math::

    y_t = c + \alpha_1 y_{t-1} + \alpha_2 y_{t-2} + \epsilon_t

where :math:`\epsilon_t` is an error term.

.. image:: ../_static/img/examples/ar2.png
    :align: center
"""

import argparse
import os
import time

import matplotlib.pyplot as plt

import jax
from jax import random
import jax.numpy as jnp

import numpyro
from numpyro.contrib.control_flow import scan
import numpyro.distributions as dist


def ar2_scan(y):
    alpha_1 = numpyro.sample("alpha_1", dist.Normal(0, 1))
    alpha_2 = numpyro.sample("alpha_2", dist.Normal(0, 1))
    const = numpyro.sample("const", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1))

    def transition(carry, _):
        y_prev, y_prev_prev = carry
        m_t = const + alpha_1 * y_prev + alpha_2 * y_prev_prev
        y_t = numpyro.sample("y", dist.Normal(m_t, sigma))
        carry = (y_t, y_prev)
        return carry, m_t

    timesteps = jnp.arange(y.shape[0] - 2)
    init = (y[1], y[0])

    with numpyro.handlers.condition(data={"y": y[2:]}):
        _, mu = scan(transition, init, timesteps)

    numpyro.deterministic("mu", mu)


def ar2_for_loop(y):
    alpha_1 = numpyro.sample("alpha_1", dist.Normal(0, 1))
    alpha_2 = numpyro.sample("alpha_2", dist.Normal(0, 1))
    const = numpyro.sample("const", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1))

    y_prev = y[1]
    y_prev_prev = y[0]
    mu = []
    for i in range(2, len(y)):
        m_t = const + alpha_1 * y_prev + alpha_2 * y_prev_prev
        mu.append(m_t)
        y_t = numpyro.sample("y_{}".format(i), dist.Normal(m_t, sigma), obs=y[i])
        y_prev_prev = y_prev
        y_prev = y_t

    numpyro.deterministic("mu", jnp.asarray(mu))


def run_inference(model, args, rng_key, y):
    start = time.time()
    sampler = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(
        sampler,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(rng_key, y=y)
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()


def main(args):
    # generate artificial dataset
    num_data = args.num_data
    rng_key = jax.random.PRNGKey(0)
    t = jnp.arange(0, num_data)
    y = jnp.sin(t) + random.normal(rng_key, (num_data,)) * 0.1

    # do inference
    if args.unroll_loop:
        # slower
        model = ar2_for_loop
    else:
        # faster
        model = ar2_scan

    samples = run_inference(model, args, rng_key, y)

    # do prediction
    mean_prediction = samples["mu"].mean(axis=0)

    # make plots
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    # plot training data
    ax.plot(t, y, color="blue", label="True values")
    # plot mean prediction
    # note that we can't make predictions for the first two points,
    # because they don't have lagged values to use for prediction.
    ax.plot(t[2:], mean_prediction, color="orange", label="Mean predictions")
    ax.set(xlabel="time", ylabel="y", title="AR2 process")
    ax.legend()

    plt.savefig("ar2.png")


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.17.0")
    parser = argparse.ArgumentParser(description="AR2 example")
    parser.add_argument("--num-data", nargs="?", default=142, type=int)
    parser.add_argument("-n", "--num-samples", nargs="?", default=1000, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=1000, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    parser.add_argument(
        "--unroll-loop",
        action="store_true",
        help="whether to unroll for-loop (note: slower)",
    )
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
