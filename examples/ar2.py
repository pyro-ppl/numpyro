# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: AR2 process
====================

In this example we show how to use ``jax.lax.scan``
to avoid writing a (slow) Python for-loop.

.. image:: ../_static/img/examples/ar2.png
    :align: center
"""

import argparse
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import jax
from jax import random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

matplotlib.use("Agg")  # noqa: E402


def ar2(y, unroll_loop=False):
    alpha_1 = numpyro.sample(
        "alpha_1",
        dist.TruncatedDistribution(dist.Normal(0, 1), low=-1, high=1),
    )
    alpha_2 = numpyro.sample(
        "alpha_2",
        dist.TruncatedDistribution(dist.Normal(0, 1), low=-1, high=1),
    )
    const = numpyro.sample(
        "const",
        dist.Normal(0, 1),
    )
    sigma = numpyro.sample(
        "sigma",
        dist.Normal(0, 1),
    )

    if unroll_loop:
        preds = []
        for i in range(2, len(y)):
            preds.append(const + alpha_1 * y[i - 1] + alpha_2 * y[i - 2])
        preds = jnp.asarray(preds)
    else:

        def transition_fn(carry, y):
            y_1, y_2 = carry
            pred = alpha_1 * y_1 + alpha_2 * y_2
            return (y, y_1), pred

        _, preds = jax.lax.scan(transition_fn, (y[1], y[0]), y[2:])

    mu = numpyro.deterministic("mu", preds)
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y[2:])


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
    mcmc.run(
        rng_key,
        y=y,
    )
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()


def main(args):

    # generate artifical dataset
    num_data = 142
    t = np.arange(0, num_data)
    y = np.sin(t) + np.random.randn(num_data) * 0.1

    # do inference
    rng_key, _ = random.split(random.PRNGKey(0))
    samples = run_inference(ar2, args, rng_key, y)

    # do prediction
    mean_prediction = samples["mu"].mean(axis=0)

    # make plots
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    # plot training data
    ax.plot(t, y, color="blue", label="True labels")
    # plot mean prediction
    # note that we can't make predictions for the first two points,
    # because they don't have lagged values to use prediction.
    ax.plot(t[2:], mean_prediction, color="orange", label="Mean predictions")
    ax.set(xlabel="time", ylabel="y", title="AR2 process")
    ax.legend()

    plt.savefig("ar2_plot.pdf")
    fig.savefig("docs/source/_static/img/examples/ar2.png")


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.8.0")
    parser = argparse.ArgumentParser(description="AR2 example")
    parser.add_argument("-n", "--num-samples", nargs="?", default=1000, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=1000, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
