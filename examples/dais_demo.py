# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: AutoDAIS
=================

AutoDAIS constructs a guide that combines elements of Hamiltonian Monte Carlo,
Annealed Importance Sampling, and Variational Inference.

In this demo script we construct a somewhat artificial example involving a gaussian
process binary classifier. We aim to demonstrate that:

- DAIS can achieve better ELBOs than e.g. mean field variational inference
- DAIS can achieve better posterior approximations than e.g. mean field variational inference
- DAIS improves as you increase K, the number of HMC steps used in the sampler

References:

[1] "MCMC Variational Inference via Uncorrected Hamiltonian Annealing,"
    Tomas Geffner, Justin Domke.
[2] "Differentiable Annealed Importance Sampling and the Perils of Gradient Noise,"
    Guodong Zhang, Kyle Hsu, Jianing Li, Chelsea Finn, Roger Grosse.

.. image:: ../_static/img/examples/dais_demo.png
    :align: center
"""

import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
import seaborn as sns

from jax import random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, autoguide
from numpyro.util import enable_x64

matplotlib.use("Agg")  # noqa: E402


# squared exponential kernel
def kernel(X, Z, length, jitter=1.0e-6):
    deltaXsq = jnp.power((X[:, None] - Z) / length, 2.0)
    k = jnp.exp(-0.5 * deltaXsq) + jitter * jnp.eye(X.shape[0])
    return k


def model(X, Y, length=0.2):
    # compute kernel
    k = kernel(X, X, length)

    # sample from gaussian process prior
    f = numpyro.sample(
        "f",
        dist.MultivariateNormal(loc=jnp.zeros(X.shape[0]), covariance_matrix=k),
    )
    # we use a non-standard link function to induce extra non-gaussianity
    numpyro.sample("obs", dist.Bernoulli(logits=jnp.power(f, 3.0)), obs=Y)


# create artificial binary classification dataset
def get_data(N=16):
    np.random.seed(0)
    X = np.linspace(-1, 1, N)
    Y = X + 0.2 * np.power(X, 3.0) + 0.5 * np.power(0.5 + X, 2.0) * np.sin(4.0 * X)
    Y -= np.mean(Y)
    Y /= np.std(Y)
    Y = np.random.binomial(1, expit(Y))

    assert X.shape == (N,)
    assert Y.shape == (N,)

    return X, Y


# helper function for running SVI with a particular autoguide
def run_svi(
    rng_key, X, Y, guide_family="AutoDiagonalNormal", K=8
):
    assert guide_family in ["AutoDiagonalNormal", "AutoDAIS"]

    if guide_family == "AutoDAIS":
        guide = autoguide.AutoDAIS(model, K=K, eta_init=0.02, eta_max=0.5)
        step_size = 5e-4
    elif guide_family == "AutoDiagonalNormal":
        guide = autoguide.AutoDiagonalNormal(model)
        step_size = 3e-3

    optimizer = numpyro.optim.Adam(step_size=step_size)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    svi_result = svi.run(rng_key, args.num_svi_steps, X, Y)
    params = svi_result.params

    final_elbo = -Trace_ELBO(num_particles=1000).loss(
        rng_key, params, model, guide, X, Y
    )

    guide_name = guide_family
    if guide_family == "AutoDAIS":
        guide_name += "-{}".format(K)

    print("[{}] final elbo: {:.2f}".format(guide_name, final_elbo))

    return guide.sample_posterior(
        random.PRNGKey(1), params, sample_shape=(args.num_samples,)
    )


# helper function for running mcmc
def run_nuts(mcmc_key, args, X, Y):
    mcmc = MCMC(NUTS(model), num_warmup=args.num_warmup, num_samples=args.num_samples)
    mcmc.run(mcmc_key, X, Y)
    mcmc.print_summary()
    return mcmc.get_samples()


def main(args):
    X, Y = get_data()

    rng_keys = random.split(random.PRNGKey(0), 4)

    # run SVI with an AutoDAIS guide for two values of K
    dais8_samples = run_svi(rng_keys[1], X, Y, guide_family="AutoDAIS", K=8)
    dais128_samples = run_svi(rng_keys[2], X, Y, guide_family="AutoDAIS", K=128)

    # run SVI with an AutoDiagonalNormal guide
    meanfield_samples = run_svi(rng_keys[3], X, Y, guide_family="AutoDiagonalNormal")

    # run MCMC inference
    nuts_samples = run_nuts(rng_keys[0], args, X, Y)

    # make 2d density plots of the (f_0, f_1) marginal posterior
    if args.num_samples >= 1000:
        sns.set_style("white")

        coord1, coord2 = 0, 1

        fig, axes = plt.subplots(
            2, 2, sharex=True, figsize=(6, 6), constrained_layout=True
        )

        xlim = (-3, 3)
        ylim = (-3, 3)

        def add_fig(samples, title, ax):
            sns.kdeplot(
                x=samples["f"][:, coord1],
                y=samples["f"][:, coord2],
                ax=ax
            )
            ax.set(title=title, xlim=xlim, ylim=ylim)

        add_fig(dais8_samples, "AutoDAIS (K=8)", axes[0][0])
        add_fig(dais128_samples, "AutoDAIS (K=128)", axes[0][1])
        add_fig(meanfield_samples, "AutoDiagonalNormal", axes[1][0])
        add_fig(nuts_samples, "NUTS", axes[1][1])

        plt.savefig("dais_demo.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Usage example for AutoDAIS guide.")
    parser.add_argument("--num-svi-steps", type=int, default=80 * 1000)
    parser.add_argument("--num-warmup", type=int, default=2000)
    parser.add_argument("--num-samples", type=int, default=10 * 1000)
    parser.add_argument("--device", default="cpu", type=str, choices=["cpu", "gpu"])

    args = parser.parse_args()

    enable_x64()
    numpyro.set_platform(args.device)

    main(args)
