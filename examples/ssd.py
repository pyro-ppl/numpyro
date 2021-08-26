import argparse

from functools import partial

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

from numpyro.examples.datasets import HIGGS, load_dataset


matplotlib.use("Agg")  # noqa: E402



# helper function for running SVI with a particular autoguide
def run_svi(rng_key, model, X, Y, surrogate_model=None, guide_family="AutoDiagonalNormal", K=8):
    assert guide_family in ["AutoDiagonalNormal", "AutoDAIS", "AutoSSDAIS"]

    if guide_family == "AutoDAIS":
        guide = autoguide.AutoDAIS(model, K=K, eta_init=0.02, eta_max=0.5)
        step_size = 5e-4
    elif guide_family == "AutoSSDAIS":
        guide = autoguide.AutoSSDAIS(model, surrogate_model, K=K, eta_init=0.02, eta_max=0.5)
        step_size = 5e-4
    elif guide_family == "AutoDiagonalNormal":
        guide = autoguide.AutoDiagonalNormal(model)
        step_size = 3e-3

    optimizer = numpyro.optim.Adam(step_size=step_size)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    svi_result = svi.run(rng_key, args.num_svi_steps, X, Y)
    params = svi_result.params

    print('\nparams\n', params)
    if guide_family == "AutoSSDAIS":
        print("omegas sum", params['omegas'].sum())


    final_elbo = -Trace_ELBO(num_particles=2000).loss(
        rng_key, params, model, guide, X, Y
    )

    guide_name = guide_family
    if guide_family in ["AutoDAIS", "AutoSSDAIS"]:
        guide_name += "-{}".format(K)

    print("[{}] final elbo: {:.2f}".format(guide_name, final_elbo))

    return

    return guide.sample_posterior(
        random.PRNGKey(1), params, sample_shape=(args.num_samples,)
    )


def model(X, Y):
    N, D = X.shape
    theta = numpyro.sample("theta", dist.Normal(jnp.zeros(D), jnp.ones(D)))
    with numpyro.plate("N", N):
        numpyro.sample("obs", dist.Bernoulli(logits=theta @ X.T), obs=Y)


def _surrogate_model(X, Y, omega_init):
    N, D = X.shape
    theta = numpyro.sample("theta", dist.Normal(jnp.zeros(D), jnp.ones(D)))
    omegas = numpyro.param("omegas", omega_init * jnp.ones(N), constraint=dist.constraints.positive)
    with numpyro.plate("N", N):
        with numpyro.handlers.scale(scale=omegas):
                numpyro.sample("obs", dist.Bernoulli(logits=theta @ X.T), obs=Y)


def main(args):
    _, fetch = load_dataset(
	HIGGS, shuffle=False, num_datapoints=2000
    )
    X, Y = fetch()
    print("X/Y", X.shape, Y.shape)

    num_ind = 200
    N = X.shape[0]
    perm = np.random.permutation(N)[:num_ind]

    rng_key = random.PRNGKey(0)

    surrogate_model = partial(_surrogate_model, X[perm], Y[perm], float(N) / float(num_ind))

    run_svi(rng_key, model, X, Y, surrogate_model=surrogate_model, guide_family="AutoSSDAIS", K=8)

    run_svi(rng_key, model, X, Y, guide_family="AutoDAIS", K=8)

    run_svi(rng_key, model, X, Y, guide_family="AutoDiagonalNormal")


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
