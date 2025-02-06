# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Zero-Inflated Poisson regression model
================================================

In this example, we model and predict how many fish are caught by visitors to a state park.
Many groups of visitors catch zero fish, either because they did not fish at all or because
they were unlucky. We would like to explicitly model this bimodal behavior (zero versus non-zero)
and ascertain which variables contribute to each behavior.

We answer this question by fitting a zero-inflated poisson regression model. We use MAP,
VI and MCMC as estimation methods. Finally, from the MCMC samples, we identify the variables that
contribute to the zero and non-zero components of the zero-inflated poisson likelihood.
"""

import argparse
import os
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import jax.numpy as jnp
from jax.random import PRNGKey
import jax.scipy as jsp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO, autoguide

matplotlib.use("Agg")  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def model(X, Y):
    D_X = X.shape[1]
    b1 = numpyro.sample("b1", dist.Normal(0.0, 1.0).expand([D_X]).to_event(1))
    b2 = numpyro.sample("b2", dist.Normal(0.0, 1.0).expand([D_X]).to_event(1))

    q = jsp.special.expit(jnp.dot(X, b1[:, None])).reshape(-1)
    lam = jnp.exp(jnp.dot(X, b2[:, None]).reshape(-1))

    with numpyro.plate("obs", X.shape[0]):
        numpyro.sample("Y", dist.ZeroInflatedPoisson(gate=q, rate=lam), obs=Y)


def run_mcmc(model, args, X, Y):
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(PRNGKey(1), X, Y)
    mcmc.print_summary()
    return mcmc.get_samples()


def run_svi(model, guide_family, args, X, Y):
    if guide_family == "AutoDelta":
        guide = autoguide.AutoDelta(model)
    elif guide_family == "AutoDiagonalNormal":
        guide = autoguide.AutoDiagonalNormal(model)

    optimizer = numpyro.optim.Adam(0.001)
    svi = SVI(model, guide, optimizer, Trace_ELBO())
    svi_results = svi.run(PRNGKey(1), args.maxiter, X=X, Y=Y)
    params = svi_results.params

    return params, guide


def main(args):
    set_seed(args.seed)

    # prepare dataset
    df = pd.read_stata("http://www.stata-press.com/data/r11/fish.dta")
    df["intercept"] = 1
    cols = ["livebait", "camper", "persons", "child", "intercept"]

    mask = np.random.randn(len(df)) < args.train_size
    df_train = df[mask]
    df_test = df[~mask]
    X_train = jnp.asarray(df_train[cols].values)
    y_train = jnp.asarray(df_train["count"].values)
    X_test = jnp.asarray(df_test[cols].values)
    y_test = jnp.asarray(df_test["count"].values)

    print("run MAP.")
    map_params, map_guide = run_svi(model, "AutoDelta", args, X_train, y_train)

    print("run VI.")
    vi_params, vi_guide = run_svi(model, "AutoDiagonalNormal", args, X_train, y_train)

    print("run MCMC.")
    posterior_samples = run_mcmc(model, args, X_train, y_train)

    # evaluation

    def svi_predict(model, guide, params, args, X):
        predictive = Predictive(
            model=model, guide=guide, params=params, num_samples=args.num_samples
        )
        predictions = predictive(PRNGKey(1), X=X, Y=None)
        svi_predictions = jnp.rint(predictions["Y"].mean(0))
        return svi_predictions

    map_predictions = svi_predict(model, map_guide, map_params, args, X_test)
    vi_predictions = svi_predict(model, vi_guide, vi_params, args, X_test)

    predictive = Predictive(model, posterior_samples=posterior_samples)
    predictions = predictive(PRNGKey(1), X=X_test, Y=None)
    mcmc_predictions = jnp.rint(predictions["Y"].mean(0))

    print(
        "MAP RMSE: ",
        mean_squared_error(y_test.to_py(), map_predictions.to_py(), squared=False),
    )
    print(
        "VI RMSE: ",
        mean_squared_error(y_test.to_py(), vi_predictions.to_py(), squared=False),
    )
    print(
        "MCMC RMSE: ",
        mean_squared_error(y_test.to_py(), mcmc_predictions.to_py(), squared=False),
    )

    # make plot
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), constrained_layout=True)

    def add_fig(var_name, title, ax):
        ax.set_title(title)
        ax.violinplot(
            [posterior_samples[var_name][:, i].to_py() for i in range(len(cols))]
        )
        ax.set_xticks(np.arange(1, len(cols) + 1))
        ax.set_xticklabels(cols, rotation=45, fontsize=10)

    add_fig("b1", "Coefficients for probability of catching fish", axes[0])
    add_fig("b2", "Coefficients for the number of fish caught", axes[1])

    plt.savefig("zip_fish.png")


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.17.0")
    parser = argparse.ArgumentParser("Zero-Inflated Poisson Regression")
    parser.add_argument("--seed", nargs="?", default=42, type=int)
    parser.add_argument("-n", "--num-samples", nargs="?", default=2000, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=1000, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--num-data", nargs="?", default=100, type=int)
    parser.add_argument("--maxiter", nargs="?", default=5000, type=int)
    parser.add_argument("--train-size", nargs="?", default=0.8, type=float)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
