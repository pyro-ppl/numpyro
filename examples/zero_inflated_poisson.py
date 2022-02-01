# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Zero-Inflated Poisson regression model
=============================================
We evaluate Zero-Inflated Poisson regression model with MLE and MCMC in terms of their predictive
performances of the number of fish caught.
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

from jax import random
import jax.numpy as jnp
import jax.scipy as jsp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import Predictive

matplotlib.use("Agg")  # noqa: E402


def model(X, Y):
    D_X = X.shape[1]
    b1 = numpyro.sample("b1", dist.Normal(0.0, 1.0).expand([D_X]).to_event(1))
    b2 = numpyro.sample("b2", dist.Normal(0.0, 1.0).expand([D_X]).to_event(1))

    q = jsp.special.expit(jnp.dot(X, b1[:, None])).reshape(-1)
    lam = jnp.exp(jnp.dot(X, b2[:, None]).reshape(-1))

    with numpyro.plate("obs", X.shape[0]):
        numpyro.sample("Y", dist.ZeroInflatedPoisson(gate=q, rate=lam), obs=Y)


def run_inference(model, args, X, Y):
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(random.PRNGKey(1), X, Y)
    mcmc.print_summary()
    return mcmc.get_samples()


def fit_statsmodel_zip(args, X, Y):
    zip_model = sm.ZeroInflatedPoisson(endog=Y, exog=X, exog_infl=X, inflation="logit")
    zip_results = zip_model.fit(maxiter=args.maxiter)
    print(zip_results.summary())
    return zip_results


def main(args):

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

    # run inference
    posterior_samples = run_inference(model, args, X_train, y_train)

    # fit zip model
    zip_results = fit_statsmodel_zip(args, X_train.to_py(), y_train.to_py())

    # evaluation
    predictive = Predictive(model, posterior_samples=posterior_samples)
    predictions = predictive(random.PRNGKey(1), X=X_test, Y=None)
    mcmc_predictions = jnp.rint(predictions["Y"].mean(0))

    sm_predictions = np.rint(
        zip_results.predict(X_test.to_py(), exog_infl=X_test.to_py())
    )
    print(
        "MCMC RMSE: ",
        mean_squared_error(y_test.to_py(), mcmc_predictions.to_py(), squared=False),
    )
    print(
        "Statsmodels RMSE: ",
        mean_squared_error(y_test.to_py(), sm_predictions, squared=False),
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

    add_fig("b1", "coeffs for probability of catching fishes", axes[0])
    add_fig("b2", "coeffs for the number of fishes caught", axes[1])

    plt.savefig("zip_fish.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Zero-Inflated Poisson Regression")
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
