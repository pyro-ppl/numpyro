# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Modelling mortality over space and time
================================================

This example is adapted from [1]. The model in the paper estimates death rates for 6791 small
areas in England for 19 age groups (0, 1-4, 5-9, 10-14, ..., 80-84, 85+ years) from 2002-19.

When modelling mortality at high spatial resolutions, the number of deaths in each age group,
spatial unit and year is small, meaning that death rates calculated from observed data have an
apparent variability which is larger than the true differences in the risk of dying. A Bayesian
multilevel modelling framework can overcome small number issues by sharing information across ages,
space and time to obtain smoothed death rates and capture the uncertainty in the estimate.

As well as a global intercept (:math:`\\alpha_0`) and slope (:math:`\\beta_0`), the model includes
the following effects:

    - Age (:math:`\\alpha_{2a}`, :math:`\\beta_{2a}`). Each age group has a different intercept
      and slope with a random-walk structure over age groups to allow for non-linear age associations.
    - Space (:math:`\\alpha_{1s}`). Each spatial unit has an intercept.
      The spatial effects are defined by a nested hierarchy of random effects following the
      administrative hierarchy of local government. The spatial term at the lower level unit is
      centred on the spatial term of the higher level unit (e.g., :math:`\\alpha_{1s_1}`) containing
      that lower level unit.

The model also has a random walk effect over time (:math:`\\pi_{t}`).

Death rates are linked to the death and population data using a binomial likelihood. The
full generative model of death rates is written as

.. math:: :nowrap:

    \\begin{align}
        \\alpha_{1s_1} & \\sim \\text{Normal}(0,\\sigma_{\\alpha_{s_1}}^2) \\\\
        \\alpha_{1s} & \\sim \\text{Normal}(\\alpha_{1s_1(s_2)},\\sigma_{\\alpha_{s_2}}^2) \\\\
        \\alpha_{2a} & \\sim \\text{Normal}(\\alpha_{2,a-1},\\sigma_{\\alpha_a}^2) \\quad \\alpha_{2,0} = \\alpha_0 \\\\
        \\beta_{2a} & \\sim \\text{Normal}(\\beta_{2,a-1},\\sigma_{\\beta_a}^2) \\quad \\beta_{2,0} = \\beta_0 \\\\
        \\pi_{t} & \\sim \\text{Normal}(\\pi_{t-1},\\sigma_{\\pi}^2), \\quad \\pi_{0} = 0 \\\\
        \\text{logit}(m_{ast}) & = \\alpha_{1s} + \\alpha_{2a} + \\beta_{2a} t + \\pi_{t}
    \\end{align}

with the hyperpriors

.. math:: :nowrap:

    \\begin{align}
        \\alpha_0 & \\sim \\text{Normal}(0,10), \\\\
        \\beta_0 & \\sim \\text{Normal}(0,10), \\\\
        \\sigma_i & \\sim \\text{Half-Normal}(1)
    \\end{align}

Further detail about the model terms can be found in [1].

The NumPyro implementation below uses :class:`~numpyro.primitives.plate` notation to declare the batch
dimensions of the age, space and time variables. This allows us to efficiently broadcast arrays
in the likelihood.

As written above, the model includes a lot of centred random effects. The NUTS algorithm benefits
from a non-centred reparamatrisation to overcome difficult posterior geometries [2]. Rather than
manually writing out the non-centred parametrisation, we make use of the NumPyro's automatic
reparametrisation in :class:`~numpyro.infer.reparam.LocScaleReparam`.

Death data at the spatial resolution in [1] are identifiable, so in this example we are using
simulated data. Compared to [1], the simulated data have fewer spatial units and a two-tier (rather than
three-tier) spatial hierarchy. There are still 19 age groups and 18 years as in the original study.
The data here have (event) dimensions of ``(19, 113, 18)`` (age, space, time).

The original implementation in nimble is at [3].

**References**

    1. Rashid, T., Bennett, J.E. et al. (2021).
       Life expectancy and risk of death in 6791 communities in England from 2002
       to 2019: high-resolution spatiotemporal analysis of civil registration data.
       The Lancet Public Health, 6, e805 - e816.
    2. Stan User's Guide. https://mc-stan.org/docs/2_28/stan-users-guide/reparameterization.html
    3. Mortality using Bayesian hierarchical models. https://github.com/theorashid/mortality-statsmodel

"""

import argparse
import os

import numpy as np

from jax import random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.examples.datasets import MORTALITY, load_dataset
from numpyro.infer import MCMC, NUTS
from numpyro.infer.reparam import LocScaleReparam


def create_lookup(s1, s2):
    """
    Create a map between s1 indices and unique s2 indices
    """
    lookup = np.column_stack([s1, s2])
    lookup = np.unique(lookup, axis=0)
    lookup = lookup[lookup[:, 1].argsort()]
    return lookup[:, 0]


reparam_config = {
    k: LocScaleReparam(0)
    for k in [
        "alpha_s1",
        "alpha_s2",
        "alpha_age_drift",
        "beta_age_drift",
        "pi_drift",
    ]
}


@numpyro.handlers.reparam(config=reparam_config)
def model(age, space, time, lookup, population, deaths=None):
    N_s1 = len(np.unique(lookup))
    N_s2 = len(np.unique(space))
    N_age = len(np.unique(age))
    N_t = len(np.unique(time))
    N = len(population)

    # plates
    age_plate = numpyro.plate("age_groups", N_age, dim=-3)
    space_plate = numpyro.plate("space", N_s2, dim=-2)
    year_plate = numpyro.plate("year", N_t - 1, dim=-1)

    # hyperparameters
    sigma_alpha_s1 = numpyro.sample("sigma_alpha_s1", dist.HalfNormal(1.0))
    sigma_alpha_s2 = numpyro.sample("sigma_alpha_s2", dist.HalfNormal(1.0))
    sigma_alpha_age = numpyro.sample("sigma_alpha_age", dist.HalfNormal(1.0))
    sigma_beta_age = numpyro.sample("sigma_beta_age", dist.HalfNormal(1.0))
    sigma_pi = numpyro.sample("sigma_pi", dist.HalfNormal(1.0))

    # spatial hierarchy
    with numpyro.plate("s1", N_s1, dim=-2):
        alpha_s1 = numpyro.sample("alpha_s1", dist.Normal(0, sigma_alpha_s1))
    with space_plate:
        alpha_s2 = numpyro.sample(
            "alpha_s2", dist.Normal(alpha_s1[lookup], sigma_alpha_s2)
        )

    # age
    with age_plate:
        alpha_age_drift_scale = jnp.pad(
            jnp.broadcast_to(sigma_alpha_age, N_age - 1),
            (1, 0),
            constant_values=10.0,  # pad so first term is alpha0, prior N(0, 10)
        )[:, jnp.newaxis, jnp.newaxis]
        alpha_age_drift = numpyro.sample(
            "alpha_age_drift", dist.Normal(0, alpha_age_drift_scale)
        )
        alpha_age = jnp.cumsum(alpha_age_drift, -3)

        beta_age_drift_scale = jnp.pad(
            jnp.broadcast_to(sigma_beta_age, N_age - 1), (1, 0), constant_values=10.0
        )[:, jnp.newaxis, jnp.newaxis]
        beta_age_drift = numpyro.sample(
            "beta_age_drift", dist.Normal(0, beta_age_drift_scale)
        )
        beta_age = jnp.cumsum(beta_age_drift, -3)
        beta_age_cum = jnp.outer(beta_age, jnp.arange(N_t))[:, jnp.newaxis, :]

    # random walk over time
    with year_plate:
        pi_drift = numpyro.sample("pi_drift", dist.Normal(0, sigma_pi))
        pi = jnp.pad(jnp.cumsum(pi_drift, -1), (1, 0))

    # likelihood
    latent_rate = alpha_age + beta_age_cum + alpha_s2 + pi
    with numpyro.plate("N", N):
        mu_logit = latent_rate[age, space, time]
        numpyro.sample("deaths", dist.Binomial(population, logits=mu_logit), obs=deaths)


def print_model_shape(model, age, space, time, lookup, population):
    with numpyro.handlers.seed(rng_seed=1):
        trace = numpyro.handlers.trace(model).get_trace(
            age=age,
            space=space,
            time=time,
            lookup=lookup,
            population=population,
        )
    print(numpyro.util.format_shapes(trace))


def run_inference(model, age, space, time, lookup, population, deaths, rng_key, args):
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(rng_key, age, space, time, lookup, population, deaths)
    mcmc.print_summary()
    return mcmc.get_samples()


def main(args):
    print("Fetching simulated data...")
    _, fetch = load_dataset(MORTALITY, shuffle=False)
    a, s1, s2, t, deaths, population = fetch()

    lookup = create_lookup(s1, s2)

    print("Model shape:")
    print_model_shape(model, a, s2, t, lookup, population)

    print("Starting inference...")
    rng_key = random.PRNGKey(args.rng_seed)
    run_inference(model, a, s2, t, lookup, population, deaths, rng_key, args)


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.17.0")

    parser = argparse.ArgumentParser(description="Mortality regression model")
    parser.add_argument("-n", "--num-samples", nargs="?", default=500, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=200, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    parser.add_argument(
        "--rng_seed", default=21, type=int, help="random number generator seed"
    )
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.enable_x64()

    main(args)
