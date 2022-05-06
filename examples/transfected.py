import argparse
import os

from functools import partial

import numpy as np
from scipy.special import expit

import jax.numpy as jnp
import jax.random as random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMCGibbs


def model(data, fixed_params):
    counts = data['counts']
    total_reads = data['total_reads']
    guide_observed = data['guide_observed']

    num_cells, num_genes = counts.shape
    assert guide_observed.shape == total_reads.shape == (num_cells,)

    transfection_prob = fixed_params['transfection_prob']
    target_prob = fixed_params['target_prob']
    concentration = fixed_params['concentration']

    gamma = numpyro.sample("gamma", dist.Normal(0.0, 0.1 * jnp.ones(3)).to_event(1))
    beta = numpyro.sample("beta", dist.Normal(0.0, 0.1 * jnp.ones(3)).to_event(1))

    with numpyro.plate("cells", num_cells, dim=-2):
        transfected = numpyro.sample("transfected", dist.Bernoulli(transfection_prob))

        logits = numpyro.deterministic("logits", beta[0] + beta[1] * total_reads[:, None] + beta[2] * transfected)
        numpyro.sample(
                "guide_observed", dist.Bernoulli(logits=logits), obs=guide_observed[:, None]
        )

    with numpyro.plate("genes", num_genes, dim=-1):
        target_genes = numpyro.sample("target_genes", dist.Bernoulli(target_prob))

    with numpyro.plate("genes", num_genes, dim=-1):
        with numpyro.plate("cells", num_cells, dim=-2):
            log_mu = numpyro.deterministic("log_mu",
                gamma[0]
                + gamma[1] * total_reads[:, None]
                + gamma[2] * target_genes * transfected
            )
            numpyro.sample(
                "counts",
                dist.NegativeBinomial2(concentration=concentration, mean=jnp.exp(log_mu)),
                obs=counts,
            )


def _gibbs_fn(data, fixed_params, rng_key, gibbs_sites, hmc_sites):
    transfection_prob = fixed_params['transfection_prob']
    target_prob = fixed_params['target_prob']
    concentration = fixed_params['concentration']

    counts = data['counts']
    total_reads = data['total_reads']
    guide_observed = data['guide_observed']

    transfected = gibbs_sites['transfected']
    target_genes = gibbs_sites['target_genes']
    beta, gamma = hmc_sites['beta'], hmc_sites['gamma']
    #logits, log_mu = hmc_sites['logits'], hmc_sites['log_mu']
    binary_values = jnp.array([0, 1])

    rng_key1, rng_key2 = random.split(rng_key)

    transfected_factor1 = jnp.log(transfection_prob) - jnp.log(1.0 - transfection_prob)
    logits = jnp.log(beta[0] + beta[1] * total_reads[:, None] + beta[2]) - \
             jnp.log(beta[0] + beta[1] * total_reads[:, None])

#
#        logits = numpyro.deterministic("logits", beta[0] + beta[1] * total_reads[:, None] + beta[2] * transfected)
#        numpyro.sample(
#                "guide_observed", dist.Bernoulli(logits=logits), obs=guide_observed[:, None]
#        )

    return {'transfected': transfected, 'target_genes': target_genes}

    x_factor = dist.Bernoulli(0.5).log_prob(values)
    x_factor = x_factor[1] - x_factor[0]
    beta_factor = dist.Normal(beta + values[:, None, None] + y_curr, 1.0).log_prob(jnp.ones((num_y, num_x))).sum([-2])
    beta_factor = beta_factor[1] - beta_factor[0]
    x_new = dist.Bernoulli(logits=x_factor + beta_factor).sample(rng_key_x)

    y_factor = dist.Bernoulli(0.5).log_prob(values)
    y_factor = y_factor[1] - y_factor[0]
    beta_factor = dist.Normal(beta + values[:, None, None] + x_curr, 1.0).log_prob(jnp.ones((num_y, num_x))).sum([-1])
    beta_factor = beta_factor[1] - beta_factor[0]
    y_new = dist.Bernoulli(logits=y_factor + beta_factor).sample(rng_key_y)

    return {'x': x_new, 'y': y_new[:, None]}


def run_inference(model, args, rng_key, data, fixed_params):
    gibbs_fn = partial(_gibbs_fn, data, fixed_params)
    hmc_kernel = NUTS(model)
    kernel = HMCGibbs(hmc_kernel, gibbs_fn=gibbs_fn, gibbs_sites=['transfected', 'target_genes'])

    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(rng_key, data, fixed_params)
    mcmc.print_summary()
    return mcmc.get_samples()


def get_data(
    num_cells=100,
    num_genes=20,
    beta2=3.45,
    gamma0=1.23,
    gamma2=2.34,
    concentration=20.0,
    seed=0,
):
    total_reads = np.log(np.random.RandomState(seed + 1).binomial(40, 0.5, size=(num_cells),))
    transfected = jnp.concatenate(
        [jnp.zeros(num_cells // 4), jnp.ones(3 * num_cells // 4)]
    )
    assert transfected.shape == total_reads.shape == (num_cells,)

    logits = beta2 * transfected
    guide_observed = np.random.RandomState(seed).binomial(1, expit(logits))
    assert guide_observed.shape == (num_cells,)

    target_genes = jnp.concatenate([jnp.ones(20), jnp.zeros(num_genes - 20)])
    log_mu = gamma0 + gamma2 * target_genes * transfected[:, None]
    counts = dist.NegativeBinomial2(concentration=concentration, mean=jnp.exp(log_mu)).sample(
        random.PRNGKey(seed)
    )
    assert counts.shape == (num_cells, num_genes)

    return {'counts': counts, 'total_reads': total_reads, 'guide_observed': guide_observed}


def main(args):
    fixed_params = {'transfection_prob': 0.75,
                    'target_prob': 0.2,
                    'concentration': 20.0}

    data = get_data(concentration=fixed_params['concentration'])
    print(data['counts'].shape)
    print(data['counts'])

    rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    run_inference(model, args, rng_key, data, fixed_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gaussian Process example")
    parser.add_argument("-n", "--num-samples", nargs="?", default=300, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=200, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
