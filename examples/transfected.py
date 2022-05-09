import argparse

from functools import partial

import numpy as np
from jax.scipy.special import expit

import jax.numpy as jnp
import jax.random as random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMCGibbs

import pandas as pd


def model(data, fixed_params):
    counts = data["counts"]
    total_reads = data["total_reads"]
    guide_observed = data["guide_observed"]

    num_cells, num_genes = counts.shape
    assert guide_observed.shape == total_reads.shape == (num_cells,)

    transfection_prob = fixed_params["transfection_prob"]
    target_prob = fixed_params["target_prob"]
    concentration = fixed_params["concentration"]

    # what are reasonable prior assumptions here?
    gamma = numpyro.sample("gamma", dist.Normal(0.0, 0.1 * jnp.ones(3)).to_event(1))
    beta = numpyro.sample("beta", dist.Normal(0.0, 0.1 * jnp.ones(3)).to_event(1))

    with numpyro.plate("cells", num_cells, dim=-2):
        transfected = numpyro.sample("transfected", dist.Bernoulli(transfection_prob))

        logits = numpyro.deterministic(
            "logits", beta[0] + beta[1] * total_reads[:, None] + beta[2] * transfected
        )
        # noisy link between observed guide and transfection status of cell
        numpyro.sample(
            "guide_observed", dist.Bernoulli(logits=logits), obs=guide_observed[:, None]
        )

    with numpyro.plate("genes", num_genes, dim=-1):
        target_genes = numpyro.sample("target_genes", dist.Bernoulli(target_prob))

    with numpyro.plate("genes", num_genes, dim=-1):
        with numpyro.plate("cells", num_cells, dim=-2):
            log_mu = numpyro.deterministic(
                "log_mu",
                gamma[0]
                + gamma[1] * total_reads[:, None]  # should this be log(total_reads) ?
                + gamma[2] * target_genes * transfected,
            )
            numpyro.sample(
                "counts",
                dist.NegativeBinomial2(
                    concentration=concentration, mean=jnp.exp(log_mu)
                ),
                obs=counts,
            )


# custom gibbs function; note this is closely tied to the specific structure of model
def _gibbs_fn(data, fixed_params, rng_key, gibbs_sites, hmc_sites):
    transfection_prob = fixed_params["transfection_prob"]
    target_prob = fixed_params["target_prob"]
    concentration = fixed_params["concentration"]

    counts = data["counts"]
    total_reads = data["total_reads"]
    guide_observed = data["guide_observed"]
    num_cells, num_genes = counts.shape

    transfected = gibbs_sites["transfected"]
    target_genes = gibbs_sites["target_genes"]
    beta = hmc_sites["beta"]
    gamma = hmc_sites["gamma"]

    rng_key1, rng_key2 = random.split(rng_key)

    # sample transfected from conditional posterior
    factor1 = jnp.log(transfection_prob) - jnp.log(1.0 - transfection_prob)
    factor2 = dist.Bernoulli(logits=beta[0] + beta[1] * total_reads + beta[2]).log_prob(guide_observed) -\
        dist.Bernoulli(logits=beta[0] + beta[1] * total_reads).log_prob(guide_observed)

    mean1 = jnp.exp(gamma[0] + gamma[1] * total_reads[:, None] + gamma[2] * target_genes)
    mean0 = jnp.exp(gamma[0] + gamma[1] * total_reads[:, None])
    factor3 = dist.NegativeBinomial2(concentration=concentration, mean=mean1).log_prob(counts) -\
        dist.NegativeBinomial2(concentration=concentration, mean=mean0).log_prob(counts)
    logits = factor1 + factor2 + factor3.sum(-1)
    transfected = dist.Bernoulli(logits=logits).sample(rng_key1)[:, None]
    assert transfected.shape == (num_cells, 1)

    # sample target_gene from conditional posterior
    factor1 = jnp.log(target_prob) - jnp.log(1.0 - target_prob)
    mean1 = jnp.exp(gamma[0] + gamma[1] * total_reads[:, None] + gamma[2] * transfected)
    factor2 = dist.NegativeBinomial2(concentration=concentration, mean=mean1).log_prob(counts) -\
        dist.NegativeBinomial2(concentration=concentration, mean=mean0).log_prob(counts)
    logits = factor1 + factor2.sum(0)
    target_genes = dist.Bernoulli(logits=logits).sample(rng_key2)
    assert target_genes.shape == (num_genes,)

    return {"transfected": transfected, "target_genes": target_genes}


# we use HMCGibbs which combines HMC moves with custom Gibbs moves
def run_inference(model, args, rng_key, data, fixed_params):
    gibbs_fn = partial(_gibbs_fn, data, fixed_params)
    hmc_kernel = NUTS(model)
    kernel = HMCGibbs(
        hmc_kernel, gibbs_fn=gibbs_fn, gibbs_sites=["transfected", "target_genes"]
    )

    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        progress_bar=True,
    )
    mcmc.run(rng_key, data, fixed_params)
    mcmc.print_summary()
    return mcmc.get_samples()


# create fake data that is similar to the generative process
def get_data(
    num_cells=400,
    num_genes=20,
    active_genes=10,
    beta2=2.00,
    gamma0=3.45,
    gamma2=2.77,
    concentration=10.0,
    seed=0,
):
    assert num_cells % 4 == 0
    assert num_genes > active_genes

    total_reads = np.random.RandomState(seed + 1).binomial(100, 0.5, size=(num_cells))
    total_reads = np.log(1 + total_reads)
    transfected = jnp.concatenate([jnp.zeros(num_cells // 4), jnp.ones(3 * num_cells // 4)])
    assert transfected.shape == total_reads.shape == (num_cells,)

    logits = beta2 * transfected
    guide_observed = np.random.RandomState(seed).binomial(1, expit(logits))
    assert guide_observed.shape == (num_cells,)

    target_genes = jnp.concatenate([jnp.ones(active_genes), jnp.zeros(num_genes - active_genes)])
    log_mu = gamma0 + gamma2 * target_genes * transfected[:, None]
    counts = dist.NegativeBinomial2(concentration=concentration, mean=jnp.exp(log_mu)).sample(random.PRNGKey(seed))
    assert counts.shape == (num_cells, num_genes)

    return {
        "counts": counts,
        "total_reads": total_reads,
        "guide_observed": guide_observed,
    }


def main(args):
    fixed_params = {
        "transfection_prob": 0.2,
        "target_prob": 0.1,
        "concentration": 10.0,
    }

    use_r_data = False

    if not use_r_data:
        data = get_data(concentration=fixed_params["concentration"])
    else:
        total_reads = pd.read_csv('total_reads.csv').values[:, 0]
        counts = pd.read_csv('counts_tbl.tsv', sep='\t').values
        guide_observed = pd.read_csv('cell_annot.tsv', sep='\t').values[:, 1]
        data = {'counts': counts, 'total_reads': total_reads, 'guide_observed': guide_observed}

    print("Starting inference with {} cells and {} genes...".format(*data['counts'].shape))

    run_inference(model, args, random.PRNGKey(args.seed), data, fixed_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="guide transfection model")
    parser.add_argument("-n", "--num-samples", nargs="?", default=1000, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=1000, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    # should generally use 64 bit precision when doing HMC
    numpyro.enable_x64()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
