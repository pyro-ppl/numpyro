import argparse
import os

import numpy as np
from scipy.special import expit

import jax.numpy as jnp
import jax.random as random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


def model(counts, total_reads, guide_observed, disp=20.0):
    num_cells, num_genes = counts.shape
    assert guide_observed.shape == total_reads.shape == (num_cells,)

    transfection_prob = 0.75
    target_prob = 0.2

    gamma = numpyro.sample("gamma", dist.Normal(0.0, jnp.ones(3)).to_event(1))
    beta = numpyro.sample("beta", dist.Normal(0.0, jnp.ones(3)).to_event(1))

    with numpyro.plate("cells", num_cells):
        transfected = numpyro.sample("transfected", dist.Bernoulli(transfection_prob))

        logits = beta[0] + beta[1] * total_reads + beta[2] * transfected
        numpyro.sample(
            "guide_observed", dist.Bernoulli(logits=logits), obs=guide_observed
        )

    with numpyro.plate("genes", num_genes):
        target_genes = numpyro.sample("target_genes", dist.Bernoulli(target_prob))

    with numpyro.plate("genes", num_genes):
        with numpyro.plate("cells", num_cells):
            mu = (
                gamma[0]
                + gamma[1] * total_reads[:, None]
                + gamma[2] * target_genes * transfected[:, None]
            )
            numpyro.sample(
                "counts",
                dist.NegativeBinomial2(concentration=disp, mean=mu),
                obs=counts,
            )


def run_inference(model, args, rng_key, counts, total_reads, guide_observed):
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(rng_key, counts, total_reads, guide_observed)
    mcmc.print_summary()
    return mcmc.get_samples()


def get_data(
    num_cells=1000,
    num_genes=100,
    beta2=1.23,
    gamma0=3.45,
    gamma2=2.34,
    disp=20.0,
    seed=0,
):
    total_reads = jnp.ones(num_cells)
    transfected = jnp.concatenate(
        [jnp.zeros(num_cells // 4), jnp.ones(3 * num_cells // 4)]
    )
    assert transfected.shape == total_reads.shape == (num_cells,)

    logits = beta2 * transfected
    guide_observed = np.random.RandomState(seed).binomial(1, expit(logits))
    assert guide_observed.shape == (num_cells,)

    target_genes = jnp.concatenate([jnp.ones(20), jnp.zeros(num_genes - 20)])
    mu = gamma0 + gamma2 * target_genes * transfected[:, None]
    counts = dist.NegativeBinomial2(concentration=disp, mean=mu).sample(
        random.PRNGKey(seed)
    )
    assert counts.shape == (num_cells, num_genes)

    return counts, total_reads, guide_observed


def main(args):
    counts, total_reads, guide_observed = get_data()

    rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    run_inference(model, args, rng_key, counts, total_reads, guide_observed)


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
