# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: 2d toy distribution with SteinVI
===================================================
In this example we infer a 2d star distribution with the different kernels available for SteinVI.
"""
import argparse
from functools import partial
from math import ceil

import matplotlib.pyplot as plt
import numpy as np

from jax import lax, random, scipy as jscipy, vmap
import jax.numpy as jnp

import numpyro
from numpyro import distributions as dist
from numpyro.contrib.einstein import SteinVI, kernels
from numpyro.infer import Trace_ELBO, init_to_value, init_with_noise
from numpyro.infer.autoguide import AutoDelta
from numpyro.optim import Adagrad

kernels = {
    "Linear": kernels.LinearKernel(),
    "IMQ": kernels.IMQKernel(),
    "RBF": kernels.RBFKernel(),
    "Random Feature": kernels.RandomFeatureKernel(),
    "Mixture": kernels.MixtureKernel(
        [0.5, 0.5], [kernels.LinearKernel(), kernels.RandomFeatureKernel()]
    ),
    "Matrix Const.": kernels.PrecondMatrixKernel(
        kernels.HessianPrecondMatrix(),
        kernels.RBFKernel(mode="matrix"),
        precond_mode="const",
    ),
}


def model(target_distribution):
    numpyro.sample("x", target_distribution)


def mmd(p_samples, q_samples):
    """Maximum Mean Discrepancy"""
    np = p_samples.shape[0]
    nq = q_samples.shape[0]
    q_samples = q_samples.squeeze()
    pmask = jnp.ones((np, np)) - jnp.eye(np)
    qmask = jnp.ones((nq, nq)) - jnp.eye(nq)
    qq_dist = jnp.linalg.norm(
        (q_samples[None, :] - q_samples[:, None]) * qmask[..., None], axis=-1
    )
    pp_dist = jnp.linalg.norm(
        (p_samples[None, :] - p_samples[:, None]) * pmask[..., None], axis=-1
    )
    pq_dist = jnp.linalg.norm(p_samples[None, :] - q_samples[:, None], axis=-1)
    return (
        jnp.mean(jnp.exp(-(qq_dist ** 2)))
        + jnp.mean(jnp.exp(-(pp_dist ** 2)))
        - 2 * jnp.mean(jnp.exp(-(pq_dist ** 2)))
    )


def add_visualization(ax, particles, title, xs_bounds, ys_bounds, target_probs):
    ax.imshow(
        target_probs,
        origin="lower",
        interpolation="bicubic",
        extent=[*xs_bounds, *ys_bounds],
    )
    ax.scatter(
        particles[..., 0],
        particles[..., 1],
        c="orange",
        marker="x",
        s=150,
        linewidths=3,
    )
    ax.set_xlim(xs_bounds)
    ax.set_ylim(ys_bounds)
    ax.set_title(title, fontsize=26)


def main(args):
    target_distribution = Star()

    num_cols = ceil((len(kernels) + 1) / 2.0)
    fig, axs = plt.subplots(2, num_cols, figsize=(20, 10), dpi=300)

    rng_key = random.PRNGKey(args.rng_seed)

    data = target_distribution.sample(rng_key, (10_000,))

    lower_threshold = jnp.min(data, axis=0) - 3
    upper_threshold = jnp.max(data, axis=0) + 3
    xs_bounds = (lower_threshold[0], upper_threshold[0])
    ys_bounds = (lower_threshold[1], upper_threshold[1])

    guide = AutoDelta(model)
    xs = np.linspace(*xs_bounds, num=100)
    ys = np.linspace(*ys_bounds, num=100)
    zs = np.stack(np.meshgrid(xs, ys), axis=-1)
    target_probs = np.exp(target_distribution.log_prob(zs))

    add_viz = partial(
        add_visualization,
        xs_bounds=xs_bounds,
        ys_bounds=ys_bounds,
        target_probs=target_probs,
    )
    stein_constr = partial(
        SteinVI,
        model=model,
        guide=guide,
        optim=Adagrad(step_size=0.1),
        loss=Trace_ELBO(),
        num_particles=args.num_particles,
        init_strategy=init_with_noise(
            init_to_value(values={"x_auto_loc": np.array([[0.0, 0.0]])}),
            noise_scale=3.0,
        ),
    )

    steinvi = stein_constr(kernel_fn=kernels["Linear"])
    state = steinvi.init(rng_key, target_distribution)
    add_viz(
        ax=axs[0, 0], particles=steinvi.get_params(state)["x_auto_loc"], title="Initial"
    )

    mmds = {}
    for i, (name, kernel) in enumerate(kernels.items()):
        j = i + 1
        ax = axs[j // num_cols, j % num_cols]
        steinvi = stein_constr(kernel_fn=kernel)
        results = steinvi.run(
            rng_key,
            args.max_iter,
            target_distribution,
            collect_fn=lambda val: val[0],
            progress_bar=args.progress_bar,
        )
        add_viz(ax=ax, particles=results.params["x_auto_loc"], title=name)
        mmds.update(
            {
                name: vmap(lambda q_samples: mmd(data, q_samples))(
                    steinvi.get_params(results.losses)["x_auto_loc"]
                )
            }
        )

    for ax in axs:
        for a in ax:
            a.set_axis_off()
    ax = axs[-1, -1]
    for name, mmd_vals in mmds.items():
        ax.plot(mmd_vals, label=name, linewidth=3)

    ax.legend(fontsize=16)
    ax.set_xlabel("Iterations", fontsize=22)
    ax.set_ylabel("MMD", fontsize=22)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_axis_on()

    fig.tight_layout()
    fig.savefig("stein_2d_toy.pdf")


class Star(dist.Distribution):
    support = dist.constraints.independent(dist.constraints.real, 1)

    def __init__(
        self,
        mu0=jnp.array([0.0, 1.5]),
        cov0=jnp.diag(jnp.array([1e-2, 1.0])),
        n_comp=5,
        validate_args=None,
    ):
        batch_shape = lax.broadcast_shapes(jnp.shape(mu0)[:-1], jnp.shape(cov0)[:-2])
        mu0 = jnp.broadcast_to(mu0, batch_shape + jnp.shape(mu0)[-1:])
        cov0 = jnp.broadcast_to(cov0, batch_shape + jnp.shape(cov0)[-2:])
        self.n_comp = n_comp
        mus = [mu0]
        covs = [cov0]
        theta = 2 * jnp.pi / n_comp
        rot = jnp.array(
            [[jnp.cos(theta), jnp.sin(theta)], [-jnp.sin(theta), jnp.cos(theta)]]
        )
        for i in range(n_comp - 1):
            mui = rot @ mus[-1]
            covi = rot @ covs[-1] @ rot.transpose()
            mus.append(mui)
            covs.append(covi)
        self.mus = jnp.stack(mus)
        self.covs = jnp.stack(covs)
        super(Star, self).__init__(
            event_shape=(2,), batch_shape=batch_shape, validate_args=validate_args
        )

    def log_prob(self, value):
        lps = []
        for i in range(self.n_comp):
            lps.append(
                dist.MultivariateNormal(self.mus[i], self.covs[i]).log_prob(value)
            )
        return jscipy.special.logsumexp(jnp.stack(lps, axis=0), axis=0) / self.n_comp

    def sample(self, key, sample_shape=()):
        assign_key, x_key = random.split(key)

        assign = dist.Categorical(
            probs=jnp.array([1 / self.n_comp] * self.n_comp)
        ).sample(assign_key, sample_shape)
        return dist.MultivariateNormal(self.mus[assign], self.covs[assign]).sample(
            x_key
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-distribution", default="sine", choices=["star"])
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--num-particles", type=int, default=20)
    parser.add_argument("--progress-bar", type=bool, default=True)
    parser.add_argument("--rng-key", type=int, default=142)
    parser.add_argument("--device", default="cpu", choices=["gpu", "cpu"])
    parser.add_argument("--rng-seed", default=142, type=int)

    args = parser.parse_args()

    numpyro.set_platform(args.device)
    main(args)
