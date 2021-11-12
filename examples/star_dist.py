"""change                 transform=lambda val: val[1],  to                transform=lambda val: val[0], in VI """
from math import ceil

import matplotlib.pyplot as plt
import numpy as np

from jax import lax, random, scipy as jscipy, vmap
import jax.numpy as jnp

import numpyro
from numpyro import distributions as dist
from numpyro.contrib.einstein import Stein, kernels
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
lrs = {
    "Linear": 0.04,
    "IMQ": 0.08,
    "RBF": 0.09,
    "Random Feature": 0.09,
    "Mixture": 0.1,
    "Matrix Const.": 1.0,
}


class Star(dist.Distribution):
    support = dist.constraints.real

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
        super(Star, self).__init__(batch_shape=batch_shape, validate_args=validate_args)

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


def model():
    numpyro.sample("x", Star())


def mmd(p_samples, q_samples):
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


if __name__ == "__main__":
    l = ceil((len(kernels) + 1) / 2.0)
    fig, axs = plt.subplots(2, l, figsize=(20, 10), dpi=300)
    rng_key = random.PRNGKey(0)
    p_samples = Star().sample(rng_key, (10_000,))

    guide = AutoDelta(model)
    star_xs = np.linspace(-7, 7, num=1000)
    star_ys = np.linspace(-7, 7, num=1000)
    star_zs = np.stack(np.meshgrid(star_xs, star_ys), axis=-1)
    star_lps = np.exp(Star().log_prob(star_zs))

    svgd = Stein(
        model,
        guide,
        Adagrad(step_size=1.0),
        Trace_ELBO(),
        kernels["Linear"],
        num_particles=50,
        init_strategy=init_with_noise(
            init_to_value(values={"x_auto_loc": np.array([[0.0, 0.0]])}),
            noise_scale=3.0,
        ),
    )

    svgd_state = svgd.init(rng_key)
    res = svgd.get_params(svgd_state)["x_auto_loc"]
    axs[0, 0].imshow(
        star_lps,
        origin="lower",
        interpolation="bicubic",
        extent=[np.min(star_xs), np.max(star_xs), np.min(star_ys), np.max(star_ys)],
    )
    axs[0, 0].scatter(
        res[..., 0], res[..., 1], c="orange", marker="x", s=150, linewidths=3
    )
    axs[0, 0].set_xlim((np.min(star_xs), np.max(star_xs)))
    axs[0, 0].set_ylim((np.min(star_ys), np.max(star_ys)))
    axs[0, 0].set_title("Initial", fontsize=26)

    num_iterations = 1000
    mmds = {}
    for i, (name, kernel) in enumerate(kernels.items()):
        j = i + 1
        ax = axs[j // l, j % l]
        svgd = Stein(
            model,
            guide,
            Adagrad(step_size=lrs[name]),
            Trace_ELBO(),
            kernel,
            num_particles=50,
            init_strategy=init_with_noise(
                init_to_value(values={"x_auto_loc": np.array([[0.0, 0.0]])}),
                noise_scale=3.0,
            ),
        )
        svgd_state, all_states = svgd.run(
            rng_key, num_iterations, transform=lambda val: val[0]
        )
        res = svgd.get_params(svgd_state)["x_auto_loc"]
        mmds.update(
            {
                name: vmap(lambda q_samples: mmd(p_samples, q_samples))(
                    svgd.get_params(all_states)["x_auto_loc"]
                )
            }
        )
        ax.imshow(
            star_lps,
            origin="lower",
            interpolation="bicubic",
            extent=[np.min(star_xs), np.max(star_xs), np.min(star_ys), np.max(star_ys)],
        )
        ax.scatter(
            res[..., 0], res[..., 1], c="orange", marker="x", s=150, linewidths=3
        )
        ax.set_xlim((np.min(star_xs), np.max(star_xs)))
        ax.set_ylim((np.min(star_ys), np.max(star_ys)))
        ax.set_title(name, fontsize=26)
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
    fig.savefig("star_kernels.pdf")
