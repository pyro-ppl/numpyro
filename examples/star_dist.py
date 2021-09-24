import jax.numpy as jnp
from jax import lax, random
from jax import scipy as jscipy
from numpyro import distributions as dist
import numpy as np
import matplotlib.pyplot as plt
import numpyro
from numpyro.contrib.callbacks import Progbar
from numpyro.contrib.einstein import Stein, kernels
from numpyro.infer import Trace_ELBO, init_with_noise, init_to_value

from numpyro.infer.autoguide import AutoDelta
from numpyro.optim import Adagrad

kernels = {'Linear': kernels.LinearKernel(),
           'IMQ': kernels.IMQKernel(),
           'RBF': kernels.RBFKernel(),
           'Random Feature': kernels.RandomFeatureKernel(),
           'Mixture': kernels.MixtureKernel([.5, .5], [kernels.LinearKernel(), kernels.RandomFeatureKernel()]),
           'Matrix Const.': kernels.PrecondMatrixKernel(kernels.HessianPrecondMatrix(),
                                                        kernels.RBFKernel(mode='matrix'),
                                                        precond_mode='const'),
           'Matrix Anch.': kernels.PrecondMatrixKernel(kernels.HessianPrecondMatrix(),
                                                        kernels.RBFKernel(mode='matrix'),
                                                        precond_mode='anchor_points'),
           }


class Star(dist.Distribution):
    support = dist.constraints.real

    def __init__(self, mu0=jnp.array([0., 1.5]), cov0=jnp.diag(jnp.array([1e-2, 1.])), n_comp=5, validate_args=None):
        batch_shape = lax.broadcast_shapes(jnp.shape(mu0)[:-1], jnp.shape(cov0)[:-2])
        mu0 = jnp.broadcast_to(mu0, batch_shape + jnp.shape(mu0)[-1:])
        cov0 = jnp.broadcast_to(cov0, batch_shape + jnp.shape(cov0)[-2:])
        self.n_comp = n_comp
        mus = [mu0]
        covs = [cov0]
        theta = 2 * jnp.pi / n_comp
        rot = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])
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
            lps.append(dist.MultivariateNormal(self.mus[i], self.covs[i]).log_prob(value))
        return jscipy.special.logsumexp(jnp.stack(lps, axis=0), axis=0) / self.n_comp

    def sample(self, key, sample_shape=()):
        zs = dist.Categorical(probs=jnp.array([1 / self.n_comp] * self.n_comp)).sample(key, sample_shape)
        xs = jnp.stack([dist.MultivariateNormal(self.mus[i], self.covs[i]).sample(key, sample_shape)
                        for i in range(self.n_comp)], axis=0)
        return jnp.take_along_axis(xs, jnp.expand_dims(jnp.expand_dims(zs, axis=-1), axis=-1), axis=0)


def model():
    numpyro.sample('x', Star())


if __name__ == '__main__':
    fig, axs = plt.subplots(1, len(kernels) + 1, figsize=(30, 5), dpi=300)
    rng_key = random.PRNGKey(0)
    guide = AutoDelta(model)
    star_xs = np.linspace(-7, 7, num=1000)
    star_ys = np.linspace(-7, 7, num=1000)
    star_zs = np.stack(np.meshgrid(star_xs, star_ys), axis=-1)
    star_lps = np.exp(Star().log_prob(star_zs))

    svgd = Stein(model,
                 guide,
                 Adagrad(step_size=1.0),
                 Trace_ELBO(),
                 kernels['Linear'],
                 num_particles=50,
                 init_strategy=init_with_noise(init_to_value(values={'x_auto_loc': np.array([[0., 0.]])}),
                                               noise_scale=3.))

    svgd_state = svgd.init(rng_key)
    res = svgd.get_params(svgd_state)['x_auto_loc']
    axs[0].imshow(star_lps, origin='lower', interpolation='bicubic', extent=[np.min(star_xs), np.max(star_xs),
                                                                             np.min(star_ys), np.max(star_ys)])
    axs[0].scatter(res[..., 0], res[..., 1], c='orange', marker='x', s=150)
    axs[0].set_xlim((np.min(star_xs), np.max(star_xs)))
    axs[0].set_ylim((np.min(star_ys), np.max(star_ys)))
    axs[0].set_title('Initial', fontsize=30)

    num_iterations = 1000

    for i, (name, kernel) in enumerate(kernels.items()):
        ax = axs[i + 1]
        svgd = Stein(model,
                     guide,
                     Adagrad(step_size=1.0),
                     Trace_ELBO(),
                     kernel,
                     num_particles=50,
                     init_strategy=init_with_noise(init_to_value(values={'x_auto_loc': np.array([[0., 0.]])}),
                                                   noise_scale=3.0))
        svgd_state, loss = svgd.run(rng_key, num_iterations, callbacks=[Progbar()])
        res = svgd.get_params(svgd_state)['x_auto_loc']
        ax.imshow(star_lps, origin='lower', interpolation='bicubic', extent=[np.min(star_xs), np.max(star_xs),
                                                                             np.min(star_ys), np.max(star_ys)])
        ax.scatter(res[..., 0], res[..., 1], c='orange', marker='x', s=150)
        ax.set_xlim((np.min(star_xs), np.max(star_xs)))
        ax.set_ylim((np.min(star_ys), np.max(star_ys)))
        ax.set_title(name, fontsize=30)
    for ax in axs:
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig('star_kernels.pdf')
