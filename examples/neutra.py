import argparse
import warnings

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import seaborn as sns

from jax import jit, random, vmap
from jax.config import config as jax_config
from jax.experimental import optimizers
import jax.numpy as np
from jax.scipy.special import logsumexp
from jax.tree_util import tree_map

from numpyro.contrib.autoguide import AutoIAFNormal
from numpyro.diagnostics import summary
import numpyro.distributions as dist
from numpyro.handlers import sample
from numpyro.hmc_util import initialize_model
from numpyro.mcmc import mcmc
from numpyro.svi import elbo, svi
from numpyro.util import fori_collect

"""
This example illustrates how to use a trained AutoIAFNormal autoguide to transform a posterior to a
Gaussian-like one. The transform will be used to get better mixing rate for NUTS sampler.

[1] Hoffman, M. et al. (2019), ["NeuTra-lizing Bad Geometry in Hamiltonian Monte Carlo Using Neural Transport"]
    (https://arxiv.org/abs/1903.03704).
"""


def dual_moon_pe(x):
    term1 = 0.5 * ((np.linalg.norm(x, axis=-1) - 2) / 0.4) ** 2
    term2 = -0.5 * ((x[..., :1] + np.array([-2., 2.])) / 0.6) ** 2
    return term1 - logsumexp(term2, axis=-1)


def dual_moon_model():
    x = sample('x', dist.Uniform(-4 * np.ones(2), 4 * np.ones(2)))
    pe = dual_moon_pe(x)
    sample('log_density', dist.Delta(log_density=-pe), obs=0.)


def make_transformed_pe(potential_fn, transform, unpack_fn):
    def transformed_potential_fn(z):
        # NB: currently, intermediates for ComposeTransform is None, so this has no effect
        # see https://github.com/pyro-ppl/numpyro/issues/242
        u, intermediates = transform.call_with_intermediates(z)
        logdet = transform.log_abs_det_jacobian(z, u, intermediates=intermediates)
        return potential_fn(unpack_fn(u)) + logdet

    return transformed_potential_fn


def main(args):
    jax_config.update('jax_platform_name', args.device)

    print("Start vanilla HMC...")
    vanilla_samples = mcmc(args.num_warmup, args.num_samples, init_params=np.array([2., 0.]),
                           potential_fn=dual_moon_pe, progbar=True)

    opt_init, opt_update, get_params = optimizers.adam(0.001)
    rng_guide, rng_init, rng_train = random.split(random.PRNGKey(1), 3)
    guide = AutoIAFNormal(rng_guide, dual_moon_model, get_params, hidden_dims=[args.num_hidden])
    svi_init, svi_update, _ = svi(dual_moon_model, guide, elbo, opt_init, opt_update, get_params)
    opt_state, _ = svi_init(rng_init)

    def body_fn(val):
        i, loss, opt_state_, rng_ = val
        loss, opt_state_, rng_ = svi_update(i, rng_, opt_state_)
        return i + 1, loss, opt_state_, rng_

    print("Start training guide...")
    # TODO: remove the warning when the issue is fixed upstream
    warnings.warn('Due to the bug https://github.com/google/jax/issues/939, to'
                  ' train AutoIAFNormal we should set the environment flag'
                  ' "XLA_FLAGS=--xla_cpu_enable_fast_math=false".')
    losses, opt_states = fori_collect(0, args.num_iters, jit(body_fn),
                                      (0, 0., opt_state, rng_train),
                                      transform=lambda x: (x[1], x[2]), progbar=False)
    last_state = tree_map(lambda x: x[-1], opt_states)
    print("Finish training guide. Extract samples...")
    guide_samples = guide.sample_posterior(random.PRNGKey(0), last_state,
                                           sample_shape=(args.num_samples,))

    transform = guide.get_transform(last_state)
    unpack_fn = lambda u: guide.unpack_latent(u, transform=False)  # noqa: E731

    _, potential_fn, constrain_fn = initialize_model(random.PRNGKey(0), dual_moon_model)
    transformed_potential_fn = make_transformed_pe(potential_fn, transform, unpack_fn)
    transformed_constrain_fn = lambda x: constrain_fn(unpack_fn(transform(x)))  # noqa: E731

    # TODO: expose latent_size in autoguide
    init_params = np.zeros(np.size(guide._init_latent))
    print("\nStart NeuTra HMC...")
    zs = mcmc(args.num_warmup, args.num_samples, init_params, potential_fn=transformed_potential_fn)
    print("Transform samples into unwarped space...")
    samples = vmap(transformed_constrain_fn)(zs)
    summary(tree_map(lambda x: x[None, ...], samples))

    # make plots

    # IAF guide samples (for plotting)
    iaf_base_samples = dist.Normal(np.zeros(2), 1.).sample(random.PRNGKey(0), (1000,))
    iaf_trans_samples = vmap(transformed_constrain_fn)(iaf_base_samples)['x']

    x1 = np.linspace(-3, 3, 100)
    x2 = np.linspace(-3, 3, 100)
    X1, X2 = np.meshgrid(x1, x2)
    P = np.clip(np.exp(-dual_moon_pe(np.stack([X1, X2], axis=-1))), a_min=0.)

    fig = plt.figure(figsize=(12, 16), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])

    ax1.plot(np.log(losses[1000:]))
    ax1.set_title('Autoguide training log loss (after 1000 steps)')

    ax2.contourf(X1, X2, P, cmap='OrRd')
    sns.kdeplot(guide_samples['x'][:, 0].copy(), guide_samples['x'][:, 1].copy(), n_levels=30, ax=ax2)
    ax2.set(xlim=[-3, 3], ylim=[-3, 3],
            xlabel='x0', ylabel='x1', title='Posterior using AutoIAFNormal guide')

    sns.scatterplot(iaf_base_samples[:, 0], iaf_base_samples[:, 1], ax=ax3, hue=iaf_trans_samples[:, 0] < 0.)
    ax3.set(xlim=[-3, 3], ylim=[-3, 3],
            xlabel='x0', ylabel='x1', title='AutoIAFNormal base samples (True=left moon; False=right moon)')

    ax4.contourf(X1, X2, P, cmap='OrRd')
    sns.kdeplot(vanilla_samples[:, 0].copy(), vanilla_samples[:, 1].copy(), n_levels=30, ax=ax4)
    ax4.plot(vanilla_samples[-50:, 0], vanilla_samples[-50:, 1], 'bo-', alpha=0.5)
    ax4.set(xlim=[-3, 3], ylim=[-3, 3],
            xlabel='x0', ylabel='x1', title='Posterior using vanilla HMC sampler')

    sns.scatterplot(zs[:, 0], zs[:, 1], ax=ax5, hue=samples['x'][:, 0] < 0.,
                    s=30, alpha=0.5, edgecolor="none")
    ax5.set(xlim=[-5, 5], ylim=[-5, 5],
            xlabel='x0', ylabel='x1', title='Samples from the warped posterior - p(z)')

    ax6.contourf(X1, X2, P, cmap='OrRd')
    sns.kdeplot(samples['x'][:, 0].copy(), samples['x'][:, 1].copy(), n_levels=30, ax=ax6)
    ax6.plot(samples['x'][-50:, 0], samples['x'][-50:, 1], 'bo-', alpha=0.2)
    ax6.set(xlim=[-3, 3], ylim=[-3, 3],
            xlabel='x0', ylabel='x1', title='Posterior using NeuTra HMC sampler')

    plt.savefig("neutra.pdf")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuTra HMC")
    parser.add_argument('-n', '--num-samples', nargs='?', default=20000, type=int)
    parser.add_argument('--num-warmup', nargs='?', default=0, type=int)
    parser.add_argument('--num-hidden', nargs='?', default=20, type=int)
    parser.add_argument('--num-iters', nargs='?', default=200000, type=int)
    parser.add_argument('--device', default='cpu', type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()
    main(args)
