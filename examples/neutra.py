import argparse
from functools import partial

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import seaborn as sns

from jax import lax, random, vmap
from jax.config import config as jax_config
import jax.numpy as np
from jax.scipy.special import logsumexp
from jax.tree_util import tree_map

import numpyro
from numpyro import optim
from numpyro.contrib.autoguide import AutoBNAFNormal
from numpyro.diagnostics import summary
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.hmc_util import initialize_model
from numpyro.infer_util import transformed_potential_energy
from numpyro.mcmc import MCMC, NUTS
from numpyro.svi import SVI, elbo

"""
This example illustrates how to use a trained AutoIAFNormal autoguide to transform a posterior to a
Gaussian-like one. The transform will be used to get better mixing rate for NUTS sampler.

[1] Hoffman, M. et al. (2019), ["NeuTra-lizing Bad Geometry in Hamiltonian Monte Carlo Using Neural Transport"]
    (https://arxiv.org/abs/1903.03704).
"""


class DualMoonDistribution(dist.Distribution):
    support = constraints.real_vector

    def __init__(self):
        super(DualMoonDistribution, self).__init__(event_shape=(2,))

    def sample(self, key, sample_shape=()):
        # it is enough to return an arbitrary sample with correct shape
        return np.zeros(sample_shape + self.event_shape)

    def log_prob(self, x):
        term1 = 0.5 * ((np.linalg.norm(x, axis=-1) - 2) / 0.4) ** 2
        term2 = -0.5 * ((x[..., :1] + np.array([-2., 2.])) / 0.6) ** 2
        pe = term1 - logsumexp(term2, axis=-1)
        return -pe


def dual_moon_model():
    numpyro.sample('x', DualMoonDistribution())


def main(args):
    jax_config.update('jax_platform_name', args.device)

    print("Start vanilla HMC...")
    nuts_kernel = NUTS(dual_moon_model)
    mcmc = MCMC(nuts_kernel, args.num_warmup, args.num_samples)
    mcmc.run(random.PRNGKey(0))
    mcmc.print_summary()
    vanilla_samples = mcmc.get_samples()['x'].copy()

    adam = optim.Adam(0.01)
    guide = AutoBNAFNormal(dual_moon_model, hidden_factors=[args.hidden_factor, args.hidden_factor])
    svi = SVI(dual_moon_model, guide, elbo, adam)
    svi_state = svi.init(random.PRNGKey(1))

    print("Start training guide...")
    last_state, losses = lax.scan(lambda state, i: svi.update(state), svi_state, np.zeros(args.num_iters))
    params = svi.get_params(last_state)
    print("Finish training guide. Extract samples...")
    guide_samples = guide.sample_posterior(random.PRNGKey(0), params,
                                           sample_shape=(args.num_samples,))['x'].copy()

    transform = guide.get_transform(params)
    _, potential_fn, constrain_fn = initialize_model(random.PRNGKey(2), dual_moon_model)
    transformed_potential_fn = partial(transformed_potential_energy, potential_fn, transform)
    transformed_constrain_fn = lambda x: constrain_fn(transform(x))  # noqa: E731

    print("\nStart NeuTra HMC...")
    nuts_kernel = NUTS(potential_fn=transformed_potential_fn)
    mcmc = MCMC(nuts_kernel, args.num_warmup, args.num_samples)
    init_params = np.zeros(guide.latent_size)
    mcmc.run(random.PRNGKey(3), init_params=init_params)
    mcmc.print_summary()
    zs = mcmc.get_samples()
    print("Transform samples into unwarped space...")
    samples = vmap(transformed_constrain_fn)(zs)
    summary(tree_map(lambda x: x[None, ...], samples))
    samples = samples['x'].copy()

    # make plots

    # guide samples (for plotting)
    guide_base_samples = dist.Normal(np.zeros(2), 1.).sample(random.PRNGKey(4), (1000,))
    guide_trans_samples = vmap(transformed_constrain_fn)(guide_base_samples)['x']

    x1 = np.linspace(-3, 3, 100)
    x2 = np.linspace(-3, 3, 100)
    X1, X2 = np.meshgrid(x1, x2)
    P = np.exp(DualMoonDistribution().log_prob(np.stack([X1, X2], axis=-1)))

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
    sns.kdeplot(guide_samples[:, 0], guide_samples[:, 1], n_levels=30, ax=ax2)
    ax2.set(xlim=[-3, 3], ylim=[-3, 3],
            xlabel='x0', ylabel='x1', title='Posterior using AutoBNAFNormal guide')

    sns.scatterplot(guide_base_samples[:, 0], guide_base_samples[:, 1], ax=ax3,
                    hue=guide_trans_samples[:, 0] < 0.)
    ax3.set(xlim=[-3, 3], ylim=[-3, 3],
            xlabel='x0', ylabel='x1', title='AutoBNAFNormal base samples (True=left moon; False=right moon)')

    ax4.contourf(X1, X2, P, cmap='OrRd')
    sns.kdeplot(vanilla_samples[:, 0], vanilla_samples[:, 1], n_levels=30, ax=ax4)
    ax4.plot(vanilla_samples[-50:, 0], vanilla_samples[-50:, 1], 'bo-', alpha=0.5)
    ax4.set(xlim=[-3, 3], ylim=[-3, 3],
            xlabel='x0', ylabel='x1', title='Posterior using vanilla HMC sampler')

    sns.scatterplot(zs[:, 0], zs[:, 1], ax=ax5, hue=samples[:, 0] < 0.,
                    s=30, alpha=0.5, edgecolor="none")
    ax5.set(xlim=[-5, 5], ylim=[-5, 5],
            xlabel='x0', ylabel='x1', title='Samples from the warped posterior - p(z)')

    ax6.contourf(X1, X2, P, cmap='OrRd')
    sns.kdeplot(samples[:, 0], samples[:, 1], n_levels=30, ax=ax6)
    ax6.plot(samples[-50:, 0], samples[-50:, 1], 'bo-', alpha=0.2)
    ax6.set(xlim=[-3, 3], ylim=[-3, 3],
            xlabel='x0', ylabel='x1', title='Posterior using NeuTra HMC sampler')

    plt.savefig("neutra.pdf")
    plt.close()


if __name__ == "__main__":
    assert numpyro.__version__.startswith('0.2.0')
    parser = argparse.ArgumentParser(description="NeuTra HMC")
    parser.add_argument('-n', '--num-samples', nargs='?', default=10000, type=int)
    parser.add_argument('--num-warmup', nargs='?', default=1000, type=int)
    parser.add_argument('--hidden-factor', nargs='?', default=50, type=int)
    parser.add_argument('--num-iters', nargs='?', default=20000, type=int)
    parser.add_argument('--device', default='cpu', type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()
    main(args)
