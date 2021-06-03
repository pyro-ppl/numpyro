import pickle
import sys
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp
from jax import random

import numpyro
from numpyro.contrib.funsor import config_enumerate
from numpyro.distributions import Dirichlet, Gamma, Uniform, VonMises, Beta, Categorical, Sine, SineSkewed
from numpyro.infer import NUTS, init_to_median, MCMC

np.set_printoptions(threshold=sys.maxsize)

AMINO_ACIDS = ['M', 'N', 'I', 'F', 'E', 'L', 'R', 'D', 'G', 'K', 'Y', 'T', 'H', 'S', 'P', 'A', 'V', 'Q', 'W', 'C']


@config_enumerate
def sine_model(data, num_mix_comp=2):
    # Mixture prior
    mix_weights = numpyro.sample('mix_weights', Dirichlet(jnp.ones((num_mix_comp,))))

    # Hprior BvM
    # Bayesian Inference and Decision Theory by Kathryn Blackmond Laskey
    beta_mean_phi = numpyro.sample('beta_mean_phi', Uniform(0., 1.))
    beta_prec_phi = numpyro.sample('beta_prec_phi', Gamma(1., 1 / 20.))  # shape, rate
    halpha_phi = beta_mean_phi * beta_prec_phi
    beta_mean_psi = numpyro.sample('beta_mean_psi', Uniform(0, 1.))
    beta_prec_psi = numpyro.sample('beta_prec_psi', Gamma(1., 1 / 20.))  # shape, rate
    halpha_psi = beta_mean_psi * beta_prec_psi

    with numpyro.plate('mixture', num_mix_comp):
        # BvM priors
        phi_loc = numpyro.sample('phi_loc', VonMises(pi, 2.))
        psi_loc = numpyro.sample('psi_loc', VonMises(-pi / 2, .2))
        phi_conc = numpyro.sample('phi_conc', Beta(halpha_phi, beta_prec_phi - halpha_phi))
        psi_conc = numpyro.sample('psi_conc', Beta(halpha_psi, beta_prec_psi - halpha_psi))
        corr_scale = numpyro.sample('corr_scale', Beta(2., 15.))

    with numpyro.plate('obs_plate', len(data), dim=-1):
        assign = numpyro.sample('mix_comp', Categorical(mix_weights), infer={"enumerate": "parallel"})
        sine = Sine(phi_loc=phi_loc[assign], psi_loc=psi_loc[assign],
                    phi_concentration=750 * phi_conc[assign],
                    psi_concentration=750 * psi_conc[assign],
                    weighted_correlation=corr_scale[assign])
        return numpyro.sample('phi_psi', sine, obs=data)


@config_enumerate
def ss_model(data, num_mix_comp=2):
    # Mixture prior
    mix_weights = numpyro.sample('mix_weights', Dirichlet(jnp.ones((num_mix_comp,))))

    # Hprior BvM
    # Bayesian Inference and Decision Theory by Kathryn Blackmond Laskey
    beta_mean_phi = numpyro.sample('beta_mean_phi', Uniform(0., 1.))
    beta_prec_phi = numpyro.sample('beta_prec_phi', Gamma(1., 1 / 20.))  # shape, rate
    halpha_phi = beta_mean_phi * beta_prec_phi
    beta_mean_psi = numpyro.sample('beta_mean_psi', Uniform(0, 1.))
    beta_prec_psi = numpyro.sample('beta_prec_psi', Gamma(1., 1 / 20.))  # shape, rate
    halpha_psi = beta_mean_psi * beta_prec_psi

    with numpyro.plate('mixture', num_mix_comp):
        # BvM priors
        phi_loc = numpyro.sample('phi_loc', VonMises(pi, 2.))
        psi_loc = numpyro.sample('psi_loc', VonMises(-pi / 2, .2))
        phi_conc = numpyro.sample('phi_conc', Beta(halpha_phi, beta_prec_phi - halpha_phi))
        psi_conc = numpyro.sample('psi_conc', Beta(halpha_psi, beta_prec_psi - halpha_psi))
        corr_scale = numpyro.sample('corr_scale', Beta(2., 15.))

        skew_phi = numpyro.sample('skew_phi', Uniform(-1., 1.))
        psi_bound = 1 - jnp.abs(skew_phi)
        skew_psi = numpyro.sample('skew_psi', Uniform(-1., 1.))
        skewness = jnp.stack((skew_phi, psi_bound * skew_psi), axis=-1)
        assert skewness.shape == (num_mix_comp, 2)

    with numpyro.plate('obs_plate', len(data), dim=-1):
        assign = numpyro.sample('mix_comp', Categorical(mix_weights), infer={"enumerate": "parallel"})
        sine = Sine(phi_loc=phi_loc[assign], psi_loc=psi_loc[assign],
                    phi_concentration=150 * phi_conc[assign],
                    psi_concentration=150 * psi_conc[assign],
                    weighted_correlation=corr_scale[assign])
        return numpyro.sample('phi_psi', SineSkewed(sine, skewness[assign]), obs=data)


def run_hmc(model, data, num_mix_comp, num_samples):
    rng_key = random.PRNGKey(0)
    kernel = NUTS(model, init_strategy=init_to_median(), max_tree_depth=7)
    mcmc = MCMC(kernel, num_samples=num_samples, num_warmup=num_samples // 5)
    mcmc.run(rng_key, data, num_mix_comp)
    mcmc.print_summary()
    post_samples = mcmc.get_samples()
    return post_samples


def fetch_aa_dihedrals(split='train', subsample_to=1000_000, shuffle=None):
    file = Path(__file__).parent / 'data/9mer_fragments_processed.pkl'
    data = pickle.load(file.open('rb'))[split]['sequences']
    data_aa = np.argmax(data[..., :20], -1)
    data = {aa: data[..., -2:][data_aa == i] for i, aa in enumerate(AMINO_ACIDS)}
    if shuffle is None:
        shuffles = {k: np.random.permutation(np.arange(v.shape[0]))[:min(subsample_to, v.shape[0])] for k, v in
                    data.items()}
    data = {aa: aa_data[shuffles[aa]] for aa, aa_data in data.items()}
    data = {aa: jnp.array(aa_data, dtype=float) for aa, aa_data in data.items()}
    return data


def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den * x / number))
        com = gcd(num, den)
        (num, den) = (int(num / com), int(den / com))
        if den == 1:
            if num == 0:
                return r'$0$'
            if num == 1:
                return r'$%s$' % latex
            elif num == -1:
                return r'$-%s$' % latex
            else:
                return r'$%s%s$' % (num, latex)
        else:
            if num == 1:
                return r'$\frac{%s}{%s}$' % (latex, den)
            elif num == -1:
                return r'$\frac{-%s}{%s}$' % (latex, den)
            else:
                return r'$\frac{%s%s}{%s}$' % (num, latex, den)

    return _multiple_formatter


def kde_ramachandran_plot(pred_data, data, aas, file_name='ssbvm_mixture.pdf'):
    fig, axs = plt.subplots(1, len(aas))
    for ax, aa in zip(axs, aas):
        aa_data = data[aa]

        ax.scatter(aa_data[:, 0], aa_data[:, 1], color='k', s=1)
        aa_data = pred_data[aa]
        ax.hexbin(aa_data[:, 0], aa_data[:, 1], cmap="Purples", extent=[-pi, pi, -pi, pi], alpha=.5)

        # label the contours
        ax.set_xlabel('$\phi$')
        ax.set_xlim([-pi, pi])
        ax.set_ylim([-pi, pi])
        ax.set_aspect('equal', 'box')

        ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

    axs[0].set_ylabel('$\psi$')
    axs[0].yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    axs[0].yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    axs[0].yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

    for ax in axs[0:]:
        ax.tick_params(labelleft=False)

    fig.tight_layout()
    plt.savefig(file_name, dvi=300, bbox_inches='tight')
    plt.clf()


def main(num_mix_start=3, num_mix_end=12, num_samples=1000, aas=('P', 'S', 'G')):
    data = fetch_aa_dihedrals(subsample_to=5_000)
    kde_ramachandran_plot(data, data, aas)
    posterior_samples = {aa: {num_mix_comp: {'sine': run_hmc(sine_model, data[aa], num_mix_comp, num_samples),
                                             'ss': run_hmc(ss_model, data[aa], num_mix_comp, num_samples)} for
                              num_mix_comp in range(num_mix_start, num_mix_end)} for aa in aas}


if __name__ == '__main__':
    main()
