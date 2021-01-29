from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flax import nn
from flax.nn.activation import tanh
from jax import random, vmap

import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.contrib.module import random_flax_module
from numpyro.infer import MCMC, NUTS, init_to_median

uci_base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'


def visualize(train_data, train_obs, test_data, predictions):
    fs = 16

    m = predictions.mean(0)
    s = predictions.std(0)
    # s_al = (pred_list[200:].var(0).to('cpu') + tau_out ** -1) ** 0.5

    f, ax = plt.subplots(1, 1, figsize=(8, 4))

    # Get upper and lower confidence bounds
    lower, upper = (m - s * 2).flatten(), (m + s * 2).flatten()
    # + aleotoric
    # lower_al, upper_al = (m - s_al*2).flatten(), (m + s_al*2).flatten()

    # Plot training data as black stars
    ax.plot(train_data, train_obs, 'k*', rasterized=True)
    # Plot predictive means as blue line
    ax.plot(test_data, m, 'b', rasterized=True)
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_data, lower, upper, alpha=0.5, rasterized=True)
    # ax.fill_between(X_test.flatten().numpy(), lower_al.numpy(), upper_al.numpy(), alpha=0.2, rasterized=True)
    ax.set_ylim([-2, 2])
    ax.set_xlim([-2, 2])
    plt.grid()
    ax.legend(['Observed Data', 'Mean', 'Epistemic'], fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)

    bbox = {'facecolor': 'white', 'alpha': 0.8, 'pad': 1, 'boxstyle': 'round', 'edgecolor': 'black'}

    plt.tight_layout()
    # plt.savefig('plots/full_hmc.pdf', rasterized=True)

    plt.show()


def load_agw_1d(get_feats=False):
    def features(x):
        return np.hstack([x[:, None] / 2.0, (x[:, None] / 2.0) ** 2])

    data = np.load(str(Path(__file__).parent / 'hmcecs' / 'data' / 'data.npy'))
    x, y = data[:, 0], data[:, 1]
    y = y[:, None]
    f = features(x)

    x_means, x_stds = x.mean(axis=0), x.std(axis=0)
    y_means, y_stds = y.mean(axis=0), y.std(axis=0)
    f_means, f_stds = f.mean(axis=0), f.std(axis=0)

    X = ((x - x_means) / x_stds).astype(np.float32)
    Y = ((y - y_means) / y_stds).astype(np.float32)
    F = ((f - f_means) / f_stds).astype(np.float32)

    if get_feats:
        return F, Y

    return X[:, None], Y


def protein():
    # from hughsalimbeni/bayesian_benchmarks
    # N, D, name = 45730, 9, 'protein'
    url = uci_base_url + '00265/CASP.csv'

    data = pd.read_csv(url).values
    return data[:, 1:], data[:, 0].reshape(-1, 1)


class Network(nn.Module):
    def apply(self, x, out_channels):
        l1 = tanh(nn.Dense(x, features=100))
        l2 = tanh(nn.Dense(l1, features=100))
        l3 = tanh(nn.Dense(l2, features=100))
        means = nn.Dense(l3, features=out_channels)
        return means


def model(data, obs=None):
    module = Network.partial(out_channels=1)
    net = random_flax_module('fnn', module, dist.Normal(0, 1.), input_shape=data.shape)

    if obs is not None:
        obs = obs[..., None]

    prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / jnp.sqrt(prec_obs)  # prior

    numpyro.sample('obs', dist.Normal(net(data), sigma_obs), obs=obs)


def hmc(dataset, data, obs, warmup, num_sample):
    kernel = NUTS(model, init_strategy=init_to_median)
    mcmc = MCMC(kernel, warmup, num_sample)
    mcmc._compile(random.PRNGKey(0), data, obs, extra_fields=("num_steps",))
    mcmc.run(random.PRNGKey(0), data, obs, extra_fields=('num_steps',))
    return mcmc.get_samples()


# helper function for prediction
def predict(model, rng_key, samples, *args, **kwargs):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = handlers.trace(model).get_trace(*args, **kwargs)
    return model_trace['obs']['value']


def main():
    data, obs = load_agw_1d()
    warmup = 100
    num_samples = 100
    test_data = np.linspace(-2, 2, 500).reshape(-1, 1)
    samples = hmc('protein', data, obs, warmup, num_samples)
    vmap_args = (samples, random.split(random.PRNGKey(1), num_samples))
    predictions = vmap(lambda samples, rng_key: predict(model, rng_key, samples, test_data))(*vmap_args)
    predictions = predictions[..., 0]
    visualize(data, obs, np.squeeze(test_data), predictions)


if __name__ == '__main__':
    main()
