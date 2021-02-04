import argparse
import time
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nn
from flax.nn.activation import tanh
from jax import random
from jax import vmap

import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.contrib.module import random_flax_module
from numpyro.infer import HMC, HMCECS, MCMC, NUTS, SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.hmc_gibbs import taylor_proxy

uci_base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'


def visualize(alg, train_data, train_obs, samples, num_samples):
    # helper function for prediction
    def predict(model, rng_key, samples, *args, **kwargs):
        model = handlers.substitute(handlers.seed(model, rng_key), samples)
        # note that Y will be sampled in the model because we pass Y=None here
        model_trace = handlers.trace(model).get_trace(*args, **kwargs)
        return model_trace['obs']['value']

    test_data = np.linspace(-2, 2, 500).reshape(-1, 1)
    vmap_args = (samples, random.split(random.PRNGKey(1), num_samples))
    predictions = vmap(lambda samples, rng_key: predict(model, rng_key, samples, test_data))(*vmap_args)
    predictions = predictions[..., 0]
    fs = 14

    m = predictions.mean(0)
    percentiles = np.percentile(predictions, [2.5, 97.5], axis=0)

    f, ax = plt.subplots(1, 1, figsize=(8, 4))

    # Get upper and lower confidence bounds
    lower, upper = (percentiles[0, :]).flatten(), (percentiles[1, :]).flatten()

    # Plot training data as black stars
    ax.plot(train_data, train_obs, 'x', marker='x', color='forestgreen', rasterized=True, label='Observed Data')
    # Plot predictive means as blue line
    ax.plot(test_data, m, 'b', rasterized=True, label="Mean Prediction")
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_data, lower, upper, alpha=0.5, rasterized=True, label='95% C.I.')
    ax.set_ylim([-2.5, 2.5])
    ax.set_xlim([-2, 2])
    plt.grid()
    ax.legend(fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)

    plt.tight_layout()
    plt.savefig(f'plots/regression_{alg}.pdf', rasterized=True)

    plt.show()


def load_agw_1d(get_feats=False):
    def features(x):
        return np.hstack([x[:, None] / 2.0, (x[:, None] / 2.0) ** 2])

    data = np.load(str(Path(__file__).parent / 'data' / 'data.npy'))
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


class Network(nn.Module):
    def apply(self, x, out_channels):
        l1 = tanh(nn.Dense(x, features=100))
        l2 = tanh(nn.Dense(l1, features=100))
        means = nn.Dense(l2, features=out_channels)
        return means


def nonlin(x):
    return tanh(x)


def model(data, obs=None, subsample_size=None):
    module = Network.partial(out_channels=1)
    net = random_flax_module('fnn', module, dist.Normal(0, 1.), input_shape=data.shape[1])

    prec_obs = numpyro.sample("prec_obs", dist.LogNormal(jnp.log(110.4), .0001))
    sigma_obs = 1.0 / jnp.sqrt(prec_obs)  # prior

    with numpyro.plate('N', data.shape[0], subsample_size=subsample_size) as idx:
        numpyro.sample('obs', dist.Normal(net(data[idx]), sigma_obs), obs=obs[idx])


def benchmark_hmc(args, features, labels):
    features = jnp.array(features)
    labels = jnp.array(labels)
    start = time.time()
    rng_key, ref_key = random.split(random.PRNGKey(1))
    subsample_size = 40
    guide = AutoNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
    params, losses = svi.run(random.PRNGKey(2), 2000, features, labels, subsample_size)
    plt.plot(losses)
    plt.show()
    ref_params = svi.guide.sample_posterior(ref_key, params, (1,))
    print(ref_params)
    if args.alg == "HMC":
        step_size = jnp.sqrt(0.5 / features.shape[0])
        trajectory_length = step_size * args.num_steps
        kernel = HMC(model, step_size=step_size, trajectory_length=trajectory_length, adapt_step_size=False,
                     dense_mass=args.dense_mass)
        subsample_size = None
    elif args.alg == "NUTS":
        kernel = NUTS(model, dense_mass=args.dense_mass)
        subsample_size = None
    elif args.alg == "HMCECS":
        subsample_size = 40
        inner_kernel = NUTS(model, init_strategy=init_to_value(values=ref_params),
                            dense_mass=args.dense_mass)
        kernel = HMCECS(inner_kernel, num_blocks=100, proxy=taylor_proxy(ref_params))
    else:
        raise ValueError('Alg not in HMC, NUTS, HMCECS.')
    mcmc = MCMC(kernel, args.num_warmup, args.num_samples)
    mcmc.run(rng_key, features, labels, subsample_size, extra_fields=("accept_prob",))
    print("Mean accept prob:", jnp.mean(mcmc.get_extra_fields()["accept_prob"]))
    mcmc.print_summary(exclude_deterministic=False)
    print('\nMCMC elapsed time:', time.time() - start)
    return mcmc.get_samples()


def main(args):
    data, obs = load_agw_1d()
    samples = benchmark_hmc(args, data, obs)
    visualize(args.alg, data, obs, samples, args.num_samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-samples', default=1000, type=int, help='number of samples')
    parser.add_argument('--num-warmup', default=200, type=int, help='number of warmup steps')
    parser.add_argument('--num-steps', default=10, type=int, help='number of steps (for "HMC")')
    parser.add_argument('--num-chains', nargs='?', default=1, type=int)
    parser.add_argument('--alg', default='NUTS', type=str,
                        help='whether to run "HMC", "NUTS", or "HMCECS"')
    parser.add_argument('--dense-mass', action="store_true")
    parser.add_argument('--x64', action="store_true")
    parser.add_argument('--device', default='gpu', type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)
    if args.x64:
        numpyro.enable_x64()

    main(args)
