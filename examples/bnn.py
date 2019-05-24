import matplotlib
matplotlib.use('Agg')  # noqa: E402
import matplotlib.pyplot as plt

import argparse

import numpy as onp
import jax.numpy as np
import jax.random as random

import numpyro.distributions as dist
from numpyro.handlers import sample, seed, substitute, trace
from numpyro.hmc_util import initialize_model
from numpyro.mcmc import mcmc


"""
We demonstrate how to use NUTS to do inference on a simple (small)
Bayesian neural network with two hidden layers.
"""


# the non-linearity we use in our neural network
def nonlin(x):
    return np.tanh(x)


# a two-layer bayesian neural network with computational flow
# given by D_X => D_H => D_H => D_Y where D_H is the number of
# hidden units. (note we indicate tensor dimensions in the comments)
def model(X, Y, D_H):
    D_X = X.shape[1]
    D_Y = 1

    # sample first layer (we put unit normal priors on all weights)
    w1 = sample("w1", dist.Normal(np.zeros((D_X, D_H)), np.ones((D_X, D_H))))  # D_X D_H
    z1 = nonlin(np.matmul(X, w1))   # N D_H  <= first layer of activations

    # sample second layer
    w2 = sample("w2", dist.Normal(np.zeros((D_H, D_H)), np.ones((D_H, D_H))))  # D_H D_H
    z2 = nonlin(np.matmul(z1, w2))  # N D_H  <= second layer of activations

    # sample final layer of weights and neural network output
    w3 = sample("w3", dist.Normal(np.zeros((D_H, D_Y)), np.ones((D_H, D_Y))))  # D_H D_Y
    z3 = np.matmul(z2, w3)  # N D_Y  <= output of the neural network

    # we put a prior on the observation noise
    prec_obs = sample("prec_obs", dist.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / np.sqrt(prec_obs)

    # observe data
    sample("Y", dist.Normal(z3, sigma_obs), obs=Y)


# helper function for HMC inference
def run_inference(model, args, rng, X, Y, D_H):
    init_params, potential_fn, constrain_fn = initialize_model(rng, model, X, Y, D_H)
    hmc_states = mcmc(args.num_warmup, args.num_samples, init_params,
                      sampler='hmc', potential_fn=potential_fn, constrain_fn=constrain_fn)
    return hmc_states


# helper function for prediction
def predict(model, rng, latents, X, D_H):
    # we need to expand prec_obs so that the (hmc) samples dimension doesn't
    # conflict with the data dimension in the final model sample statement
    latents['prec_obs'] = latents['prec_obs'][:, None, None]
    model = substitute(seed(model, rng), latents)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = trace(model).get_trace(X=X, Y=None, D_H=D_H)
    return model_trace['Y']['value']


# create artificial regression dataset
def get_data(N=50, D_X=3, sigma_obs=0.05, N_test=400):
    D_Y = 1  # create 1d outputs
    onp.random.seed(0)
    X = 2.0 * np.arange(N) / N - 1.0
    X = np.power(X[:, None], np.arange(D_X))
    W = 0.5 * onp.random.randn(D_X)
    Y = np.dot(X, W) + 0.5 * np.power(0.5 + X[:, 1], 2.0) * np.sin(4.0 * X[:, 1])
    Y += sigma_obs * onp.random.randn(N)
    Y = Y[:, onp.newaxis]
    Y -= np.mean(Y)
    Y /= np.std(Y)

    assert X.shape == (N, D_X)
    assert Y.shape == (N, D_Y)

    X_test = 2.0 * np.arange(N_test) / N_test - 1.0
    X_test = np.power(X_test[:, None], np.arange(D_X))

    return X, Y, X_test


def main(args):
    N, D_X, D_H = args.num_data, 3, args.num_hidden
    X, Y, X_test = get_data(N=N, D_X=D_X)

    # do inference
    rng, rng_predict = random.split(random.PRNGKey(0))
    latents = run_inference(model, args, rng, X, Y, D_H)
    predictions = predict(model, rng_predict, latents, X_test, D_H)[:, :, 0]

    # make plots
    fig, ax = plt.subplots(1, 1)

    mean_prediction = np.mean(predictions, axis=0)
    percentiles = onp.percentile(predictions, [5.0, 95.0], axis=0)

    # plot training data
    ax.plot(X[:, 1], Y[:, 0], 'kx')
    # plot 90% confidence level of predictions
    ax.fill_between(X_test[:, 1], percentiles[0, :], percentiles[1, :], color='lightblue')
    # plot mean prediction
    ax.plot(X_test[:, 1], mean_prediction, 'blue', ls='solid', lw=2.0)

    plt.savefig('bnn_plot.pdf')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stochastic network")
    parser.add_argument("-n", "--num-samples", nargs="?", default=2000, type=int)
    parser.add_argument("--num-warmup", nargs='?', default=1000, type=int)
    parser.add_argument("--num-data", nargs='?', default=100, type=int)
    parser.add_argument("--num-hidden", nargs='?', default=5, type=int)
    args = parser.parse_args()
    main(args)
