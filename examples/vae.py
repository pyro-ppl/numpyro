# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Variational Autoencoder
=======================
"""

import argparse
import inspect
import os
import time

import matplotlib.pyplot as plt

from jax import jit, lax, random
from jax.experimental import stax
import jax.numpy as jnp
from jax.random import PRNGKey

import numpyro
from numpyro import optim
import numpyro.distributions as dist
from numpyro.examples.datasets import MNIST, load_dataset
from numpyro.infer import SVI, Trace_ELBO

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(inspect.getfile(lambda: None)),
                              '.results'))
os.makedirs(RESULTS_DIR, exist_ok=True)


def encoder(hidden_dim, z_dim):
    return stax.serial(
        stax.Dense(hidden_dim, W_init=stax.randn()), stax.Softplus,
        stax.FanOut(2),
        stax.parallel(stax.Dense(z_dim, W_init=stax.randn()),
                      stax.serial(stax.Dense(z_dim, W_init=stax.randn()), stax.Exp)),
    )


def decoder(hidden_dim, out_dim):
    return stax.serial(
        stax.Dense(hidden_dim, W_init=stax.randn()), stax.Softplus,
        stax.Dense(out_dim, W_init=stax.randn()), stax.Sigmoid,
    )


def model(batch, hidden_dim=400, z_dim=100):
    batch = jnp.reshape(batch, (batch.shape[0], -1))
    batch_dim, out_dim = jnp.shape(batch)
    decode = numpyro.module('decoder', decoder(hidden_dim, out_dim), (batch_dim, z_dim))
    z = numpyro.sample('z', dist.Normal(jnp.zeros((z_dim,)), jnp.ones((z_dim,))))
    img_loc = decode(z)
    return numpyro.sample('obs', dist.Bernoulli(img_loc), obs=batch)


def guide(batch, hidden_dim=400, z_dim=100):
    batch = jnp.reshape(batch, (batch.shape[0], -1))
    batch_dim, out_dim = jnp.shape(batch)
    encode = numpyro.module('encoder', encoder(hidden_dim, z_dim), (batch_dim, out_dim))
    z_loc, z_std = encode(batch)
    z = numpyro.sample('z', dist.Normal(z_loc, z_std))
    return z


@jit
def binarize(rng_key, batch):
    return random.bernoulli(rng_key, batch).astype(batch.dtype)


def main(args):
    encoder_nn = encoder(args.hidden_dim, args.z_dim)
    decoder_nn = decoder(args.hidden_dim, 28 * 28)
    adam = optim.Adam(args.learning_rate)
    svi = SVI(model, guide, adam, Trace_ELBO(), hidden_dim=args.hidden_dim, z_dim=args.z_dim)
    rng_key = PRNGKey(0)
    train_init, train_fetch = load_dataset(MNIST, batch_size=args.batch_size, split='train')
    test_init, test_fetch = load_dataset(MNIST, batch_size=args.batch_size, split='test')
    num_train, train_idx = train_init()
    rng_key, rng_key_binarize, rng_key_init = random.split(rng_key, 3)
    sample_batch = binarize(rng_key_binarize, train_fetch(0, train_idx)[0])
    svi_state = svi.init(rng_key_init, sample_batch)

    @jit
    def epoch_train(svi_state, rng_key):
        def body_fn(i, val):
            loss_sum, svi_state = val
            rng_key_binarize = random.fold_in(rng_key, i)
            batch = binarize(rng_key_binarize, train_fetch(i, train_idx)[0])
            svi_state, loss = svi.update(svi_state, batch)
            loss_sum += loss
            return loss_sum, svi_state

        return lax.fori_loop(0, num_train, body_fn, (0., svi_state))

    @jit
    def eval_test(svi_state, rng_key):
        def body_fun(i, loss_sum):
            rng_key_binarize = random.fold_in(rng_key, i)
            batch = binarize(rng_key_binarize, test_fetch(i, test_idx)[0])
            # FIXME: does this lead to a requirement for an rng_key arg in svi_eval?
            loss = svi.evaluate(svi_state, batch) / len(batch)
            loss_sum += loss
            return loss_sum

        loss = lax.fori_loop(0, num_test, body_fun, 0.)
        loss = loss / num_test
        return loss

    def reconstruct_img(epoch, rng_key):
        img = test_fetch(0, test_idx)[0][0]
        plt.imsave(os.path.join(RESULTS_DIR, 'original_epoch={}.png'.format(epoch)), img, cmap='gray')
        rng_key_binarize, rng_key_sample = random.split(rng_key)
        test_sample = binarize(rng_key_binarize, img)
        params = svi.get_params(svi_state)
        z_mean, z_var = encoder_nn[1](params['encoder$params'], test_sample.reshape([1, -1]))
        z = dist.Normal(z_mean, z_var).sample(rng_key_sample)
        img_loc = decoder_nn[1](params['decoder$params'], z).reshape([28, 28])
        plt.imsave(os.path.join(RESULTS_DIR, 'recons_epoch={}.png'.format(epoch)), img_loc, cmap='gray')

    for i in range(args.num_epochs):
        rng_key, rng_key_train, rng_key_test, rng_key_reconstruct = random.split(rng_key, 4)
        t_start = time.time()
        num_train, train_idx = train_init()
        _, svi_state = epoch_train(svi_state, rng_key_train)
        rng_key, rng_key_test, rng_key_reconstruct = random.split(rng_key, 3)
        num_test, test_idx = test_init()
        test_loss = eval_test(svi_state, rng_key_test)
        reconstruct_img(i, rng_key_reconstruct)
        print("Epoch {}: loss = {} ({:.2f} s.)".format(i, test_loss, time.time() - t_start))


if __name__ == '__main__':
    assert numpyro.__version__.startswith('0.4.0')
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=15, type=int, help='number of training epochs')
    parser.add_argument('-lr', '--learning-rate', default=1.0e-3, type=float, help='learning rate')
    parser.add_argument('-batch-size', default=128, type=int, help='batch size')
    parser.add_argument('-z-dim', default=50, type=int, help='size of latent')
    parser.add_argument('-hidden-dim', default=400, type=int, help='size of hidden layer in encoder/decoder networks')
    args = parser.parse_args()
    main(args)
