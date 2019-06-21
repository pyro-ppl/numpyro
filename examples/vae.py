import argparse
import os
import time

import matplotlib.pyplot as plt

from jax import jit, lax, random
from jax.experimental import optimizers, stax
import jax.numpy as np
from jax.random import PRNGKey

import numpyro.distributions as dist
from numpyro.examples.datasets import MNIST, load_dataset
from numpyro.handlers import param, sample
from numpyro.svi import elbo, svi


def sigmoid(x):
    return 1 / (1 + np.exp(x))


# TODO: move to JAX
def _elemwise_no_params(fun, **kwargs):
    def init_fun(rng, input_shape): return input_shape, ()

    def apply_fun(params, inputs, rng=None): return fun(inputs, **kwargs)

    return init_fun, apply_fun


Sigmoid = _elemwise_no_params(sigmoid)


RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
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
        stax.Dense(out_dim, W_init=stax.randn()), Sigmoid,
    )


def model(batch, **kwargs):
    decode = kwargs['decode']
    decoder_params = param('decoder', None)
    z_dim = kwargs['z_dim']
    batch = np.reshape(batch, (batch.shape[0], -1))
    z = sample('z', dist.Normal(np.zeros((z_dim,)), np.ones((z_dim,))))
    img_loc = decode(decoder_params, z)
    return sample('obs', dist.Bernoulli(img_loc), obs=batch)


def guide(batch, **kwargs):
    encode = kwargs['encode']
    encoder_params = param('encoder', None)
    batch = np.reshape(batch, (batch.shape[0], -1))
    z_loc, z_std = encode(encoder_params, batch)
    z = sample('z', dist.Normal(z_loc, z_std))
    return z


@jit
def binarize(rng, batch):
    rng, = random.split(rng, 1)
    return rng, random.bernoulli(rng, batch).astype(batch.dtype)


def main(args):
    encoder_init, encode = encoder(args.hidden_dim, args.z_dim)
    decoder_init, decode = decoder(args.hidden_dim, 28 * 28)
    opt_init, opt_update, get_params = optimizers.adam(args.learning_rate)
    svi_init, svi_update, svi_eval = svi(model, guide, elbo, opt_init, opt_update, get_params,
                                         encode=encode, decode=decode, z_dim=args.z_dim)
    svi_update = jit(svi_update)
    rng = PRNGKey(0)
    train_init, train_fetch = load_dataset(MNIST, batch_size=args.batch_size, split='train')
    test_init, test_fetch = load_dataset(MNIST, batch_size=args.batch_size, split='test')
    num_train, train_idx = train_init()
    rng, rng_enc, rng_dec = random.split(rng, 3)
    _, encoder_params = encoder_init(rng_enc, (args.batch_size, 28 * 28))
    _, decoder_params = decoder_init(rng_dec, (args.batch_size, args.z_dim))
    params = {'encoder': encoder_params, 'decoder': decoder_params}
    rng, sample_batch = binarize(rng, train_fetch(0, train_idx)[0])
    opt_state, constrain_fn = svi_init(rng, (sample_batch,), (sample_batch,), params)
    rng, = random.split(rng, 1)

    @jit
    def epoch_train(opt_state, rng):
        def body_fn(i, val):
            loss_sum, opt_state, rng = val
            rng, batch = binarize(rng, train_fetch(i, train_idx)[0])
            loss, opt_state, rng = svi_update(i, rng, opt_state, (batch,), (batch,),)
            loss_sum += loss
            return loss_sum, opt_state, rng

        return lax.fori_loop(0, num_train, body_fn, (0., opt_state, rng))

    @jit
    def eval_test(opt_state, rng):
        def body_fun(i, val):
            loss_sum, rng = val
            rng, = random.split(rng, 1)
            rng, batch = binarize(rng, test_fetch(i, test_idx)[0])
            loss = svi_eval(rng, opt_state, (batch,), (batch,)) / len(batch)
            loss_sum += loss
            return loss_sum, rng

        loss, _ = lax.fori_loop(0, num_test, body_fun, (0., rng))
        loss = loss / num_test
        return loss

    def reconstruct_img(epoch):
        img = test_fetch(0, test_idx)[0][0]
        plt.imsave(os.path.join(RESULTS_DIR, 'original_epoch={}.png'.format(epoch)), img, cmap='gray')
        _, test_sample = binarize(rng, img)
        params = get_params(opt_state)
        z_mean, z_var = encode(params['encoder'], test_sample.reshape([1, -1]))
        z = dist.Normal(z_mean, z_var).sample(rng)
        img_loc = decode(params['decoder'], z).reshape([28, 28])
        plt.imsave(os.path.join(RESULTS_DIR, 'recons_epoch={}.png'.format(epoch)), img_loc, cmap='gray')

    for i in range(args.num_epochs):
        t_start = time.time()
        num_train, train_idx = train_init()
        _, opt_state, rng = epoch_train(opt_state, rng)
        rng, rng_test = random.split(rng, 2)
        num_test, test_idx = test_init()
        test_loss = eval_test(opt_state, rng_test)
        reconstruct_img(i)
        print("Epoch {}: loss = {} ({:.2f} s.)".format(i, test_loss, time.time() - t_start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=20, type=int, help='number of training epochs')
    parser.add_argument('-lr', '--learning-rate', default=1.0e-3, type=float, help='learning rate')
    parser.add_argument('-batch-size', default=128, type=int, help='batch size')
    parser.add_argument('-z-dim', default=50, type=int, help='size of latent')
    parser.add_argument('-hidden-dim', default=400, type=int, help='size of hidden layer in encoder/decoder networks')
    args = parser.parse_args()
    main(args)
