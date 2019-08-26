import argparse
import os
import time

import matplotlib.pyplot as plt

from jax import jit, lax, random
from jax.experimental import optimizers, stax
import jax.numpy as np
from jax.random import PRNGKey

import numpyro
import numpyro.distributions as dist
from numpyro.examples.datasets import MNIST, load_dataset
from numpyro.svi import elbo, svi

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
        stax.Dense(out_dim, W_init=stax.randn()), stax.Sigmoid,
    )


def model(batch, **kwargs):
    decode = kwargs['decode']
    decoder_params = numpyro.param('decoder', None)
    z_dim = kwargs['z_dim']
    batch = np.reshape(batch, (batch.shape[0], -1))
    z = numpyro.sample('z', dist.Normal(np.zeros((z_dim,)), np.ones((z_dim,))))
    img_loc = decode(decoder_params, z)
    return numpyro.sample('obs', dist.Bernoulli(img_loc), obs=batch)


def guide(batch, **kwargs):
    encode = kwargs['encode']
    encoder_params = numpyro.param('encoder', None)
    batch = np.reshape(batch, (batch.shape[0], -1))
    z_loc, z_std = encode(encoder_params, batch)
    z = numpyro.sample('z', dist.Normal(z_loc, z_std))
    return z


@jit
def binarize(rng, batch):
    return random.bernoulli(rng, batch).astype(batch.dtype)


def main(args):
    encoder_init, encode = encoder(args.hidden_dim, args.z_dim)
    decoder_init, decode = decoder(args.hidden_dim, 28 * 28)
    opt_init, opt_update, get_opt_params = optimizers.adam(args.learning_rate)
    svi_init, svi_update, svi_eval = svi(model, guide, elbo, opt_init, opt_update, get_opt_params,
                                         encode=encode, decode=decode, z_dim=args.z_dim)
    rng = PRNGKey(0)
    train_init, train_fetch = load_dataset(MNIST, batch_size=args.batch_size, split='train')
    test_init, test_fetch = load_dataset(MNIST, batch_size=args.batch_size, split='test')
    num_train, train_idx = train_init()
    rng, rng_enc, rng_dec, rng_binarize, rng_init = random.split(rng, 5)
    _, encoder_params = encoder_init(rng_enc, (args.batch_size, 28 * 28))
    _, decoder_params = decoder_init(rng_dec, (args.batch_size, args.z_dim))
    params = {'encoder': encoder_params, 'decoder': decoder_params}
    sample_batch = binarize(rng_binarize, train_fetch(0, train_idx)[0])
    opt_state, get_params = svi_init(rng_init, (sample_batch,), (sample_batch,), params)

    @jit
    def epoch_train(opt_state, rng):
        def body_fn(i, val):
            loss_sum, opt_state, rng = val
            rng, rng_binarize = random.split(rng)
            batch = binarize(rng_binarize, train_fetch(i, train_idx)[0])
            # TODO: we will want to merge (i, rng, opt_state) into `svi_state`
            # Here the index `i` is reseted after each epoch, which causes no
            # problem for static learning rate, but it is not a right way for
            # scheduled learning rate.
            loss, opt_state, rng = svi_update(i, rng, opt_state, (batch,), (batch,),)
            loss_sum += loss
            return loss_sum, opt_state, rng

        return lax.fori_loop(0, num_train, body_fn, (0., opt_state, rng))

    @jit
    def eval_test(opt_state, rng):
        def body_fun(i, val):
            loss_sum, rng = val
            rng, rng_binarize, rng_eval = random.split(rng, 3)
            batch = binarize(rng_binarize, test_fetch(i, test_idx)[0])
            loss = svi_eval(rng_eval, opt_state, (batch,), (batch,)) / len(batch)
            loss_sum += loss
            return loss_sum, rng

        loss, _ = lax.fori_loop(0, num_test, body_fun, (0., rng))
        loss = loss / num_test
        return loss

    def reconstruct_img(epoch, rng):
        img = test_fetch(0, test_idx)[0][0]
        plt.imsave(os.path.join(RESULTS_DIR, 'original_epoch={}.png'.format(epoch)), img, cmap='gray')
        rng_binarize, rng_sample = random.split(rng)
        test_sample = binarize(rng_binarize, img)
        params = get_params(opt_state)
        z_mean, z_var = encode(params['encoder'], test_sample.reshape([1, -1]))
        z = dist.Normal(z_mean, z_var).sample(rng_sample)
        img_loc = decode(params['decoder'], z).reshape([28, 28])
        plt.imsave(os.path.join(RESULTS_DIR, 'recons_epoch={}.png'.format(epoch)), img_loc, cmap='gray')

    for i in range(args.num_epochs):
        t_start = time.time()
        num_train, train_idx = train_init()
        _, opt_state, rng = epoch_train(opt_state, rng)
        rng, rng_test, rng_reconstruct = random.split(rng, 3)
        num_test, test_idx = test_init()
        test_loss = eval_test(opt_state, rng_test)
        reconstruct_img(i, rng_reconstruct)
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
