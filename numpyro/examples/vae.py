import argparse

from jax.experimental import stax, optimizers
import jax.numpy as np
import jax.random as random
from jax.random import PRNGKey

from numpyro.examples.datasets import iter_dataset, MNIST
from numpyro.handlers import sample
import numpyro.distributions as dist
from numpyro.svi import svi, elbo


def sigmoid(x):
    return 1 / (1 + np.exp(x))


# TODO: move to JAX
def _elemwise_no_params(fun, **kwargs):
  init_fun = lambda input_shape: (input_shape, ())
  apply_fun = lambda params, inputs, rng=None: fun(inputs, **kwargs)
  return init_fun, apply_fun


Sigmoid = _elemwise_no_params(sigmoid)


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


def model(batch, decoder_params, z_dim, **kwargs):
    decode = kwargs['decode']
    batch = np.reshape(batch, (batch.shape[0], -1))
    z = sample('z', dist.norm(np.zeros(z_dim), np.ones(z_dim)))
    img_loc = decode(decoder_params, z)
    return sample('obs', dist.bernoulli(img_loc), obs=batch)


def guide(batch, encoder_params, **kwargs):
    encode = kwargs['encode']
    batch = np.reshape(batch, (batch.shape[0], -1))
    z_loc, z_std = encode(encoder_params, batch)
    z = sample('z', dist.norm(z_loc, z_std))
    return z


def main(args):
    encoder_init, encode = encoder(args.hidden_dim, args.z_dim)
    decoder_init, decode = decoder(args.hidden_dim, 28 * 28)
    opt_init, opt_update = optimizers.adam(args.learning_rate)
    svi_init, svi_update = svi(model, guide, elbo, opt_init, opt_update)
    rng = PRNGKey(0)
    for i in range(args.num_epochs):
        for j, (batch, label) in enumerate(iter_dataset(MNIST, batch_size=args.batch_size)):
            if i == 0 and j == 0:
                _, encoder_params = encoder_init((args.batch_size, 28 * 28))
                _, decoder_params = decoder_init((args.batch_size, args.z_dim))
                params = {'encoder': encoder_params, 'decoder': decoder_params}
                opt_state = svi_init(rng,
                                     model_args=(batch, decoder_params, args.z_dim),
                                     guide_args=(batch, encoder_params),
                                     kwargs={'encode': encode, 'decode': decode},
                                     params=params)
                rng, = random.split(rng, 1)
            params = optimizers.get_params(opt_state)
            loss, opt_state, rng = svi_update(i, opt_state, rng,
                                              model_args=(batch, params['decoder'], args.z_dim),
                                              guide_args=(batch, params['encoder']),
                                              kwargs={'encode': encode, 'decode': decode},
                                              )
            if i % 100 == 0:
                print("step {} loss = {}".format(i, loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=10, type=int, help='number of training epochs')
    parser.add_argument('-lr', '--learning-rate', default=1.0e-3, type=float, help='learning rate')
    parser.add_argument('-batch-size', default=128, type=int, help='batch size')
    parser.add_argument('-z-dim', default=50, type=int, help='size of latent')
    parser.add_argument('-hidden-dim', default=400, type=int, help='size of hidden layer in encoder/decoder networks')
    args = parser.parse_args()
    main(args)
