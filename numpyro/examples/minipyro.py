import argparse

import jax.numpy as np
from jax import lax, random
from jax.experimental import optimizers
from jax.random import PRNGKey

import numpyro.distributions as dist
from numpyro.handlers import param, sample
from numpyro.svi import elbo, svi


def model(data):
    loc = sample("loc", dist.norm(0., 1.))
    sample("obs", dist.norm(loc, 1.), obs=data)


# Define a guide (i.e. variational distribution) with a Normal
# distribution over the latent random variable `loc`.
def guide():
    guide_loc = param("guide_loc", 0.)
    guide_scale = np.exp(param("guide_scale_log", 0.))
    sample("loc", dist.norm(guide_loc, guide_scale))


def main(args):
    # Generate some data.
    data = random.normal(PRNGKey(0), shape=(100,)) + 3.0

    # Construct an SVI object so we can do variational inference on our
    # model/guide pair.
    opt_init, opt_update = optimizers.adam(args.learning_rate)
    svi_init, svi_update, _ = svi(model, guide, elbo, opt_init, opt_update)
    rng = PRNGKey(0)
    opt_state = svi_init(rng, model_args=(data,))

    # Training loop
    rng, = random.split(rng, 1)

    def body_fn(i, val):
        opt_state_, rng_ = val
        loss, opt_state_, rng_ = svi_update(i, opt_state_, rng_, model_args=(data,))
        return opt_state_, rng_

    opt_state, _ = lax.fori_loop(0, args.num_steps, body_fn, (opt_state, rng))

    # Report the final values of the variational parameters
    # in the guide after training.
    params = optimizers.get_params(opt_state)
    for name, value in params.items():
        print("{} = {}".format(name, value))

    # For this simple (conjugate) model we know the exact posterior. In
    # particular we know that the variational distribution should be
    # centered near 3.0. So let's check this explicitly.
    assert np.abs(params["guide_loc"] - 3.0) < 0.1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini Pyro demo")
    parser.add_argument("-f", "--full-pyro", action="store_true", default=False)
    parser.add_argument("-n", "--num-steps", default=1001, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.02, type=float)
    args = parser.parse_args()
    main(args)
