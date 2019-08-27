import argparse

from jax import random
import jax.numpy as np
from jax.random import PRNGKey

import numpyro
from numpyro import optim
import numpyro.distributions as dist
from numpyro.svi import elbo, svi
from numpyro.util import fori_loop


def model(data):
    loc = numpyro.sample("loc", dist.Normal(0., 1.))
    numpyro.sample("obs", dist.Normal(loc, 1.), obs=data)


# Define a guide (i.e. variational distribution) with a Normal
# distribution over the latent random variable `loc`.
def guide():
    guide_loc = numpyro.param("guide_loc", 0.)
    guide_scale = np.exp(numpyro.param("guide_scale_log", 0.))
    numpyro.sample("loc", dist.Normal(guide_loc, guide_scale))


def main(args):
    # Generate some data.
    data = random.normal(PRNGKey(0), shape=(100,)) + 3.0

    # Construct an SVI object so we can do variational inference on our
    # model/guide pair.
    adam = optim.Adam(args.learning_rate)
    svi_init, svi_update, _ = svi(model, guide, elbo, adam)
    rng, rng_init = random.split(PRNGKey(0))
    opt_state, _ = svi_init(rng_init, model_args=(data,))

    # Training loop
    def body_fn(i, val):
        opt_state_, rng_ = val
        loss, opt_state_, rng_ = svi_update(rng_, opt_state_, model_args=(data,))
        return opt_state_, rng_

    opt_state, _ = fori_loop(0, args.num_steps, body_fn, (opt_state, rng))

    # Report the final values of the variational parameters
    # in the guide after training.
    params = adam.get_params(opt_state)
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
