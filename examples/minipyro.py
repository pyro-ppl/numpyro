# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse

from jax import random
import jax.numpy as jnp
from jax.random import PRNGKey

import numpyro
from numpyro import optim
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.util import fori_loop


def model(data):
    loc = numpyro.sample("loc", dist.Normal(0., 1.))
    numpyro.sample("obs", dist.Normal(loc, 1.), obs=data)


# Define a guide (i.e. variational distribution) with a Normal
# distribution over the latent random variable `loc`.
def guide(data):
    guide_loc = numpyro.param("guide_loc", 0.)
    guide_scale = jnp.exp(numpyro.param("guide_scale_log", 0.))
    numpyro.sample("loc", dist.Normal(guide_loc, guide_scale))


def main(args):
    # Generate some data.
    data = random.normal(PRNGKey(0), shape=(100,)) + 3.0

    # Construct an SVI object so we can do variational inference on our
    # model/guide pair.
    adam = optim.Adam(args.learning_rate)

    svi = SVI(model, guide, adam, Trace_ELBO(num_particles=100))
    svi_state = svi.init(PRNGKey(0), data)

    # Training loop
    def body_fn(i, val):
        svi_state, loss = svi.update(val, data)
        return svi_state

    svi_state = fori_loop(0, args.num_steps, body_fn, svi_state)

    # Report the final values of the variational parameters
    # in the guide after training.
    params = svi.get_params(svi_state)
    for name, value in params.items():
        print("{} = {}".format(name, value))

    # For this simple (conjugate) model we know the exact posterior. In
    # particular we know that the variational distribution should be
    # centered near 3.0. So let's check this explicitly.
    assert jnp.abs(params["guide_loc"] - 3.0) < 0.1


if __name__ == "__main__":
    assert numpyro.__version__.startswith('0.4.0')
    parser = argparse.ArgumentParser(description="Mini Pyro demo")
    parser.add_argument("-f", "--full-pyro", action="store_true", default=False)
    parser.add_argument("-n", "--num-steps", default=1001, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.02, type=float)
    args = parser.parse_args()
    main(args)
