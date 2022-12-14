import jax.numpy as jnp
import numpyro.contrib.funsor
from pyroapi import distributions as dist
from pyroapi import handlers, infer, optim, pyro, pyro_backend
from jax import random

#  import funsor
#  funsor.set_backend("jax")

# Start from this source of randomness. We will split keys for subsequent operations.
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

data = random.uniform(rng_key, (5,))

def model(data):
    with pyro.plate("data", len(data)):
        x = pyro.sample(
            "x",
            dist.Categorical(jnp.ones(2)),
            infer={"enumerate": "parallel"},
        )
        pyro.sample("y", dist.Normal(jnp.array([1.0, 2.0])[x], 1), obs=data)

def guide(data):
    pass

optimizer = optim.Adam({"lr": 0.005})
elbo = infer.TraceEnum_ELBO(max_plate_nesting=1)
svi = infer.SVI(model, guide, optimizer, loss=elbo)

with pyro_backend("numpyro.funsor"):
    loss = svi.step(data)
    print(loss)
