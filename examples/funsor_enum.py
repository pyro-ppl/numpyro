import jax.numpy as jnp
# import numpyro.contrib.funsor
import numpyro as pyro
from numpyro import distributions as dist
from numpyro import handlers, infer, optim
from jax import random
from numpyro.contrib.funsor import plate

import funsor
funsor.set_backend("jax")

# Start from this source of randomness. We will split keys for subsequent operations.
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

data = random.uniform(rng_key, (5,))

def model(data):
    with plate("data", len(data)):
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

# with pyro_backend("numpyro.funsor"):
loss = svi.run(rng_key_, 10, data)
print(loss)
