# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Predator-Prey model with triangular transport map
# Based on "Transport map accelerated Markov chain Monte Carlo" by Parno and Marzouk (SIAM/ASA Journal on Uncertainty Quantification 2018)

# %%
# %env JAX_DEBUG_NANS=True
from jax.config import config
# config.update('jax_disable_jit', True)
import jax
import jax.numpy as np
import numpy as onp
import numpyro
from numpyro.contrib.autoguide import AutoDelta
from numpyro import distributions as dist
from numpyro.distributions import constraints
from numpyro.examples.runge_kutta import runge_kutta_4
from numpyro.infer import init_to_median, init_with_noise
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from collections import OrderedDict


# %%
rng_key = jax.random.PRNGKey(242)

# %% [markdown]
# ## Predator Prey Model

# %%
def predator_prey_step(t, state, r=0.6, k=100, s=1.2, a=25, u=0.5, v=0.3):
    prey = state[..., 0]
    predator = state[..., 1]
    sh = (prey * predator)/(a + prey + 1e-3)
    prey_upd = r * prey * (1 - prey / (k + 1e-3)) - s * sh
    predator_upd = u * sh - v * predator
    return np.stack((prey_upd, predator_upd), axis=-1)
num_time = 50
step_size = 0.1
num_steps = int(num_time / step_size)
dampening_rate = 0.9
lyapunov_scale = 1e-3
clip = lambda x: np.clip(x, -10.0, 10.0)
predator_prey = runge_kutta_4(predator_prey_step, step_size, num_steps, dampening_rate, 
                              lyapunov_scale, clip, unconstrain_fn=lambda _, x: np.where(np.logical_and(1e-30 < x, x < 10), np.log(np.expm1(np.clip(x,1e-30, 10))), x),
                              constrain_fn=lambda _, x: jax.nn.softplus(x))
predator_prey = jax.partial(predator_prey, rng_key)


# %%
indices = np.array([1, 11, 21, 31, 41])
res, lyapunov_loss = predator_prey(np.array([50., 5.]))
res = np.reshape(res, (num_time, num_steps // num_time, -1))[:, 0, :]
noise = jax.random.normal(rng_key, (1000,5,2)) * 10
data = (indices, res[indices] + noise)
data


# %%
def model(indices, observations):
    prior_dist = dist.Uniform(0.1, 50)
    prey0 = numpyro.sample('prey0', prior_dist)
    predator0 = numpyro.sample('predator0', prior_dist)
    r = numpyro.sample('r', prior_dist)
    k = numpyro.sample('k', prior_dist)
    s = numpyro.sample('s', prior_dist)
    a = numpyro.sample('a', prior_dist)
    u = numpyro.sample('u', prior_dist)
    v = numpyro.sample('v', prior_dist)
    ppres, lyapunov_loss = predator_prey(np.array([prey0, predator0]), r=r, k=k, s=s, a=a, u=u, v=v)
    ppres = np.reshape(ppres, (num_time, num_steps // num_time, -1))[:, 0, :]
    numpyro.factor('lyapunov_loss', lyapunov_loss)
    numpyro.sample('obs', dist.Normal(ppres[indices], 10.0).to_event(2), obs=observations)

# %% [markdown]
# ### SVI

# %%
svi = numpyro.infer.SVI(model, AutoDelta(model, init_strategy=init_to_median()), 
                        numpyro.optim.Adam(0.1), numpyro.infer.ELBO())
state = svi.init(rng_key, *data)
pbar = tqdm(range(1000))
for i in pbar:
    state, loss = svi.update(state, *data)
    pbar.set_description(f'SVI {loss}')


# %%
params = svi.get_params(state)
print(params)

# %% [markdown]
# ### Guide and Stein with Transport Maps

# %%
svgd = numpyro.infer.SVGD(model, AutoDelta(model, init_strategy=init_to_median()), 
                          numpyro.optim.Adam(0.01), numpyro.infer.ELBO(),
                          numpyro.infer.kernels.RBFKernel(), num_particles=100,
                          repulsion_temperature=0.001 * data[1].shape[0])
state, loss = svgd.run(rng_key, 10000, *data)  # rounds 10000


# %%
sample_frame = pd.DataFrame.from_dict(svgd.predict(state, *data))
g = sns.pairplot(sample_frame, corner=True, diag_kind='kde')
g.map_lower(sns.kdeplot, lw=2)


# %%
def transmap_guide(indices, observations):
    param_keys = {'prey0', 'preadtor0', 'r', 'k', 's', 'a', 'u', 'v'}
    params = OrderedDict()
    for k in param_keys:
        params[k] = numpyro.param(f'{k}$val', 25.)
    tmapp = numpyro.param('tmapp', np.zeros(8 * (8 + 1) // 2))
    tril_idx = np.tril_indices(8)
    tmap = jax.ops.index_update(np.zeros((8, 8)), tril_idx, tmapp)
    tloc = numpyro.param('tloc', np.zeros(8))
    tvals = jax.nn.softplus(tmap @ np.array(list(params.values())) + tloc)
    for pk, tval in zip(params.keys(), tvals):
        if pk == 'k':
            tval = tval
        numpyro.sample(pk, dist.Delta(tval))


# %%
svgd = numpyro.infer.SVGD(model, numpyro.guides.WrappedGuide(transmap_guide, init_strategy=init_with_noise(init_to_median())), 
                          numpyro.optim.Adam(0.01), numpyro.infer.ELBO(),
                          numpyro.infer.kernels.RBFKernel(), repulsion_temperature=0.01 * data[1].shape[0],
                          num_particles=100,
                          classic_guide_params_fn=lambda n: n in {'tmapp', 'tloc'})
state, loss = svgd.run(rng_key, 10000, *data)


# %%
sample_frame = pd.DataFrame(svgd.predict(state, *data))
g = sns.pairplot(sample_frame, corner=True, diag_kind='kde')
g.map_lower(sns.kdeplot, lw=2)

# %% [markdown]
# ### NUTS MCMC Sampling

# %%
mcmc = numpyro.infer.MCMC(numpyro.infer.NUTS(model), 100, 500, chain_method='vectorized')
mcmc.run(rng_key, *data)
mcmc.print_summary()


# %%
samples = mcmc.get_samples()
sample_frame = pd.DataFrame(samples)
g = sns.pairplot(sample_frame, corner=True, diag_kind='kde')
g.map_lower(sns.kdeplot, lw=2)


# %%


