# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from jax import config
config.update("jax_debug_nans", True)

import numpyro
import numpyro.infer.kernels as kernels
from numpyro.callbacks import Progbar
from numpyro.distributions import NormalMixture
from numpyro.infer import ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.infer.initialization import init_with_noise, init_to_value
from numpyro.infer.stein import SVGD


# %%
rng_key = jax.random.PRNGKey(42)
num_iterations = 1500


# %%
def model():
    numpyro.sample('x', NormalMixture(jnp.array([1/3, 2/3]), 
                                      jnp.array([-2., 2.]), jnp.array([1., 1.])))


# %%
guide = AutoDelta(model, init_strategy=init_with_noise(init_to_value(values={'x': -10.}), noise_scale=1.0))
svgd = SVGD(model, guide, numpyro.optim.Adagrad(step_size=0.001), ELBO(),
            kernels.RBFKernel(), num_particles=100)
svgd_state = svgd.init(rng_key)


# %%
sns.kdeplot(svgd.get_params(svgd_state)['auto_x'])


# %%
svgd_state, loss = svgd.train(rng_key, num_iterations, callbacks=[Progbar()])


# %%
plt.clf()
sns.kdeplot(svgd.get_params(svgd_state)['auto_x'])


# %%
svgd.get_params(svgd_state)['auto_x']


# %%
guide = AutoDelta(model, init_strategy=init_with_noise(init_to_value(values={'x': -10.}), noise_scale=1.0))
svgd = SVGD(model, guide, numpyro.optim.Adagrad(step_size=1.0), ELBO(),
            kernels.LinearKernel(), num_particles=100)
svgd_state = svgd.init(rng_key)
svgd_state, loss = svgd.run(rng_key, num_iterations)


# %%
plt.clf()
sns.kdeplot(svgd.get_params(svgd_state)['auto_x'])


# %%
guide = AutoDelta(model, init_strategy=init_with_noise(init_to_value(values={'x': -10.}), noise_scale=1.0))
svgd = SVGD(model, guide, numpyro.optim.Adagrad(step_size=1.0), ELBO(),
            kernels.RandomFeatureKernel(), num_particles=100)
svgd_state = svgd.init(rng_key)
svgd_state, loss = svgd.run(rng_key, num_iterations * 2)


# %%
plt.clf()
sns.kdeplot(svgd.get_params(svgd_state)['auto_x'])


# %%
guide = AutoDelta(model, init_strategy=init_with_noise(init_to_value(values={'x': -10.}), noise_scale=1.0))
svgd = SVGD(model, guide, numpyro.optim.Adagrad(step_size=1.0), ELBO(),
            kernels.IMQKernel(), num_particles=100)
svgd_state = svgd.init(rng_key)
svgd_state, loss = svgd.run(rng_key, num_iterations)


# %%
plt.clf()
sns.kdeplot(svgd.get_params(svgd_state)['auto_x'])


# %%
guide = AutoDelta(model, init_strategy=init_with_noise(init_to_value(values={'x': -10.}), noise_scale=1.0))
svgd = SVGD(model, guide, numpyro.optim.Adagrad(step_size=1.0), ELBO(),
            kernels.MixtureKernel([0.5, 0.5], [kernels.LinearKernel(), kernels.RandomFeatureKernel()]),
            num_particles=100)
svgd_state = svgd.init(rng_key)
svgd_state, loss = svgd.run(rng_key, num_iterations)


# %%
plt.clf()
sns.kdeplot(svgd.get_params(svgd_state)['auto_x'])


# %%


