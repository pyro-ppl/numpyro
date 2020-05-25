# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from numpyro.distributions import constraints
import jax
import jax.numpy as np
from jax.experimental import stax
import numpyro.distributions as dist
import numpyro
from numpyro.distributions.util import logsumexp
from numpyro.infer.stein import SVGD
from numpyro.infer.kernels import RBFKernel
from numpyro.contrib.autoguide import AutoDelta
from numpyro.infer.util import init_with_noise, init_to_value
from numpyro.contrib.nn.auto_reg_nn import AutoregressiveNN
from numpyro.distributions.flows import InverseAutoregressiveTransform
from numpyro.distributions.transforms import (
    ComposeTransform,
    PermuteTransform
)
from numpyro.guides import WrappedGuide
from numpyro.optim import Adam
from numpyro.infer import ELBO
import seaborn as sns
import matplotlib.pyplot as plt


# %%
class DualMoonDistribution(dist.Distribution):
    support = constraints.real_vector

    def __init__(self):
        super(DualMoonDistribution, self).__init__(event_shape=(2,))

    def sample(self, key, sample_shape=()):
        # it is enough to return an arbitrary sample with correct shape
        return np.zeros(sample_shape + self.event_shape)

    def log_prob(self, x):
        term1 = 0.5 * ((np.linalg.norm(x, axis=-1) - 2) / 0.4) ** 2
        term2 = -0.5 * ((x[..., :1] + np.array([-2., 2.])) / 0.6) ** 2
        pe = term1 - logsumexp(term2, axis=-1)
        return -pe


def dual_moon_model():
    numpyro.sample('x', DualMoonDistribution())


# %%
rng_key = jax.random.PRNGKey(142)
guide = AutoDelta(dual_moon_model, init_strategy=init_with_noise(init_to_value({'x': np.array([0.,0.])}), noise_scale=1.0))

svgd = SVGD(dual_moon_model, guide, Adam(step_size=0.02), ELBO(),
            RBFKernel(), num_stein_particles=1000, num_loss_particles=1)
svgd_state = svgd.init(rng_key)


# %%
x1 = np.linspace(-3, 3, 100)
x2 = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(x1, x2)
P = np.exp(DualMoonDistribution().log_prob(np.stack([X1, X2], axis=-1)))

fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.contourf(X1, X2, P, cmap='OrRd', )
sns.kdeplot(svgd.get_params(svgd_state)['auto_x'][:, 0], svgd.get_params(svgd_state)['auto_x'][:, 1], n_levels=30, ax=ax)


# %%
rng_key = jax.random.PRNGKey(142)

num_iterations = 250
svgd_state, loss = svgd.run(rng_key, num_iterations)


# %%
fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.contourf(X1, X2, P, cmap='OrRd', )
sns.kdeplot(svgd.get_params(svgd_state)['auto_x'][:, 0], svgd.get_params(svgd_state)['auto_x'][:, 1], n_levels=30, ax=ax)


# %%
def neural_transport_guide(num_iaf=3, hidden_dims=[2,2]):
    flows = []
    for i in range(num_iaf):
        arn = AutoregressiveNN(2, hidden_dims, permutation=np.arange(2), nonlinearity=stax.Elu)
        arnn = numpyro.module(f'arn_{i}', arn, (2,))
        flows.append(InverseAutoregressiveTransform(arnn))
        if i < num_iaf - 1:
            flows.append(PermuteTransform(np.arange(2)[::-1]))
    particle = numpyro.param('particle', init_value=np.array([0,0]))
    numpyro.sample('x', dist.TransformedDistribution(dist.Delta(particle), flows))


# %%
rng_key = jax.random.PRNGKey(142)

svgd = SVGD(dual_moon_model, 
            WrappedGuide(neural_transport_guide,
                         lambda site: not site['name'].endswith("$params")),
            Adam(step_size=0.02), ELBO(), RBFKernel(), num_stein_particles=1000, num_loss_particles=1)
svgd_state = svgd.init(rng_key)


# %%


