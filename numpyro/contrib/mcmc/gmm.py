from itertools import permutations

# import matplotlib.pyplot as plt
# import seaborn as sns
from scipy.stats import ks_2samp

from jax import disable_jit, random
import jax.numpy as jnp

import numpyro
from numpyro.diagnostics import print_summary
import numpyro.distributions as dist
from numpyro.contrib.indexing import Vindex
from numpyro.contrib.mcmc.mixed_hmc import MixedHMC
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.util import optional, control_flow_prims_disabled


def model(probs, mu_list):
    with numpyro.plate("D", 24):
        c = numpyro.sample("c", dist.Categorical(probs))
        numpyro.sample("x", dist.Normal(Vindex(mu_list)[..., c], jnp.sqrt(3)))


gibbs = False
if gibbs:
    max_times = [1, 0]  # [0, 1] or [1, 0]
    num_trajectories = 1
    thin = 20
    discrete_mass = 1
    progress_bar = False
    num_samples = int(2e6)
else:
    max_times = None
    num_trajectories = 80
    thin = 1
    discrete_mass = 0.1  # important
    progress_bar = False
    num_samples = int(1e4)
kernel = MixedHMC(NUTS(model), discrete_sites=["c"], num_trajectories=num_trajectories,
                  num_discrete_steps=None, max_times=max_times, discrete_mass=discrete_mass)
# kernel = NUTS(model, target_accept_prob=0.6)
mcmc = MCMC(kernel, int(1e4), num_samples, progress_bar=progress_bar)
probs = jnp.array([0.15, 0.3, 0.3, 0.25])
mu_list = jnp.array(list(permutations([-2, 0, 2, 4])))
with optional(__debug__, disable_jit()), optional(__debug__, control_flow_prims_disabled()):
    mcmc.run(random.PRNGKey(0), probs, mu_list)
samples = mcmc.get_samples()
samples = {k: v[::thin] for k, v in samples.items()}
print_summary(samples, group_by_chain=False)
actual_samples = Predictive(model, {}, num_samples=int(1e7), return_sites=["x"])(
    random.PRNGKey(1), probs, mu_list)["x"]
ks_stats = [ks_2samp(actual_samples[:, i], samples["x"][:, i]).statistic for i in range(24)]
print("max ks_stats:", max(ks_stats))
# sns.kdeplot(actual_samples, color="r")
# sns.kdeplot(samples["x"])
# plt.show()
