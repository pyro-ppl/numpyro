import nest_asyncio
import stan

from jax.random import PRNGKey

import numpy as np

import numpyro
from numpyro.distributions import constraints, biject_to
from numpyro.infer import MCMC, NUTS

from age_model import model
from data import get_data, transform_data


nest_asyncio.apply()

numpyro.enable_x64()


data = get_data()
transformed_data = transform_data(data)
kernel = NUTS(model, step_size=0.02, max_tree_depth=15, target_accept_prob=0.95)

mcmc = MCMC(kernel, num_warmup=0, num_samples=1)
rng_key = PRNGKey(1)
mcmc.run(rng_key, transformed_data)
z = {k: v[0] for k, v in mcmc.get_samples().items()}


with open("stan_model_main.stan") as f:
    stan_code = f.read()
stan_model = stan.build(stan_code, data=data, random_seed=1)
stan_fit = stan_model.sample(num_chains=1, num_warmup=20, stepsize=1e-6, max_depth=5, num_samples=2)


params_support = {
  "R0": constraints.positive,
  "e_cases_N0": constraints.positive,
  "sd_dip_rnde": constraints.positive,
  "phi": constraints.positive,
  "log_ifr_age_base": constraints.less_than(0.),
  "hyper_log_ifr_age_rnde_mid1": constraints.positive,
  "hyper_log_ifr_age_rnde_mid2": constraints.positive,
  "hyper_log_ifr_age_rnde_old": constraints.positive,
  "log_ifr_age_rnde_mid1": constraints.positive,
  "log_ifr_age_rnde_mid2": constraints.positive,
  "log_ifr_age_rnde_old": constraints.positive,
  "upswing_timeeff_reduced": constraints.positive,
  "sd_upswing_timeeff_reduced": constraints.positive,
  "hyper_timeeff_shift_mid1": constraints.positive,
  "timeeff_shift_mid1": constraints.positive,
  "impact_intv_children_effect": constraints.interval(.1, 1.),
  "impact_intv_onlychildren_effect": constraints.positive
}


params = stan_fit.to_frame().to_dict()
z0 = {}
z1 = {}
for k in stan_fit.param_names:
    if k not in z:
        continue
    z0[k] = np.array([v[0] for k1, v in params.items() if k1.startswith(k)])
    z1[k] = np.array([v[1] for k1, v in params.items() if k1.startswith(k)])
    shape = z[k].shape
    if len(shape) == 2:
        z0[k] = z0[k].reshape((shape[1], shape[0])).T
        z1[k] = z1[k].reshape((shape[1], shape[0])).T
    elif len(shape) == 0:
        z0[k] = z0[k].reshape(())
        z1[k] = z1[k].reshape(())
    if k in params_support:
        z0[k] = biject_to(params_support[k]).inv(z0[k])
        z1[k] = biject_to(params_support[k]).inv(z1[k])


pe0_stan = -params["lp__"][0]
pe1_stan = -params["lp__"][1]
print("stan pe0 and pe1:", pe0_stan, pe1_stan)
print("stan pe diff:", pe1_stan - pe0_stan)

potential_fn = mcmc.sampler._potential_fn_gen(transformed_data)
pe0_numpyro = potential_fn(z0)
pe1_numpyro = potential_fn(z1)
print("numpyro pe0 and pe1:", pe0_numpyro, pe1_numpyro)
print("numpyro pe diff:", pe1_numpyro - pe0_numpyro)

print("delta(pe0): %.3e" % (pe0_numpyro - pe0_stan))
print("delta(pe1): %.3e" % (pe1_numpyro - pe1_stan))


# pe_and_grad = jax.jit(jax.value_and_grad(potential_fn))
# print(pe_and_grad(z1)[0])
# a, b = pe_and_grad(z0)
# print(a)
