import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, BarkerMH


rho = 0.97

def model():
    cov = jnp.array([[1.0, rho], [rho, 1.0]])
    x = numpyro.sample("x", dist.MultivariateNormal(jnp.zeros(2), covariance_matrix=cov))

dense_mass = 1
kernel = BarkerMH(model, dense_mass=dense_mass)
mcmc = MCMC(kernel, num_warmup=10000, num_samples=10000, progress_bar=False)
mcmc.run(jax.random.PRNGKey(0))
mcmc.print_summary()
print(mcmc.last_state.adapt_state.mass_matrix_sqrt)
if dense_mass:
    print(jnp.matmul(mcmc.last_state.adapt_state.mass_matrix_sqrt, jnp.transpose(mcmc.last_state.adapt_state.mass_matrix_sqrt)))
print("step size", mcmc.last_state.adapt_state.step_size)
