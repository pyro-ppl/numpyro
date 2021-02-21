import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, BarkerMH


rho = 0.90

def model():
    cov = jnp.array([[10.0, rho], [rho, 0.1]])
    x = numpyro.sample("x", dist.MultivariateNormal(jnp.zeros(2), covariance_matrix=cov))

dense_mass = 0
kernel = BarkerMH(model, dense_mass=dense_mass)
mcmc = MCMC(kernel, num_warmup=10 * 10000, num_samples=10 * 10000, progress_bar=False)
mcmc.run(jax.random.PRNGKey(0))
mcmc.print_summary()
print(mcmc.last_state.adapt_state.mass_matrix_sqrt)
if dense_mass:
    print(jnp.matmul(mcmc.last_state.adapt_state.mass_matrix_sqrt, jnp.transpose(mcmc.last_state.adapt_state.mass_matrix_sqrt)))
    print(jnp.linalg.inv(jnp.matmul(mcmc.last_state.adapt_state.mass_matrix_sqrt, jnp.transpose(mcmc.last_state.adapt_state.mass_matrix_sqrt))))
print("step size", mcmc.last_state.adapt_state.step_size)
